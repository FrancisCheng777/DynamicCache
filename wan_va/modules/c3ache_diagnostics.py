"""Measure counterfactual cache errors while executing the complete policy.

Each native stack executes exactly once. Only pure output-head computations are
repeated; no second attention/KV forward, random draw, or approximate action is
introduced. Timings from this mode are diagnostics, never speed benchmarks.
"""
import math

import torch

from .c3ache import C3ache


def tensor_error(estimate, target):
    estimate, target = estimate.float(), target.float()
    difference = estimate - target
    target_norm = torch.linalg.vector_norm(target)
    estimate_norm = torch.linalg.vector_norm(estimate)
    denominator = target_norm * estimate_norm
    return {
        "finite": torch.isfinite(estimate).all() & torch.isfinite(target).all(),
        "target_l2": target_norm,
        "estimate_l2": estimate_norm,
        "relative_l2": torch.linalg.vector_norm(difference) / target_norm.clamp_min(1e-12),
        "rmse": difference.square().mean().sqrt(),
        "max_abs": difference.abs().max(),
        "cosine": (estimate * target).sum() / denominator.clamp_min(1e-12),
        "cosine_defined": denominator > 0,
    }


def serializable(value):
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(item) for item in value]
    # Keep failed approximations visible via their finite flag, without writing
    # nonstandard JSON NaN/Infinity or preventing the full policy from finishing.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class C3acheDiagnostics(C3ache):
    def __init__(self, config, *, action_channels=None, action_scales=None, guidance_scale=1.0):
        if not config.enabled:
            raise ValueError("Diagnostics require an enabled hypothetical cache")
        self.action_channels = tuple(action_channels) if action_channels is not None else None
        self.action_scales = tuple(action_scales) if action_scales is not None else None
        self.guidance_scale = float(guidance_scale)
        if self.action_channels is not None:
            if not self.action_channels or len(set(self.action_channels)) != len(self.action_channels) or min(self.action_channels) < 0:
                raise ValueError("Action channels must be unique, nonnegative and nonempty")
        if self.action_scales is not None:
            if self.action_channels is None or len(self.action_scales) != len(self.action_channels):
                raise ValueError("Provide one scale for each selected action channel")
            if any(not math.isfinite(scale) or scale <= 0 for scale in self.action_scales):
                raise ValueError("Action scales must be positive and finite")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale < 0:
            raise ValueError("Guidance scale must be finite and nonnegative")
        super().__init__(config)

    def reset(self):
        super().reset()
        self._rows = []
        self._pending = None

    def begin_chunk(self, frame_id, schedule, context=()):
        super().begin_chunk(frame_id, schedule, context)
        self._rows = []
        self._pending = None

    def run(self, hidden_states, compute_blocks, *, step, update_cache=0):
        self._pending = None
        original = hidden_states.detach().clone()
        reference = self._residuals.get(step)
        hits_before = self._counts["reused"]
        # The only execution of attention/FFN. Its original output drives policy.
        full = compute_blocks(hidden_states)
        approximate = super().run(original, lambda _: full, step=step, update_cache=update_cache)
        would_reuse = self._counts["reused"] > hits_before
        if (not torch.is_grad_enabled() and update_cache == 0 and not self._cold
                and step is not None and 0 < step < len(self._schedule)
                and self.config.start_step <= step <= self.config.end_step):
            self._pending = (original, full, approximate, reference if would_reuse else None, step)
        return full

    def _policy_velocity(self, prediction):
        if prediction.shape[0] == 2:
            # Exact native positive-then-negative branch ordering and arithmetic.
            prediction = (prediction[1:] + self.guidance_scale * (prediction[:1] - prediction[1:])
                          if self.guidance_scale > 1 else prediction[:1])
        elif prediction.shape[0] != 1 or self.guidance_scale > 1:
            raise ValueError("Diagnostics expect one policy batch, with two branches for CFG")
        if self.action_channels is not None:
            prediction = prediction[..., list(self.action_channels)]
        return prediction.float()

    def observe_output(self, full_prediction, output_head):
        pending, self._pending = self._pending, None
        if pending is None:
            return
        original, full, approximate, reference, step = pending
        timestep, sigma = self._schedule[step]
        next_sigma = self._schedule[step + 1][1] if step + 1 < len(self._schedule) else 0.0
        delta = next_sigma - sigma
        velocity = self._policy_velocity(full_prediction)
        # Same-state BF16 subtract/add isolates arithmetic reconstruction error
        # from the extra error of moving a reference between chunks.
        roundtrip = original + (full.detach() - original)
        roundtrip_velocity = self._policy_velocity(output_head(roundtrip))
        row = {
            "step": step, "timestep": float(timestep), "sigma": float(sigma),
            "delta_sigma": float(delta), "frame_id": self._frame_id,
            "would_reuse": reference is not None,
            "reference_frame_id": reference.frame_id if reference is not None else self._frame_id,
            "same_state_roundtrip_velocity": tensor_error(roundtrip_velocity, velocity),
        }
        if reference is not None:
            estimated_velocity = self._policy_velocity(output_head(approximate))
            update_error = (estimated_velocity - velocity) * delta
            scales = (torch.tensor(self.action_scales, device=velocity.device, dtype=torch.float32)
                      if self.action_scales is not None else torch.ones(velocity.shape[-1], device=velocity.device))
            scaled_error = update_error * scales
            row.update(
                residual=tensor_error(reference.value, full.float() - original.float()),
                hidden=tensor_error(approximate, full),
                velocity=tensor_error(estimated_velocity, velocity),
                update_rmse=update_error.square().mean().sqrt(),
                scaled_update_rmse=scaled_error.square().mean().sqrt(),
                scaled_update_channel_rmse=scaled_error.square().mean(dim=(0, 1)).sqrt(),
                scaled_update_channel_max_abs=scaled_error.abs().amax(dim=(0, 1)),
            )
        self._rows.append(row)

    def stats(self):
        result = super().stats()
        hypothetical_hits = result["reused_calls"]
        if hypothetical_hits:
            result["full_reasons"]["shadow_full_on_hypothetical_hit"] = hypothetical_hits
        result.update(
            execution_mode="shadow", reused_calls=0,
            full_calls=result["full_calls"] + hypothetical_hits,
            hypothetical_reused_calls=hypothetical_hits,
            timing_is_benchmark=False,
            diagnostics={
                "schema_version": 1, "trajectory": "full_policy",
                "action_channels": self.action_channels,
                "action_scales": self.action_scales,
                "guidance_scale": self.guidance_scale,
                "error_scope": "one_step_on_full_policy_state_not_final_action_error",
                "steps": serializable(self._rows),
            },
        )
        return result
