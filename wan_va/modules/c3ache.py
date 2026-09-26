"""Training-free C³ache: reuse h_L - h_0 at matched denoising steps.

This cache is independent of attention KV caches and is never a model parameter.
Call ``begin_chunk`` once per action chunk, then ``run`` in denoising order.
Cold chunks, step zero, and all KV commits always execute the complete stack.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Hashable, Sequence

import torch


@dataclass(frozen=True)
class C3acheConfig:
    enabled: bool = False
    start_step: int = 5
    end_step: int = 39
    refresh_interval: int = 2

    def __post_init__(self):
        if self.start_step < 0 or self.end_step < self.start_step:
            raise ValueError("Require 0 <= start_step <= end_step (inclusive)")
        if self.refresh_interval < 0:
            raise ValueError("refresh_interval must be nonnegative; 0 means no periodic refresh")


@dataclass
class _Residual:
    value: torch.Tensor
    frame_id: int


class C3ache:
    def __init__(self, config=C3acheConfig()):
        self.config = config
        self.reset()

    def reset(self):
        """Clear references on episode/prompt/model changes, including all GPU tensors."""
        self._residuals = {}
        self._signature = None
        self._frame_id = None
        self._regular_index = -1
        self._cold = True
        self._refresh = True
        self._step_zero_seen = False
        self._schedule = ()
        self._counts = Counter()

    def begin_chunk(
        self, frame_id: int, schedule: Sequence[tuple[float, float]], context: Hashable = ()
    ):
        """Match the complete native (timestep, sigma) schedule and CFG/layout context.

        ``context`` must be immutable and identify the checkpoint, action layout,
        CFG branch ordering/scales, and KV namespace. Do NOT include the current
        observation or absolute frame position: reuse is deliberately cross-chunk.
        The special first chunk has a pinned action frame and cannot be a reference.
        """
        schedule = tuple(tuple(pair) for pair in schedule)
        if not schedule or any(len(pair) != 2 for pair in schedule):
            raise ValueError("schedule must contain (timestep, sigma) pairs")
        signature = (schedule, context)
        hash(signature)  # Fail early for mutable context rather than comparing tensors.
        discontinuity = self._frame_id is not None and frame_id <= self._frame_id
        if signature != self._signature or discontinuity or frame_id == 0:
            self.reset()
        self._signature = signature
        self._schedule = schedule
        self._frame_id = frame_id
        self._cold = frame_id == 0
        if not self._cold:
            self._regular_index += 1
        interval = self.config.refresh_interval
        self._refresh = self._regular_index == 0 or (
            interval > 0 and self._regular_index % interval == 0
        )
        self._step_zero_seen = False
        self._counts = Counter()
        if self._refresh:
            self._residuals.clear()

    def run(
        self,
        hidden_states: torch.Tensor,
        compute_blocks: Callable[[torch.Tensor], torch.Tensor],
        *,
        step: int | None,
        update_cache: int = 0,
    ) -> torch.Tensor:
        """Return the unchanged full output, or current h_0 plus an earlier residual.

        No reused output is ever written back into the residual cache. ``step=None``
        is the appended zero-noise commit forward, not a sampler step.
        """
        reason = None
        if not self.config.enabled:
            reason = "disabled"
        elif torch.is_grad_enabled():
            reason = "grad_enabled"
        elif update_cache != 0:
            reason = "kv_commit"
        elif self._frame_id is None:
            reason = "no_chunk"
        elif self._cold:
            reason = "cold_chunk"
        elif step is None or not 0 <= step < len(self._schedule):
            reason = "not_sampler_step"
        elif step == 0:
            reason = "first_step"
        elif not self._step_zero_seen:
            reason = "first_step_not_run"
        elif not self.config.start_step <= step <= self.config.end_step:
            reason = "outside_window"

        eligible = reason is None
        entry = self._residuals.get(step) if eligible else None
        if eligible and not self._refresh and entry is not None:
            residual = entry.value
            if (residual.shape == hidden_states.shape
                    and residual.dtype == hidden_states.dtype
                    and residual.device == hidden_states.device):
                self._counts["reused"] += 1
                return hidden_states + residual
            reason = "tensor_mismatch"
        elif eligible:
            reason = "refresh" if self._refresh else "miss"

        # Clone before executing the callable so in-place implementations cannot
        # corrupt h_0. Keep its dtype; never reconstruct a full-path output from R.
        original = hidden_states.detach().clone() if eligible else None
        output = compute_blocks(hidden_states)
        self._counts["full"] += 1
        self._counts[reason] += 1
        if step == 0 and update_cache == 0 and not torch.is_grad_enabled():
            self._step_zero_seen = True
        if eligible:
            self._residuals[step] = _Residual(output.detach() - original, self._frame_id)
            self._counts["stored"] += 1
        return output

    def stats(self) -> dict:
        """Small JSON-serializable per-chunk counters; no tensor synchronization."""
        return {
            "enabled": self.config.enabled,
            "frame_id": self._frame_id,
            "cold": self._cold,
            "refresh": self._refresh,
            "full_calls": self._counts["full"],
            "reused_calls": self._counts["reused"],
            "stored_residuals": self._counts["stored"],
            "full_reasons": {k: v for k, v in self._counts.items()
                             if k not in {"full", "reused", "stored"}},
            "reference_frames": {str(k): v.frame_id for k, v in self._residuals.items()},
            "cache_bytes": sum(v.value.numel() * v.value.element_size()
                               for v in self._residuals.values()),
        }
