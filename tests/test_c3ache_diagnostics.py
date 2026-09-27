import json

import pytest
import torch

from wan_va.modules.c3ache import C3acheConfig
from wan_va.modules.c3ache_diagnostics import C3acheDiagnostics


SCHEDULE = ((1000.0, 1.0), (200.0, 0.2), (1.0, 0.001))


@torch.no_grad()
def test_shadow_executes_each_stack_once_and_returns_the_real_output():
    diagnostic = C3acheDiagnostics(C3acheConfig(True, 1, 1, 0))
    calls = []
    outputs = []

    def blocks(x):
        calls.append(x.clone())
        result = x.square()
        outputs.append(result)
        return result

    for frame in [4, 8, 12]:
        diagnostic.begin_chunk(frame, SCHEDULE)
        x = torch.full((2, 2, 3), float(frame))
        diagnostic.run(x, lambda value: value, step=0)
        result = diagnostic.run(x, blocks, step=1)
        assert result is outputs[-1]
    assert len(calls) == 3


@torch.no_grad()
def test_shadow_reports_native_step_weighting_selected_channels_and_frozen_reference():
    diagnostic = C3acheDiagnostics(C3acheConfig(True, 1, 1, 0),
                                  action_channels=[0, 2], action_scales=[2.0, 4.0])
    x = torch.ones(2, 2, 3)
    for frame in [0, 4, 8, 12]:
        diagnostic.begin_chunk(frame, SCHEDULE)
        diagnostic.run(x, lambda value: value, step=0)
        increment = 1 if frame <= 4 else 3
        full = diagnostic.run(x, lambda value: value + increment, step=1)
        diagnostic.observe_output(full, lambda hidden: hidden)
        stats = diagnostic.stats()
        assert stats["reused_calls"] == 0
        assert stats["full_calls"] == 2
        if frame == 0:
            assert not stats["diagnostics"]["steps"]
        elif frame > 4:
            row = stats["diagnostics"]["steps"][0]
            assert row["reference_frame_id"] == 4
            assert row["would_reuse"]
            assert row["update_rmse"] == pytest.approx(0.398)
            assert row["scaled_update_channel_rmse"] == pytest.approx([0.796, 1.592])
            assert row["residual"]["relative_l2"] == pytest.approx(2 / 3)
            assert row["same_state_roundtrip_velocity"]["max_abs"] == 0
    diagnostic.reset()
    assert not diagnostic.stats()["diagnostics"]["steps"]
    assert not diagnostic.stats()["reference_frames"]


@torch.no_grad()
@pytest.mark.parametrize("guidance,expected_error", [(1, 2.0), (5, 6.0)])
def test_shadow_error_uses_the_executed_cfg_branch(guidance, expected_error):
    diagnostic = C3acheDiagnostics(C3acheConfig(True, 1, 1, 0), guidance_scale=guidance)
    x = torch.zeros(2, 1, 1)
    for frame in [4, 8]:
        diagnostic.begin_chunk(frame, SCHEDULE)
        diagnostic.run(x, lambda value: value, step=0)
        delta = torch.tensor([[[1.0]], [[1.0]]]) if frame == 4 else torch.tensor([[[3.0]], [[5.0]]])
        full = diagnostic.run(x, lambda value: value + delta, step=1)
        diagnostic.observe_output(full, lambda hidden: hidden)
    # CFG=5: old velocity=1; current=5+5*(3-5)=-5, difference=6.
    assert diagnostic.stats()["diagnostics"]["steps"][0]["velocity"]["max_abs"] == expected_error


@torch.no_grad()
def test_bf16_roundtrip_error_is_separate_from_cross_chunk_error():
    diagnostic = C3acheDiagnostics(C3acheConfig(True, 1, 1, 0))
    x = torch.ones(1, 1, 1, dtype=torch.bfloat16)
    diagnostic.begin_chunk(4, SCHEDULE)
    diagnostic.run(x, lambda value: value, step=0)
    full = diagnostic.run(x, lambda value: torch.full_like(value, 0.001), step=1)
    diagnostic.observe_output(full, lambda hidden: hidden)
    row = diagnostic.stats()["diagnostics"]["steps"][0]
    assert not row["would_reuse"]
    assert row["same_state_roundtrip_velocity"]["max_abs"] > 0
    assert "velocity" not in row


@torch.no_grad()
def test_nonfinite_shadow_is_reported_without_corrupting_full_output_or_json():
    diagnostic = C3acheDiagnostics(C3acheConfig(True, 1, 1, 0))
    x = torch.zeros(1, 1, 1)
    for frame in [4, 8]:
        diagnostic.begin_chunk(frame, SCHEDULE)
        diagnostic.run(x, lambda value: value, step=0)
        full = diagnostic.run(x, lambda value: value + frame, step=1)
        diagnostic.observe_output(full, lambda hidden: hidden / 0)
    assert torch.isfinite(full).all()
    stats = diagnostic.stats()
    assert not stats["diagnostics"]["steps"][0]["velocity"]["finite"]
    json.dumps(stats, allow_nan=False)
