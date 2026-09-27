import json
import csv
import subprocess
import sys

import pytest

from evaluation.libero.evaluate_c3ache import validate_server_mode
from evaluation.libero.result_io import episode_path, write_json
from evaluation.libero.summarize_c3ache_diagnostics import summarize


def diagnostic_run(root, *, mode="shadow"):
    metadata = {
        "schema_version": 1, "complete": True,
        "protocol": {"task_ids": [2], "episodes_per_task": 2, "base_seed": 0},
        "client_runtime": {"test": True},
        "server": {"execution_mode": mode, "c3ache": {"enabled": mode != "baseline"},
                   "native": {}, "upstream_commit": "test", "code": {}, "checkpoint": {}, "runtime": {}},
    }
    write_json(root / "manifest.json", metadata)
    for episode, error in enumerate([1.0, 3.0]):
        row = {"step": 5, "sigma": 0.3, "delta_sigma": -0.1, "would_reuse": True,
               "reference_frame_id": 4, "frame_id": 8,
               "velocity": {"finite": True, "relative_l2": error / 2, "rmse": error * 10, "max_abs": error * 20},
               "residual": {"relative_l2": error / 3}, "hidden": {"relative_l2": error / 4},
               "same_state_roundtrip_velocity": {"rmse": 0.001}, "update_rmse": error,
               "scaled_update_rmse": error * 2, "scaled_update_channel_rmse": [error, error * 2]}
        write_json(episode_path(root, 2, episode), {
            "task_id": 2, "episode_id": episode, "seed": 200000 + episode,
            "initial_state_index": episode, "status": "ok", "success": True, "env_steps": 30,
            "chunks": [{"chunk_index": 0, "action_sha256": f"action-{episode}", "action": [1.0, 2.0],
                        "c3ache": {"execution_mode": mode, "reused_calls": 0, "hypothetical_reused_calls": 1,
                                   "diagnostics": {"schema_version": 1, "action_channels": [0, 6],
                                                   "action_scales": [1.0, 1.0], "guidance_scale": 1.0,
                                                   "steps": [row]}}}],
        })
    return metadata


def test_report_groups_by_sigma_and_records_trace_agreement(tmp_path):
    diagnostic_run(tmp_path / "shadow")
    diagnostic_run(tmp_path / "baseline", mode="baseline")
    report = summarize(tmp_path / "shadow", {"baseline": tmp_path / "baseline"})
    assert report["hypothetical_reuse_samples"] == 2
    row = report["by_step"][0]
    assert row["step"] == 5
    assert row["update_rmse"]["mean"] == 2.0
    assert row["update_rmse"]["p95"] == 3.0
    assert row["scaled_update_channel_rmse"][1]["mean"] == 4.0
    assert report["trace_comparisons"]["baseline"]["all_exact"]


def test_report_flags_nonfinite_approximations_and_first_trace_difference(tmp_path):
    diagnostic_run(tmp_path / "shadow")
    diagnostic_run(tmp_path / "baseline", mode="baseline")
    location = episode_path(tmp_path / "shadow", 2, 1)
    record = json.loads(location.read_text())
    record["chunks"][0]["action_sha256"] = "changed"
    record["chunks"][0]["action"] = [1.0, 2.5]
    record["chunks"][0]["c3ache"]["diagnostics"]["steps"][0]["velocity"]["finite"] = False
    write_json(location, record)
    report = summarize(tmp_path / "shadow", {"baseline": tmp_path / "baseline"})
    assert report["nonfinite_reuse_samples"] == 1
    comparison = report["trace_comparisons"]["baseline"]
    assert not comparison["all_exact"]
    assert comparison["episodes"][1]["first_different_chunk"] == 0
    assert comparison["episodes"][1]["first_difference_action_max_abs"] == 0.5


def test_report_rejects_missing_observations_and_non_shadow_input(tmp_path):
    manifest = diagnostic_run(tmp_path)
    path = episode_path(tmp_path, 2, 0)
    record = json.loads(path.read_text())
    record["chunks"][0]["c3ache"]["diagnostics"]["steps"] = []
    write_json(path, record)
    with pytest.raises(ValueError, match="[Oo]bservation|[Cc]ount|[Ss]ample"):
        summarize(tmp_path)
    manifest["server"]["execution_mode"] = "cached"
    write_json(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="shadow"):
        summarize(tmp_path)


def test_client_requires_explicit_shadow_mode():
    metadata = {"execution_mode": "shadow", "c3ache": {"enabled": True}}
    validate_server_mode(metadata, "shadow")
    for mode in ["baseline", "cached"]:
        with pytest.raises(ValueError, match="expected-mode"):
            validate_server_mode(metadata, mode)


def test_report_command_writes_json_and_csv(tmp_path):
    diagnostic_run(tmp_path / "shadow")
    diagnostic_run(tmp_path / "baseline", mode="baseline")
    output = tmp_path / "summary" / "diagnostics.json"
    result = subprocess.run([sys.executable, "-m", "evaluation.libero.summarize_c3ache_diagnostics",
                             "--shadow", str(tmp_path / "shadow"), "--baseline", str(tmp_path / "baseline"),
                             "--output", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["hypothetical_reuse_samples"] == 2
    with output.with_suffix(".csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["action_6_scaled_update_rmse_mean"] == "4.0"
