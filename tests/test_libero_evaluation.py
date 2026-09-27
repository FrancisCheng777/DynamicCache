import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from evaluation.libero.compare_c3ache import compare_runs, paired_success
from evaluation.libero.evaluate_c3ache import client_runtime, rollout
from evaluation.libero.result_io import episode_path, load_run, write_json


def test_small_perfect_agreement_does_not_establish_one_percentage_point_margin():
    result = paired_success([True] * 20, [True] * 20)
    assert result["point_estimate_within_margin"]
    assert result["noninferiority"] == "not_established"
    assert result["interval_95_pp"][0] < -1


def test_metadata_supports_official_libero_namespace_package(tmp_path):
    # Official LIBERO's outer package has no __init__.py, hence __file__ is None.
    package = SimpleNamespace(__file__=None, __path__=[str(tmp_path)])
    assert client_runtime(package)["libero_code"]["commit"] == "unknown"


def test_paired_success_counts_losses_in_percentage_points():
    result = paired_success([True] * 500, [False] * 10 + [True] * 490)
    assert result["difference_pp"] == -2
    assert result["losses"] == 10
    assert result["gains"] == 0
    assert not result["point_estimate_within_margin"]


def make_run(path, enabled):
    manifest = {
        "schema_version": 1, "complete": True,
        "protocol": {"task_ids": [0], "episodes_per_task": 2, "base_seed": 3},
        "client_runtime": {"version": "test"},
        "server": {"upstream_commit": "test", "code": "test", "checkpoint": "same",
                   "native": {"steps": 50}, "runtime": "test", "profile_inference": True,
                   "c3ache": {"enabled": enabled}},
    }
    write_json(path / "manifest.json", manifest)
    for episode in range(2):
        write_json(episode_path(path, 0, episode), {
            "task_id": 0, "episode_id": episode, "seed": 3 + episode,
            "initial_state_index": episode, "status": "ok", "success": True,
            "episode_wall_ms": 100,
            "chunks": [{"chunk_index": 0, "infer_rpc_ms": 10, "policy_cycle_ms": 10,
                        "c3ache": {"reused_calls": int(enabled)}}],
        })
    return manifest


@pytest.mark.parametrize("fault", ["missing", "extra", "error", "incomplete", "wrong_seed"])
def test_partial_or_invalid_results_cannot_be_reported_as_complete(tmp_path, fault):
    manifest = make_run(tmp_path, False)
    path = episode_path(tmp_path, 0, 0)
    record = json.loads(path.read_text())
    if fault == "missing":
        path.unlink()
    elif fault == "extra":
        write_json(episode_path(tmp_path, 0, 8), record)
    elif fault == "incomplete":
        manifest["complete"] = False
        write_json(tmp_path / "manifest.json", manifest)
    else:
        record["status" if fault == "error" else "seed"] = "error" if fault == "error" else 88
        write_json(path, record)
    with pytest.raises(ValueError):
        load_run(tmp_path)


def test_comparison_rejects_changed_checkpoint_or_native_settings(tmp_path):
    make_run(tmp_path / "a", False)
    manifest = make_run(tmp_path / "b", True)
    assert compare_runs(tmp_path / "a", tmp_path / "b")["success"]["episodes"] == 2
    for key in ["checkpoint", "native", "runtime"]:
        changed = copy.deepcopy(manifest)
        changed["server"][key] = "changed"
        write_json(tmp_path / "b" / "manifest.json", changed)
        with pytest.raises(ValueError, match=key):
            compare_runs(tmp_path / "a", tmp_path / "b")


def test_shadow_measurements_cannot_be_used_as_cached_speedup_results(tmp_path):
    make_run(tmp_path / "a", False)
    manifest = make_run(tmp_path / "b", True)
    manifest["server"]["execution_mode"] = "shadow"
    write_json(tmp_path / "b" / "manifest.json", manifest)
    with pytest.raises(ValueError, match="[Dd]iagnostic|[Ss]hadow"):
        compare_runs(tmp_path / "a", tmp_path / "b")


class Environment:
    """LIBERO is an external boundary; use a deterministic environment contract."""
    def __init__(self, success_at=None):
        self.env = SimpleNamespace(timestep=0)
        self.success_at = success_at
        self.actions = []

    def seed(self, value):
        self.seed_value = value

    def reset(self):
        self.env.timestep = 0

    def set_init_state(self, state):
        self.state = state

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.env.timestep += 1
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        return {"agentview_image": image, "robot0_eye_in_hand_image": image}, 0, self.env.timestep == self.success_at, {}


class Policy:
    def __init__(self):
        self.calls = []
        self.action = np.arange(7 * 4 * 4, dtype=np.float32).reshape(7, 4, 4)

    def infer(self, obs):
        self.calls.append(obs)
        if obs.get("reset"):
            return {"seed": obs["seed"]}
        if obs.get("compute_kv_cache"):
            return {"timings_ms": {"history_ms": 1}}
        return {"action": self.action, "c3ache": {"reused_calls": 0}, "timings_ms": {"action_loop_ms": 1}}


def test_rollout_preserves_cold_action_skip_feedback_and_chunk_boundary_limit():
    env, policy = Environment(), Policy()
    result = rollout(policy, env, np.array([2]), "test prompt", 42, max_env_steps=20)
    assert env.seed_value == 42
    assert result["env_steps"] == 5 + 12 + 16  # Native check is once per chunk.
    assert not result["success"]
    assert len(result["chunks"]) == 2
    np.testing.assert_array_equal(env.actions[5], policy.action[:, 1, 0])
    feedback = [call for call in policy.calls if call.get("compute_kv_cache")]
    assert [len(call["obs"]) for call in feedback] == [12, 16]
    assert all(call["state"] is policy.action for call in feedback)
    first_image = policy.calls[1]["obs"]["observation.images.agentview_rgb"]
    np.testing.assert_array_equal(first_image[0], np.arange(12, dtype=np.uint8).reshape(2, 2, 3)[-1])


def test_terminal_chunk_has_no_feedback_and_success_is_recorded():
    env, policy = Environment(success_at=7), Policy()
    result = rollout(policy, env, np.array([2]), "test prompt", 42, max_env_steps=20)
    assert result["success"]
    assert result["env_steps"] == 7
    assert not any(call.get("compute_kv_cache") for call in policy.calls)
    assert "history_rpc_ms" not in result["chunks"][0]
