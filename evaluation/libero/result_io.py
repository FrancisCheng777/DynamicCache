"""Atomic, resumable episode records and strict completeness checks (stdlib only)."""
import json
import os
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def expected_keys(protocol):
    return [(task, episode) for task in protocol["task_ids"]
            for episode in range(protocol["episodes_per_task"])]


def episode_seed(base_seed, task, episode):
    if not (0 <= episode < 100_000 and task >= 0):
        raise ValueError("Episode index must be in [0, 100000); task index must be nonnegative")
    seed = base_seed + task * 100_000 + episode
    if not 0 <= seed < 2**32:
        raise ValueError("Derived episode seed must be in [0, 2**32)")
    return seed


def episode_path(root, task, episode):
    return Path(root) / "episodes" / f"{task:03d}_{episode:05d}.json"


def validate_record(record, task, episode, protocol):
    if (record["task_id"], record["episode_id"]) != (task, episode):
        raise ValueError("Episode file has the wrong task/episode key")
    if record["seed"] != episode_seed(protocol["base_seed"], task, episode):
        raise ValueError("Episode seed differs from the manifest")
    if record["status"] != "ok" or type(record.get("success")) is not bool:
        raise ValueError(f"Episode {task}/{episode} did not finish successfully as a measurement")


def load_run(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError(f"Incomplete run: {root}")
    keys = expected_keys(manifest["protocol"])
    expected_paths = {episode_path(root, *key) for key in keys}
    actual_paths = set((root / "episodes").glob("*.json"))
    if actual_paths != expected_paths:
        raise ValueError(f"Missing or unexpected episode records in {root}")
    records = {}
    for key in keys:
        record = json.loads(episode_path(root, *key).read_text())
        validate_record(record, *key, manifest["protocol"])
        records[key] = record
    if not records:
        raise ValueError("No episodes in run")
    return manifest, records
