"""Summarize baseline-driven residual errors; this is not a speed benchmark."""
import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path

from .result_io import load_run, write_json


def distribution(values):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {"count": len(values), "mean": sum(values) / len(values),
            "p95": values[max(0, math.ceil(0.95 * len(values)) - 1)], "max": values[-1]}


def by_step(rows, channel_count):
    groups = defaultdict(list)
    for row in rows:
        groups[row["step"]].append(row)
    output = []
    metrics = ["update_rmse", "scaled_update_rmse", "velocity.relative_l2", "velocity.rmse",
               "velocity.max_abs", "residual.relative_l2", "hidden.relative_l2"]
    for step, group in sorted(groups.items()):
        if len({(row["sigma"], row["delta_sigma"]) for row in group}) != 1:
            raise ValueError("A sampler step has inconsistent sigma/delta values")
        samples = [row for row in group if row["would_reuse"]]
        result = {"step": step, "sigma": group[0]["sigma"], "delta_sigma": group[0]["delta_sigma"],
                  "samples": len(samples), "nonfinite_samples": sum(not row["velocity"]["finite"] for row in samples),
                  "same_state_roundtrip_rmse": distribution([row["same_state_roundtrip_velocity"]["rmse"] for row in group])}
        for key in metrics:
            def get(row):
                value = row
                for part in key.split("."):
                    value = value.get(part) if isinstance(value, dict) else None
                return value
            result[key.replace(".", "_")] = distribution([get(row) for row in samples])
        result["scaled_update_channel_rmse"] = [
            distribution([row["scaled_update_channel_rmse"][channel] for row in samples])
            for channel in range(channel_count)]
        output.append(result)
    return output


def flatten(value):
    if isinstance(value, list):
        return [item for child in value for item in flatten(child)]
    return [value]


def compare_traces(reference_manifest, reference, manifest, observed):
    for key in ["protocol", "client_runtime"]:
        if reference_manifest[key] != manifest[key]:
            raise ValueError(f"Trace comparison has different {key}")
    for key in ["code", "checkpoint", "native", "runtime", "upstream_commit"]:
        if reference_manifest["server"][key] != manifest["server"][key]:
            raise ValueError(f"Trace comparison has different server {key}")
    if reference.keys() != observed.keys():
        raise ValueError("Trace comparison requires identical episode keys")
    episodes = []
    for key in sorted(reference):
        a, b = reference[key], observed[key]
        if a["initial_state_index"] != b["initial_state_index"]:
            raise ValueError("Trace comparison has different initial states")
        first, max_abs = None, None
        for index, (left, right) in enumerate(zip(a["chunks"], b["chunks"])):
            if not left.get("action_sha256") or not right.get("action_sha256"):
                raise ValueError("Trace comparison requires recorded action hashes")
            if left["action_sha256"] != right["action_sha256"]:
                first = index
                if "action" in left and "action" in right:
                    x, y = flatten(left["action"]), flatten(right["action"])
                    if len(x) != len(y):
                        raise ValueError("Action trace shapes differ")
                    max_abs = max(abs(i - j) for i, j in zip(x, y))
                break
        if first is None and len(a["chunks"]) != len(b["chunks"]):
            first = min(len(a["chunks"]), len(b["chunks"]))
        exact = first is None and a["success"] == b["success"] and a["env_steps"] == b["env_steps"]
        episodes.append({"task_id": key[0], "episode_id": key[1], "exact": exact,
                         "first_different_chunk": first, "first_difference_action_max_abs": max_abs,
                         "reference_success": a["success"], "observed_success": b["success"],
                         "reference_env_steps": a["env_steps"], "observed_env_steps": b["env_steps"]})
    return {"all_exact": all(row["exact"] for row in episodes), "episodes": episodes,
            "interpretation": "Compare baseline-repeat variation before attributing any mismatch to instrumentation."}


def summarize(root, comparisons=None):
    manifest, records = load_run(root)
    if manifest["server"].get("execution_mode") != "shadow":
        raise ValueError("This report requires a shadow diagnostic run")
    rows = []
    layout = None
    for (task, episode), record in sorted(records.items()):
        for chunk in record["chunks"]:
            cache = chunk["c3ache"]
            if cache.get("execution_mode") != "shadow" or cache["reused_calls"] != 0:
                raise ValueError("Diagnostic run executed approximate stack outputs")
            diagnostic = cache["diagnostics"]
            current_layout = (diagnostic["action_channels"], diagnostic["action_scales"], diagnostic["guidance_scale"])
            if layout is not None and layout != current_layout:
                raise ValueError("Diagnostic action scaling/CFG layout changed")
            layout = current_layout
            measurements = diagnostic["steps"]
            if sum(row["would_reuse"] for row in measurements) != cache["hypothetical_reused_calls"]:
                raise ValueError("Missing diagnostic observations: hypothetical hit and sample counts differ")
            if len({row["step"] for row in measurements}) != len(measurements):
                raise ValueError("Duplicate diagnostic sampler steps in a chunk")
            rows.extend({**row, "task_id": task, "episode_id": episode, "chunk_index": chunk["chunk_index"]}
                        for row in measurements)
    samples = [row for row in rows if row["would_reuse"]]
    comparisons = comparisons or {}
    channels = layout[0] if layout else []
    if channels is None:
        channels = list(range(len(samples[0]["scaled_update_channel_rmse"]))) if samples else []
    traces = {}
    loaded = {}
    for name, path in comparisons.items():
        other_manifest, other_records = load_run(path)
        if other_manifest["server"].get("execution_mode") == "shadow":
            raise ValueError("Reference traces must come from baseline or full-refresh runs")
        if any(chunk["c3ache"]["reused_calls"] for record in other_records.values() for chunk in record["chunks"]):
            raise ValueError("Reference trace actually reused approximate outputs")
        loaded[name] = (other_manifest, other_records)
        traces[name] = compare_traces(other_manifest, other_records, manifest, records)
    controls = {}
    if "baseline" in loaded:
        for name, (other_manifest, other_records) in loaded.items():
            if name != "baseline":
                controls[name] = compare_traces(*loaded["baseline"], other_manifest, other_records)
    return {
        "schema_version": 1, "source": str(Path(root).resolve()), "manifest": manifest,
        "status": "measured" if samples else "no_reuse_observations",
        "episodes": len(records), "hypothetical_reuse_samples": len(samples),
        "nonfinite_reuse_samples": sum(not row["velocity"]["finite"] for row in samples),
        "action_channels": channels, "action_scales": layout[1] if layout else [],
        "by_step": by_step(rows, len(channels)),
        "by_task": {str(task): by_step([row for row in rows if row["task_id"] == task], len(channels))
                    for task in manifest["protocol"]["task_ids"]},
        "trace_comparisons": traces, "baseline_control_comparisons": controls,
        "notes": [
            "The executed policy is always full computation. Diagnostic timing is not a speed benchmark.",
            "Errors are local counterfactuals on the full-policy trajectory, not closed-loop cached-policy outcomes.",
            "Scaled updates use native action normalization ranges; they are not final actions or physical displacement guarantees.",
            "Channel errors should be inspected separately, especially translation, rotation and gripper.",
            "A low error at one step does not certify a whole window or a one-percentage-point success bound.",
            "Nonfinite values are counted; null metric values are excluded from descriptive distributions.",
        ],
    }


def write_csv(path, report):
    columns = ["step", "sigma", "delta_sigma", "samples", "nonfinite_samples",
               "velocity_relative_l2_mean", "velocity_rmse_mean", "update_rmse_mean", "update_rmse_p95",
               "scaled_update_rmse_mean", "scaled_update_rmse_p95", "same_state_roundtrip_rmse_mean"]
    columns += [f"action_{channel}_scaled_update_rmse_mean" for channel in report["action_channels"]]
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in report["by_step"]:
            result = {key: row[key] for key in columns[:5]}
            for metric in ["velocity_relative_l2", "velocity_rmse", "update_rmse", "scaled_update_rmse", "same_state_roundtrip_rmse"]:
                for stat in ["mean", "p95"]:
                    name = f"{metric}_{stat}"
                    if name in columns:
                        result[name] = row[metric][stat]
            for channel, values in zip(report["action_channels"], row["scaled_update_channel_rmse"]):
                result[f"action_{channel}_scaled_update_rmse_mean"] = values["mean"]
            writer.writerow(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--baseline-repeat")
    parser.add_argument("--full-refresh")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    comparisons = {name: getattr(args, name) for name in ["baseline", "baseline_repeat", "full_refresh"] if getattr(args, name)}
    report = summarize(args.shadow, comparisons)
    output = Path(args.output)
    write_json(output, report)
    write_csv(output.with_suffix(".csv"), report)
    print(f"Diagnostic samples={report['hypothetical_reuse_samples']}, nonfinite={report['nonfinite_reuse_samples']}; {output}")


if __name__ == "__main__":
    main()
