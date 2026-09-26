"""Compare complete, paired LIBERO runs; never infer success from log text."""
import argparse
import json
import math
from statistics import NormalDist, median

from .result_io import load_run, write_json


def paired_success(baseline, cached, margin_pp=1.0):
    if len(baseline) != len(cached) or not baseline:
        raise ValueError("Need equally sized, nonempty paired outcomes")
    if any(type(value) is not bool for value in [*baseline, *cached]):
        raise ValueError("Success outcomes must be booleans")
    n = len(baseline)
    gains = sum(b is False and c is True for b, c in zip(baseline, cached))
    losses = sum(b is True and c is False for b, c in zip(baseline, cached))
    # Bonferroni-adjusted Wilson bounds for the two discordant probabilities.
    # This is an approximate, conservative interval for their difference. It
    # stays nonzero-width even if a small pilot happens to have no discordance.
    z = NormalDist().inv_cdf(1 - 0.05 / 4)
    def wilson(count):
        p = count / n
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        return max(0.0, center - radius), min(1.0, center + radius)
    gain_lo, gain_hi = wilson(gains)
    loss_lo, loss_hi = wilson(losses)
    lower, upper = 100 * (gain_lo - loss_hi), 100 * (gain_hi - loss_lo)
    difference = 100 * (gains - losses) / n
    decision = ("supported" if lower >= -margin_pp else
                "exceeds_margin" if upper < -margin_pp else "not_established")
    return {"episodes": n, "baseline_success_percent": 100 * sum(baseline) / n,
            "cached_success_percent": 100 * sum(cached) / n,
            "difference_pp": difference, "gains": gains, "losses": losses,
            "interval_95_pp": [lower, upper],
            "interval_method": "approximate Bonferroni-Wilson on paired discordances",
            "margin_pp": margin_pp, "point_estimate_within_margin": difference >= -margin_pp,
            "noninferiority": decision}


def distribution(values):
    if not values:
        return None
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Latency samples must be finite and nonnegative")
    values = sorted(values)
    return {"count": len(values), "median_ms": median(values),
            "p95_ms": values[max(0, math.ceil(0.95 * len(values)) - 1)],
            "mean_ms": sum(values) / len(values)}


def latency(records):
    chunks = [chunk for record in records.values() for chunk in record["chunks"]]
    result = {}
    for key in ["infer_rpc_ms", "history_rpc_ms", "policy_cycle_ms"]:
        result[key] = distribution([chunk[key] for chunk in chunks if key in chunk])
    for key in ["prepare_ms", "video_loop_ms", "action_loop_ms", "infer_ms", "history_ms"]:
        result["server_" + key] = distribution([chunk["timings_ms"][key] for chunk in chunks
                                                 if key in chunk.get("timings_ms", {})])
    # Cold-start chunks always run full; report steady-state separately as well.
    steady = [chunk for chunk in chunks if chunk["chunk_index"] > 0]
    result["steady_infer_rpc_ms"] = distribution([chunk["infer_rpc_ms"] for chunk in steady])
    result["episode_wall_ms"] = distribution([record["episode_wall_ms"] for record in records.values()])
    return result


def compare_runs(baseline_dir, cached_dir, margin_pp=1.0):
    left, a = load_run(baseline_dir)
    right, b = load_run(cached_dir)
    for key in ["schema_version", "protocol", "client_runtime"]:
        if left[key] != right[key]:
            raise ValueError(f"Run mismatch in {key}; paired comparison would be confounded")
    for key in ["upstream_commit", "code", "checkpoint", "native", "runtime", "profile_inference"]:
        if left["server"][key] != right["server"][key]:
            raise ValueError(f"Server mismatch in {key}; paired comparison would be confounded")
    if left["server"]["c3ache"]["enabled"] or not right["server"]["c3ache"]["enabled"]:
        raise ValueError("Expected an uncached baseline and a cache-enabled treatment")
    if a.keys() != b.keys():
        raise ValueError("Episode keys do not match")
    for key in a:
        if a[key]["initial_state_index"] != b[key]["initial_state_index"]:
            raise ValueError(f"Initial-state index mismatch for {key}")
    outcome = paired_success([row["success"] for row in a.values()],
                             [b[key]["success"] for key in a], margin_pp)
    by_task = {}
    for task in left["protocol"]["task_ids"]:
        keys = [key for key in a if key[0] == task]
        by_task[str(task)] = paired_success([a[k]["success"] for k in keys], [b[k]["success"] for k in keys], margin_pp)
    la, lb = latency(a), latency(b)
    speedups = {key: la[key]["median_ms"] / lb[key]["median_ms"]
                for key in la if la[key] and lb[key] and lb[key]["median_ms"] > 0}
    hits = sum(chunk.get("c3ache", {}).get("reused_calls", 0)
               for row in b.values() for chunk in row["chunks"])
    return {"success": outcome, "per_task": by_task,
            "baseline_latency": la, "cached_latency": lb, "ratio_of_latency_medians": speedups,
            "cached_stack_hits": hits, "cache_exercised": hits > 0,
            "acceptance_supported": hits > 0 and outcome["noninferiority"] == "supported",
            "notes": ["Success intervals assume independent episode pairs on this fixed task suite.",
                      "Reproduce a meaningful full baseline before interpreting noninferiority.",
                      "Latency is observed on separate closed-loop trajectories; action/episode lengths can differ.",
                      "policy_cycle_ms = infer RPC + feedback/KV RPC; terminal chunks have no feedback RPC.",
                      "A small smoke run cannot establish a one-percentage-point bound."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--cached", required=True)
    parser.add_argument("--margin-pp", type=float, default=1.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not 0 <= args.margin_pp <= 100:
        parser.error("--margin-pp must be between 0 and 100")
    result = compare_runs(args.baseline, args.cached, args.margin_pp)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
