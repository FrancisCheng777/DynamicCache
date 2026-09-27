#!/usr/bin/env python3
"""Run bounded full-policy controls and shadow diagnostics on one existing GPU."""
import argparse
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from script.run_libero_pair import ROOT, stop_process, wait_ready


MODES = ("baseline", "baseline_repeat", "full_refresh", "shadow")


def build_commands(args, mode):
    if mode not in MODES:
        raise ValueError(f"Unknown diagnostic mode: {mode}")
    root = Path(args.out_dir).resolve()
    server = [args.server_python, "-m", "torch.distributed.run", "--nproc_per_node=1",
              "--master_port", str(args.master_port), "wan_va/wan_va_server.py",
              "--config-name", "libero", "--checkpoint", str(Path(args.checkpoint).resolve()),
              "--port", str(args.port), "--save_root", str(root / "debug" / mode), "--no-save-debug"]
    if args.offload is not None:
        server.append("--offload" if args.offload else "--no-offload")
    expected = "baseline"
    if mode in ("full_refresh", "shadow"):
        expected = "cached" if mode == "full_refresh" else "shadow"
        server += ["--c3ache" if mode == "full_refresh" else "--diagnose-c3ache",
                   "--cache-start-step", str(args.cache_start_step), "--cache-end-step", str(args.cache_end_step),
                   "--cache-refresh-interval", str(1 if mode == "full_refresh" else args.cache_refresh_interval)]
    client = [args.client_python, "-m", "evaluation.libero.evaluate_c3ache", "--host", "127.0.0.1",
              "--port", str(args.port), "--suite", "libero_10", "--task-start", str(args.task_start),
              "--task-end", str(args.task_end), "--episodes", str(args.episodes), "--base-seed", str(args.base_seed),
              "--expected-mode", expected, "--out-dir", str(root / mode), "--record-actions"]
    if args.save_videos:
        client.append("--save-videos")
    if args.resume:
        client.append("--resume")
    return server, client


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gpu", default="0", help="One existing CUDA device index/UUID")
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--client-python", default=sys.executable)
    parser.add_argument("--task-start", type=int, default=1)
    parser.add_argument("--task-end", type=int, default=4, help="Exclusive; defaults cover tasks 01, 02, 03")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--cache-start-step", type=int, default=1)
    parser.add_argument("--cache-end-step", type=int, default=49)
    parser.add_argument("--cache-refresh-interval", type=int, default=2)
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--master-port", type=int, default=29061)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-baseline-repeat", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.task_start < args.task_end <= 10 or not 0 < args.episodes <= 50:
        parser.error("Require a nonempty task range within [0,10) and 1..50 episodes")
    if not 1 <= args.cache_start_step <= args.cache_end_step < 50:
        parser.error("Diagnostic window must lie in 1..49; step 0 is protected by native KV semantics")
    if args.cache_refresh_interval < 0 or args.cache_refresh_interval == 1:
        parser.error("Use interval 0 or >=2 for hypothetical reuse; interval=1 is already a separate control")
    if args.startup_timeout <= 0 or not args.gpu or "," in args.gpu:
        parser.error("Use a positive timeout and one GPU per diagnostic run")
    if any(not 0 < port < 65536 for port in [args.port, args.master_port]) or args.port == args.master_port:
        parser.error("Use two distinct valid ports")
    if not 0 <= args.base_seed <= 2**32 - 1 - (args.task_end - 1) * 100_000 - (args.episodes - 1):
        parser.error("Derived episode seeds must fit uint32")
    return args


def main():
    args = parse_args()
    modes = [mode for mode in MODES if not (args.skip_baseline_repeat and mode == "baseline_repeat")]
    commands = {mode: build_commands(args, mode) for mode in modes}
    for mode, (server, client) in commands.items():
        print(f"{mode} server: CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)} {shlex.join(server)}", flush=True)
        print(f"{mode} client: {shlex.join(client)}", flush=True)
    print(f"Full-policy episodes: {len(modes) * (args.task_end - args.task_start) * args.episodes}; no cached-policy rollout or full queue.", flush=True)
    if args.dry_run:
        return
    if not Path(args.checkpoint).is_dir():
        raise ValueError("--checkpoint must be an existing local snapshot")
    root = Path(args.out_dir).resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise ValueError("Use a new output directory or --resume with unchanged metadata")
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("MUJOCO_GL", "egl")
    if args.gpu.isdecimal():
        env.setdefault("MUJOCO_EGL_DEVICE_ID", args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    for port in [args.port, args.master_port]:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    subprocess.run(commands["baseline"][1] + ["--check-env"], cwd=ROOT, env=env, check=True)
    for mode, (server, client) in commands.items():
        (root / f"{mode}_commands.json").write_text(json.dumps({"server": server, "client": client,
                                                                "cuda_visible_devices": args.gpu}, indent=2) + "\n")
        with (root / f"{mode}_server.log").open("a" if args.resume else "w") as log:
            process = subprocess.Popen(server, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                wait_ready(process, args.port, args.startup_timeout)
                subprocess.run(client, cwd=ROOT, env=env, check=True)
            finally:
                stop_process(process)
    command = [args.client_python, "-m", "evaluation.libero.summarize_c3ache_diagnostics",
               "--shadow", str(root / "shadow"), "--baseline", str(root / "baseline"),
               "--full-refresh", str(root / "full_refresh"), "--output", str(root / "diagnostics.json")]
    if "baseline_repeat" in modes:
        command += ["--baseline-repeat", str(root / "baseline_repeat")]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
