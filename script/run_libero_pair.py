#!/usr/bin/env python3
"""Run baseline and C³ache sequentially on one existing GPU; never rents hardware."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]


def build_commands(args, mode):
    task_end, episodes = {"smoke": (1, 2), "pilot": (10, 2), "full": (10, 50)}[args.preset]
    server = [args.server_python, "-m", "torch.distributed.run", "--nproc_per_node=1",
              "--master_port", str(args.master_port), "wan_va/wan_va_server.py",
              "--config-name", "libero", "--checkpoint", str(Path(args.checkpoint).resolve()),
              "--port", str(args.port), "--save_root", str(Path(args.out_dir).resolve() / "debug" / mode),
              "--profile-inference", "--no-save-debug"]
    if args.offload is not None:
        server.append("--offload" if args.offload else "--no-offload")
    if mode == "cached":
        server += ["--c3ache", "--cache-start-step", str(args.cache_start_step),
                   "--cache-end-step", str(args.cache_end_step),
                   "--cache-refresh-interval", str(args.cache_refresh_interval)]
    client = [args.client_python, "-m", "evaluation.libero.evaluate_c3ache", "--host", "127.0.0.1",
              "--port", str(args.port), "--suite", "libero_10", "--task-start", "0",
              "--task-end", str(task_end), "--episodes", str(episodes),
              "--base-seed", str(args.base_seed), "--expected-mode", mode,
              "--out-dir", str(Path(args.out_dir).resolve() / mode)]
    if args.resume:
        client.append("--resume")
    if args.save_videos:
        client.append("--save-videos")
    return server, client


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)


def wait_ready(process, port, timeout):
    deadline = time.monotonic() + timeout
    # Do not route localhost readiness checks through proxy environment variables.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Server exited before becoming ready; inspect the saved server log")
        try:
            with opener.open(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Server did not become ready in {timeout} seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local robbyant/lingbot-va-posttrain-libero-long snapshot")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--preset", choices=["smoke", "pilot", "full"], default="smoke")
    parser.add_argument("--gpu", default="0", help="One existing CUDA device index/UUID")
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--client-python", default=sys.executable)
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--master-port", type=int, default=29061)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--cache-start-step", type=int, default=5)
    parser.add_argument("--cache-end-step", type=int, default=39)
    parser.add_argument("--cache-refresh-interval", type=int, default=2)
    parser.add_argument("--offload", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without starting any processes")
    args = parser.parse_args()
    if not 0 <= args.cache_start_step <= args.cache_end_step < 50 or args.cache_refresh_interval < 0:
        parser.error("Require 0 <= cache-start-step <= cache-end-step < 50 and nonnegative refresh interval")
    if args.startup_timeout <= 0 or "," in args.gpu:
        parser.error("Use a positive startup timeout and one GPU per paired run")
    commands = {mode: build_commands(args, mode) for mode in ["baseline", "cached"]}
    for mode, (server, client) in commands.items():
        print(f"{mode} server: CUDA_VISIBLE_DEVICES={shlex.quote(args.gpu)} {shlex.join(server)}", flush=True)
        print(f"{mode} client: {shlex.join(client)}", flush=True)
    if args.dry_run:
        return
    if not Path(args.checkpoint).is_dir():
        parser.error("--checkpoint must be an existing local checkpoint directory")
    root = Path(args.out_dir).resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        parser.error("Output directory is not empty; choose another directory or use --resume")
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONUNBUFFERED"] = "1"
    # Fail before launching if either port is already owned by another process.
    for port in [args.port, args.master_port]:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", port))
    # Discover simulator/dependency problems before loading the large model.
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
    subprocess.run([args.client_python, "-m", "evaluation.libero.compare_c3ache",
                    "--baseline", str(root / "baseline"), "--cached", str(root / "cached"),
                    "--output", str(root / "comparison.json")], cwd=ROOT, env=env, check=True)
    print(f"Completed paired {args.preset} run: {root / 'comparison.json'}", flush=True)


if __name__ == "__main__":
    main()
