"""Seeded LIBERO evaluation using LingBot-VA's upstream closed-loop protocol.

Run from the repository root with ``python -m evaluation.libero.evaluate_c3ache``.
LIBERO and rendering dependencies are loaded only when actually evaluating.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

from .result_io import episode_path, episode_seed, expected_keys, validate_record, write_json


class BenchmarkClient:
    """Bounded connection/RPC timeouts, using the upstream wire format."""
    def __init__(self, host, port, timeout=1800):
        import websockets.sync.client
        # Load the original codec directly. Importing the wan_va package eagerly
        # imports the diffusion model, which must NOT be required in a LIBERO env.
        codec_path = Path(__file__).resolve().parents[2] / "wan_va/utils/Simple_Remote_Infer/deploy/msgpack_numpy.py"
        spec = importlib.util.spec_from_file_location("dynamiccache_wire", codec_path)
        codec = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(codec)
        self._unpack = codec.unpackb
        self._packer = codec.Packer()
        self.timeout = timeout
        self.connection = websockets.sync.client.connect(
            f"ws://{host}:{port}", compression=None, max_size=None,
            open_timeout=15, ping_interval=None,
        )
        self._unpack(self.connection.recv(timeout=15))  # Upstream greeting.

    def infer(self, value):
        self.connection.send(self._packer.pack(value))
        response = self.connection.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError(f"Inference server error:\n{response}")
        return self._unpack(response)

    def close(self):
        self.connection.close()


def extract_obs(obs):
    # Same vertical flip, resolution and camera names as upstream client.py.
    import numpy as np
    return {
        "observation.images.agentview_rgb": np.ascontiguousarray(obs["agentview_image"][::-1]),
        "observation.images.eye_in_hand_rgb": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1]),
    }


def rollout(model, env, initial_state, prompt, seed, *, max_env_steps=800, save_video_path=None):
    import numpy as np

    start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    env.reset()
    env.set_init_state(initial_state)
    for _ in range(5):
        observation, _, _, _ = env.step([0.0] * 7)
    first_obs = extract_obs(observation)
    reset_reply = model.infer(dict(reset=True, prompt=prompt, seed=seed))
    if reset_reply.get("seed") != seed:
        raise RuntimeError("Server did not acknowledge the episode seed; use the DynamicCache server")
    done = False
    first = True
    chunks = []
    frames = []
    # Keep upstream's chunk-boundary limit, including the five settling steps.
    while env.env.timestep < max_env_steps:
        before = time.perf_counter()
        reply = model.infer(dict(obs=first_obs, prompt=prompt))
        rpc_ms = (time.perf_counter() - before) * 1000
        action = reply["action"]
        if action.shape != (7, 4, 4) or not np.isfinite(action).all():
            raise ValueError(f"Expected finite LIBERO actions of shape (7, 4, 4), got {action.shape}")
        if "c3ache" not in reply:
            raise RuntimeError("Server did not report cache counters")
        chunk = {"chunk_index": len(chunks), "infer_rpc_ms": rpc_ms,
                 "policy_cycle_ms": rpc_ms, "c3ache": reply["c3ache"],
                 "timings_ms": reply.get("timings_ms", {})}
        chunks.append(chunk)
        key_frames = []
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                observation, _, done, _ = env.step(action[:, i, j])
                if done:
                    break
                observes = extract_obs(observation)
                # Upstream collects every action for the native 4 actions/frame.
                if (j + 1) % (action.shape[2] // 4) == 0:
                    key_frames.append(observes)
                    if save_video_path is not None:
                        frames.append(observes)
            if done:
                break
        first = False
        if done:
            break
        before = time.perf_counter()
        feedback = model.infer(dict(obs=key_frames, compute_kv_cache=True, imagine=False, state=action))
        chunk["history_rpc_ms"] = (time.perf_counter() - before) * 1000
        chunk["policy_cycle_ms"] += chunk["history_rpc_ms"]
        chunk["timings_ms"].update(feedback.get("timings_ms", {}))

    episode_wall_ms = (time.perf_counter() - start) * 1000
    if save_video_path is not None and frames:
        import imageio.v2 as imageio
        Path(save_video_path).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(save_video_path, [np.hstack(list(frame.values())) for frame in frames], fps=60)
    return {"status": "ok", "success": bool(done), "env_steps": int(env.env.timestep),
            "episode_wall_ms": episode_wall_ms, "chunks": chunks}


def state_digest(states):
    import numpy as np
    array = np.ascontiguousarray(states)
    hasher = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def client_runtime(libero):
    def git_state(root):
        try:
            return {
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                   stderr=subprocess.DEVNULL, text=True).strip(),
                "tracked_files_dirty": bool(subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
                    stderr=subprocess.DEVNULL, text=True).strip()),
            }
        except (OSError, subprocess.CalledProcessError):
            return {"commit": "unknown", "tracked_files_dirty": None}
    versions = {}
    for package in ["numpy", "torch", "mujoco", "robosuite", "libero", "websockets", "msgpack"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    module_dir = (Path(libero.__file__).resolve().parent if libero.__file__
                  else Path(next(iter(libero.__path__))).resolve())
    return {"python": platform.python_version(), "packages": versions,
            "mujoco_gl": os.environ.get("MUJOCO_GL"),
            "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            "libero_code": git_state(module_dir),
            "evaluator_code": git_state(Path(__file__).resolve().parents[2])}


def evaluate(args):
    import numpy as np
    import libero
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    root = Path(args.out_dir)
    bench = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = list(range(args.task_start, args.task_end))
    if not task_ids or min(task_ids) < 0 or max(task_ids) >= bench.get_num_tasks():
        raise ValueError("Requested task range is invalid for this benchmark")
    initial_states = {task: bench.get_task_init_states(task) for task in task_ids}
    if any(args.episodes > len(states) for states in initial_states.values()):
        raise ValueError("Requested episodes exceed available initial states; implicit repetition is not allowed")
    protocol = {
        "name": "lingbot-va-native-libero-v1", "suite": args.suite, "task_ids": task_ids,
        "episodes_per_task": args.episodes, "base_seed": args.base_seed,
        "max_env_steps": args.max_env_steps, "limit_check": "chunk_boundary", "settling_steps": 5,
        "resolution": [128, 128], "first_chunk_skip_frames": 1, "save_videos": args.save_videos,
        "tasks": {str(task): {"prompt": bench.get_task(task).language,
                              "initial_states_sha256": state_digest(initial_states[task]),
                              "bddl_sha256": hashlib.sha256(Path(bench.get_task_bddl_file_path(task)).read_bytes()).hexdigest()}
                  for task in task_ids},
    }
    for task, episode in expected_keys(protocol):
        episode_seed(args.base_seed, task, episode)
    if args.check_env:
        task = task_ids[0]
        env = OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(task),
                                 camera_heights=128, camera_widths=128)
        try:
            env.seed(episode_seed(args.base_seed, task, 0))
            env.reset()
            env.set_init_state(initial_states[task][0])
            observation, _, _, _ = env.step([0.0] * 7)
            if any(image.shape != (128, 128, 3) for image in extract_obs(observation).values()):
                raise ValueError("Unexpected LIBERO camera shape")
        finally:
            env.close()
        print("LIBERO environment preflight passed (assets, initial states and rendering).", flush=True)
        return
    model = BenchmarkClient(args.host, args.port, args.rpc_timeout)
    try:
        metadata = model.infer({"get_metadata": True})["metadata"]
        if metadata["c3ache"]["enabled"] != (args.expected_mode == "cached"):
            raise ValueError("Connected server does not match --expected-mode")
        manifest = {"schema_version": 1, "protocol": protocol, "server": metadata,
                    "client_runtime": client_runtime(libero), "complete": False}
        manifest_path = root / "manifest.json"
        if manifest_path.exists():
            if not args.resume:
                raise ValueError("Output already has a manifest; use --resume or a new output directory")
            previous = json.loads(manifest_path.read_text())
            if {k: v for k, v in previous.items() if k != "complete"} != {k: v for k, v in manifest.items() if k != "complete"}:
                raise ValueError("Resume metadata differs from the original run")
        elif root.exists() and any(root.iterdir()):
            raise ValueError("New evaluation requires an empty output directory")
        write_json(manifest_path, manifest)
        total = len(expected_keys(protocol))
        completed = 0
        for task, episode in expected_keys(protocol):
            path = episode_path(root, task, episode)
            seed = episode_seed(args.base_seed, task, episode)
            if args.resume and path.exists():
                record = json.loads(path.read_text())
                if record.get("status") == "ok":
                    validate_record(record, task, episode, protocol)
                    completed += 1
                    print(f"Resume: {completed}/{total} task={task} episode={episode}", flush=True)
                    continue
            record = {"task_id": task, "episode_id": episode, "seed": seed, "initial_state_index": episode}
            env = None
            try:
                # Seed before construction as well as before reset.
                random.seed(seed)
                np.random.seed(seed)
                env = OffScreenRenderEnv(bddl_file_name=bench.get_task_bddl_file_path(task),
                                         camera_heights=128, camera_widths=128)
                video_path = str(root / "videos" / f"{task:03d}_{episode:05d}.mp4") if args.save_videos else None
                record.update(rollout(model, env, initial_states[task][episode], bench.get_task(task).language,
                                      seed, max_env_steps=args.max_env_steps, save_video_path=video_path))
                write_json(path, record)
            except Exception as error:
                record.update(status="error", error=f"{type(error).__name__}: {error}")
                write_json(path, record)
                raise
            finally:
                if env is not None:
                    env.close()
            completed += 1
            print(f"{completed}/{total} task={task} episode={episode} success={record['success']}", flush=True)
        manifest["complete"] = True
        write_json(manifest_path, manifest)
    finally:
        model.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--suite", default="libero_10", choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"])
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--max-env-steps", type=int, default=800)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-mode", required=True, choices=["baseline", "cached"])
    parser.add_argument("--rpc-timeout", type=float, default=1800)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-env", action="store_true", help="Validate simulator assets/rendering without contacting a model server")
    args = parser.parse_args()
    if not 0 < args.episodes < 100_000 or args.max_env_steps <= 5 or args.rpc_timeout <= 0:
        parser.error("Require positive episodes/RPC timeout and max-env-steps > 5")
    evaluate(args)


if __name__ == "__main__":
    main()
