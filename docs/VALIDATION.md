# Validation record

## Implemented validation

Local validation was performed on macOS arm64, Python 3.12, PyTorch 2.9.0,
diffusers 0.36.0 and transformers 4.55.2, using CPU tensors and PyTorch SDPA.
The original local suite passed **36 tests** on 2026-09-26. The shadow-diagnostic
extension passes **59 tests** in the same local environment. Linux CPU CI covers
Python 3.10 and 3.12.

```bash
python -m pytest tests -q
python -m compileall -q wan_va evaluation/libero script/run_libero_pair.py
python wan_va/wan_va_server.py --help
python -m evaluation.libero.evaluate_c3ache --help
python script/run_libero_pair.py --checkpoint /models/libero --out-dir /tmp/dynamiccache-smoke --dry-run
python script/run_c3ache_diagnostics.py --checkpoint /models/libero --out-dir /tmp/dynamiccache-diagnostics --dry-run
git diff --check
```

Tests exercise:

- Current embedding + earlier residual, separately for both CFG branches.
- Exact full-path output preservation, cache disabled, and refresh interval 1.
- Cold chunks, protected first/tail steps, refresh schedules, cache misses/reset,
  changed sigma/CFG/layout/dtype, and no recursive reuse as a new reference.
- Autograd bypass and invalid configuration handling.
- A real small `WanTransformer3DModel`, using actual blocks, head, norms, rotary
  embedding, attention, and KV allocator; not a mock replacement for the model.
- Persistent KV correctness when the actual upstream allocator is full, and
  preservation of video/history/commit computation.
- The actual `VA_Server._infer` loops with a real tiny transformer and native
  schedulers. Checkpoint/VAE I/O is replaced at its external boundary.
- LIBERO protocol control flow against deterministic environment/policy boundary
  doubles, including cold action slicing, observation orientation, seed reset,
  feedback, terminal handling and the upstream chunk-boundary step limit.
- Strict result completeness/identity checks and paired success-rate arithmetic.
- An actual local WebSocket/NumPy round trip while explicitly disallowing model
  package imports in the simulator client.

The first residual-reuse test failed on the full-only implementation before the
cache was added. The real-model integration test then failed with six block calls
instead of five before the model hook was connected. A namespace-package regression
test exposed and verified the fix for official LIBERO's outer `__file__ = None`.

## Shadow diagnostics

New tests first demonstrated that an ordinary cache changes outputs and skips
stack calls, and that the original speed comparator would accept shadow-marked
measurements. The diagnostic implementation now:

- Executes each real stack exactly once and returns the original full output.
- Preserves real tiny-model action outputs, valid KV, and RNG state against the
  full path, including the native 20-video/50-action-step server loop and commits.
- Keeps reference refresh scheduling independent of executing the full policy.
- Measures selected action channels after the native CFG branch combination,
  applies sigma increments and action normalization scales, and separates
  same-state BF16 reconstruction error from cross-chunk error.
- Serializes nonfinite approximation diagnostics without feeding them to actions.
- Rejects diagnostic runs as cache speed benchmarks; checks sample counts and
  per-episode action traces, and produces tested JSON/CSV summaries.
- Generates a bounded 12-episode default command sequence without launching any
  process in dry-run mode.

The actual full-checkpoint CUDA/LIBERO diagnostic run is deferred to an available
user GPU. No numeric residual-error results are claimed here. Separately supplied
small-run feedback motivated this diagnostic extension; its raw artifacts were
not re-executed or independently audited in this local validation session.

## Not yet measured

No full released checkpoint was loaded on CUDA during this local implementation
session. The new diagnostic mode has no measured full-model GPU memory or latency,
no multi-GPU FSDP validation, and no one-percentage-point noninferiority finding.

CPU tests of small random models verify execution/state contracts; they cannot
establish that learned LingBot-VA residuals transfer accurately between chunks.
The code defaults to cache off, and the public README labels the initial cache
window as unvalidated. No GPU rental or training was started.

Before treating the method as usable, run environment preflight and the bounded
diagnostics in DIAGNOSTICS.md, then test a quality-screened cache configuration
with paired closed-loop evaluation. Inspect videos, episode lengths and cumulative
inference time as well as success, using held-out initial states for confirmation.
