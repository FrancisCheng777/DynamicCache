# Validation record

## Implemented validation

Local validation was performed on macOS arm64, Python 3.12, PyTorch 2.9.0,
diffusers 0.36.0 and transformers 4.55.2, using CPU tensors and PyTorch SDPA.
The local suite passed **36 tests** on 2026-09-26. It is also configured for Linux
CPU CI on Python 3.10 and 3.12.

```bash
python -m pytest tests -q
python -m compileall -q wan_va evaluation/libero script/run_libero_pair.py
python wan_va/wan_va_server.py --help
python -m evaluation.libero.evaluate_c3ache --help
python script/run_libero_pair.py --checkpoint /models/libero --out-dir /tmp/dynamiccache-smoke --dry-run
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

## Not yet measured

No full released checkpoint was loaded on CUDA during this implementation session.
No real LIBERO closed-loop benchmark results, 4090/A800 memory measurements,
multi-GPU FSDP validation, speedups, success rates, or one-percentage-point
noninferiority findings are claimed.

CPU tests of small random models verify execution/state contracts; they cannot
establish that learned LingBot-VA residuals transfer accurately between chunks.
The code defaults to cache off, and the public README labels the initial cache
window as unvalidated. No GPU rental or training was started.

Before treating the method as usable, run environment preflight, the paired smoke
test, a pilot over all ten tasks, and the complete paired benchmark with fixed
parameters. Inspect saved videos and baseline success as well as aggregate numbers.
