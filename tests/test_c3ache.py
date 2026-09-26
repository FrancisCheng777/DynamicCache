import pytest
import torch

from wan_va.modules.c3ache import C3ache, C3acheConfig


SCHEDULE = ((1000.0, 1.0), (200.0, 0.2), (1.0, 0.001))


@torch.no_grad()
def test_reuses_residual_with_current_embedding_in_both_cfg_branches():
    cache = C3ache(C3acheConfig(True, 1, 1, 0))
    old = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])
    current = old + 10
    calls = []

    def blocks(x):
        calls.append(x.clone())
        return x + x.square()

    cache.begin_chunk(4, SCHEDULE)
    cache.run(old, lambda x: x, step=0)
    reference = cache.run(old, blocks, step=1)
    cache.begin_chunk(8, SCHEDULE)
    cache.run(current, lambda x: x, step=0)
    actual = cache.run(current, blocks, step=1)

    torch.testing.assert_close(actual, current + (reference - old))
    assert len(calls) == 1, "A cache hit must skip the entire transformer stack"


def start(cache, frame, x, *, schedule=SCHEDULE, context=()):
    cache.begin_chunk(frame, schedule, context)
    return cache.run(x, lambda y: y + 1, step=0)


@torch.no_grad()
@pytest.mark.parametrize("interval,expected", [(0, [1, 0, 0, 0]), (1, [1, 1, 1, 1]), (2, [1, 0, 1, 0])])
def test_refresh_interval_counts_regular_chunks(interval, expected):
    cache = C3ache(C3acheConfig(True, 1, 1, interval))
    calls = []
    for frame in [4, 8, 12, 16]:
        x = torch.tensor([float(frame)])
        start(cache, frame, x)
        def blocks(y):
            calls.append(frame)
            return y.square()
        result = cache.run(x, blocks, step=1)
        if expected[frame // 4 - 1]:
            residual = x.square() - x
        torch.testing.assert_close(result, x + residual)
    assert calls == [4 * (i + 1) for i, full in enumerate(expected) if full]


@torch.no_grad()
def test_cold_chunk_is_never_a_reference_and_full_outputs_are_returned_directly():
    cache = C3ache(C3acheConfig(True, 1, 1, 0))
    x = torch.tensor([1.0], dtype=torch.bfloat16)
    full_output = torch.tensor([0.001], dtype=torch.bfloat16)
    for frame in [0, 4]:
        start(cache, frame, x)
        result = cache.run(x, lambda _: full_output, step=1)
        assert result is full_output  # Avoid subtract/add rounding on the full path.
        assert cache.stats()["stored_residuals"] == (0 if frame == 0 else 1)


@torch.no_grad()
def test_first_step_tail_and_kv_commits_always_execute_and_do_not_replace_reference():
    cache = C3ache(C3acheConfig(True, 0, 1, 0))
    x = torch.ones(1)
    start(cache, 4, x)
    cache.run(x, lambda y: y + 3, step=1)
    start(cache, 8, x)
    calls = []
    def full(y):
        calls.append(True)
        return y + 99
    for step, update in [(0, 0), (2, 0), (1, 1), (1, 2), (None, 1)]:
        assert cache.run(x, full, step=step, update_cache=update).item() == 100
    assert len(calls) == 5
    assert cache.run(x, full, step=1).item() == 4
    assert cache.stats()["reference_frames"] == {"1": 4}


@torch.no_grad()
@pytest.mark.parametrize("change", ["reset", "sigma", "cfg", "rewind", "dtype", "shape"])
def test_incompatible_reference_is_not_reused(change):
    cache = C3ache(C3acheConfig(True, 1, 1, 0))
    x = torch.ones(1)
    start(cache, 4, x)
    cache.run(x, lambda y: y + 3, step=1)
    if change == "reset":
        cache.reset()
    schedule = SCHEDULE if change != "sigma" else ((1000.0, 1.0), (200.0, 0.25), (1.0, 0.001))
    if change == "dtype":
        x = x.double()
    if change == "shape":
        x = x.repeat(2)
    start(cache, 4 if change == "rewind" else 8, x, schedule=schedule, context=("cfg",) if change == "cfg" else ())
    result = cache.run(x, lambda y: y + 7, step=1)
    torch.testing.assert_close(result, x + 7)
    assert cache.stats()["reused_calls"] == 0


@torch.no_grad()
def test_reuse_requires_first_step_and_never_recursively_updates_residual():
    cache = C3ache(C3acheConfig(True, 1, 1, 0))
    x = torch.ones(1)
    start(cache, 4, x)
    cache.run(x, lambda y: y + 3, step=1)
    cache.begin_chunk(8, SCHEDULE)
    assert cache.run(x, lambda y: y + 77, step=1).item() == 78
    for frame in [12, 16]:
        start(cache, frame, x * frame)
        result = cache.run(x * frame, lambda y: y + 77, step=1)
        assert result.item() == frame + 3
        result.add_(100)  # Mutating a returned output must not mutate the cache.
    assert cache.stats()["reference_frames"] == {"1": 4}


def test_grad_enabled_bypasses_cache_and_keeps_autograd():
    cache = C3ache(C3acheConfig(True, 1, 1, 0))
    with torch.no_grad():
        start(cache, 4, torch.ones(1))
        cache.run(torch.ones(1), lambda y: y + 3, step=1)
        start(cache, 8, torch.ones(1))
    x = torch.ones(1, requires_grad=True)
    cache.run(x, lambda y: y.square() * 7, step=1).sum().backward()
    assert x.grad.item() == 14
    assert cache.stats()["reused_calls"] == 0


@pytest.mark.parametrize("kwargs", [{"start_step": -1}, {"start_step": 5, "end_step": 4}, {"refresh_interval": -1}])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        C3acheConfig(**kwargs)
