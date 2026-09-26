"""Exercise the real upstream transformer and KV allocator on small CPU tensors."""
import torch
import copy
import pytest

from wan_va.modules.c3ache import C3ache, C3acheConfig
from wan_va.modules.model import WanTransformer3DModel


SCHEDULE = ((1000.0, 1.0), (200.0, 0.2), (1.0, 0.001))


def tiny_model():
    torch.manual_seed(17)
    model = WanTransformer3DModel(
        patch_size=[1, 1, 1], num_attention_heads=2, attention_head_dim=12,
        in_channels=3, out_channels=3, action_dim=3, text_dim=8,
        freq_dim=16, ffn_dim=48, num_layers=2, attn_mode="torch",
    ).to(torch.bfloat16).eval()
    model.create_empty_cache("pos", 4, 2, 2, "cpu", torch.bfloat16, 2)
    return model


def inputs(frame=4, step=1):
    generator = torch.Generator().manual_seed(frame * 10 + step)
    return {
        "noisy_latents": torch.randn(2, 3, 2, 1, 1, generator=generator).bfloat16(),
        "text_emb": torch.randn(2, 3, 8, generator=generator).bfloat16(),
        "timesteps": torch.full((2, 2), SCHEDULE[step][0]),
        "grid_id": torch.tensor([[[frame, frame + 1], [0, 0], [0, 0]]]).expand(2, -1, -1),
    }


@torch.no_grad()
def test_cache_skips_real_blocks_but_runs_current_output_head():
    model = tiny_model()
    cache = C3ache(C3acheConfig(True, 1, 1, 0))
    counts = {"block": 0, "head": 0}
    def block_hook(*_):
        counts["block"] += 1
    def head_hook(*_):
        counts["head"] += 1
    model.blocks[0].register_forward_hook(block_hook)
    model.action_proj_out.register_forward_hook(head_hook)
    for frame in [4, 8]:
        cache.begin_chunk(frame, SCHEDULE)
        for step in range(3):
            output = model(inputs(frame, step), action_mode=True,
                           action_cache=cache, cache_step=step)
            assert output.shape == (2, 2, 3)
            assert torch.isfinite(output).all()
    assert counts == {"block": 5, "head": 6}
    assert cache.stats()["reused_calls"] == 1


def assert_active_kv_equal(left, right):
    for a, b in zip(left.blocks, right.blocks):
        a, b = a.attn1.attn_caches["pos"], b.attn1.attn_caches["pos"]
        assert torch.equal(a["mask"], b["mask"])
        active = a["mask"]
        for key in ["id", "is_pred"]:
            assert torch.equal(a[key][active], b[key][active])
        for key in ["k", "v"]:
            assert torch.equal(a[key][:, active], b[key][:, active])


@torch.no_grad()
@pytest.mark.parametrize("enabled,interval", [(False, 2), (True, 1)])
def test_disabled_or_every_chunk_refresh_matches_uncached_path_exactly(enabled, interval):
    full = tiny_model()
    cached = copy.deepcopy(full)
    cache = C3ache(C3acheConfig(enabled, 1, 1, interval))
    for frame in [0, 4, 8]:
        cache.begin_chunk(frame, SCHEDULE)
        for step in range(3):
            data = inputs(frame, step)
            expected = full(data, action_mode=True)
            actual = cached(data, action_mode=True, action_cache=cache, cache_step=step)
            assert torch.equal(actual, expected)
        for action_mode, update_cache in [(False, 1), (True, 1), (False, 2), (True, 2)]:
            data = inputs(frame)
            expected = full(data, action_mode=action_mode, update_cache=update_cache)
            actual = cached(data, action_mode=action_mode, update_cache=update_cache,
                            action_cache=cache, cache_step=1)
            assert torch.equal(actual, expected)
        assert_active_kv_equal(full, cached)
        assert cache.stats()["reused_calls"] == 0


@torch.no_grad()
def test_reuse_preserves_persistent_kv_on_identical_inputs_even_when_slots_are_full():
    full = tiny_model()
    cached = copy.deepcopy(full)
    cache = C3ache(C3acheConfig(True, 1, 1, 0))
    # Fill all eight slots; the next temporary attention call must evict two.
    for frame in [0, 2, 4, 6]:
        for model in [full, cached]:
            model(inputs(frame), action_mode=True, update_cache=2)
    assert int(full.blocks[0].attn1.attn_caches["pos"]["mask"].sum()) == 8
    hits = 0
    for frame in [8, 12, 16]:
        cache.begin_chunk(frame, SCHEDULE)
        for step in range(3):
            data = inputs(frame, step)
            full(data, action_mode=True)
            cached(data, action_mode=True, action_cache=cache, cache_step=step)
            assert_active_kv_equal(full, cached)
            assert int(cached.blocks[0].attn1.attn_caches["pos"]["mask"].sum()) == 6
        hits += cache.stats()["reused_calls"]
        # Same commit input in both traces isolates the cache-state contract.
        for model in [full, cached]:
            model(inputs(frame), action_mode=True, update_cache=1,
                  action_cache=cache if model is cached else None, cache_step=None)
            model.clear_pred_cache("pos")
            model(inputs(frame), action_mode=True, update_cache=2)
        assert_active_kv_equal(full, cached)
    assert hits == 2


@torch.no_grad()
def test_video_calls_never_read_or_overwrite_action_residuals():
    model = tiny_model()
    cache = C3ache(C3acheConfig(True, 1, 1, 0))
    for frame in [4, 8]:
        cache.begin_chunk(frame, SCHEDULE)
        model(inputs(frame, 0), action_mode=True, action_cache=cache, cache_step=0)
        model(inputs(frame, 1), action_mode=True, action_cache=cache, cache_step=1)
    before = cache.stats()
    expected = copy.deepcopy(model)(inputs(8), action_mode=False)
    actual = model(inputs(8), action_mode=False, action_cache=cache, cache_step=1)
    assert torch.equal(actual, expected)
    assert cache.stats() == before
