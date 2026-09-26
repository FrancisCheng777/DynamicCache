"""Real server sampling loops with a tiny real model; VAE/checkpoint I/O is external."""
from types import SimpleNamespace

import torch
import pytest

from wan_va.modules.c3ache import C3ache, C3acheConfig
from wan_va.modules.model import WanTransformer3DModel
from wan_va.utils.scheduler import FlowMatchScheduler
from wan_va.wan_va_server import VA_Server


def server(action_steps=3, video_steps=2, window=(1, 1)):
    instance = VA_Server.__new__(VA_Server)
    instance.job_config = SimpleNamespace(
        frame_chunk_size=2, action_dim=3, action_num_inference_steps=action_steps,
        num_inference_steps=video_steps, video_exec_step=-1, guidance_scale=5,
        action_guidance_scale=1, wan22_pretrained_model_name_or_path="test",
        patch_size=(1, 2, 2), save_debug=False,
    )
    instance.device = torch.device("cpu")
    instance.dtype = torch.bfloat16
    instance.cache_name = "pos"
    instance.profile_inference = False
    instance.action_per_frame = 1
    instance.latent_height = instance.latent_width = 2
    instance.use_cfg = True
    instance.prompt_embeds = torch.zeros(1, 3, 8).bfloat16()
    instance.negative_prompt_embeds = torch.ones(1, 3, 8).bfloat16()
    instance.action_mask = torch.ones(3).bool()
    instance.action_cache = C3ache(C3acheConfig(True, *window, 0))
    instance.scheduler = FlowMatchScheduler(shift=5, sigma_min=0, extra_one_step=True)
    instance.action_scheduler = FlowMatchScheduler(shift=0.05, sigma_min=0, extra_one_step=True)
    instance.transformer = WanTransformer3DModel(
        patch_size=[1, 2, 2], num_attention_heads=2, attention_head_dim=12,
        in_channels=48, out_channels=48, action_dim=3, text_dim=8,
        freq_dim=16, ffn_dim=48, num_layers=2, attn_mode="torch",
    ).to(torch.bfloat16).eval()
    instance.transformer.create_empty_cache("pos", 8, 2, 2, "cpu", torch.bfloat16, 2)
    instance._encode_obs = lambda _: torch.zeros(1, 48, 1, 2, 2).bfloat16()
    instance.postprocess_action = lambda action: action
    return instance


@torch.no_grad()
@pytest.mark.parametrize("action_steps,video_steps,window", [(3, 2, (1, 1)), (50, 20, (5, 39))])
def test_server_distinguishes_sampler_steps_cold_chunk_video_and_zero_noise_commit(action_steps, video_steps, window):
    instance = server(action_steps, video_steps, window)
    calls = []
    def hook(_module, _args, kwargs):
        calls.append((kwargs["action_mode"], kwargs["update_cache"], kwargs.get("cache_step")))
    instance.transformer.register_forward_pre_hook(hook, with_kwargs=True)
    hits = []
    for frame in [0, 2, 4]:
        actions, video = instance._infer({}, frame_st_id=frame)
        assert torch.isfinite(actions).all() and torch.isfinite(video).all()
        hits.append(instance.action_cache.stats()["reused_calls"])
    assert hits == [0, 0, window[1] - window[0] + 1]
    one_chunk = ([(False, 0, None)] * video_steps + [(False, 1, None)]
                 + [(True, 0, step) for step in range(action_steps)] + [(True, 1, None)])
    assert calls == one_chunk * 3
    assert instance.action_cache.stats()["full_reasons"]["kv_commit"] == 1
