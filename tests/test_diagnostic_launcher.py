import subprocess
import sys

import pytest

from script.run_c3ache_diagnostics import MODES, build_commands, parse_args


def test_launcher_covers_full_policy_controls_and_shadow_with_correct_modes():
    args = parse_args(["--checkpoint", "/models/libero", "--out-dir", "/tmp/diagnostic-test"])
    for mode in MODES:
        server, client = build_commands(args, mode)
        assert "--nproc_per_node=1" in server
        assert "--record-actions" in client and "--save-videos" in client
        assert client[client.index("--task-start") + 1] == "1"
        assert client[client.index("--task-end") + 1] == "4"
        if mode == "full_refresh":
            assert "--c3ache" in server and "--diagnose-c3ache" not in server
            assert server[server.index("--cache-refresh-interval") + 1] == "1"
        elif mode == "shadow":
            assert "--diagnose-c3ache" in server and "--c3ache" not in server
            assert client[client.index("--expected-mode") + 1] == "shadow"
        else:
            assert "--c3ache" not in server and "--diagnose-c3ache" not in server


@pytest.mark.parametrize("options", [["--cache-start-step", "0"], ["--cache-end-step", "50"],
                                     ["--cache-refresh-interval", "1"], ["--gpu", "0,1"],
                                     ["--task-end", "11"], ["--base-seed", "4294967295"]])
def test_invalid_diagnostic_requests_fail_before_loading_models(options):
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "/models/libero", "--out-dir", "/tmp/diagnostic-test", *options])


def test_dry_run_needs_neither_checkpoint_nor_cuda(tmp_path):
    output = tmp_path / "not-created"
    result = subprocess.run([sys.executable, "script/run_c3ache_diagnostics.py", "--checkpoint", "/missing/checkpoint",
                             "--out-dir", str(output), "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "Full-policy episodes: 12" in result.stdout
    assert not output.exists()
