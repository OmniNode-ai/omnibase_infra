# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression guard for the Qwen3.8-27B vLLM systemd unit and its provisioning.

OMN-18570. The operator brought Qwen3.8-27B up on ``.201:8000`` by hand on
2026-09-17. Doctrine is that nothing on a governed host is managed by hand, so
the unit text and the two toolchain repairs it depends on are declared in this
repo and pinned here.

This file guards the unit's invariants ONLY. The co-residency arithmetic it
shares with the 35B unit it replaced is pinned by
``test_vllm_qwen_coder_service_unit.py``; the numbers are restated here rather
than imported because the two units may diverge and a shared constant would let
one drift behind the other silently.
"""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path

import pytest

SERVICE_PATH = Path("deploy/systemd/vllm-gpu0-qwen38-27b.service")
PROVISION_PATH = Path("deploy/systemd/provision-vllm-0.27-venv.sh")

# --- Measured cuda:0 facts, .201 (OMN-14379, unchanged by the swap) ---------
# nvidia-smi --query-gpu=memory.total -> 32607 MiB; torch.cuda.mem_get_info(0)
# -> (29853679616, 33668726784). Re-read live 2026-09-17: 29349/32607 MiB used
# with both units resident.
TOTAL_GIB = 33_668_726_784 / 2**30  # 31.3564
FREE_TO_A_STARTING_VLLM_GIB = 29_853_679_616 / 2**30  # 27.8034

# The fraction the live unit runs at. Pinned as an exact value, not a range:
# raising it breaks the co-residency budget with vllm-embeddings.service (:8002)
# on the same card, and lowering it drops --max-model-len 131072 below one full
# context of KV.
EXPECTED_GPU_MEMORY_UTILIZATION = 0.86
EXPECTED_MAX_MODEL_LEN = 131_072

# Measured at the 2026-09-17 bind, recorded in the unit's own header.
MEASURED_KV_CACHE_TOKENS = 216_946
MEASURED_MAX_CONCURRENCY = 1.66

# The nvidia-nvvm version cuda-toolkit 13.0.3.0 pins. Above this, cicc emits a
# PTX ISA ptxas refuses.
EXPECTED_NVVM_PIN = "nvidia-nvvm==13.0.88"

# The three FlashInfer warmup switches the unit turns off, all of which must
# stay off or a cold start pays close to an hour of kernel autotuning.
KERNEL_CONFIG_KEYS = (
    "enable_flashinfer_autotune",
    "enable_jit_warmup",
    "enable_cutedsl_warmup",
)


def _service_text() -> str:
    return SERVICE_PATH.read_text(encoding="utf-8")


def _provision_text() -> str:
    return PROVISION_PATH.read_text(encoding="utf-8")


def _exec_start() -> str:
    """The ExecStart line with systemd's backslash continuations folded out."""
    service = _service_text()
    match = re.search(r"^ExecStart=(.*?)(?=\n[A-Z][A-Za-z]*=)", service, re.M | re.S)
    assert match is not None, "unit must declare ExecStart"
    return re.sub(r"\\\s*\n\s*", " ", match.group(1))


def _kernel_config() -> dict[str, object]:
    """Parse the --kernel-config argument out of ExecStart as real JSON.

    This is the assertion that would have caught a single-quoted JSON body.
    systemd passes the value inside single quotes, so the JSON *inside* them
    must use double-quoted keys; single quotes there look plausible to a human
    reader and are not JSON, and vLLM rejects the whole argument at startup.
    """
    match = re.search(r"--kernel-config\s+'([^']*)'", _exec_start())
    assert match is not None, (
        "unit must declare --kernel-config as a single-quoted JSON object; "
        "without it a cold start pays the full sm120 autotune sweep."
    )
    return json.loads(match.group(1))


@pytest.mark.unit
def test_unit_serves_qwen38_27b_on_the_stability_endpoint() -> None:
    """The identity the delegation chain resolves against must not move."""
    exec_start = _exec_start()

    assert "--port 8000" in exec_start
    assert "--served-model-name Qwen3.8-27B" in exec_start
    assert "--model /data/inference/hf-cache/Qwen3.8-27B-NVFP4-RTX5090" in exec_start, (
        "must load the ONE complete weight set; three other Qwen3.8-27B "
        "directories under hf-cache are partial, a drafter, or GGUF."
    )


@pytest.mark.unit
def test_unit_declares_the_measured_max_model_len() -> None:
    """131072 HELD at 0.86 on the 2026-09-17 bind; a silent step-down is a regression."""
    match = re.search(r"--max-model-len\s+(\d+)", _exec_start())
    assert match is not None, "unit must declare --max-model-len"

    assert int(match.group(1)) == EXPECTED_MAX_MODEL_LEN


@pytest.mark.unit
def test_gpu_memory_utilization_is_the_measured_co_residency_fraction() -> None:
    """0.86 is load-bearing in both directions, so it is pinned exactly.

    Upward it breaks the OMN-14379 budget shared with vllm-embeddings.service
    on the same RTX 5090; downward it drops the KV pool below one full
    --max-model-len context. The startup ceiling is restated so the reason the
    value cannot simply be raised travels with the assertion.
    """
    match = re.search(r"--gpu-memory-utilization\s+([0-9.]+)", _exec_start())
    assert match is not None, "unit must declare --gpu-memory-utilization"
    fraction = float(match.group(1))

    assert fraction == EXPECTED_GPU_MEMORY_UTILIZATION

    startup_ceiling = FREE_TO_A_STARTING_VLLM_GIB / TOTAL_GIB  # 0.8867
    assert fraction < startup_ceiling, (
        f"--gpu-memory-utilization {fraction} exceeds the {startup_ceiling:.4f} "
        "ceiling a starting vLLM sees while vllm-embeddings holds cuda:0."
    )


@pytest.mark.unit
def test_kernel_config_is_valid_json_with_double_quoted_keys() -> None:
    """The three warmup switches must parse as JSON and all be false."""
    config = _kernel_config()

    assert set(config) == set(KERNEL_CONFIG_KEYS), (
        f"--kernel-config declares {sorted(config)}, expected "
        f"{sorted(KERNEL_CONFIG_KEYS)}"
    )
    for key in KERNEL_CONFIG_KEYS:
        assert config[key] is False, (
            f"{key} must stay false; turning it on costs close to an hour of "
            "local-serving downtime on every cold start, paid again on any "
            "interruption because the object cache is written only at the end."
        )

    raw = re.search(r"--kernel-config\s+'([^']*)'", _exec_start())
    assert raw is not None
    for key in KERNEL_CONFIG_KEYS:
        assert f'"{key}"' in raw.group(1), (
            f"{key} must be double-quoted inside the JSON object. Single quotes "
            "there read as plausible JSON to a human and are rejected by vLLM."
        )


@pytest.mark.unit
def test_unit_is_type_simple() -> None:
    """vLLM's api_server never forks or notifies; anything else hangs the boot.

    Type=notify would wait out TimeoutStartSec=600 and then kill a server that
    was already serving; Type=forking would mis-track the main PID.
    """
    assert re.search(r"^Type=simple$", _service_text(), re.M), (
        "unit must declare Type=simple"
    )


@pytest.mark.unit
def test_unit_records_the_measured_capacity_ceiling() -> None:
    """1.66x is the planning figure, so it travels with the unit, not a ticket.

    The KV pool holds 216,946 tokens against a 131,072-token context: past one
    full-context request the server serializes rather than degrading.
    """
    service = _service_text()

    assert f"{MEASURED_KV_CACHE_TOKENS:,}" in service, (
        "unit must record the measured GPU KV cache size in tokens"
    )
    assert f"{MEASURED_MAX_CONCURRENCY}x" in service, (
        "unit must record the measured maximum concurrency at full context"
    )
    assert "SERIALIZES" in service, (
        "unit must say what happens past the ceiling, not only what it is"
    )


@pytest.mark.unit
def test_unit_points_at_its_provisioning_script() -> None:
    """The venv is hand-built; the unit must not imply otherwise."""
    service = _service_text()

    assert PROVISION_PATH.name in service
    assert "NOT created by any repo path" in service, (
        "the unit must state plainly that its venv is not repo-provisioned"
    )


@pytest.mark.unit
def test_provisioning_script_pins_nvidia_nvvm() -> None:
    """PTX ISA skew: cicc at 13.4.92 emits 9.4, ptxas accepts at most 9.0."""
    provision = _provision_text()

    assert EXPECTED_NVVM_PIN in provision, (
        f"provisioning must pin {EXPECTED_NVVM_PIN}; an unconstrained "
        "`pip install -U` silently reintroduces the skew."
    )
    assert "13.4.92" in provision, (
        "the version that broke must be named, so the next reader can "
        "recognise the state rather than only the remedy"
    )


@pytest.mark.unit
def test_provisioning_script_recreates_both_cuda_symlinks() -> None:
    """Both are wiped by a reinstall of nvidia-cuda-runtime, with no error."""
    provision = _provision_text()

    assert 'ensure_symlink "${CU13}/lib64" "lib"' in provision, (
        "FlashInfer links against <cu13>/lib64 but the wheel ships lib/"
    )
    assert 'ensure_symlink "${CU13}/lib/libcudart.so" "libcudart.so.13"' in provision, (
        "the wheel ships no unversioned libcudart.so"
    )


@pytest.mark.unit
def test_provisioning_script_is_idempotent_and_has_a_check_mode() -> None:
    """Re-running after any pip operation must be safe and must be possible.

    A repair script that can only be run blind is one nobody runs on a live
    host. ``--check`` is what makes the state auditable without mutating it.
    """
    provision = _provision_text()

    assert "--check" in provision
    assert "ln -sfn" in provision, (
        "symlink creation must overwrite, not fail, on re-run"
    )
    assert "refusing to replace a real file" in provision, (
        "the script must not clobber a real file that occupies a symlink path"
    )
    assert provision.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in provision

    mode = PROVISION_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, f"{PROVISION_PATH} must be executable"
