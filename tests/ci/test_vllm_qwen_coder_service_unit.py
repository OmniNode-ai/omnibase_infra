# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression guard for the Qwen coder vLLM systemd unit."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

SERVICE_PATH = Path("deploy/systemd/vllm-gpu0-qwen-coder.service")

# --- Measured cuda:0 facts, .201, 2026-09-03 (OMN-14379) --------------------
# Source of every number below is recorded in the unit file's own header, which
# is the artifact under test. They are duplicated here so the invariant is
# mechanically enforced rather than merely written down in a comment.
#
#   nvidia-smi --query-gpu=memory.total          -> 32607 MiB
#   torch.cuda.mem_get_info(0)                   -> (29853679616, 33668726784)
#   nvidia-smi --query-compute-apps=pid,used_gpu_memory -> 966377, 3112 MiB
TOTAL_GIB = 33_668_726_784 / 2**30  # 31.3564 — torch's view of the RTX 5090
FREE_TO_A_STARTING_VLLM_GIB = 29_853_679_616 / 2**30  # 27.8034, embeddings resident

# vllm-embeddings.service (:8002) shares cuda:0 since the 4090 RMA (OMN-16407).
# That unit is NOT repo-owned yet; its fraction is pinned here so a change to
# either side of the co-residency budget breaks this test.
EMBEDDINGS_FRACTION = 0.08

# vLLM v1 refuses to start unless free >= gpu_memory_utilization * total.
STARTUP_CEILING = FREE_TO_A_STARTING_VLLM_GIB / TOTAL_GIB  # 0.8867

# A fraction may sit at most this close to the hard ceiling. Below this the fix
# degrades from "permanently broken" to "intermittently broken" — the embedding
# server grows under load and the allocator fragments across restarts.
MIN_STARTUP_MARGIN_GIB = 0.5

# One full --max-model-len 131072 sequence of KV must still fit inside the
# budget. 10 full_attention layers x 2 (K+V) x 2 kv-heads x 256 head_dim x 2 B
# = 20,480 B/token; the other 30 layers are linear_attention (constant state).
KV_BYTES_PER_TOKEN = 10 * 2 * 2 * 256 * 2
# Everything in the budget that is NOT KV cache: 22.743 GiB safetensors less the
# 1.573 GiB MTP module (not loaded without --speculative-config), plus CUDA
# context and activation peak. MEASURED, not estimated: at 0.86 vLLM 0.21.0
# reported "Available KV cache memory: 3.71 GiB" on 2026-09-03, so the non-KV
# footprint is 0.86 * TOTAL_GIB - 3.71.
NON_KV_FOOTPRINT_GIB = 23.26


def _service_text() -> str:
    return SERVICE_PATH.read_text(encoding="utf-8")


def _gpu_memory_utilization() -> float:
    match = re.search(r"--gpu-memory-utilization\s+([0-9.]+)", _service_text())
    assert match is not None, "unit must declare --gpu-memory-utilization"
    return float(match.group(1))


@pytest.mark.unit
def test_qwen_coder_vllm_unit_declares_tool_call_parser_flags() -> None:
    service = _service_text()

    assert "--enable-auto-tool-choice" in service
    assert "--tool-call-parser qwen3_coder" in service


@pytest.mark.unit
def test_qwen_coder_vllm_unit_preserves_stability_endpoint_identity() -> None:
    service = _service_text()

    assert "--port 8000" in service
    assert "--served-model-name Qwen3.6-35B-A3B" in service
    assert "--max-model-len 131072" in service


@pytest.mark.unit
def test_gpu_memory_utilization_clears_the_vllm_startup_check() -> None:
    """OMN-14379: 0.92 could never start while vllm-embeddings held cuda:0.

    vLLM v1 raises ValueError when free memory at startup is below
    `gpu_memory_utilization * total`. Free is measured AFTER this process's own
    CUDA context exists, so the check is stricter than the physical fit — which
    is exactly why 0.92 + 0.08 "summing to 1.00" was not merely tight but
    impossible.
    """
    fraction = _gpu_memory_utilization()
    requested_gib = fraction * TOTAL_GIB
    margin_gib = FREE_TO_A_STARTING_VLLM_GIB - requested_gib

    assert fraction < STARTUP_CEILING, (
        f"--gpu-memory-utilization {fraction} requests {requested_gib:.3f} GiB but only "
        f"{FREE_TO_A_STARTING_VLLM_GIB:.3f} GiB is free to a starting vLLM process on "
        f"cuda:0 while vllm-embeddings.service is resident. Hard ceiling is "
        f"{STARTUP_CEILING:.4f}."
    )
    assert margin_gib >= MIN_STARTUP_MARGIN_GIB, (
        f"--gpu-memory-utilization {fraction} leaves only {margin_gib:.3f} GiB of startup "
        f"margin; at least {MIN_STARTUP_MARGIN_GIB} GiB is required so the embedding "
        f"server's growth under load cannot reintroduce the crash-loop."
    )


@pytest.mark.unit
def test_gpu_memory_utilization_is_co_resident_with_the_embedding_server() -> None:
    """The two cuda:0 fractions plus BOTH CUDA contexts must fit in one device."""
    fraction = _gpu_memory_utilization()

    assert fraction + EMBEDDINGS_FRACTION < 1.0, (
        f"--gpu-memory-utilization {fraction} plus vllm-embeddings' "
        f"{EMBEDDINGS_FRACTION} leaves no room for either process's CUDA context, "
        "which is allocated outside the fraction."
    )


@pytest.mark.unit
def test_gpu_memory_utilization_still_holds_one_full_context_of_kv() -> None:
    """Shrinking the fraction must not silently make --max-model-len unservable."""
    service = _service_text()
    max_model_len = int(re.search(r"--max-model-len\s+(\d+)", service).group(1))  # type: ignore[union-attr]

    kv_needed_gib = (max_model_len * KV_BYTES_PER_TOKEN) / 2**30
    kv_available_gib = _gpu_memory_utilization() * TOTAL_GIB - NON_KV_FOOTPRINT_GIB

    assert kv_available_gib >= kv_needed_gib, (
        f"budget leaves {kv_available_gib:.3f} GiB of KV cache but --max-model-len "
        f"{max_model_len} needs {kv_needed_gib:.3f} GiB for a single sequence; vLLM "
        "would refuse to start."
    )


@pytest.mark.unit
def test_unit_records_the_co_residency_arithmetic() -> None:
    """The numbers must travel with the unit, not only in a ticket comment."""
    service = _service_text()

    assert "OMN-14379" in service
    assert "vllm-embeddings" in service
    assert "mem_get_info" in service
