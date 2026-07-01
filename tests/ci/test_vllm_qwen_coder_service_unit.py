# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Regression guard for the Qwen coder vLLM systemd unit."""

from __future__ import annotations

from pathlib import Path

import pytest

SERVICE_PATH = Path("deploy/systemd/vllm-gpu0-qwen-coder.service")


@pytest.mark.unit
def test_qwen_coder_vllm_unit_declares_tool_call_parser_flags() -> None:
    service = SERVICE_PATH.read_text(encoding="utf-8")

    assert "--enable-auto-tool-choice" in service
    assert "--tool-call-parser qwen3_coder" in service


@pytest.mark.unit
def test_qwen_coder_vllm_unit_preserves_stability_endpoint_identity() -> None:
    service = SERVICE_PATH.read_text(encoding="utf-8")

    assert "--port 8000" in service
    assert "--served-model-name Qwen3.6-35B-A3B" in service
    assert "--max-model-len 131072" in service
