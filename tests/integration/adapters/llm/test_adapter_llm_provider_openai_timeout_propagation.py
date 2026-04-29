# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""End-to-end integration test for OMN-9869 LLM adapter timeout propagation.

Exercises the full adapter -> handler -> transport call chain with a
recording transport substitute, asserting that the contract-owned
``ModelLlmAdapterRequest.timeout_seconds`` value reaches the HTTP
transport layer's ``_execute_llm_http_call`` (the boundary where
``httpx.Client.post`` would issue the network call).

This is the integration counterpart to the unit translation test in
``tests/unit/adapters/llm/test_adapter_llm_provider_openai.py``: the unit
test verifies the field hand-off in ``_translate_request``; this test
verifies the same value survives the handler delegation path that runs
on every real generation call.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest

from omnibase_core.types import JsonType
from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
    AdapterLlmProviderOpenai,
    TransportHolderLlmHttp,
)
from omnibase_infra.adapters.llm.model_llm_adapter_request import (
    ModelLlmAdapterRequest,
)


class _RecordingTransport(TransportHolderLlmHttp):
    """Transport double that records ``timeout_seconds`` on each call.

    Bypasses real HTTP, allowlist, and HMAC checks so the test stays
    hermetic and observes the exact value the handler forwards from the
    inference request.
    """

    def __init__(self) -> None:
        super().__init__(
            target_name="omn-9869-recorder",
            max_timeout_seconds=600.0,
        )
        self.calls: list[dict[str, Any]] = []

    async def _execute_llm_http_call(
        self,
        url: str,
        payload: dict[str, JsonType],
        correlation_id: UUID,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
    ) -> dict[str, JsonType]:
        self.calls.append(
            {
                "url": url,
                "timeout_seconds": timeout_seconds,
                "model": payload.get("model"),
            }
        )
        # Minimal OpenAI-shape chat completion response so the handler can
        # finish parsing without raising.
        return {
            "id": "chatcmpl-omn-9869",
            "object": "chat.completion",
            "model": payload.get("model", "qwen2.5-coder-14b"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }


@pytest.mark.asyncio
async def test_explicit_timeout_reaches_http_transport_layer() -> None:
    """Caller-provided ``timeout_seconds`` propagates to ``_execute_llm_http_call``.

    Regression guard for OMN-9869: prior to the fix the adapter dropped
    the contract-owned timeout, so the inference layer always used its
    30-second default and long-running local-model calls failed before
    the node-contract timeout window.
    """
    adapter = AdapterLlmProviderOpenai(
        base_url="http://localhost:8000",
        default_model="qwen2.5-coder-14b",
    )
    recorder = _RecordingTransport()
    adapter._transport = recorder
    # Re-bind the handler so it uses the recording transport instead of the
    # one created during ``AdapterLlmProviderOpenai.__init__``.
    adapter._handler._transport = recorder

    try:
        await adapter.generate_async(
            ModelLlmAdapterRequest(
                prompt="Long-running prompt",
                model_name="qwen2.5-coder-14b",
                timeout_seconds=240.0,
            )
        )
    finally:
        await recorder.close()

    assert recorder.calls, "expected at least one HTTP call"
    assert recorder.calls[0]["timeout_seconds"] == 240.0


@pytest.mark.asyncio
async def test_default_timeout_reaches_http_transport_layer() -> None:
    """Omitting ``timeout_seconds`` propagates the 30.0 default end-to-end."""
    adapter = AdapterLlmProviderOpenai(
        base_url="http://localhost:8000",
        default_model="qwen2.5-coder-14b",
    )
    recorder = _RecordingTransport()
    adapter._transport = recorder
    adapter._handler._transport = recorder

    try:
        await adapter.generate_async(
            ModelLlmAdapterRequest(
                prompt="Hello",
                model_name="qwen2.5-coder-14b",
            )
        )
    finally:
        await recorder.close()

    assert recorder.calls, "expected at least one HTTP call"
    assert recorder.calls[0]["timeout_seconds"] == 30.0
