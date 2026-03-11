# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for AdapterLlmProviderOpenai.generate_stream() sync method (OMN-4483)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
    AdapterLlmProviderOpenai,
)
from omnibase_infra.adapters.llm.model_llm_adapter_request import ModelLlmAdapterRequest


def _make_request() -> ModelLlmAdapterRequest:
    return ModelLlmAdapterRequest(
        prompt="Say hello",
        model_name="test-model",
        max_tokens=64,
        temperature=0.0,
    )


@pytest.mark.unit
def test_generate_stream_yields_chunks() -> None:
    """Sync stream must yield all chunks from the async generator."""
    adapter = AdapterLlmProviderOpenai.__new__(AdapterLlmProviderOpenai)

    async def fake_stream(request: ModelLlmAdapterRequest) -> object:
        for chunk in ["Hello", " world", "!"]:
            yield chunk

    request = _make_request()
    with patch.object(adapter, "generate_stream_async", side_effect=fake_stream):
        results = list(adapter.generate_stream(request))

    assert results == ["Hello", " world", "!"]


@pytest.mark.unit
def test_generate_stream_closes_generator_on_completion() -> None:
    """Async generator finalizer must run after stream is consumed."""
    adapter = AdapterLlmProviderOpenai.__new__(AdapterLlmProviderOpenai)
    close_called: list[bool] = []

    async def fake_stream_with_close_check(
        request: ModelLlmAdapterRequest,
    ) -> object:
        try:
            for chunk in ["A", "B"]:
                yield chunk
        finally:
            close_called.append(True)

    request = _make_request()
    with patch.object(
        adapter, "generate_stream_async", side_effect=fake_stream_with_close_check
    ):
        list(adapter.generate_stream(request))

    assert close_called, "Async generator finalizer must run after stream is consumed"
