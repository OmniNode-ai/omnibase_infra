# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for overseer scoped adapter package exports."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, cast

import pytest

from omnibase_infra.adapters.llm.model_llm_adapter_request import ModelLlmAdapterRequest
from omnibase_infra.adapters.overseer import AdapterLlmProviderScoped
from omnibase_infra.errors import InvariantViolation

if TYPE_CHECKING:
    from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
        AdapterLlmProviderOpenai,
    )

pytestmark = pytest.mark.integration


class _StreamingProvider:
    provider_name = "integration-provider"
    provider_type = "local"
    is_available = True

    def __init__(self) -> None:
        self.supports_streaming_called = False
        self.stream_started = False

    def supports_streaming(self) -> bool:
        self.supports_streaming_called = True
        return True

    async def generate_stream_async(
        self,
        request: ModelLlmAdapterRequest,
    ) -> AsyncGenerator[str, None]:
        self.stream_started = True
        yield "first"
        yield "second"


@pytest.mark.asyncio
async def test_scoped_llm_public_export_preserves_async_streaming() -> None:
    inner = _StreamingProvider()
    scoped = AdapterLlmProviderScoped(
        inner=cast("AdapterLlmProviderOpenai", inner),
        allowed_actions=frozenset({"generate_stream", "generate_stream_async"}),
    )

    assert scoped.supports_streaming() is True

    chunks = [
        chunk
        async for chunk in scoped.generate_stream_async(
            cast("ModelLlmAdapterRequest", object())
        )
    ]

    assert chunks == ["first", "second"]
    assert inner.supports_streaming_called is True
    assert inner.stream_started is True


def test_scoped_llm_public_export_denies_stream_capability_probe() -> None:
    inner = _StreamingProvider()
    scoped = AdapterLlmProviderScoped(
        inner=cast("AdapterLlmProviderOpenai", inner),
        allowed_actions=frozenset(),
    )

    with pytest.raises(InvariantViolation):
        scoped.supports_streaming()

    assert inner.supports_streaming_called is False
