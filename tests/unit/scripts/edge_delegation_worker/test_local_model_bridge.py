# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts.edge_delegation_worker.local_model_bridge."""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.errors import (
    InfraProtocolError,
    InfraRequestRejectedError,
    InfraUnavailableError,
)
from scripts.edge_delegation_worker.local_model_bridge import run_chat_completion
from scripts.edge_delegation_worker.models import ModelLocalInferenceRequest

pytestmark = pytest.mark.unit


@pytest.fixture
async def http_client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        yield client


def _request() -> ModelLocalInferenceRequest:
    return ModelLocalInferenceRequest(
        correlation_id=uuid4(),
        model="qwen-local",
        messages=({"role": "user", "content": "hello"},),
    )


@pytest.mark.asyncio
async def test_run_chat_completion_success(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/chat/completions", method="POST"
    ).respond_with_json(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi there"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
    )

    request = _request()
    result = await run_chat_completion(
        httpserver.url_for(""),  # type: ignore[attr-defined]
        request,
        http_client=http_client,
    )
    assert result.correlation_id == request.correlation_id
    assert result.content == "hi there"
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 2


@pytest.mark.asyncio
async def test_run_chat_completion_4xx_raises_request_rejected(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/chat/completions", method="POST"
    ).respond_with_json({"error": "context length exceeded"}, status=400)

    with pytest.raises(InfraRequestRejectedError):
        await run_chat_completion(
            httpserver.url_for(""),  # type: ignore[attr-defined]
            _request(),
            http_client=http_client,
        )


@pytest.mark.asyncio
async def test_run_chat_completion_5xx_raises_protocol_error(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/chat/completions", method="POST"
    ).respond_with_data("internal error", status=500)

    with pytest.raises(InfraProtocolError):
        await run_chat_completion(
            httpserver.url_for(""),  # type: ignore[attr-defined]
            _request(),
            http_client=http_client,
        )


@pytest.mark.asyncio
async def test_run_chat_completion_missing_choices_fails_closed(
    httpserver: object, http_client: httpx.AsyncClient
) -> None:
    httpserver.expect_request(  # type: ignore[attr-defined]
        "/v1/chat/completions", method="POST"
    ).respond_with_json({"choices": []})

    with pytest.raises(InfraProtocolError):
        await run_chat_completion(
            httpserver.url_for(""),  # type: ignore[attr-defined]
            _request(),
            http_client=http_client,
        )


@pytest.mark.asyncio
async def test_run_chat_completion_unreachable_host_raises_unavailable(
    http_client: httpx.AsyncClient,
) -> None:
    with pytest.raises(InfraUnavailableError):
        await run_chat_completion(
            "http://127.0.0.1:1",
            _request(),
            http_client=http_client,
        )
