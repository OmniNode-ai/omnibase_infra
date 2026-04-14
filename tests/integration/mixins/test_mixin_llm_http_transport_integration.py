# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for MixinLlmHttpTransport (OMN-8697).

Validates the LLM HTTP transport mixin initializes correctly, enforces
CIDR allowlist validation, and maps HTTP status codes to typed exceptions.
Tests do not require an external LLM server.
"""

from __future__ import annotations

import json
import os
from ipaddress import IPv4Network
from uuid import UUID, uuid4

import httpx
import pytest

from omnibase_infra.errors import InfraAuthenticationError, InfraUnavailableError
from omnibase_infra.mixins.mixin_async_circuit_breaker import MixinAsyncCircuitBreaker
from omnibase_infra.mixins.mixin_llm_http_transport import MixinLlmHttpTransport
from omnibase_infra.mixins.mixin_retry_execution import MixinRetryExecution

_CID: UUID = uuid4()


class _LlmTransportHarness(
    MixinLlmHttpTransport, MixinAsyncCircuitBreaker, MixinRetryExecution
):
    """Minimal test harness for MixinLlmHttpTransport."""


def _make_harness(
    mock_transport: httpx.MockTransport | None = None,
) -> _LlmTransportHarness:
    client = httpx.AsyncClient(transport=mock_transport) if mock_transport else None
    h = _LlmTransportHarness()
    h._init_llm_http_transport(
        target_name="test-llm",
        max_timeout_seconds=5.0,
        http_client=client,
    )
    return h


def test_init_sets_target_name() -> None:
    h = _make_harness()
    assert h._llm_target_name == "test-llm"


def test_init_sets_max_timeout() -> None:
    h = _make_harness()
    assert h._max_timeout_seconds == 5.0


@pytest.mark.asyncio
async def test_cidr_allowlist_rejects_external_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ENDPOINT_CIDR_ALLOWLIST", "192.168.86.0/24")
    h = _make_harness()
    h._reload_cidr_allowlist()

    with pytest.raises(InfraAuthenticationError):
        await h._validate_endpoint_allowlist("http://8.8.8.8:8000/v1/completions", _CID)


@pytest.mark.asyncio
async def test_cidr_allowlist_accepts_local_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_ENDPOINT_CIDR_ALLOWLIST", "192.168.86.0/24")
    monkeypatch.setenv("LOCAL_LLM_SHARED_SECRET", "test-secret-value-1234")
    h = _make_harness()
    h._reload_cidr_allowlist()
    # Should not raise for local subnet
    await h._validate_endpoint_allowlist(
        "http://192.168.86.201:8000/v1/completions", _CID
    )


@pytest.mark.asyncio
async def test_http_500_maps_to_infra_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_ENDPOINT_CIDR_ALLOWLIST", "0.0.0.0/0")
    monkeypatch.setenv("LOCAL_LLM_SHARED_SECRET", "test-secret-value-1234")

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    h = _make_harness(mock_transport=httpx.MockTransport(_handler))
    h._reload_cidr_allowlist()

    with pytest.raises(InfraUnavailableError):
        await h._execute_llm_http_call(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"model": "test", "messages": []},
            correlation_id=_CID,
            max_retries=0,
            timeout_seconds=2.0,
        )
