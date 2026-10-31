# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declared timeouts reach HTTPX through real adapter/handler requests."""

from __future__ import annotations

import httpx
import pytest
from pytest_httpserver import HTTPServer

from omnibase_infra.adapters.llm.adapter_code_review_analysis import (
    AdapterCodeReviewAnalysis,
)
from omnibase_infra.adapters.llm.adapter_documentation_generation import (
    AdapterDocumentationGeneration,
)
from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
    TransportHolderLlmHttp,
)
from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
    AdapterLinearGraphQLProjectTracker,
)
from omnibase_infra.models.outbound_timeouts import load_outbound_timeouts

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("override", [None, 7.25])
async def test_tracker_timeout_reaches_a_real_graphql_request(
    httpserver: HTTPServer, override: float | None
) -> None:
    httpserver.expect_oneshot_request("/graphql", method="POST").respond_with_json(
        {"data": {"viewer": {"id": "timeout-probe"}}}
    )
    observed: list[object] = []

    async def record_timeout(request: httpx.Request) -> None:
        observed.append(request.extensions["timeout"])

    adapter = (
        AdapterLinearGraphQLProjectTracker(
            api_key="loopback-test", endpoint=httpserver.url_for("/graphql")
        )
        if override is None
        else AdapterLinearGraphQLProjectTracker(
            api_key="loopback-test",
            endpoint=httpserver.url_for("/graphql"),
            timeout_seconds=override,
        )
    )
    adapter._client.event_hooks["request"].append(record_timeout)
    try:
        assert await adapter.connect() is True
        assert (await adapter.health_check()).status == "healthy"
    finally:
        await adapter.close()

    expected = (
        load_outbound_timeouts().linear_graphql_request_seconds
        if override is None
        else override
    )
    assert observed == [httpx.Timeout(expected).as_dict()]
    assert adapter._client.is_closed
    httpserver.check_assertions()


@pytest.mark.parametrize(
    "documentation", [False, True], ids=["review", "documentation"]
)
async def test_llm_timeout_reaches_the_http_transport(
    httpserver: HTTPServer, monkeypatch: pytest.MonkeyPatch, documentation: bool
) -> None:
    monkeypatch.setenv("LLM_ENDPOINT_CIDR_ALLOWLIST", "127.0.0.0/8")
    monkeypatch.setenv("LOCAL_LLM_SHARED_SECRET", "loopback-timeout-probe")
    monkeypatch.setattr(TransportHolderLlmHttp, "_LOCAL_LLM_CIDRS", None)
    httpserver.expect_oneshot_request("/completion", method="POST").respond_with_json(
        {
            "id": "timeout-probe",
            "model": "timeout-probe-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Probe response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }
    )
    endpoint = httpserver.url_for("/completion")
    adapter = (
        AdapterDocumentationGeneration(base_url=endpoint)
        if documentation
        else AdapterCodeReviewAnalysis(base_url=endpoint)
    )
    observed: list[object] = []

    async def record_timeout(request: httpx.Request) -> None:
        observed.append(request.extensions["timeout"])

    client = await adapter._transport._get_http_client()
    client.event_hooks["request"].append(record_timeout)
    try:
        if isinstance(adapter, AdapterDocumentationGeneration):
            result = await adapter.generate(
                task_type="docstring", source="def probe(): pass"
            )
        else:
            result = await adapter.review(
                review_type="naming", code_diff="+def probe(): pass"
            )
        assert result.rendered_text == "Probe response"
    finally:
        await adapter.close()

    timeouts = load_outbound_timeouts()
    expected = (
        timeouts.llm_documentation_request_seconds
        if documentation
        else timeouts.llm_request_seconds
    )
    assert observed == [httpx.Timeout(expected).as_dict()]
    assert client.is_closed
    httpserver.check_assertions()
