# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A 402 from a tool endpoint is a typed payment refusal (OMN-20610).

Driven through both adapters, ``ONEXToMCPAdapter.invoke_tool`` and
``AdapterONEXToolExecution.execute``, with an ``httpx.MockTransport`` standing
in for the endpoint. The 402 body carries a ``payTo`` value that must never
reach the tool result.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest

from omnibase_infra.adapters.adapter_onex_tool_execution import (
    PAYMENT_REFUSED_MESSAGE,
    AdapterONEXToolExecution,
)
from omnibase_infra.errors import InfraConnectionError, InfraPaymentRequiredError
from omnibase_infra.handlers.mcp.adapter_onex_to_mcp import (
    MCPToolParameter,
    ONEXToMCPAdapter,
)
from omnibase_infra.models.mcp.model_mcp_tool_definition import ModelMCPToolDefinition

pytestmark = [pytest.mark.unit]

PAY_TO = "0xPAYTOSECRET1234567890abcdef"
BODY = json.dumps({"x402Version": 2, "accepts": [{"payTo": PAY_TO}]}).encode()
ENDPOINT = "http://tool.invalid/execute"


def _client(status: int, hits: list[int]) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(1)
        return httpx.Response(
            status,
            content=BODY,
            headers={"payment-required": f"hdr-{PAY_TO}"} if status == 402 else {},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _tool() -> ModelMCPToolDefinition:
    return ModelMCPToolDefinition(
        name="paid_tool", description="d", endpoint=ENDPOINT, timeout_seconds=5
    )


class TestExecutePaymentRefusal:
    async def test_402_records_no_breaker_failure(self) -> None:
        hits: list[int] = []
        adapter = AdapterONEXToolExecution(
            container=MagicMock(), http_client=_client(402, hits)
        )
        before = adapter._circuit_breaker_failures
        raw = await adapter.execute(_tool(), {"a": 1}, uuid4())
        assert adapter._circuit_breaker_failures == before == 0
        assert len(hits) == 1
        assert raw["success"] is False
        assert raw["payment_refused"] is True

    async def test_402_result_has_fixed_sentence_and_no_body_byte(self) -> None:
        adapter = AdapterONEXToolExecution(
            container=MagicMock(), http_client=_client(402, [])
        )
        raw = await adapter.execute(_tool(), {}, uuid4())
        assert raw["error"] == PAYMENT_REFUSED_MESSAGE
        assert PAY_TO not in json.dumps(raw, default=str)

    async def test_500_still_records_breaker_failure(self) -> None:
        adapter = AdapterONEXToolExecution(
            container=MagicMock(), http_client=_client(500, [])
        )
        raw = await adapter.execute(_tool(), {}, uuid4())
        assert adapter._circuit_breaker_failures == 1
        assert "payment_refused" not in raw

    async def test_dispatch_raises_typed_error_with_response_digest(self) -> None:
        adapter = AdapterONEXToolExecution(
            container=MagicMock(), http_client=_client(402, [])
        )
        with pytest.raises(InfraPaymentRequiredError) as exc_info:
            await adapter._http_dispatch(ENDPOINT, {}, 5.0, uuid4())
        assert not isinstance(exc_info.value, InfraConnectionError)
        assert PAY_TO not in str(exc_info.value)
        assert exc_info.value.body_byte_length == len(BODY)


class TestInvokeToolPaymentRefusal:
    async def _invoke(self, hits: list[int]) -> tuple[dict[str, object], int]:
        execution = AdapterONEXToolExecution(
            container=MagicMock(), http_client=_client(402, hits)
        )
        adapter = ONEXToMCPAdapter(node_executor=execution)
        await adapter.register_node_as_tool(
            node_name="paid_tool",
            description="d",
            parameters=[
                MCPToolParameter(
                    name="a", parameter_type="string", description="a", required=True
                )
            ],
            version="1.0.0",
            timeout_seconds=5,
        )
        adapter._tool_cache["paid_tool"].execution_endpoint = ENDPOINT
        result = await adapter.invoke_tool("paid_tool", {"a": "x"}, uuid4())
        return result, execution._circuit_breaker_failures

    async def test_fixed_sentence_is_error_and_not_rewrapped(self) -> None:
        result, failures = await self._invoke([])
        assert result["isError"] is True
        assert result["content"] == [{"type": "text", "text": PAYMENT_REFUSED_MESSAGE}]
        text = json.dumps(result)
        assert "Unexpected error" not in text
        assert "Connection error" not in text
        assert failures == 0

    async def test_no_body_byte_in_tool_result(self) -> None:
        result, _ = await self._invoke([])
        assert PAY_TO not in json.dumps(result)

    async def test_typed_error_escaping_executor_is_also_fixed_sentence(self) -> None:
        executor = AdapterONEXToolExecution(container=MagicMock())
        adapter = ONEXToMCPAdapter(node_executor=executor)
        await adapter.register_node_as_tool(
            node_name="paid_tool",
            description="d",
            parameters=[],
            version="1.0.0",
            timeout_seconds=5,
        )
        adapter._tool_cache["paid_tool"].execution_endpoint = ENDPOINT
        raising = AsyncMock(
            side_effect=InfraPaymentRequiredError(
                f"Payment required {PAY_TO}", raw_body=BODY
            )
        )
        with patch.object(executor, "execute", new=raising):
            result = await adapter.invoke_tool("paid_tool", {}, uuid4())
        assert result["isError"] is True
        assert result["content"] == [{"type": "text", "text": PAYMENT_REFUSED_MESSAGE}]
        assert PAY_TO not in json.dumps(result)
