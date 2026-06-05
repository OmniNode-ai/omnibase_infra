# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.models.llm import ModelLlmInferenceResponse, ModelLlmUsage
from omnibase_infra.models.model_backend_result import ModelBackendResult
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_inference_command import (
    HandlerLlmInferenceCommand,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers.handler_llm_openai_compatible import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.models.model_llm_inference_command import (
    ModelLlmInferenceCommand,
)

pytestmark = pytest.mark.unit


class _FakeInferenceHandler:
    def __init__(self) -> None:
        self.request = None
        self.last_call_metrics = _Metrics()

    async def handle(
        self, request: object, correlation_id: UUID | None = None
    ) -> object:
        self.request = request
        return ModelLlmInferenceResponse(
            generated_text="ok",
            model_used="gemini-2.5-pro",
            operation_type=EnumLlmOperationType.CHAT_COMPLETION,
            finish_reason=EnumLlmFinishReason.STOP,
            usage=ModelLlmUsage(tokens_input=3, tokens_output=2),
            latency_ms=12.0,
            backend_result=ModelBackendResult(success=True, duration_ms=12.0),
            correlation_id=correlation_id or uuid4(),
            execution_id=uuid4(),
            timestamp=datetime.now(UTC),
        )


class _Metrics:
    model_id = "gemini-2.5-pro"
    prompt_tokens = 3
    completion_tokens = 2
    total_tokens = 5
    latency_ms = 12.0

    def model_dump_json(self) -> str:
        return (
            '{"schema_version":"1.0","model_id":"gemini-2.5-pro",'
            '"prompt_tokens":3,"completion_tokens":2,"total_tokens":5,'
            '"estimated_cost_usd":null,"latency_ms":12.0,'
            '"usage_raw":{},"usage_normalized":{},'
            '"usage_is_estimated":false,"input_hash":"abc",'
            '"code_version":"","contract_version":"",'
            '"timestamp_iso":"2026-06-04T18:00:00+00:00",'
            '"reporting_source":"test","extensions":{}}'
        )


@pytest.mark.asyncio
async def test_command_handler_preserves_full_gemini_endpoint_url() -> None:
    fake_handler = _FakeInferenceHandler()
    handler = HandlerLlmInferenceCommand(
        inference_handler=fake_handler,  # type: ignore[arg-type]
    )
    command = ModelLlmInferenceCommand(
        correlation_id=UUID("11111111-1111-4111-8111-111111111111"),
        model="gemini-2.5-pro",
        messages=({"role": "user", "content": "ping"},),
        provider_config={
            "endpoint_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            "api_key": "test-key",
        },
    )

    output = await handler.handle(command)

    assert fake_handler.request is not None
    request = fake_handler.request
    assert request.endpoint_url == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert request.base_url == request.endpoint_url
    assert HandlerLlmOpenaiCompatible._build_url(request) == request.endpoint_url
    assert len(output.events) == 3
    assert isinstance(output.events[0], ModelLlmInferenceResponse)
    assert output.events[0].generated_text == "ok"
    assert output.events[0].correlation_id == command.correlation_id


def test_command_handler_rejects_missing_endpoint_contract() -> None:
    handler = HandlerLlmInferenceCommand(
        inference_handler=_FakeInferenceHandler(),  # type: ignore[arg-type]
    )
    command = ModelLlmInferenceCommand(
        model="gemini-2.5-pro",
        messages=({"role": "user", "content": "ping"},),
    )

    with pytest.raises(ValueError, match="requires endpoint_url or base_url"):
        handler._build_request(command)
