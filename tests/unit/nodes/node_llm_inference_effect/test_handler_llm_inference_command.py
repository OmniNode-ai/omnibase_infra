# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.enums import EnumLlmFinishReason, EnumLlmOperationType
from omnibase_infra.errors import SecretResolutionError
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
from omnibase_infra.runtime.models.model_secret_mapping import ModelSecretMapping
from omnibase_infra.runtime.models.model_secret_resolver_config import (
    ModelSecretResolverConfig,
)
from omnibase_infra.runtime.models.model_secret_source_spec import (
    ModelSecretSourceSpec,
)
from omnibase_infra.runtime.secret_resolver import SecretResolver

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


_REF = "llm.gemini.api_key"
_ENV_VAR = "OMN17106_TEST_LLM_KEY"
_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def _resolver() -> SecretResolver:
    return SecretResolver(
        config=ModelSecretResolverConfig(
            mappings=[
                ModelSecretMapping(
                    logical_name=_REF,
                    source=ModelSecretSourceSpec(
                        source_type="env", source_path=_ENV_VAR
                    ),
                )
            ]
        )
    )


def _build_handler(
    fake: _FakeInferenceHandler, resolver: SecretResolver | None = None
) -> HandlerLlmInferenceCommand:
    return HandlerLlmInferenceCommand(
        inference_handler=cast("HandlerLlmOpenaiCompatible", fake),
        secret_resolver=resolver,
    )


def _ref_command(**overrides: object) -> ModelLlmInferenceCommand:
    fields: dict[str, object] = {
        "model": "gemini-2.5-pro",
        "messages": ({"role": "user", "content": "ping"},),
        "endpoint_url": _ENDPOINT,
        "api_key_ref": _REF,
    }
    fields.update(overrides)
    return ModelLlmInferenceCommand.model_validate(fields)


def test_command_refuses_a_literal_api_key_field() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ModelLlmInferenceCommand.model_validate(
            {"model": "m", "endpoint_url": _ENDPOINT, "api_key": "sk-literal"}
        )
    assert "sk-literal" not in str(excinfo.value)


@pytest.mark.parametrize(
    "key", ["api_key", "apiKey", "authorization", "access_token", "client_secret"]
)
def test_command_refuses_a_literal_credential_in_provider_config(key: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        ModelLlmInferenceCommand.model_validate(
            {
                "model": "m",
                "endpoint_url": _ENDPOINT,
                "provider_config": {key: "sk-literal"},
            }
        )
    assert "sk-literal" not in str(excinfo.value)


def test_command_allows_non_credential_provider_config_keys() -> None:
    command = ModelLlmInferenceCommand.model_validate(
        {
            "model": "m",
            "endpoint_url": _ENDPOINT,
            "provider_config": {"max_tokens": 5, "api_key_ref": _REF},
        }
    )
    assert command.provider_config["max_tokens"] == 5


def test_command_serialises_the_ref_never_a_value() -> None:
    dumped = _ref_command().model_dump_json()
    assert _REF in dumped
    assert '"api_key"' not in dumped


@pytest.mark.asyncio
async def test_handler_resolves_api_key_ref_at_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_ENV_VAR, "resolved-secret-value")
    fake_handler = _FakeInferenceHandler()
    handler = _build_handler(fake_handler, _resolver())

    await handler.handle(_ref_command())

    assert fake_handler.request is not None
    assert fake_handler.request.api_key.get_secret_value() == "resolved-secret-value"


@pytest.mark.asyncio
async def test_handler_refuses_a_ref_it_cannot_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_ENV_VAR, raising=False)
    fake_handler = _FakeInferenceHandler()
    handler = _build_handler(fake_handler, _resolver())

    with pytest.raises(SecretResolutionError):
        await handler.handle(_ref_command())

    assert fake_handler.request is None


@pytest.mark.asyncio
async def test_handler_without_a_resolver_refuses_a_ref_instead_of_sending_unauthenticated() -> (
    None
):
    fake_handler = _FakeInferenceHandler()
    handler = _build_handler(fake_handler)

    with pytest.raises(SecretResolutionError):
        await handler.handle(_ref_command())

    assert fake_handler.request is None


@pytest.mark.asyncio
async def test_handler_sends_no_auth_when_the_command_names_no_ref() -> None:
    fake_handler = _FakeInferenceHandler()
    handler = _build_handler(fake_handler)

    await handler.handle(_ref_command(api_key_ref=None))

    assert fake_handler.request is not None
    assert fake_handler.request.api_key is None
