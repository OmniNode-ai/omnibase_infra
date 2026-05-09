# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Scope-enforcing wrapper adapter for ProtocolLLMProvider.

Wraps AdapterLlmProviderOpenai with an allowed-action set. Any call to an
action not in the set raises InvariantViolation before the underlying adapter
is reached.

Related Tickets:
    - OMN-8065: Task 7 — Scoped wrapper adapters for TicketService, EventBus, LLMProvider
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Iterator
from typing import TYPE_CHECKING

from omnibase_infra.adapters.llm.adapter_llm_provider_openai import (
    AdapterLlmProviderOpenai,
)
from omnibase_infra.adapters.llm.model_llm_adapter_request import ModelLlmAdapterRequest
from omnibase_infra.adapters.llm.model_llm_adapter_response import (
    ModelLlmAdapterResponse,
)
from omnibase_infra.adapters.llm.model_llm_health_response import ModelLlmHealthResponse
from omnibase_infra.errors.error_invariant_violation import InvariantViolation

if TYPE_CHECKING:
    from omnibase_core.types import JsonType
    from omnibase_infra.adapters.llm.model_llm_model_capabilities import (
        ModelLlmModelCapabilities,
    )
    from omnibase_infra.adapters.llm.model_llm_provider_config import (
        ModelLlmProviderConfig,
    )

_PROTOCOL_DOMAIN = "llm_provider"

_ACTION_GENERATE = "generate"
_ACTION_GENERATE_ASYNC = "generate_async"
_ACTION_GENERATE_STREAM = "generate_stream"
_ACTION_GENERATE_STREAM_ASYNC = "generate_stream_async"
_ACTION_HEALTH_CHECK = "health_check"
_ACTION_CLOSE = "close"
_ACTION_CONFIGURE = "configure"
_ACTION_GET_AVAILABLE_MODELS = "get_available_models"
_ACTION_GET_MODEL_CAPABILITIES = "get_model_capabilities"
_ACTION_ESTIMATE_COST = "estimate_cost"
_ACTION_VALIDATE_REQUEST = "validate_request"
_ACTION_GET_PROVIDER_INFO = "get_provider_info"


class AdapterLlmProviderScoped:
    """Scope-enforcing wrapper around AdapterLlmProviderOpenai.

    Delegates all method calls to the wrapped adapter after checking whether
    the action is present in ``allowed_actions``.  Raises InvariantViolation
    on any disallowed action before any network I/O occurs.

    Args:
        inner: The AdapterLlmProviderOpenai to delegate to.
        allowed_actions: Frozenset of action name strings that callers may
            invoke.  Pass an empty frozenset to deny all actions.
    """

    def __init__(
        self,
        inner: AdapterLlmProviderOpenai,
        allowed_actions: frozenset[str],
    ) -> None:
        self._inner = inner
        self._allowed_actions = allowed_actions

    def _check(self, action_name: str) -> None:
        if action_name not in self._allowed_actions:
            raise InvariantViolation(
                action_name=action_name,
                protocol_domain=_PROTOCOL_DOMAIN,
                allowed_actions=tuple(sorted(self._allowed_actions)),
            )

    # ── ProtocolLLMProvider properties ────────────────────────────────

    @property
    def provider_name(self) -> str:
        return self._inner.provider_name

    @property
    def provider_type(self) -> str:
        return self._inner.provider_type

    @property
    def is_available(self) -> bool:
        return self._inner.is_available

    # ── Actions ──────────────────────────────────────────────────────

    def configure(self, config: ModelLlmProviderConfig) -> None:
        self._check(_ACTION_CONFIGURE)
        self._inner.configure(config)

    async def get_available_models(self) -> list[str]:
        self._check(_ACTION_GET_AVAILABLE_MODELS)
        return await self._inner.get_available_models()

    async def get_model_capabilities(
        self, model_name: str
    ) -> ModelLlmModelCapabilities:
        self._check(_ACTION_GET_MODEL_CAPABILITIES)
        return await self._inner.get_model_capabilities(model_name)

    def validate_request(self, request: ModelLlmAdapterRequest) -> bool:
        self._check(_ACTION_VALIDATE_REQUEST)
        return self._inner.validate_request(request)

    async def generate(
        self, request: ModelLlmAdapterRequest
    ) -> ModelLlmAdapterResponse:
        self._check(_ACTION_GENERATE)
        return await self._inner.generate(request)

    async def generate_async(
        self, request: ModelLlmAdapterRequest
    ) -> ModelLlmAdapterResponse:
        self._check(_ACTION_GENERATE_ASYNC)
        return await self._inner.generate_async(request)

    def generate_stream(self, request: ModelLlmAdapterRequest) -> Iterator[str]:
        self._check(_ACTION_GENERATE_STREAM)
        return self._inner.generate_stream(request)

    async def generate_stream_async(
        self,
        request: ModelLlmAdapterRequest,
    ) -> AsyncGenerator[str, None]:
        self._check(_ACTION_GENERATE_STREAM_ASYNC)
        async for chunk in self._inner.generate_stream_async(request):
            yield chunk

    def estimate_cost(self, request: ModelLlmAdapterRequest) -> float:
        self._check(_ACTION_ESTIMATE_COST)
        return self._inner.estimate_cost(request)

    async def health_check(self) -> ModelLlmHealthResponse:
        self._check(_ACTION_HEALTH_CHECK)
        return await self._inner.health_check()

    async def get_provider_info(self) -> JsonType:
        self._check(_ACTION_GET_PROVIDER_INFO)
        return await self._inner.get_provider_info()

    def supports_streaming(self) -> bool:
        self._check(_ACTION_GENERATE_STREAM)
        return self._inner.supports_streaming()

    async def close(self) -> None:
        self._check(_ACTION_CLOSE)
        await self._inner.close()


__all__: list[str] = ["AdapterLlmProviderScoped"]
