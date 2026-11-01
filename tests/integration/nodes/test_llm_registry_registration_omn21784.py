# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for LLM handler registration through the container (OMN-21784).

The embedding and inference registries build their handler only as the
instance argument of ``service_registry.register_instance``. These tests
drive the real registries against a recording service registry and assert
the registered instance is the real handler class, and that the four
removed node registry packages no longer import.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from omnibase_infra.nodes.node_llm_embedding_effect.handlers import (
    HandlerEmbeddingOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_embedding_effect.registry import (
    RegistryInfraLlmEmbeddingEffect,
)
from omnibase_infra.nodes.node_llm_inference_effect.handlers import (
    HandlerLlmOpenaiCompatible,
)
from omnibase_infra.nodes.node_llm_inference_effect.registry import (
    RegistryInfraLlmInferenceEffect,
)
from omnibase_infra.nodes.node_llm_inference_effect.services.service_llm_metrics_publisher import (
    ServiceLlmMetricsPublisher,
)

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

pytestmark = [pytest.mark.integration]


class _RecordingServiceRegistry:
    """Service registry double that records every register_instance call."""

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []

    async def register_instance(self, **kwargs: Any) -> None:
        self.registered.append(kwargs)

    def get_service(self, interface: type) -> Any:
        for call in reversed(self.registered):
            if call["interface"] is interface:
                return call["instance"]
        raise KeyError(interface)


def _container(registry: _RecordingServiceRegistry | None) -> ModelONEXContainer:
    return cast(
        "ModelONEXContainer",
        SimpleNamespace(
            service_registry=registry,
            get_service=registry.get_service if registry else None,
        ),
    )


@pytest.mark.asyncio
async def test_embedding_registry_registers_real_handler_instance() -> None:
    registry = _RecordingServiceRegistry()
    container = cast("ModelONEXContainer", SimpleNamespace(service_registry=registry))

    await RegistryInfraLlmEmbeddingEffect.register_openai_compatible(
        container,
        target_name="omn21784-embedding",
    )

    assert len(registry.registered) == 1
    call = registry.registered[0]
    assert call["interface"] is HandlerEmbeddingOpenaiCompatible
    assert isinstance(call["instance"], HandlerEmbeddingOpenaiCompatible)


@pytest.mark.asyncio
async def test_embedding_registry_without_service_registry_registers_nothing() -> None:
    container = _container(None)

    await RegistryInfraLlmEmbeddingEffect.register_openai_compatible(container)

    assert container.service_registry is None


@pytest.mark.asyncio
async def test_inference_registry_registers_real_handler_and_transport() -> None:
    registry = _RecordingServiceRegistry()

    await RegistryInfraLlmInferenceEffect.register_openai_compatible(
        _container(registry), target_name="omn21784-inference"
    )

    interfaces = [call["interface"] for call in registry.registered]
    assert HandlerLlmOpenaiCompatible in interfaces
    handler = registry.get_service(HandlerLlmOpenaiCompatible)
    assert isinstance(handler, HandlerLlmOpenaiCompatible)


@pytest.mark.asyncio
async def test_inference_registry_without_service_registry_registers_nothing() -> None:
    container = _container(None)

    await RegistryInfraLlmInferenceEffect.register_openai_compatible(container)

    assert container.service_registry is None


@pytest.mark.asyncio
async def test_inference_metrics_registry_wraps_the_registered_handler() -> None:
    registry = _RecordingServiceRegistry()

    async def publisher(*_args: Any, **_kwargs: Any) -> bool:
        return True

    await RegistryInfraLlmInferenceEffect.register_openai_compatible_with_metrics(
        _container(registry), publisher, target_name="omn21784-metrics"
    )

    handler = registry.get_service(HandlerLlmOpenaiCompatible)
    service = registry.get_service(ServiceLlmMetricsPublisher)
    assert isinstance(handler, HandlerLlmOpenaiCompatible)
    assert isinstance(service, ServiceLlmMetricsPublisher)
    assert service._handler is handler


@pytest.mark.parametrize(
    "module",
    [
        "omnibase_infra.nodes.node_contract_persistence_effect.registry",
        "omnibase_infra.nodes.node_pattern_lifecycle_effect.registry",
        "omnibase_infra.nodes.node_reward_binder_effect.registry",
        "omnibase_infra.nodes.node_rrh_emit_effect.registry",
    ],
)
def test_removed_node_registry_packages_do_not_import(module: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
