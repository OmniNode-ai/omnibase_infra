# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration tests for LLM handler registration through the container (OMN-21784).

The embedding and inference registries build their handler only as the
instance argument of ``service_registry.register_instance``. These tests
drive the real registries against a real ``ModelONEXContainer`` and resolve
the handlers back through ``get_service_async``, assert the unset-registry
paths build nothing, and assert the four removed node registry packages no
longer import.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, cast

import pytest

from omnibase_core.models.container.model_onex_container import ModelONEXContainer
from omnibase_infra.mixins import MixinLlmHttpTransport
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
from omnibase_infra.nodes.node_llm_inference_effect.registry import (
    registry_infra_llm_inference_effect as registry_module,
)
from omnibase_infra.nodes.node_llm_inference_effect.services.service_llm_metrics_publisher import (
    ServiceLlmMetricsPublisher,
)

pytestmark = [pytest.mark.integration]


def _unset_registry_container() -> ModelONEXContainer:
    return cast("ModelONEXContainer", SimpleNamespace(service_registry=None))


@pytest.mark.asyncio
async def test_embedding_registry_registers_real_handler_instance() -> None:
    container = ModelONEXContainer()

    await RegistryInfraLlmEmbeddingEffect.register_openai_compatible(
        container,
        target_name="omn21784-embedding",
    )

    handler = await container.get_service_async(HandlerEmbeddingOpenaiCompatible)
    assert isinstance(handler, HandlerEmbeddingOpenaiCompatible)


@pytest.mark.asyncio
async def test_embedding_registry_without_service_registry_registers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("handler constructed with no service registry")

    monkeypatch.setattr(HandlerEmbeddingOpenaiCompatible, "__init__", _forbidden)

    await RegistryInfraLlmEmbeddingEffect.register_openai_compatible(
        _unset_registry_container()
    )


@pytest.mark.asyncio
async def test_inference_registry_registers_real_handler_and_transport() -> None:
    container = ModelONEXContainer()

    await RegistryInfraLlmInferenceEffect.register_openai_compatible(
        container, target_name="omn21784-inference"
    )

    handler = await container.get_service_async(HandlerLlmOpenaiCompatible)
    transport = await container.get_service_async(MixinLlmHttpTransport)
    assert isinstance(handler, HandlerLlmOpenaiCompatible)
    assert isinstance(transport, MixinLlmHttpTransport)


@pytest.mark.asyncio
async def test_inference_registry_without_service_registry_registers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("transport built with no service registry")

    monkeypatch.setattr(registry_module, "_create_transport_adapter", _forbidden)

    await RegistryInfraLlmInferenceEffect.register_openai_compatible(
        _unset_registry_container()
    )


@pytest.mark.asyncio
async def test_inference_metrics_registry_wraps_the_registered_handler() -> None:
    container = ModelONEXContainer()

    async def publisher(*_args: Any, **_kwargs: Any) -> bool:
        return True

    await RegistryInfraLlmInferenceEffect.register_openai_compatible_with_metrics(
        container, publisher, target_name="omn21784-metrics"
    )

    handler = await container.get_service_async(HandlerLlmOpenaiCompatible)
    service = await container.get_service_async(ServiceLlmMetricsPublisher)
    assert isinstance(handler, HandlerLlmOpenaiCompatible)
    assert isinstance(service, ServiceLlmMetricsPublisher)
    assert service._handler is handler


@pytest.mark.asyncio
async def test_inference_metrics_registry_without_service_registry_registers_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("transport built with no service registry")

    monkeypatch.setattr(registry_module, "_create_transport_adapter", _forbidden)

    async def publisher(*_args: Any, **_kwargs: Any) -> bool:
        return True

    await RegistryInfraLlmInferenceEffect.register_openai_compatible_with_metrics(
        _unset_registry_container(), publisher
    )


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
