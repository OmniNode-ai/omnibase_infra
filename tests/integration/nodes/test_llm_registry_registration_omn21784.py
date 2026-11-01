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

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

pytestmark = [pytest.mark.integration]


class _RecordingServiceRegistry:
    """Service registry double that records every register_instance call."""

    def __init__(self) -> None:
        self.registered: list[dict[str, Any]] = []

    async def register_instance(self, **kwargs: Any) -> None:
        self.registered.append(kwargs)


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
    container = cast("ModelONEXContainer", SimpleNamespace(service_registry=None))

    await RegistryInfraLlmEmbeddingEffect.register_openai_compatible(
        container,
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
