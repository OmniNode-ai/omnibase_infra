# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for LLM completion node DI registry [OMN-7410]."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_completion_registry_binds_handler() -> None:
    """Registry must bind HandlerLLMCompletion into the DI container."""
    from omnibase_infra.nodes.node_llm_completion_effect.registry import (
        RegistryInfraLlmCompletionEffect,
    )

    # Create mock container with service_registry
    mock_registry = MagicMock()
    mock_registry.register_instance = AsyncMock()

    mock_container = MagicMock()
    mock_container.service_registry = mock_registry

    await RegistryInfraLlmCompletionEffect.register(mock_container)

    # Verify register_instance was called with the correct handler type
    mock_registry.register_instance.assert_called_once()
    call_kwargs = mock_registry.register_instance.call_args

    from omnibase_infra.nodes.node_llm_completion_effect.handlers.handler_llm_completion import (
        HandlerLLMCompletion,
    )

    assert call_kwargs.kwargs["interface"] is HandlerLLMCompletion
    assert isinstance(call_kwargs.kwargs["instance"], HandlerLLMCompletion)


@pytest.mark.asyncio
async def test_completion_registry_noop_without_service_registry() -> None:
    """Registry should not fail when service_registry is None."""
    from omnibase_infra.nodes.node_llm_completion_effect.registry import (
        RegistryInfraLlmCompletionEffect,
    )

    mock_container = MagicMock()
    mock_container.service_registry = None

    # Should not raise
    await RegistryInfraLlmCompletionEffect.register(mock_container)
