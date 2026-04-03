# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Registry for LLM Completion Effect Node Dependencies [OMN-7410].

Registers HandlerLLMCompletion with the ONEX container for DI resolution.
Pattern adapted from the embedding node's registry.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer

logger = logging.getLogger(__name__)


class RegistryInfraLlmCompletionEffect:
    """Registry for LLM completion effect node dependencies."""

    @staticmethod
    async def register(container: ModelONEXContainer) -> None:
        """Register completion handler with the ONEX container.

        Args:
            container: ONEX dependency injection container.
        """
        from omnibase_core.enums import EnumInjectionScope
        from omnibase_infra.nodes.node_llm_completion_effect.handlers.handler_llm_completion import (
            HandlerLLMCompletion,
        )

        handler = HandlerLLMCompletion()

        if container.service_registry is None:
            return

        await container.service_registry.register_instance(
            interface=HandlerLLMCompletion,
            instance=handler,
            scope=EnumInjectionScope.GLOBAL,
        )
        logger.info("Registered LLM completion handler")


__all__ = ["RegistryInfraLlmCompletionEffect"]
