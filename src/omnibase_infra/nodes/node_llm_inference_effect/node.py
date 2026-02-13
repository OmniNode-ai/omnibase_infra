# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Declarative LLM inference effect node.

All behavior is defined in contract.yaml. This node coordinates LLM
inference operations by delegating to provider-specific handlers.

Related:
    - contract.yaml: Node contract definition
    - handlers/: Provider-specific handler implementations
    - OMN-2108: Phase 8 Ollama inference handler
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeLlmInferenceEffect(NodeEffect):
    """Declarative LLM inference effect node.

    All behavior is defined in contract.yaml and delegated to
    provider-specific handlers. This node contains NO custom logic.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__: list[str] = ["NodeLlmInferenceEffect"]
