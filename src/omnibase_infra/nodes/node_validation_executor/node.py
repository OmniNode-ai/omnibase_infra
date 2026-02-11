# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeValidationExecutor - Declarative EFFECT node for running validation checks.

Runs actual checks: typecheck, lint, unit tests, integration tests,
risk assessment, cost measurement. Each check produces a ModelCheckResult.

This node is declarative -- all behavior is defined in contract.yaml
and implemented through handlers.

Ticket: OMN-2147
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeValidationExecutor(NodeEffect):
    """Effect node for running validation checks.

    All behavior is defined in contract.yaml and delegated to handlers.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__: list[str] = ["NodeValidationExecutor"]
