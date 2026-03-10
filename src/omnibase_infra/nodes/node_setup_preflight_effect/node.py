# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Declarative EFFECT node for platform setup preflight validation.

Validates all prerequisites before platform provisioning: Docker version,
Compose version, Python version, env vars, Docker daemon, omnibase dir,
and port availability.

Ticket: OMN-3492
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodePreflightEffect(NodeEffect):
    """Validates prerequisites before platform provisioning.

    Handlers:
        - ``HandlerPreflightCheck``: Runs all 7 preflight checks and returns
          aggregated pass/fail results.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the preflight effect node."""
        super().__init__(container)


__all__: list[str] = ["NodePreflightEffect"]
