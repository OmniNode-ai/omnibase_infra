# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Registry for NodeBaselineCapture dependencies.

Ticket: OMN-7484
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer
    from omnibase_infra.nodes.node_baseline_capture.node import NodeBaselineCapture


class RegistryInfraBaselineCapture:
    """Registry for NodeBaselineCapture dependency injection."""

    def __init__(self, container: ModelONEXContainer) -> None:
        self._container = container

    def create_effect(self) -> NodeBaselineCapture:
        from omnibase_infra.nodes.node_baseline_capture.node import NodeBaselineCapture

        return NodeBaselineCapture(self._container)


__all__: list[str] = ["RegistryInfraBaselineCapture"]
