# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""NodeBaselineCapture — EFFECT node for raw baseline measurement capture.

SOW Phase 2 — Track B4a. Reads agent_actions within a lookback window and
emits onex.evt.omnibase-infra.baselines-computed.v1 with raw per-agent
measurements. No delta or ROI computation (deferred to B4b).

Ticket: OMN-7484
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_effect import NodeEffect

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeBaselineCapture(NodeEffect):
    """EFFECT node for raw baseline measurement capture.

    Reads agent_actions and agent_routing_decisions rows within a configurable
    lookback window and emits a baselines-computed snapshot event for omnidash.

    All behavior is defined in contract.yaml and implemented through
    HandlerBaselineCapture. No custom logic exists in this class.

    Attributes:
        container: ONEX dependency injection container.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)


__all__: list[str] = ["NodeBaselineCapture"]
