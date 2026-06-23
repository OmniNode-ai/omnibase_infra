# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeBrokerDiskWatermarkCompute — declarative compute node.

Probes docker data-root + named broker volumes against disk watermarks
and classifies each as CLEAN / WARN / P0.  Downstream orchestrators
act on the result (Slack alert for WARN, Urgent Linear ticket for P0).

Eliminates the class of demo-day outages described in OMN-13009
(error system:28 No space left on device).

Handlers:
    - ``HandlerBrokerDiskWatermark``: Probe and classify disk usage.

Ticket: OMN-13009
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container.model_onex_container import ModelONEXContainer


class NodeBrokerDiskWatermarkCompute(NodeCompute):
    """Compute node for broker disk watermark probing.

    Capability: broker.disk.probe

    All behavior defined in contract.yaml and implemented through handlers.
    No custom logic in this class.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        """Initialize the broker disk watermark compute node."""
        super().__init__(container)


__all__: list[str] = ["NodeBrokerDiskWatermarkCompute"]
