# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Ledger Projection Compute - Platform event ledger projection node.

This package provides the NodeLedgerProjectionCompute, a COMPUTE node that
subscribes to 7 platform event topics for event ledger persistence.

Core Purpose:
    Projects events from the platform event bus into the audit ledger,
    enabling complete traceability and debugging support across all
    node lifecycle events, FSM transitions, and runtime coordination.

Subscribed Topics:
    - onex.evt.platform.node-registration.v1
    - onex.evt.platform.node-introspection.v1
    - onex.evt.platform.node-heartbeat.v1
    - onex.cmd.platform.request-introspection.v1
    - onex.evt.platform.fsm-state-transitions.v1
    - onex.intent.platform.runtime-tick.v1
    - onex.snapshot.platform.registration-snapshots.v1

Consumer Configuration:
    - consumer_purpose: "audit" (non-processing, read-only)
    - auto_offset_reset: "earliest" (capture all historical events)

Related Tickets:
    - OMN-1648: Ledger Projection Compute Node

Example:
    >>> from omnibase_core.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.node_ledger_projection_compute import (
    ...     NodeLedgerProjectionCompute,
    ...     RegistryInfraLedgerProjection,
    ... )
    >>>
    >>> container = ModelONEXContainer()
    >>> node = RegistryInfraLedgerProjection.create_node(container)
"""

from omnibase_infra.nodes.node_ledger_projection_compute.node import (
    NodeLedgerProjectionCompute,
)
from omnibase_infra.nodes.node_ledger_projection_compute.registry import (
    RegistryInfraLedgerProjection,
)

__all__ = [
    "NodeLedgerProjectionCompute",
    "RegistryInfraLedgerProjection",
]
