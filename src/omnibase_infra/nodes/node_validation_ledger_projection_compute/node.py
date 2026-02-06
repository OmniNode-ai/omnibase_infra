# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""NodeValidationLedgerProjectionCompute - Declarative COMPUTE node for validation ledger projection.

Subscribes to 3 cross-repo validation event topics and projects events
into the validation_event_ledger for deterministic replay.

All business logic is delegated to HandlerValidationLedgerProjection.

Subscribed Topics (via contract.yaml):
    - onex.validation.cross_repo.run.started.v1
    - onex.validation.cross_repo.violations.batch.v1
    - onex.validation.cross_repo.run.completed.v1

Ticket: OMN-1908
"""

from __future__ import annotations

from omnibase_core.nodes.node_compute import NodeCompute


class NodeValidationLedgerProjectionCompute(NodeCompute):
    """Declarative COMPUTE node for validation ledger projection.

    All behavior is defined in contract.yaml and delegated to
    HandlerValidationLedgerProjection. This node contains no custom logic.

    See Also:
        - handlers/handler_validation_ledger_projection.py: Contains all compute logic
        - contract.yaml: Node subscription and I/O configuration
    """

    # Declarative node - all behavior defined in contract.yaml


__all__ = ["NodeValidationLedgerProjectionCompute"]
