# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Validation Ledger Projection Compute - Cross-repo validation event projection.

This package provides the NodeValidationLedgerProjectionCompute, a declarative
COMPUTE node that subscribes to 3 cross-repo validation event topics for
validation event ledger persistence.

Architecture:
    This node follows the ONEX declarative pattern where:
    - NodeValidationLedgerProjectionCompute is a declarative shell (no custom logic)
    - HandlerValidationLedgerProjection contains all compute logic
    - contract.yaml defines behavior via handler_routing

Core Purpose:
    Projects cross-repo validation events into the validation_event_ledger
    table, enabling deterministic replay and dashboard queries.

Subscribed Topics:
    - onex.validation.cross_repo.run.started.v1
    - onex.validation.cross_repo.violations.batch.v1
    - onex.validation.cross_repo.run.completed.v1

Ticket: OMN-1908
"""

from omnibase_infra.nodes.node_validation_ledger_projection_compute.handlers import (
    HandlerValidationLedgerProjection,
)
from omnibase_infra.nodes.node_validation_ledger_projection_compute.node import (
    NodeValidationLedgerProjectionCompute,
)
from omnibase_infra.nodes.node_validation_ledger_projection_compute.registry import (
    RegistryInfraValidationLedgerProjection,
)

__all__ = [
    "HandlerValidationLedgerProjection",
    "NodeValidationLedgerProjectionCompute",
    "RegistryInfraValidationLedgerProjection",
]
