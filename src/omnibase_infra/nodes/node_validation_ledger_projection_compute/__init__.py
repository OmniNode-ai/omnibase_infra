# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Node Validation Ledger Projection Compute - Cross-repo validation ledger projection node.

This package provides the NodeValidationLedgerProjectionCompute, a declarative COMPUTE
node that subscribes to 3 cross-repo validation event topics for validation ledger
persistence.

Architecture:
    This node follows the ONEX declarative pattern where:
    - NodeValidationLedgerProjectionCompute is a declarative shell (no custom logic)
    - HandlerValidationLedgerProjection contains all compute logic
    - contract.yaml defines behavior via handler_routing

Core Purpose:
    Projects validation events from the event bus into the validation_event_ledger,
    enabling deterministic replay and complete traceability of cross-repo validation
    runs, violations, and completions.

Subscribed Topics:
    - onex.validation.cross_repo.run.started.v1
    - onex.validation.cross_repo.violations.batch.v1
    - onex.validation.cross_repo.run.completed.v1

Consumer Configuration:
    - consumer_purpose: "projection" (non-processing, read-only)
    - auto_offset_reset: "earliest" (capture all historical events)

Related Tickets:
    - OMN-1908: Cross-Repo Validation Ledger Persistence
"""

from omnibase_infra.nodes.node_validation_ledger_projection_compute.handlers import (
    HandlerValidationLedgerProjection,
)
from omnibase_infra.nodes.node_validation_ledger_projection_compute.node import (
    NodeValidationLedgerProjectionCompute,
)

__all__ = [
    "HandlerValidationLedgerProjection",
    "NodeValidationLedgerProjectionCompute",
]
