# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handlers for validation ledger projection compute operations.

This package provides handlers for the validation ledger projection compute node:
    - HandlerValidationLedgerProjection: Transforms Kafka events to ledger entries

Ticket: OMN-1908
"""

from omnibase_infra.nodes.node_validation_ledger_projection_compute.handlers.handler_validation_ledger_projection import (
    HandlerValidationLedgerProjection,
)

__all__ = [
    "HandlerValidationLedgerProjection",
]
