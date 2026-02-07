# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handlers for validation ledger projection compute operations.

This package provides handlers for the validation ledger projection compute node:
    - HandlerValidationLedgerProjection: Extracts metadata from Kafka messages
      and prepares validation ledger entries with base64 encoding and SHA-256 hashing.

The handler implements best-effort metadata extraction, ensuring validation events
are never dropped due to parsing failures.
"""

from omnibase_infra.nodes.node_validation_ledger_projection_compute.handlers.handler_validation_ledger_projection import (
    HandlerValidationLedgerProjection,
)

__all__ = [
    "HandlerValidationLedgerProjection",
]
