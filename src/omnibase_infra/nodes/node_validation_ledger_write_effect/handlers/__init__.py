# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Handlers for validation event ledger persistence operations.

This package provides the handler for the validation ledger write effect
node:
    - HandlerValidationLedgerAppend: Idempotent INSERT with duplicate
      detection

Composes with HandlerDb for PostgreSQL operations.
"""

from omnibase_infra.nodes.node_validation_ledger_write_effect.handlers.handler_validation_ledger_append import (
    HandlerValidationLedgerAppend,
)

__all__ = [
    "HandlerValidationLedgerAppend",
]
