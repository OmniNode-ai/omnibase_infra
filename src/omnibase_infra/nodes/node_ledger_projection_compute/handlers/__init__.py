# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handlers for ledger projection compute operations.

This package provides handlers for the ledger projection compute node:
    - HandlerLedgerProjection: Transforms ModelEventMessage to ModelIntent

The handler implements best-effort metadata extraction, ensuring events
are never dropped due to parsing failures.
"""

from omnibase_infra.nodes.node_ledger_projection_compute.handlers.handler_ledger_projection import (
    HandlerLedgerProjection,
)

__all__ = [
    "HandlerLedgerProjection",
]
