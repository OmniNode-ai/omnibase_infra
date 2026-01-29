# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Protocols for event ledger persistence operations.

This package defines the protocol interfaces for ledger persistence,
enabling duck typing and testability for handlers.
"""

from omnibase_infra.nodes.node_ledger_write_effect.protocols.protocol_ledger_persistence import (
    ProtocolLedgerPersistence,
)

__all__ = ["ProtocolLedgerPersistence"]
