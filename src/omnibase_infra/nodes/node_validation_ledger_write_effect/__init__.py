# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Node Validation Ledger Write Effect - validation ledger write node (OMN-14524).

This package provides the NodeValidationLedgerWriteEffect, an effect node
for validation event ledger write operations using PostgreSQL. Mirrors
node_ledger_write_effect's pattern for the generic event_ledger.

Capabilities:
    - validation_ledger.write: Append validation events to the validation
      ledger

Available Exports:
    - NodeValidationLedgerWriteEffect: The declarative effect node
    - HandlerValidationLedgerAppend: Handler for idempotent append operations
    - RegistryInfraValidationLedgerWrite: Dependency injection registry

Related Modules:
    - handlers: Handler implementation
    - registry: Dependency injection registration
"""

from omnibase_infra.nodes.node_validation_ledger_write_effect.handlers import (
    HandlerValidationLedgerAppend,
)
from omnibase_infra.nodes.node_validation_ledger_write_effect.node import (
    NodeValidationLedgerWriteEffect,
)
from omnibase_infra.nodes.node_validation_ledger_write_effect.registry import (
    RegistryInfraValidationLedgerWrite,
)

__all__ = [
    "HandlerValidationLedgerAppend",
    "NodeValidationLedgerWriteEffect",
    "RegistryInfraValidationLedgerWrite",
]
