# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Registry for validation event ledger write effect node components.

This registry exports all public symbols for the
node_validation_ledger_write_effect node, providing a single import point
for consumers of this node. Mirrors RegistryInfraLedgerWrite.

Exported Components:
    - NodeValidationLedgerWriteEffect: The declarative effect node
    - HandlerValidationLedgerAppend: Handler for idempotent append operations
    - Models: Data models for validation ledger operations

Example:
    >>> from omnibase_infra.nodes.node_validation_ledger_write_effect.registry import (
    ...     RegistryInfraValidationLedgerWrite,
    ... )
    >>> # Access all components via registry
    >>> node_cls = RegistryInfraValidationLedgerWrite.node
    >>> append_handler = RegistryInfraValidationLedgerWrite.handler_append
"""

from __future__ import annotations

# Models
from omnibase_infra.models.validation_ledger import (
    ModelPayloadValidationLedgerAppend,
    ModelValidationLedgerAppendResult,
)

# Handler
from omnibase_infra.nodes.node_validation_ledger_write_effect.handlers import (
    HandlerValidationLedgerAppend,
)

# Node
from omnibase_infra.nodes.node_validation_ledger_write_effect.node import (
    NodeValidationLedgerWriteEffect,
)


class RegistryInfraValidationLedgerWrite:
    """Registry providing access to all validation ledger write effect node components.

    A centralized access point for all components of the
    node_validation_ledger_write_effect node. Use this registry for
    dependency injection and container registration.

    Class Attributes:
        node: The NodeValidationLedgerWriteEffect class
        handler_append: The HandlerValidationLedgerAppend class
        models: Tuple of all model classes

    Example:
        >>> from omnibase_infra.nodes.node_validation_ledger_write_effect.registry import (
        ...     RegistryInfraValidationLedgerWrite,
        ... )
        >>> node = RegistryInfraValidationLedgerWrite.node(container)
        >>> append_handler = RegistryInfraValidationLedgerWrite.handler_append(container)
    """

    # Node
    node = NodeValidationLedgerWriteEffect

    # Handler
    handler_append = HandlerValidationLedgerAppend

    # Models (as tuple for iteration)
    models = (
        ModelValidationLedgerAppendResult,
        ModelPayloadValidationLedgerAppend,
    )

    # Individual model references
    model_append_result = ModelValidationLedgerAppendResult
    model_payload_append = ModelPayloadValidationLedgerAppend


# Re-export all components at module level for convenience
__all__ = [
    "RegistryInfraValidationLedgerWrite",
    # Node
    "NodeValidationLedgerWriteEffect",
    # Handler
    "HandlerValidationLedgerAppend",
    # Models
    "ModelValidationLedgerAppendResult",
    "ModelPayloadValidationLedgerAppend",
]
