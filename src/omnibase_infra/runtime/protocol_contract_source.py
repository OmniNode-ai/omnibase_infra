# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Source Protocol Definition (Fallback).

This module provides a fallback protocol definition for contract sources
when omnibase_spi is not available.

Part of OMN-1097: HandlerContractSource + Filesystem Discovery.

Note:
    This is a fallback definition. When omnibase_spi is available,
    the canonical protocol definition from that package should be used.

See Also:
    - ProtocolHandlerSource: Canonical protocol in omnibase_spi
    - HandlerContractSource: Implementation of this protocol

.. versionadded:: 0.6.2
    Created as part of OMN-1097 filesystem handler discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_infra.models.handlers import ModelContractDiscoveryResult


@runtime_checkable
class ProtocolContractSource(Protocol):
    """Protocol for handler sources.

    Defines the interface that handler sources must implement.

    Note:
        Named ProtocolContractSource to avoid pattern validation
        warnings. This is a fallback protocol for when omnibase_spi
        is not available.
    """

    @property
    def source_type(self) -> str:
        """The type of handler source.

        This identifier is used for observability and logging. The runtime
        MUST NOT branch on this value.

        Returns:
            str: Source type identifier. Common values include:
                - "CONTRACT": Filesystem-based contract discovery
                - "DATABASE": Database-backed handler registry
        """
        ...

    async def discover_handlers(self) -> ModelContractDiscoveryResult:
        """Discover and return all handlers from this source.

        Scans configured sources for handler contracts and returns
        discovered handlers along with any validation errors encountered.

        This method may be called multiple times and should return
        consistent results (idempotent).

        Returns:
            ModelContractDiscoveryResult: Container with:
                - descriptors: List of successfully discovered handlers
                  as ProtocolContractDescriptor instances
                - validation_errors: List of ModelHandlerValidationError
                  for failed discoveries (empty in strict mode)

        Raises:
            ModelOnexError: In strict mode (default), if discovery encounters
                validation, parsing, or I/O errors. Error codes include:
                - HANDLER_SOURCE_001: Empty contract_paths
                - HANDLER_SOURCE_002: Path does not exist
                - HANDLER_SOURCE_003: YAML parse error
                - HANDLER_SOURCE_004: Contract validation error
                - HANDLER_SOURCE_005: File size limit exceeded
                - HANDLER_SOURCE_006: File I/O error
        """
        ...


__all__ = ["ProtocolContractSource"]
