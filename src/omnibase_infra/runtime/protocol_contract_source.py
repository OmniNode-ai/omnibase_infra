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

        Returns:
            str: Source type identifier (e.g., "CONTRACT", "DATABASE").
        """
        ...

    async def discover_handlers(self) -> ModelContractDiscoveryResult:
        """Discover and return all handlers from this source.

        Scans configured sources for handler contracts and returns
        discovered handlers along with any validation errors encountered.

        Args:
            None - This method takes no arguments.

        Returns:
            ModelContractDiscoveryResult: Container with:
                - descriptors: List of successfully discovered handlers
                - validation_errors: List of errors for failed discoveries

        Raises:
            ModelOnexError: In strict mode, if discovery encounters
                validation or parsing errors.
        """
        ...


__all__ = ["ProtocolContractSource"]
