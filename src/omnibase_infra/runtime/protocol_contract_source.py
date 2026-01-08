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
        """The type of handler source."""
        ...

    async def discover_handlers(self) -> ModelContractDiscoveryResult:
        """Discover and return all handlers from this source.

        Returns:
            ModelContractDiscoveryResult containing discovered handler
            descriptors and any validation errors encountered during discovery.
        """
        ...


__all__ = ["ProtocolContractSource"]
