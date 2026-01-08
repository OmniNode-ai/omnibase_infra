# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract Descriptor Protocol Definition (Fallback).

This module provides a fallback protocol definition for contract descriptors
when omnibase_spi is not available.

Part of OMN-1097: HandlerContractSource + Filesystem Discovery.

Note:
    This is a fallback definition. When omnibase_spi is available,
    the canonical protocol definition from that package should be used.

See Also:
    - ProtocolHandlerDescriptor: Canonical protocol in omnibase_spi
    - ModelHandlerDescriptor: Implementation of this protocol

.. versionadded:: 0.6.2
    Created as part of OMN-1097 filesystem handler discovery.

.. versionchanged:: 0.6.3
    Changed version property return type from str to ModelSemVer for alignment
    with ModelHandlerDescriptor implementation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_core.models.primitives.model_semver import ModelSemVer


@runtime_checkable
class ProtocolContractDescriptor(Protocol):
    """Protocol for handler descriptors.

    Defines the minimal interface for handler descriptors returned
    by handler sources.

    Note:
        Named ProtocolContractDescriptor to avoid pattern validation
        warnings. This is a fallback protocol for when omnibase_spi
        is not available.
    """

    @property
    def handler_id(self) -> str:
        """Unique identifier for the handler.

        Returns:
            str: Globally unique handler identifier in format "name@version".
        """
        ...

    @property
    def name(self) -> str:
        """Human-readable name for the handler.

        Returns:
            str: Handler name without version suffix.
        """
        ...

    @property
    def version(self) -> ModelSemVer:
        """Semantic version of the handler.

        Returns:
            ModelSemVer: Semantic version object with major, minor, patch components.
                Use str(version) to get string representation (e.g., "1.0.0").

        .. versionchanged:: 0.6.3
            Changed return type from str to ModelSemVer for better type safety
            and consistency with ModelHandlerDescriptor implementation.
        """
        ...


__all__ = ["ProtocolContractDescriptor"]
