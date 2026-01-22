# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Source Resolver for Multi-Source Handler Discovery.

This module provides the HandlerSourceResolver class, which resolves handlers
from multiple sources (bootstrap, contract) based on the configured mode.

Part of OMN-1095: Handler Source Mode Hybrid Resolution.

Resolution Modes:
    - BOOTSTRAP: Only use hardcoded bootstrap handlers.
    - CONTRACT: Only use YAML contract-discovered handlers.
    - HYBRID: Per-handler resolution with contract precedence.

In HYBRID mode, the resolver performs per-handler identity resolution:
    1. Discovers handlers from both bootstrap and contract sources
    2. Builds a handler map keyed by handler_id
    3. Contract handlers override bootstrap handlers with same handler_id
    4. Bootstrap handlers serve as fallback for handlers not in contracts

See Also:
    - EnumHandlerSourceMode: Defines the resolution modes
    - HandlerBootstrapSource: Provides bootstrap handlers
    - HandlerContractSource: Provides contract-discovered handlers
    - ProtocolContractSource: Protocol for handler sources

.. versionadded:: 0.7.0
    Created as part of OMN-1095 handler source mode hybrid resolution.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from omnibase_infra.enums.enum_handler_source_mode import EnumHandlerSourceMode

if TYPE_CHECKING:
    from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource

# Import models after TYPE_CHECKING to avoid circular imports
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import (
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
)

# Rebuild to resolve forward reference to ModelHandlerValidationError
# This is required because ModelContractDiscoveryResult uses TYPE_CHECKING
# to defer the import of ModelHandlerValidationError for circular import avoidance.
ModelContractDiscoveryResult.model_rebuild()

logger = logging.getLogger(__name__)


class HandlerSourceResolver:
    """Resolver for multi-source handler discovery with configurable modes.

    This class resolves handlers from bootstrap and contract sources based on
    the configured mode. It supports three resolution strategies:

    - BOOTSTRAP: Use only bootstrap handlers, ignore contracts.
    - CONTRACT: Use only contract handlers, ignore bootstrap.
    - HYBRID: Per-handler resolution where contract handlers take precedence
      over bootstrap handlers with the same handler_id, and bootstrap handlers
      serve as fallback for handlers not defined in contracts.

    Attributes:
        mode: The configured resolution mode.

    Example:
        >>> resolver = HandlerSourceResolver(
        ...     bootstrap_source=bootstrap_source,
        ...     contract_source=contract_source,
        ...     mode=EnumHandlerSourceMode.HYBRID,
        ... )
        >>> result = await resolver.resolve_handlers()
        >>> print(f"Discovered {len(result.descriptors)} handlers")

    .. versionadded:: 0.7.0
        Created as part of OMN-1095 handler source mode hybrid resolution.
    """

    def __init__(
        self,
        bootstrap_source: ProtocolContractSource,
        contract_source: ProtocolContractSource,
        mode: EnumHandlerSourceMode,
    ) -> None:
        """Initialize the handler source resolver.

        Args:
            bootstrap_source: Source for bootstrap handlers. Must implement
                ProtocolContractSource with discover_handlers() method.
            contract_source: Source for contract-discovered handlers. Must
                implement ProtocolContractSource with discover_handlers() method.
            mode: Resolution mode determining which sources are used and how
                handlers are merged.
        """
        self._bootstrap_source = bootstrap_source
        self._contract_source = contract_source
        self._mode = mode

    @property
    def mode(self) -> EnumHandlerSourceMode:
        """Get the configured resolution mode.

        Returns:
            EnumHandlerSourceMode: The mode used for handler resolution.
        """
        return self._mode

    async def resolve_handlers(self) -> ModelContractDiscoveryResult:
        """Resolve handlers based on the configured mode.

        Discovers handlers from the appropriate source(s) based on mode:
        - BOOTSTRAP: Only queries bootstrap source
        - CONTRACT: Only queries contract source
        - HYBRID: Queries both sources and merges with contract precedence

        Returns:
            ModelContractDiscoveryResult: Container with discovered descriptors
            and any validation errors from the queried source(s).
        """
        if self._mode == EnumHandlerSourceMode.BOOTSTRAP:
            return await self._resolve_bootstrap()
        elif self._mode == EnumHandlerSourceMode.CONTRACT:
            return await self._resolve_contract()
        else:
            # HYBRID mode
            return await self._resolve_hybrid()

    async def _resolve_bootstrap(self) -> ModelContractDiscoveryResult:
        """Resolve handlers using only the bootstrap source.

        Returns:
            ModelContractDiscoveryResult: Handlers from bootstrap source only.
        """
        result = await self._bootstrap_source.discover_handlers()

        logger.info(
            "Handler resolution completed (BOOTSTRAP mode)",
            extra={
                "mode": self._mode.value,
                "bootstrap_handler_count": len(result.descriptors),
                "resolved_handler_count": len(result.descriptors),
            },
        )

        return result

    async def _resolve_contract(self) -> ModelContractDiscoveryResult:
        """Resolve handlers using only the contract source.

        Returns:
            ModelContractDiscoveryResult: Handlers from contract source only.
        """
        result = await self._contract_source.discover_handlers()

        logger.info(
            "Handler resolution completed (CONTRACT mode)",
            extra={
                "mode": self._mode.value,
                "contract_handler_count": len(result.descriptors),
                "resolved_handler_count": len(result.descriptors),
            },
        )

        return result

    async def _resolve_hybrid(self) -> ModelContractDiscoveryResult:
        """Resolve handlers using both sources with contract precedence.

        In HYBRID mode:
        1. Discover handlers from both bootstrap and contract sources
        2. Build a handler map keyed by handler_id
        3. Contract handlers override bootstrap handlers with same handler_id
        4. Bootstrap handlers serve as fallback when no matching contract handler

        Returns:
            ModelContractDiscoveryResult: Merged handlers with contract precedence
            and combined validation errors from both sources.
        """
        # Get handlers from both sources
        bootstrap_result = await self._bootstrap_source.discover_handlers()
        contract_result = await self._contract_source.discover_handlers()

        # Build handler map - contract handlers first (they take precedence)
        handlers_by_id: dict[str, ModelHandlerDescriptor] = {}

        # Add contract handlers (they win conflicts)
        for descriptor in contract_result.descriptors:
            handlers_by_id[descriptor.handler_id] = descriptor

        # Add bootstrap handlers only if not already present (fallback)
        fallback_count = 0
        override_count = 0
        for descriptor in bootstrap_result.descriptors:
            if descriptor.handler_id in handlers_by_id:
                # Contract handler wins - this is an override
                override_count += 1
            else:
                # No contract handler with this ID - use bootstrap as fallback
                handlers_by_id[descriptor.handler_id] = descriptor
                fallback_count += 1

        # Merge validation errors from both sources
        all_errors: list[ModelHandlerValidationError] = list(
            bootstrap_result.validation_errors
        ) + list(contract_result.validation_errors)

        # Log structured counts for observability
        logger.info(
            "Handler resolution completed (HYBRID mode)",
            extra={
                "mode": self._mode.value,
                "contract_handler_count": len(contract_result.descriptors),
                "bootstrap_handler_count": len(bootstrap_result.descriptors),
                "fallback_handler_count": fallback_count,
                "override_count": override_count,
                "resolved_handler_count": len(handlers_by_id),
            },
        )

        return ModelContractDiscoveryResult(
            descriptors=list(handlers_by_id.values()),
            validation_errors=all_errors,
        )


__all__ = ["HandlerSourceResolver"]
