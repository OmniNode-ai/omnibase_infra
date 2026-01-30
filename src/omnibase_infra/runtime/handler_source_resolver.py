# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler Source Resolver for Multi-Source Handler Discovery.

This module provides the HandlerSourceResolver class, which resolves handlers
from multiple sources (bootstrap, contract, Kafka events) based on the configured mode.

Part of OMN-1095: Handler Source Mode Hybrid Resolution.

Resolution Modes:
    - BOOTSTRAP: Only use hardcoded bootstrap handlers.
    - CONTRACT: Only use YAML contract-discovered handlers.
    - HYBRID: Per-handler resolution with configurable precedence.
    - KAFKA_EVENTS: Use Kafka-based contract source for cache-based discovery.

In HYBRID mode, the resolver performs per-handler identity resolution:
    1. Discovers handlers from both bootstrap and contract sources
    2. Builds a handler map keyed by handler_id
    3. Resolves conflicts based on allow_bootstrap_override:
       - False (default): Contract handlers override bootstrap handlers
       - True: Bootstrap handlers override contract handlers
    4. Non-conflicting handlers are included from both sources

In KAFKA_EVENTS mode, the resolver delegates to a KafkaContractSource instance
that returns cached descriptors from contract registration events. This is a
beta cache-only implementation where discovered contracts take effect on the
next runtime restart.

See Also:
    - EnumHandlerSourceMode: Defines the resolution modes
    - HandlerBootstrapSource: Provides bootstrap handlers
    - HandlerContractSource: Provides contract-discovered handlers
    - KafkaContractSource: Provides Kafka cache-based handler discovery
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

# Forward Reference Resolution:
# ModelContractDiscoveryResult uses a forward reference to ModelHandlerValidationError.
# Since we import ModelHandlerValidationError above, we can call model_rebuild() here
# to resolve the forward reference. This call is idempotent - multiple calls are harmless.
ModelContractDiscoveryResult.model_rebuild()

logger = logging.getLogger(__name__)


class HandlerSourceResolver:
    """Resolver for multi-source handler discovery with configurable modes.

    This class resolves handlers from bootstrap and contract sources based on
    the configured mode. It supports four resolution strategies:

    - BOOTSTRAP: Use only bootstrap handlers, ignore contracts.
    - CONTRACT: Use only contract handlers, ignore bootstrap.
    - HYBRID: Per-handler resolution with configurable precedence:
        - allow_bootstrap_override=False (default): Contract handlers take
          precedence over bootstrap handlers with the same handler_id.
        - allow_bootstrap_override=True: Bootstrap handlers take precedence
          over contract handlers with the same handler_id.
      In both cases, handlers without conflicts are included from both sources.
    - KAFKA_EVENTS: Use Kafka-based contract source for cache-based discovery.
      Delegates to a KafkaContractSource that returns cached descriptors from
      contract registration events. This is a beta cache-only implementation.

    Attributes:
        mode: The configured resolution mode.
        allow_bootstrap_override: If True, bootstrap handlers take precedence
            in HYBRID mode. Default is False (contract precedence).

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
        *,
        allow_bootstrap_override: bool = False,
    ) -> None:
        """Initialize the handler source resolver.

        Args:
            bootstrap_source: Source for bootstrap handlers. Must implement
                ProtocolContractSource with discover_handlers() method.
            contract_source: Source for contract-discovered handlers. Must
                implement ProtocolContractSource with discover_handlers() method.
            mode: Resolution mode determining which sources are used and how
                handlers are merged.
            allow_bootstrap_override: If True, bootstrap handlers override
                contract handlers with the same handler_id in HYBRID mode.
                Default is False (contract handlers take precedence).
                Has no effect in BOOTSTRAP or CONTRACT modes.
        """
        self._bootstrap_source = bootstrap_source
        self._contract_source = contract_source
        self._mode = mode
        self._allow_bootstrap_override = allow_bootstrap_override

    @property
    def mode(self) -> EnumHandlerSourceMode:
        """Get the configured resolution mode.

        Returns:
            EnumHandlerSourceMode: The mode used for handler resolution.
        """
        return self._mode

    @property
    def allow_bootstrap_override(self) -> bool:
        """Get the bootstrap override configuration.

        Returns:
            bool: True if bootstrap handlers take precedence in HYBRID mode,
                False if contract handlers take precedence (default).
        """
        return self._allow_bootstrap_override

    async def resolve_handlers(self) -> ModelContractDiscoveryResult:
        """Resolve handlers based on the configured mode.

        Discovers handlers from the appropriate source(s) based on mode:
        - BOOTSTRAP: Only queries bootstrap source
        - CONTRACT: Only queries contract source
        - HYBRID: Queries both sources and merges with contract precedence
        - KAFKA_EVENTS: Queries Kafka-based contract cache

        Returns:
            ModelContractDiscoveryResult: Container with discovered descriptors
            and any validation errors from the queried source(s).
        """
        if self._mode == EnumHandlerSourceMode.BOOTSTRAP:
            return await self._resolve_bootstrap()
        elif self._mode == EnumHandlerSourceMode.CONTRACT:
            return await self._resolve_contract()
        elif self._mode == EnumHandlerSourceMode.KAFKA_EVENTS:
            return await self._resolve_kafka_events()
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

    async def _resolve_kafka_events(self) -> ModelContractDiscoveryResult:
        """Resolve handlers using Kafka-based contract source.

        For KAFKA_EVENTS mode, the contract_source is expected to be a
        KafkaContractSource instance that returns cached descriptors from
        contract registration events.

        Note:
            This is a beta cache-only implementation. Discovered contracts
            take effect on the next runtime restart.

        Returns:
            ModelContractDiscoveryResult: Discovery result from the Kafka
            contract cache.
        """
        result = await self._contract_source.discover_handlers()

        logger.info(
            "Handler resolution completed (KAFKA_EVENTS mode)",
            extra={
                "mode": self._mode.value,
                "kafka_handler_count": len(result.descriptors),
                "resolved_handler_count": len(result.descriptors),
            },
        )

        return result

    async def _resolve_hybrid(self) -> ModelContractDiscoveryResult:
        """Resolve handlers using both sources with configurable precedence.

        In HYBRID mode:
        1. Discover handlers from both bootstrap and contract sources
        2. Build a handler map keyed by handler_id
        3. Resolve conflicts based on allow_bootstrap_override:
           - False (default): Contract handlers override bootstrap handlers
           - True: Bootstrap handlers override contract handlers
        4. Non-conflicting handlers are included from both sources

        Returns:
            ModelContractDiscoveryResult: Merged handlers with configured
            precedence and combined validation errors from both sources.
        """
        # Get handlers from both sources
        bootstrap_result = await self._bootstrap_source.discover_handlers()
        contract_result = await self._contract_source.discover_handlers()

        # Build lookup maps for both sources
        bootstrap_by_id: dict[str, ModelHandlerDescriptor] = {
            d.handler_id: d for d in bootstrap_result.descriptors
        }
        contract_by_id: dict[str, ModelHandlerDescriptor] = {
            d.handler_id: d for d in contract_result.descriptors
        }

        # Determine which source takes precedence
        if self._allow_bootstrap_override:
            # Bootstrap wins conflicts: add bootstrap first, then contract fallbacks
            primary_source = bootstrap_result.descriptors
            primary_by_id = bootstrap_by_id
            secondary_source = contract_result.descriptors
            secondary_by_id = contract_by_id
            primary_name = "bootstrap"
            secondary_name = "contract"
        else:
            # Contract wins conflicts (default): add contract first, then bootstrap fallbacks
            primary_source = contract_result.descriptors
            primary_by_id = contract_by_id
            secondary_source = bootstrap_result.descriptors
            secondary_by_id = bootstrap_by_id
            primary_name = "contract"
            secondary_name = "bootstrap"

        # Build handler map - primary source handlers first (they take precedence)
        handlers_by_id: dict[str, ModelHandlerDescriptor] = {}

        # Add primary handlers (they win conflicts)
        for descriptor in primary_source:
            handlers_by_id[descriptor.handler_id] = descriptor

            # Log primary-only handlers (no secondary equivalent)
            if descriptor.handler_id not in secondary_by_id:
                logger.debug(
                    f"Adding {primary_name}-only handler (no {secondary_name} equivalent)",
                    extra={
                        "handler_id": descriptor.handler_id,
                        "handler_name": descriptor.name,
                        "source": primary_name,
                    },
                )

        # Add secondary handlers only if not already present (fallback)
        fallback_count = 0
        override_count = 0
        for descriptor in secondary_source:
            if descriptor.handler_id in handlers_by_id:
                # Primary handler wins - this is an override
                override_count += 1
                primary_handler = handlers_by_id[descriptor.handler_id]
                logger.debug(
                    f"{primary_name.capitalize()} handler overrides {secondary_name} handler",
                    extra={
                        "handler_id": descriptor.handler_id,
                        "primary_name": primary_handler.name,
                        "secondary_name": descriptor.name,
                        "primary_source": primary_name,
                        "secondary_source": secondary_name,
                        "contract_path": (
                            primary_handler.contract_path
                            if primary_name == "contract"
                            else descriptor.contract_path
                        ),
                    },
                )
            else:
                # No primary handler with this ID - use secondary as fallback
                handlers_by_id[descriptor.handler_id] = descriptor
                fallback_count += 1
                logger.debug(
                    f"Using {secondary_name} handler as fallback (no {primary_name} match)",
                    extra={
                        "handler_id": descriptor.handler_id,
                        "handler_name": descriptor.name,
                        "source": secondary_name,
                    },
                )

        # NOTE: Validation errors from bootstrap and contract sources are intentionally
        # combined WITHOUT deduplication. During migration, the same error appearing from
        # BOTH sources helps distinguish handler-level issues (error in both) from
        # source-specific configuration problems (error in only one). This preserves
        # diagnostic signal that would be lost if we deduplicated.
        all_errors: list[ModelHandlerValidationError] = list(
            bootstrap_result.validation_errors
        ) + list(contract_result.validation_errors)

        # Log structured counts for observability
        logger.info(
            "Handler resolution completed (HYBRID mode)",
            extra={
                "mode": self._mode.value,
                "allow_bootstrap_override": self._allow_bootstrap_override,
                "precedence": primary_name,
                "contract_handler_count": len(contract_result.descriptors),
                "bootstrap_handler_count": len(bootstrap_result.descriptors),
                "fallback_handler_count": fallback_count,
                "override_count": override_count,
                "resolved_handler_count": len(handlers_by_id),
                "validation_error_count": len(all_errors),
            },
        )

        return ModelContractDiscoveryResult(
            descriptors=list(handlers_by_id.values()),
            validation_errors=all_errors,
        )


__all__ = ["HandlerSourceResolver"]
