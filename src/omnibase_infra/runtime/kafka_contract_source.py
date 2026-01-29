# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka-Based Contract Source for Event-Driven Discovery.

This module provides KafkaContractSource for discovering handler contracts
via Kafka events instead of filesystem or registry polling.

Part of OMN-1654: KafkaContractSource (cache + discovery).

**Beta Implementation**: Cache-only model. Does NOT wire business subscriptions
dynamically. Discovered contracts take effect on next restart.

Contract Event Flow:
    1. External system publishes ModelContractRegisteredEvent to platform topic
    2. KafkaContractSource receives event via baseline-wired subscription
    3. Contract YAML is parsed and cached as ModelHandlerDescriptor
    4. Next call to discover_handlers() returns cached descriptors
    5. Runtime restart applies new handler configuration

Event Topics (Platform Reserved):
    - Registration: {env}.{TOPIC_SUFFIX_CONTRACT_REGISTERED}
    - Deregistration: {env}.{TOPIC_SUFFIX_CONTRACT_DEREGISTERED}

    Topic suffixes are imported from omnibase_core.constants for single source of truth.

See Also:
    - HandlerContractSource: Filesystem-based discovery
    - RegistryContractSource: Consul KV-based discovery
    - ProtocolContractSource: Protocol definition

.. versionadded:: 0.8.0
    Created as part of OMN-1654 Kafka-based contract discovery.
"""

from __future__ import annotations

import logging
import threading
from typing import cast
from uuid import UUID, uuid4

import yaml
from pydantic import ValidationError

from omnibase_core.constants import (
    TOPIC_SUFFIX_CONTRACT_DEREGISTERED,
    TOPIC_SUFFIX_CONTRACT_REGISTERED,
)
from omnibase_core.models.contracts.model_handler_contract import ModelHandlerContract
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.events import (
    ModelContractDeregisteredEvent,
    ModelContractRegisteredEvent,
)
from omnibase_infra.enums import EnumHandlerErrorType, EnumHandlerSourceType
from omnibase_infra.models.errors import ModelHandlerValidationError
from omnibase_infra.models.handlers import (
    LiteralHandlerKind,
    ModelContractDiscoveryResult,
    ModelHandlerDescriptor,
    ModelHandlerIdentifier,
)
from omnibase_infra.runtime.protocol_contract_source import ProtocolContractSource

logger = logging.getLogger(__name__)

# Forward Reference Resolution:
# ModelContractDiscoveryResult uses a forward reference to ModelHandlerValidationError.
# Since we import ModelHandlerValidationError above, we can call model_rebuild() here
# to resolve the forward reference. This call is idempotent - multiple calls are harmless.
ModelContractDiscoveryResult.model_rebuild()

# Maximum contract size (same as other sources)
MAX_CONTRACT_SIZE = 10 * 1024 * 1024  # 10MB


class KafkaContractSource(ProtocolContractSource):
    """Kafka-based contract source - cache + discovery only.

    Subscribes to platform-reserved contract topics (baseline-wired).
    Maintains in-memory cache of descriptors derived from contract YAML.

    Does NOT wire business subscriptions dynamically.
    For beta: discover + next restart applies.

    This source maintains an in-memory cache of handler descriptors that is
    populated by contract registration events received via Kafka. The cache
    is read-only from the discover_handlers() perspective - it simply returns
    whatever has been cached from events.

    Thread Safety:
        This class is thread-safe. All access to the internal cache
        (``_cached_descriptors``) and error list (``_pending_errors``) is
        protected by a ``threading.Lock``. Multiple Kafka consumer threads
        may safely call ``on_contract_registered()`` and
        ``on_contract_deregistered()`` concurrently.

    Attributes:
        source_type: Returns "KAFKA_EVENTS" as the source type identifier.

    Example:
        >>> source = KafkaContractSource(environment="dev")
        >>>
        >>> # Event handler wiring (done by runtime)
        >>> source.on_contract_registered(event)
        >>>
        >>> # Discovery returns cached descriptors
        >>> result = await source.discover_handlers()
        >>> for desc in result.descriptors:
        ...     print(f"Cached: {desc.handler_id}")

    Note:
        This class does NOT handle Kafka subscription setup. The runtime is
        responsible for wiring the platform-reserved contract topics to the
        on_contract_registered/on_contract_deregistered methods.

    .. versionadded:: 0.8.0
        Created as part of OMN-1654 Kafka-based contract discovery.
    """

    def __init__(
        self,
        environment: str = "dev",
        graceful_mode: bool = True,
    ) -> None:
        """Initialize the Kafka contract source.

        Args:
            environment: Environment name for topic prefix (e.g., "dev", "prod").
                Used for observability logging only - actual topic wiring is
                done by the runtime.
            graceful_mode: If True (default), collect errors instead of raising.
                For cache-based sources, graceful mode is typically preferred
                since individual event failures should not crash the runtime.
        """
        self._environment = environment
        self._graceful_mode = graceful_mode
        self._correlation_id = uuid4()

        # In-memory cache of discovered descriptors
        # Key: node_name (from contract event)
        # Value: ModelHandlerDescriptor parsed from contract YAML
        self._cached_descriptors: dict[str, ModelHandlerDescriptor] = {}

        # Track validation errors from event processing
        # These are cleared on each discover_handlers() call
        self._pending_errors: list[ModelHandlerValidationError] = []

        # Lock for thread-safe access to _cached_descriptors and _pending_errors
        # Required because Kafka consumer callbacks may run on multiple threads
        self._lock = threading.Lock()

        logger.info(
            "KafkaContractSource initialized",
            extra={
                "environment": environment,
                "graceful_mode": graceful_mode,
                "correlation_id": str(self._correlation_id),
            },
        )

    @property
    def source_type(self) -> str:
        """Return source type identifier.

        Returns:
            "KAFKA_EVENTS" as the source type.
        """
        return "KAFKA_EVENTS"

    @property
    def cached_count(self) -> int:
        """Return the number of cached descriptors.

        Returns:
            Number of handler descriptors currently cached.
        """
        with self._lock:
            return len(self._cached_descriptors)

    async def discover_handlers(self) -> ModelContractDiscoveryResult:
        """Return cached descriptors from contract events.

        This method returns whatever descriptors have been cached from
        contract registration events. It does not perform any I/O or
        network operations - it simply returns the current cache state.

        Returns:
            ModelContractDiscoveryResult with cached descriptors and any
            validation errors encountered during event processing.
        """
        # Atomically collect and clear pending state
        with self._lock:
            errors = list(self._pending_errors)
            self._pending_errors.clear()
            descriptors = list(self._cached_descriptors.values())

        logger.info(
            "Handler discovery completed (KAFKA_EVENTS mode)",
            extra={
                "cached_descriptor_count": len(descriptors),
                "validation_error_count": len(errors),
                "environment": self._environment,
                "correlation_id": str(self._correlation_id),
            },
        )

        return ModelContractDiscoveryResult(
            descriptors=descriptors,
            validation_errors=errors,
        )

    def on_contract_registered(
        self,
        node_name: str,
        contract_yaml: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Cache descriptor from contract registration event.

        Called by the runtime when a contract registration event is received
        on the platform-reserved contract topic.

        Args:
            node_name: Unique identifier for the node (used as cache key).
            contract_yaml: Full YAML content of the handler contract.
            correlation_id: Optional correlation ID from the event for tracing.

        Returns:
            True if the contract was successfully cached, False if parsing failed.

        Note:
            In graceful mode, parsing errors are collected in pending_errors
            and returned on the next discover_handlers() call. In strict mode,
            errors are raised immediately.
        """
        event_correlation = correlation_id or uuid4()

        logger.debug(
            "Processing contract registration event",
            extra={
                "node_name": node_name,
                "contract_size": len(contract_yaml),
                "correlation_id": str(event_correlation),
                "source_correlation_id": str(self._correlation_id),
            },
        )

        try:
            descriptor = self._yaml_to_descriptor(
                node_name=node_name,
                contract_yaml=contract_yaml,
                correlation_id=event_correlation,
            )
            with self._lock:
                self._cached_descriptors[node_name] = descriptor

            logger.info(
                "Contract registered and cached",
                extra={
                    "node_name": node_name,
                    "handler_id": descriptor.handler_id,
                    "handler_version": str(descriptor.version),
                    "correlation_id": str(event_correlation),
                },
            )
            return True

        except (yaml.YAMLError, ValidationError, ModelOnexError, ValueError) as e:
            error = self._create_parse_error(
                node_name=node_name,
                error=e,
                correlation_id=event_correlation,
            )

            if self._graceful_mode:
                with self._lock:
                    self._pending_errors.append(error)
                logger.warning(
                    "Contract registration failed (graceful mode)",
                    extra={
                        "node_name": node_name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "correlation_id": str(event_correlation),
                    },
                )
                return False
            else:
                raise ModelOnexError(
                    f"Failed to parse contract for node '{node_name}': {e}",
                    error_code="KAFKA_CONTRACT_001",
                ) from e

    def on_contract_deregistered(
        self,
        node_name: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Remove descriptor from cache on deregistration event.

        Called by the runtime when a contract deregistration event is received
        on the platform-reserved contract topic.

        Args:
            node_name: Unique identifier for the node to remove.
            correlation_id: Optional correlation ID from the event for tracing.

        Returns:
            True if a descriptor was removed, False if not found in cache.
        """
        event_correlation = correlation_id or uuid4()

        with self._lock:
            removed = self._cached_descriptors.pop(node_name, None)

        if removed:
            logger.info(
                "Contract deregistered and removed from cache",
                extra={
                    "node_name": node_name,
                    "handler_id": removed.handler_id,
                    "correlation_id": str(event_correlation),
                },
            )
            return True
        else:
            logger.debug(
                "Contract deregistration for unknown node (no-op)",
                extra={
                    "node_name": node_name,
                    "correlation_id": str(event_correlation),
                },
            )
            return False

    def handle_registered_event(
        self,
        event: ModelContractRegisteredEvent,
    ) -> bool:
        """Handle a typed contract registration event.

        This is the preferred method for processing registration events when
        using the typed event models from omnibase_core. It extracts the
        relevant fields and delegates to on_contract_registered().

        Args:
            event: The typed contract registration event from Kafka.

        Returns:
            True if the contract was successfully cached, False if parsing failed.

        Example:
            >>> source = KafkaContractSource()
            >>> event = ModelContractRegisteredEvent(
            ...     node_name="my.handler",
            ...     contract_yaml="...",
            ...     # ... other fields
            ... )
            >>> success = source.handle_registered_event(event)
        """
        return self.on_contract_registered(
            node_name=event.node_name,
            contract_yaml=event.contract_yaml,
            correlation_id=event.correlation_id,
        )

    def handle_deregistered_event(
        self,
        event: ModelContractDeregisteredEvent,
    ) -> bool:
        """Handle a typed contract deregistration event.

        This is the preferred method for processing deregistration events when
        using the typed event models from omnibase_core. It extracts the
        relevant fields and delegates to on_contract_deregistered().

        Args:
            event: The typed contract deregistration event from Kafka.

        Returns:
            True if a descriptor was removed, False if not found in cache.

        Example:
            >>> source = KafkaContractSource()
            >>> event = ModelContractDeregisteredEvent(
            ...     node_name="my.handler",
            ...     reason=EnumDeregistrationReason.SHUTDOWN,
            ...     # ... other fields
            ... )
            >>> removed = source.handle_deregistered_event(event)
        """
        return self.on_contract_deregistered(
            node_name=event.node_name,
            correlation_id=event.correlation_id,
        )

    def _yaml_to_descriptor(
        self,
        node_name: str,
        contract_yaml: str,
        correlation_id: UUID,
    ) -> ModelHandlerDescriptor:
        """Convert contract YAML to ModelHandlerDescriptor.

        Args:
            node_name: Node identifier (used for error context).
            contract_yaml: Full YAML content of the handler contract.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelHandlerDescriptor parsed from the contract.

        Raises:
            yaml.YAMLError: If YAML parsing fails.
            ValidationError: If Pydantic validation fails.
            ModelOnexError: If contract exceeds size limit.
            ValueError: If contract data is invalid.
        """
        # Check size limit
        contract_bytes = contract_yaml.encode("utf-8")
        if len(contract_bytes) > MAX_CONTRACT_SIZE:
            raise ModelOnexError(
                f"Contract exceeds size limit: {len(contract_bytes)} bytes "
                f"(max: {MAX_CONTRACT_SIZE} bytes)",
                error_code="KAFKA_CONTRACT_002",
            )

        # Parse YAML
        contract_data = yaml.safe_load(contract_yaml)
        if not contract_data:
            raise ValueError("Contract YAML is empty or invalid")

        # Validate against ModelHandlerContract
        contract = ModelHandlerContract.model_validate(contract_data)

        # Extract handler_class from metadata section
        # NOTE: handler_class must be in metadata per ModelHandlerContract schema
        # (root-level extra fields are forbidden by Pydantic extra='forbid')
        # TODO [OMN-1420]: Use contract.handler_class once available in schema
        handler_class = None
        if isinstance(contract_data, dict):
            metadata = contract_data.get("metadata", {})
            if isinstance(metadata, dict):
                handler_class = metadata.get("handler_class")

        if handler_class is None:
            logger.debug(
                "handler_class missing from contract, handler may not be loadable",
                extra={
                    "node_name": node_name,
                    "handler_id": contract.handler_id,
                    "correlation_id": str(correlation_id),
                },
            )

        # Build descriptor
        handler_kind = cast(
            "LiteralHandlerKind", contract.descriptor.node_archetype.value
        )

        return ModelHandlerDescriptor(
            handler_id=contract.handler_id,
            name=contract.name,
            version=contract.contract_version,
            handler_kind=handler_kind,
            input_model=contract.input_model,
            output_model=contract.output_model,
            description=contract.description,
            handler_class=handler_class,
            contract_path=f"kafka://{self._environment}/contracts/{node_name}",
            contract_config=contract_data,
        )

    def _create_parse_error(
        self,
        node_name: str,
        error: Exception,
        correlation_id: UUID,
    ) -> ModelHandlerValidationError:
        """Create a validation error for contract parse failures.

        Args:
            node_name: The node identifier.
            error: The parsing error.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelHandlerValidationError with parse error details.
        """
        handler_identity = ModelHandlerIdentifier.from_handler_id(
            f"kafka://{node_name}"
        )

        return ModelHandlerValidationError(
            error_type=EnumHandlerErrorType.CONTRACT_PARSE_ERROR,
            rule_id="KAFKA-001",
            handler_identity=handler_identity,
            source_type=EnumHandlerSourceType.CONTRACT,
            message=f"Failed to parse contract from Kafka event for node '{node_name}': {error}",
            remediation_hint="Check YAML syntax and required contract fields in the registration event",
            correlation_id=correlation_id,
        )

    def clear_cache(self) -> int:
        """Clear all cached descriptors.

        Utility method for testing and runtime reset scenarios.

        Returns:
            Number of descriptors that were cleared.
        """
        with self._lock:
            count = len(self._cached_descriptors)
            self._cached_descriptors.clear()
            self._pending_errors.clear()

        logger.info(
            "Contract cache cleared",
            extra={
                "cleared_count": count,
                "correlation_id": str(self._correlation_id),
            },
        )
        return count


__all__ = [
    "KafkaContractSource",
    "MAX_CONTRACT_SIZE",
    # Re-exported from omnibase_core for convenience
    "ModelContractDeregisteredEvent",
    "ModelContractRegisteredEvent",
    "TOPIC_SUFFIX_CONTRACT_DEREGISTERED",
    "TOPIC_SUFFIX_CONTRACT_REGISTERED",
]
