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

Error Codes:
    This module uses structured error codes for precise error classification:

    **Parse Errors (KAFKA_CONTRACT_001)**:
        - ``KAFKA_CONTRACT_001`` (PARSE_FAILURE_STRICT_MODE): Contract parsing failed
          in strict mode (graceful_mode=False). Raised when YAML parsing, Pydantic
          validation, or contract schema validation fails.

          **When Raised**: Only in strict mode. In graceful mode, parse errors are
          logged and cached as ``ModelParseError`` instead of raising.

          **Remediation**:
            1. Check contract YAML syntax with a YAML linter
            2. Validate contract against ``ModelHandlerContract`` schema
            3. Enable ``graceful_mode=True`` if partial failures are acceptable
            4. Check logs for the underlying error (YAML, validation, etc.)

    **Size Limit Errors (KAFKA_CONTRACT_002)**:
        - ``KAFKA_CONTRACT_002`` (CONTRACT_SIZE_EXCEEDED): Contract YAML exceeds
          the 10MB size limit. This is a hard limit applied before YAML parsing
          to prevent memory exhaustion attacks.

          **When Raised**: During contract parsing when the UTF-8 encoded contract
          exceeds ``MAX_CONTRACT_SIZE`` (10,485,760 bytes).

          **Remediation**:
            1. Reduce contract size by removing unnecessary fields
            2. Split large handler configurations into multiple contracts
            3. Move large static data (e.g., schemas) to external files
            4. Verify the contract isn't corrupted or duplicated

    Error codes are accessible via ``error.error_code`` on raised ``ModelOnexError``
    exceptions. Include the correlation ID from logs when reporting errors.

.. versionadded:: 0.8.0
    Created as part of OMN-1654 Kafka-based contract discovery.
"""

from __future__ import annotations

import logging
import threading
from typing import Protocol, cast, get_args, runtime_checkable
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


class ContractYamlParser:
    """Parse contract YAML into ModelHandlerDescriptor.

    This class handles YAML parsing, validation, and error creation
    for contract documents received via Kafka events. It encapsulates
    all YAML-to-descriptor conversion logic in a single responsibility
    class.

    Attributes:
        environment: The environment name used for constructing contract paths.

    Example:
        >>> parser = ContractYamlParser(environment="dev")
        >>> descriptor = parser.parse("my.node", yaml_content, correlation_id)
        >>> print(descriptor.handler_id)

    .. versionadded:: 0.8.0
        Extracted from KafkaContractSource for single-responsibility.
    """

    __slots__ = ("_environment",)

    def __init__(self, environment: str = "dev") -> None:
        """Initialize the parser.

        Args:
            environment: Environment name for topic prefix (e.g., "dev", "prod").
                Used for constructing contract_path URIs.
        """
        self._environment = environment

    @property
    def environment(self) -> str:
        """Return the environment name.

        Returns:
            The environment name (e.g., "dev", "prod").
        """
        return self._environment

    def parse(
        self,
        node_name: str,
        contract_yaml: str,
        correlation_id: UUID,
    ) -> ModelHandlerDescriptor:
        """Parse contract YAML into ModelHandlerDescriptor.

        Args:
            node_name: Node identifier (used for error context and path).
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

        # Build descriptor - validate handler_kind before casting
        archetype_value = contract.descriptor.node_archetype.value
        valid_handler_kinds = get_args(LiteralHandlerKind)
        if archetype_value not in valid_handler_kinds:
            raise ValueError(
                f"Invalid node_archetype value '{archetype_value}'. "
                f"Expected one of: {valid_handler_kinds}"
            )
        # Value validated above, cast is now safe
        handler_kind = cast("LiteralHandlerKind", archetype_value)

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

    def create_parse_error(
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


class KafkaContractCache:
    """Thread-safe cache for contract descriptors and validation errors.

    This class manages the in-memory storage of handler descriptors discovered
    from Kafka contract registration events. All operations are thread-safe
    and can be called concurrently from multiple Kafka consumer threads.

    The cache provides atomic operations for:
    - Adding/removing handler descriptors by node name
    - Collecting validation errors during event processing
    - Retrieving all cached data with proper thread synchronization

    Attributes:
        count: Number of cached descriptors (thread-safe read).
        error_count: Number of pending validation errors (thread-safe read).

    Example:
        >>> cache = KafkaContractCache()
        >>> cache.add("my.node", descriptor)
        >>> assert cache.count == 1
        >>> removed = cache.remove("my.node")
        >>> assert removed is not None

    .. versionadded:: 0.8.0
        Extracted from KafkaContractSource for single-responsibility.
    """

    __slots__ = ("_descriptors", "_errors", "_lock")

    def __init__(self) -> None:
        """Initialize an empty cache with thread-safe locking."""
        self._descriptors: dict[str, ModelHandlerDescriptor] = {}
        self._errors: list[ModelHandlerValidationError] = []
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        """Return the number of cached descriptors.

        Returns:
            Number of handler descriptors currently in the cache.
        """
        with self._lock:
            return len(self._descriptors)

    @property
    def error_count(self) -> int:
        """Return the number of pending validation errors.

        Returns:
            Number of validation errors awaiting retrieval.
        """
        with self._lock:
            return len(self._errors)

    def add(self, node_name: str, descriptor: ModelHandlerDescriptor) -> None:
        """Add or update a descriptor in the cache.

        If a descriptor with the same node_name exists, it is replaced.

        Args:
            node_name: Unique identifier for the node (cache key).
            descriptor: The handler descriptor to cache.
        """
        with self._lock:
            self._descriptors[node_name] = descriptor

    def remove(self, node_name: str) -> ModelHandlerDescriptor | None:
        """Remove a descriptor from the cache.

        Args:
            node_name: Unique identifier for the node to remove.

        Returns:
            The removed descriptor if found, None otherwise.
        """
        with self._lock:
            return self._descriptors.pop(node_name, None)

    def get_all(self) -> list[ModelHandlerDescriptor]:
        """Return all cached descriptors.

        Returns:
            A copy of all cached descriptors. Modifying the returned
            list does not affect the cache.
        """
        with self._lock:
            return list(self._descriptors.values())

    def add_error(self, error: ModelHandlerValidationError) -> None:
        """Add a validation error to the pending errors list.

        Args:
            error: The validation error to add.
        """
        with self._lock:
            self._errors.append(error)

    def get_errors(self) -> list[ModelHandlerValidationError]:
        """Return pending errors WITHOUT clearing them.

        This method provides read-only access to pending errors for
        inspection without consuming them.

        Returns:
            A copy of the pending validation errors.
        """
        with self._lock:
            return list(self._errors)

    def get_and_clear_errors(self) -> list[ModelHandlerValidationError]:
        """Atomically get and clear all pending errors.

        This method returns the current errors and clears the internal
        list in a single atomic operation, ensuring no errors are lost
        in concurrent access scenarios.

        Returns:
            The list of pending errors that were cleared.
        """
        with self._lock:
            errors = list(self._errors)
            self._errors.clear()
            return errors

    def clear(self) -> int:
        """Clear all cached descriptors and pending errors.

        Returns:
            Number of descriptors that were cleared.
        """
        with self._lock:
            count = len(self._descriptors)
            self._descriptors.clear()
            self._errors.clear()
            return count


@runtime_checkable
class ProtocolContractEventCallbacks(Protocol):
    """Protocol defining callbacks for contract registration events.

    This protocol defines the interface required by MixinTypedContractEvents
    for delegating typed event handling to the host class. Any class using
    MixinTypedContractEvents must implement these methods.

    The protocol enables type-safe cross-mixin method access without
    inheritance conflicts or type: ignore comments.

    Methods:
        on_contract_registered: Process a contract registration event.
        on_contract_deregistered: Process a contract deregistration event.

    Example:
        >>> class MySource(MixinTypedContractEvents, ProtocolContractSource):
        ...     def on_contract_registered(
        ...         self, node_name: str, contract_yaml: str, correlation_id: UUID | None
        ...     ) -> bool:
        ...         # Implementation here
        ...         return True
        ...
        ...     def on_contract_deregistered(
        ...         self, node_name: str, correlation_id: UUID | None
        ...     ) -> bool:
        ...         # Implementation here
        ...         return True

    .. versionadded:: 0.8.0
        Created for type-safe mixin composition in MixinTypedContractEvents.
    """

    def on_contract_registered(
        self,
        node_name: str,
        contract_yaml: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Cache descriptor from contract registration event.

        Args:
            node_name: Unique identifier for the node (used as cache key).
            contract_yaml: Full YAML content of the handler contract.
            correlation_id: Optional correlation ID from the event for tracing.

        Returns:
            True if the contract was successfully cached, False if parsing failed.
        """
        ...

    def on_contract_deregistered(
        self,
        node_name: str,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Remove descriptor from cache on deregistration event.

        Args:
            node_name: Unique identifier for the node to remove.
            correlation_id: Optional correlation ID from the event for tracing.

        Returns:
            True if a descriptor was removed, False if not found in cache.
        """
        ...


class MixinTypedContractEvents:
    """Mixin providing typed event handler and error inspection methods.

    This mixin provides convenience methods for handling typed contract
    events from omnibase_core and inspecting pending validation errors.
    It delegates to the primitive methods (on_contract_registered,
    on_contract_deregistered) and the cache which must be available
    in the class using this mixin.

    This mixin reduces method count in KafkaContractSource by extracting
    the typed event handling and error inspection into a separate concern.

    Requirements:
        Classes using this mixin MUST implement ProtocolContractEventCallbacks
        (i.e., provide on_contract_registered and on_contract_deregistered methods).

    .. versionadded:: 0.8.0
        Extracted from KafkaContractSource for single-responsibility.
    """

    # Type hints for mixin - these must be provided by the using class
    _cache: KafkaContractCache

    def _as_event_handler(self) -> ProtocolContractEventCallbacks:
        """Cast self to ProtocolContractEventCallbacks for type-safe method access.

        Returns:
            Self cast as ProtocolContractEventCallbacks for type checker.

        Note:
            Only call this method when the host class implements
            ProtocolContractEventCallbacks (provides on_contract_registered
            and on_contract_deregistered methods).
        """
        return cast("ProtocolContractEventCallbacks", self)

    def get_pending_errors(self) -> list[ModelHandlerValidationError]:
        """Return pending validation errors WITHOUT clearing them.

        This method provides read-only access to pending errors for inspection
        purposes. Unlike ``discover_handlers()`` which clears errors after
        returning them (one-shot behavior), this method allows repeated
        inspection of the same errors.

        Use this method when you need to:
        - Check error state without consuming errors
        - Log or display errors before a discover_handlers() call
        - Implement custom error handling before the next discovery cycle

        Returns:
            A copy of the pending validation errors list. Modifying the
            returned list does not affect the internal state.

        Thread Safety:
            This method is thread-safe. The returned list is a copy created
            while holding the internal lock.

        Example:
            >>> source = KafkaContractSource()
            >>> # Inspect errors without consuming
            >>> errors = source.get_pending_errors()
            >>> print(f"Found {len(errors)} pending errors")
            >>>
            >>> # Errors are still available for discover_handlers()
            >>> result = await source.discover_handlers()
            >>> assert len(result.validation_errors) == len(errors)
            >>>
            >>> # Now errors are cleared (one-shot)
            >>> assert source.pending_error_count == 0

        See Also:
            - ``pending_error_count``: Quick count check without list copy.
            - ``discover_handlers()``: Consumes errors (one-shot retrieval).
        """
        return self._cache.get_errors()

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
        return self._as_event_handler().on_contract_registered(
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
        return self._as_event_handler().on_contract_deregistered(
            node_name=event.node_name,
            correlation_id=event.correlation_id,
        )


class KafkaContractSource(MixinTypedContractEvents, ProtocolContractSource):
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

    __slots__ = (
        "_cache",
        "_correlation_id",
        "_environment",
        "_graceful_mode",
        "_parser",
    )

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
        self._cache = KafkaContractCache()
        self._parser = ContractYamlParser(environment=environment)

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
        return self._cache.count

    @property
    def environment(self) -> str:
        """Return the environment name.

        Returns:
            The environment name (e.g., "dev", "prod").
        """
        return self._environment

    @property
    def graceful_mode(self) -> bool:
        """Return whether graceful mode is enabled.

        Returns:
            True if graceful mode is enabled, False otherwise.
        """
        return self._graceful_mode

    @property
    def correlation_id(self) -> UUID:
        """Return the source correlation ID.

        Returns:
            The unique correlation ID for this source instance.
        """
        return self._correlation_id

    @property
    def pending_error_count(self) -> int:
        """Return the number of pending validation errors.

        This property provides a quick way to check if there are pending
        errors without consuming them. Use ``get_pending_errors()`` to
        inspect errors without clearing, or ``discover_handlers()`` to
        consume errors (one-shot retrieval).

        Returns:
            Number of validation errors currently pending.

        Example:
            >>> source = KafkaContractSource()
            >>> if source.pending_error_count > 0:
            ...     errors = source.get_pending_errors()
            ...     for err in errors:
            ...         print(f"Pending error: {err.message}")
        """
        return self._cache.error_count

    async def discover_handlers(self) -> ModelContractDiscoveryResult:
        """Return cached descriptors from contract events.

        This method returns whatever descriptors have been cached from
        contract registration events. It does not perform any I/O or
        network operations - it simply returns the current cache state.

        Warning:
            **One-Shot Error Retrieval**: Validation errors are cleared after
            being returned. Calling this method twice will return an empty
            error list on the second call (unless new errors occurred between
            calls). Use ``get_pending_errors()`` to inspect errors without
            consuming them, or ``pending_error_count`` for a quick count check.

        Returns:
            ModelContractDiscoveryResult with cached descriptors and any
            validation errors encountered during event processing. Errors
            are cleared from internal state after this call returns.

        Example:
            >>> source = KafkaContractSource()
            >>> # First call returns accumulated errors
            >>> result1 = await source.discover_handlers()
            >>> print(f"Errors: {len(result1.validation_errors)}")
            >>>
            >>> # Second call has empty errors (already consumed)
            >>> result2 = await source.discover_handlers()
            >>> assert len(result2.validation_errors) == 0

        Implementation Detail:
            This method uses a two-step retrieval pattern: errors are cleared
            first, then descriptors are fetched. Each step is individually
            atomic (protected by the cache lock), but there is a brief window
            between the two operations where new events could arrive.

            If a contract event fails parsing between ``get_and_clear_errors()``
            and ``get_all()``, the resulting error will NOT be included in this
            discovery result - it will appear in the next ``discover_handlers()``
            call. Similarly, a successful registration between the two calls
            will include the new descriptor but any error from a concurrent
            failed registration will be deferred.

            This eventual consistency is acceptable for the cache-only beta
            model where discovered contracts take effect on the next restart
            anyway. The design prioritizes simplicity over perfect atomicity,
            avoiding the complexity of a single lock spanning both operations
            (which would increase lock contention with concurrent Kafka
            consumer threads).

        See Also:
            - ``get_pending_errors()``: Inspect errors without clearing.
            - ``pending_error_count``: Quick count check.
        """
        # Two-step retrieval: errors cleared first, then descriptors fetched.
        # See "Implementation Detail" in docstring for timing window behavior.
        errors = self._cache.get_and_clear_errors()
        descriptors = self._cache.get_all()

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
            descriptor = self._parser.parse(
                node_name=node_name,
                contract_yaml=contract_yaml,
                correlation_id=event_correlation,
            )
            self._cache.add(node_name, descriptor)

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
            error = self._parser.create_parse_error(
                node_name=node_name,
                error=e,
                correlation_id=event_correlation,
            )

            if self._graceful_mode:
                self._cache.add_error(error)
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
                    correlation_id=event_correlation,
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

        removed = self._cache.remove(node_name)

        if removed is not None:
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

    def clear_cache(self) -> int:
        """Clear all cached descriptors.

        Utility method for testing and runtime reset scenarios.

        Returns:
            Number of descriptors that were cleared.
        """
        count = self._cache.clear()

        logger.info(
            "Contract cache cleared",
            extra={
                "cleared_count": count,
                "correlation_id": str(self._correlation_id),
            },
        )
        return count


__all__ = [
    "ContractYamlParser",
    "KafkaContractCache",
    "KafkaContractSource",
    "MAX_CONTRACT_SIZE",
    "MixinTypedContractEvents",
    "ProtocolContractEventCallbacks",
    # Re-exported from omnibase_core for convenience
    "ModelContractDeregisteredEvent",
    "ModelContractRegisteredEvent",
    "TOPIC_SUFFIX_CONTRACT_DEREGISTERED",
    "TOPIC_SUFFIX_CONTRACT_REGISTERED",
]
