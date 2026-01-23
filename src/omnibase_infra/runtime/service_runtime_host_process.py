# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Runtime Host Process implementation for ONEX Infrastructure.

This module implements the RuntimeHostProcess class, which is responsible for:
- Owning and managing an event bus instance (EventBusInmemory or EventBusKafka)
- Registering handlers via the wiring module
- Subscribing to event bus topics and routing envelopes to handlers
- Handling errors by producing success=False response envelopes
- Processing envelopes sequentially (no parallelism in MVP)
- Basic shutdown (no graceful drain in MVP)

The RuntimeHostProcess is the central coordinator for infrastructure runtime,
bridging event-driven message routing with protocol handlers.

Event Bus Support:
    The RuntimeHostProcess supports two event bus implementations:
    - EventBusInmemory: For local development and testing
    - EventBusKafka: For production use with Kafka/Redpanda

    The event bus can be injected via constructor or auto-created based on config.

Example Usage:
    ```python
    from omnibase_infra.runtime import RuntimeHostProcess

    async def main() -> None:
        process = RuntimeHostProcess()
        await process.start()
        try:
            # Process handles messages via event bus subscription
            await asyncio.sleep(60)
        finally:
            await process.stop()
    ```

Integration with Handlers:
    Handlers are registered during start() via the wiring module. Each handler
    processes envelopes for a specific protocol type (e.g., "http", "db").
    The handler_type field in envelopes determines routing.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pydantic import BaseModel

from omnibase_infra.enums import EnumHandlerSourceMode, EnumInfraTransportType
from omnibase_infra.errors import (
    EnvelopeValidationError,
    ModelInfraErrorContext,
    ProtocolConfigurationError,
    RuntimeHostError,
    UnknownHandlerTypeError,
)
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.runtime.envelope_validator import (
    normalize_correlation_id,
    validate_envelope,
)
from omnibase_infra.runtime.handler_registry import RegistryProtocolBinding
from omnibase_infra.runtime.models import ModelDuplicateResponse
from omnibase_infra.runtime.protocol_lifecycle_executor import ProtocolLifecycleExecutor
from omnibase_infra.runtime.util_wiring import wire_default_handlers
from omnibase_infra.utils.util_env_parsing import parse_env_float

if TYPE_CHECKING:
    from omnibase_core.container import ModelONEXContainer
    from omnibase_infra.event_bus.models import ModelEventMessage
    from omnibase_infra.idempotency import ModelIdempotencyGuardConfig
    from omnibase_infra.idempotency.protocol_idempotency_store import (
        ProtocolIdempotencyStore,
    )
    from omnibase_infra.models.handlers import (
        ModelHandlerDescriptor,
        ModelHandlerSourceConfig,
    )
    from omnibase_infra.nodes.architecture_validator import ProtocolArchitectureRule
    from omnibase_infra.protocols import ProtocolContainerAware
    from omnibase_infra.runtime.contract_handler_discovery import (
        ContractHandlerDiscovery,
    )

from omnibase_infra.models.types import JsonDict

# Expose wire_default_handlers as wire_handlers for test patching compatibility
# Tests patch "omnibase_infra.runtime.service_runtime_host_process.wire_handlers"
wire_handlers = wire_default_handlers

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_INPUT_TOPIC = "requests"
DEFAULT_OUTPUT_TOPIC = "responses"
DEFAULT_GROUP_ID = "runtime-host"

# Health check timeout bounds (per ModelLifecycleSubcontract)
MIN_HEALTH_CHECK_TIMEOUT = 1.0
MAX_HEALTH_CHECK_TIMEOUT = 60.0
DEFAULT_HEALTH_CHECK_TIMEOUT: float = parse_env_float(
    "ONEX_HEALTH_CHECK_TIMEOUT",
    5.0,
    min_value=MIN_HEALTH_CHECK_TIMEOUT,
    max_value=MAX_HEALTH_CHECK_TIMEOUT,
    transport_type=EnumInfraTransportType.RUNTIME,
    service_name="runtime_host_process",
)

# Drain timeout bounds for graceful shutdown (OMN-756)
# Controls how long to wait for in-flight messages to complete before shutdown
MIN_DRAIN_TIMEOUT_SECONDS = 1.0
MAX_DRAIN_TIMEOUT_SECONDS = 300.0
DEFAULT_DRAIN_TIMEOUT_SECONDS: float = parse_env_float(
    "ONEX_DRAIN_TIMEOUT",
    30.0,
    min_value=MIN_DRAIN_TIMEOUT_SECONDS,
    max_value=MAX_DRAIN_TIMEOUT_SECONDS,
    transport_type=EnumInfraTransportType.RUNTIME,
    service_name="runtime_host_process",
)


class RuntimeHostProcess:
    """Runtime host process that owns event bus and coordinates handlers.

    The RuntimeHostProcess is the central coordinator for ONEX infrastructure
    runtime. It owns an event bus instance (EventBusInmemory or EventBusKafka),
    registers handlers via the wiring module, and routes incoming envelopes to
    appropriate handlers.

    Container Integration:
        RuntimeHostProcess now accepts a ModelONEXContainer parameter for
        dependency injection. The container provides access to:
        - RegistryProtocolBinding: Handler registry for protocol routing

        This follows ONEX container-based DI patterns for better testability
        and lifecycle management. The legacy singleton pattern is deprecated
        in favor of container resolution.

    Attributes:
        event_bus: The owned event bus instance (EventBusInmemory or EventBusKafka)
        is_running: Whether the process is currently running
        input_topic: Topic to subscribe to for incoming envelopes
        output_topic: Topic to publish responses to
        group_id: Consumer group identifier

    Example:
        ```python
        from omnibase_core.container import ModelONEXContainer
        from omnibase_infra.runtime.util_container_wiring import wire_infrastructure_services

        # Container-based initialization (preferred)
        container = ModelONEXContainer()
        wire_infrastructure_services(container)
        process = RuntimeHostProcess(container=container)
        await process.start()
        health = await process.health_check()
        await process.stop()

        # Direct initialization (without container)
        process = RuntimeHostProcess()  # Uses singleton registries
        ```

    Graceful Shutdown:
        The stop() method implements graceful shutdown with a configurable drain
        period. After unsubscribing from topics, it waits for in-flight messages
        to complete before shutting down handlers and closing the event bus.
        See stop() docstring for configuration details.
    """

    def __init__(
        self,
        container: ModelONEXContainer | None = None,
        event_bus: EventBusInmemory | EventBusKafka | None = None,
        input_topic: str = DEFAULT_INPUT_TOPIC,
        output_topic: str = DEFAULT_OUTPUT_TOPIC,
        config: dict[str, object] | None = None,
        handler_registry: RegistryProtocolBinding | None = None,
        architecture_rules: tuple[ProtocolArchitectureRule, ...] | None = None,
        contract_paths: list[str] | None = None,
    ) -> None:
        """Initialize the runtime host process.

        Args:
            container: Optional ONEX dependency injection container. When provided,
                the runtime host can resolve dependencies from the container if they
                are not explicitly provided. This follows the ONEX container-based
                DI pattern for better testability and explicit dependency management.

                Container Resolution (during async start()):
                    - If handler_registry is None and container is provided, resolves
                      RegistryProtocolBinding from container.service_registry
                    - Event bus must be provided explicitly or defaults to EventBusInmemory
                      (required immediately during __init__)

                Usage:
                    ```python
                    from omnibase_core.container import ModelONEXContainer
                    from omnibase_infra.runtime.util_container_wiring import wire_infrastructure_services

                    container = ModelONEXContainer()
                    await wire_infrastructure_services(container)
                    process = RuntimeHostProcess(container=container)
                    await process.start()
                    ```

            event_bus: Optional event bus instance (EventBusInmemory or EventBusKafka).
                       If None, creates EventBusInmemory.
            input_topic: Topic to subscribe to for incoming envelopes.
            output_topic: Topic to publish responses to.
            config: Optional configuration dict that can override topics and group_id.
                Supported keys:
                    - input_topic: Override input topic
                    - output_topic: Override output topic
                    - group_id: Override consumer group identifier
                    - health_check_timeout_seconds: Timeout for individual handler
                      health checks (default: 5.0 seconds, valid range: 1-60 per
                      ModelLifecycleSubcontract). Values outside this range are
                      clamped to the nearest bound with a warning logged.
                      Invalid string values fall back to the default with a warning.
                    - drain_timeout_seconds: Maximum time to wait for in-flight
                      messages to complete during graceful shutdown (default: 30.0
                      seconds, valid range: 1-300). Values outside this range are
                      clamped to the nearest bound with a warning logged.
            handler_registry: Optional RegistryProtocolBinding instance for handler lookup.
                Type: RegistryProtocolBinding | None

                Purpose:
                    Provides the registry that maps handler_type strings (e.g., "http", "db")
                    to their corresponding ProtocolContainerAware classes. The registry is queried
                    during start() to instantiate and initialize all registered handlers.

                Resolution Order:
                    1. If handler_registry is provided, uses this pre-resolved registry
                    2. If container is provided, resolves from container.service_registry
                    3. If None, falls back to singleton via get_handler_registry()

                Container Integration:
                    When using container-based DI (recommended), resolve the registry from
                    the container and pass it to RuntimeHostProcess:

                    ```python
                    async def create_runtime() -> RuntimeHostProcess:
                        container = ModelONEXContainer()
                        await wire_infrastructure_services(container)
                        registry = await container.service_registry.resolve_service(
                            RegistryProtocolBinding
                        )
                        return RuntimeHostProcess(handler_registry=registry)
                    ```

                    This follows ONEX container-based DI patterns for better testability
                    and explicit dependency management.

            container: Optional ONEX container for dependency injection. Required for
                architecture validation. If None and architecture validation is requested,
                a minimal container will be created.

            architecture_rules: Optional tuple of architecture rules to validate at startup.
                Type: tuple[ProtocolArchitectureRule, ...] | None

                Purpose:
                    Architecture rules are validated BEFORE the runtime starts. Violations
                    with ERROR severity will prevent startup. Violations with WARNING
                    severity are logged but don't block startup.

                    Rules implementing ProtocolArchitectureRule can be:
                    - Custom rules specific to your application
                    - Standard rules from OMN-1099 validators

                Example:
                    ```python
                    from my_rules import NoHandlerPublishingRule, NoAnyTypesRule

                    process = RuntimeHostProcess(
                        container=container,
                        architecture_rules=(
                            NoHandlerPublishingRule(),
                            NoAnyTypesRule(),
                        ),
                    )
                    await process.start()  # Validates architecture first
                    ```

            contract_paths: Optional list of paths to scan for handler contracts.
                Type: list[str] | None

                Purpose:
                    Enables contract-based handler discovery. When provided, the runtime
                    will auto-discover and register handlers from these paths during
                    start() instead of using wire_default_handlers().

                    Paths can be:
                    - Directories: Recursively scanned for handler contracts
                    - Files: Directly loaded as contract files

                Behavior:
                    - If contract_paths is provided: Uses ContractHandlerDiscovery
                      to auto-discover and register handlers from the specified paths.
                    - If contract_paths is None or empty: Falls back to the existing
                      wire_default_handlers() behavior.

                Error Handling:
                    Discovery errors are logged but do not block startup. This enables
                    graceful degradation where some handlers can be registered even
                    if others fail to load.

                Example:
                    ```python
                    # Contract-based handler discovery
                    process = RuntimeHostProcess(
                        contract_paths=["src/nodes/handlers", "plugins/"]
                    )
                    await process.start()

                    # Or with explicit file paths
                    process = RuntimeHostProcess(
                        contract_paths=[
                            "handlers/auth/handler_contract.yaml",
                            "handlers/db/handler_contract.yaml",
                        ]
                    )
                    ```
        """
        # Store container reference for dependency resolution
        self._container: ModelONEXContainer | None = container
        # Handler registry (container-based DI or singleton fallback)
        self._handler_registry: RegistryProtocolBinding | None = handler_registry

        # Architecture rules for startup validation
        self._architecture_rules: tuple[ProtocolArchitectureRule, ...] = (
            architecture_rules or ()
        )

        # Contract paths for handler discovery (OMN-1133)
        # Convert strings to Path objects for consistent filesystem operations
        self._contract_paths: list[Path] = (
            [Path(p) for p in contract_paths] if contract_paths else []
        )

        # Handler discovery service (lazy-created if contract_paths provided)
        self._handler_discovery: ContractHandlerDiscovery | None = None

        # Create or use provided event bus
        self._event_bus: EventBusInmemory | EventBusKafka = (
            event_bus or EventBusInmemory()
        )

        # Extract configuration with defaults
        config = config or {}

        # Topic configuration (config overrides constructor args)
        self._input_topic: str = str(config.get("input_topic", input_topic))
        self._output_topic: str = str(config.get("output_topic", output_topic))
        # Note: ModelRuntimeConfig uses field name "consumer_group" with alias "group_id".
        # When config.model_dump() is called, it outputs "consumer_group" by default.
        # We check both keys to support either field name or alias.
        # Empty strings and whitespace-only strings fall through to the next option.
        consumer_group = config.get("consumer_group")
        group_id = config.get("group_id")
        self._group_id: str = str(
            (consumer_group if consumer_group and str(consumer_group).strip() else None)
            or (group_id if group_id and str(group_id).strip() else None)
            or DEFAULT_GROUP_ID
        )

        # Health check configuration (from lifecycle subcontract pattern)
        # Default: 5.0 seconds, valid range: 1-60 seconds per ModelLifecycleSubcontract
        # Values outside bounds are clamped with a warning
        _timeout_raw = config.get("health_check_timeout_seconds")
        timeout_value: float = DEFAULT_HEALTH_CHECK_TIMEOUT
        if isinstance(_timeout_raw, int | float):
            timeout_value = float(_timeout_raw)
        elif isinstance(_timeout_raw, str):
            try:
                timeout_value = float(_timeout_raw)
            except ValueError:
                logger.warning(
                    "Invalid health_check_timeout_seconds string value, using default",
                    extra={
                        "invalid_value": _timeout_raw,
                        "default_value": DEFAULT_HEALTH_CHECK_TIMEOUT,
                    },
                )
                timeout_value = DEFAULT_HEALTH_CHECK_TIMEOUT

        # Validate bounds and clamp if necessary
        if (
            timeout_value < MIN_HEALTH_CHECK_TIMEOUT
            or timeout_value > MAX_HEALTH_CHECK_TIMEOUT
        ):
            logger.warning(
                "health_check_timeout_seconds out of valid range, clamping",
                extra={
                    "original_value": timeout_value,
                    "min_value": MIN_HEALTH_CHECK_TIMEOUT,
                    "max_value": MAX_HEALTH_CHECK_TIMEOUT,
                    "clamped_value": max(
                        MIN_HEALTH_CHECK_TIMEOUT,
                        min(timeout_value, MAX_HEALTH_CHECK_TIMEOUT),
                    ),
                },
            )
            timeout_value = max(
                MIN_HEALTH_CHECK_TIMEOUT,
                min(timeout_value, MAX_HEALTH_CHECK_TIMEOUT),
            )

        self._health_check_timeout_seconds: float = timeout_value

        # Drain timeout configuration for graceful shutdown (OMN-756)
        # Default: 30.0 seconds, valid range: 1-300 seconds
        # Values outside bounds are clamped with a warning
        _drain_timeout_raw = config.get("drain_timeout_seconds")
        drain_timeout_value: float = DEFAULT_DRAIN_TIMEOUT_SECONDS
        if isinstance(_drain_timeout_raw, int | float):
            drain_timeout_value = float(_drain_timeout_raw)
        elif isinstance(_drain_timeout_raw, str):
            try:
                drain_timeout_value = float(_drain_timeout_raw)
            except ValueError:
                logger.warning(
                    "Invalid drain_timeout_seconds string value, using default",
                    extra={
                        "invalid_value": _drain_timeout_raw,
                        "default_value": DEFAULT_DRAIN_TIMEOUT_SECONDS,
                    },
                )
                drain_timeout_value = DEFAULT_DRAIN_TIMEOUT_SECONDS

        # Validate drain timeout bounds and clamp if necessary
        if (
            drain_timeout_value < MIN_DRAIN_TIMEOUT_SECONDS
            or drain_timeout_value > MAX_DRAIN_TIMEOUT_SECONDS
        ):
            logger.warning(
                "drain_timeout_seconds out of valid range, clamping",
                extra={
                    "original_value": drain_timeout_value,
                    "min_value": MIN_DRAIN_TIMEOUT_SECONDS,
                    "max_value": MAX_DRAIN_TIMEOUT_SECONDS,
                    "clamped_value": max(
                        MIN_DRAIN_TIMEOUT_SECONDS,
                        min(drain_timeout_value, MAX_DRAIN_TIMEOUT_SECONDS),
                    ),
                },
            )
            drain_timeout_value = max(
                MIN_DRAIN_TIMEOUT_SECONDS,
                min(drain_timeout_value, MAX_DRAIN_TIMEOUT_SECONDS),
            )

        self._drain_timeout_seconds: float = drain_timeout_value

        # Handler executor for lifecycle operations (shutdown, health check)
        self._lifecycle_executor = ProtocolLifecycleExecutor(
            health_check_timeout_seconds=self._health_check_timeout_seconds
        )

        # Store full config for handler initialization
        self._config: dict[str, object] | None = config

        # Runtime state
        self._is_running: bool = False

        # Subscription handle (callable to unsubscribe)
        self._subscription: Callable[[], Awaitable[None]] | None = None

        # Handler registry (handler_type -> handler instance)
        # This will be populated from the singleton registry during start()
        self._handlers: dict[str, ProtocolContainerAware] = {}

        # Track failed handler instantiations (handler_type -> error message)
        # Used by health_check() to report degraded state
        self._failed_handlers: dict[str, str] = {}

        # Handler descriptors (handler_type -> descriptor with contract_config)
        # Stored during registration for use during handler initialization
        # Enables contract config to be passed to handlers via initialize()
        self._handler_descriptors: dict[str, ModelHandlerDescriptor] = {}

        # Pending message tracking for graceful shutdown (OMN-756)
        # Tracks count of in-flight messages currently being processed
        self._pending_message_count: int = 0
        self._pending_lock: asyncio.Lock = asyncio.Lock()

        # Drain state tracking for graceful shutdown (OMN-756)
        # True when stop() has been called and we're waiting for messages to drain
        self._is_draining: bool = False

        # Idempotency guard for duplicate message detection (OMN-945)
        # None = disabled, otherwise points to configured store
        self._idempotency_store: ProtocolIdempotencyStore | None = None
        self._idempotency_config: ModelIdempotencyGuardConfig | None = None

        logger.debug(
            "RuntimeHostProcess initialized",
            extra={
                "input_topic": self._input_topic,
                "output_topic": self._output_topic,
                "group_id": self._group_id,
                "health_check_timeout_seconds": self._health_check_timeout_seconds,
                "drain_timeout_seconds": self._drain_timeout_seconds,
                "has_container": self._container is not None,
                "has_handler_registry": self._handler_registry is not None,
                "has_contract_paths": len(self._contract_paths) > 0,
                "contract_path_count": len(self._contract_paths),
            },
        )

    @property
    def container(self) -> ModelONEXContainer | None:
        """Return the optional ONEX dependency injection container.

        Returns:
            The container if provided during initialization, None otherwise.
        """
        return self._container

    @property
    def event_bus(self) -> EventBusInmemory | EventBusKafka:
        """Return the owned event bus instance.

        Returns:
            The event bus instance managed by this process.
        """
        return self._event_bus

    @property
    def is_running(self) -> bool:
        """Return True if runtime is started.

        Returns:
            Boolean indicating whether the process is running.
        """
        return self._is_running

    @property
    def input_topic(self) -> str:
        """Return the input topic for envelope subscription.

        Returns:
            The topic name to subscribe to for incoming envelopes.
        """
        return self._input_topic

    @property
    def output_topic(self) -> str:
        """Return the output topic for response publishing.

        Returns:
            The topic name to publish responses to.
        """
        return self._output_topic

    @property
    def group_id(self) -> str:
        """Return the consumer group identifier.

        Returns:
            The consumer group ID for this process.
        """
        return self._group_id

    @property
    def is_draining(self) -> bool:
        """Return True if the process is draining pending messages during shutdown.

        This property indicates whether the runtime host is in the graceful shutdown
        drain period - the phase where stop() has been called, new messages are no
        longer being accepted, and the process is waiting for in-flight messages to
        complete before shutting down handlers and the event bus.

        Drain State Transitions:
            - False: Normal operation (accepting and processing messages)
            - True: Drain period active (stop() called, waiting for pending messages)
            - False: After drain completes and shutdown finishes

        Use Cases:
            - Health check reporting (indicate service is shutting down)
            - Load balancer integration (remove from rotation during drain)
            - Monitoring dashboards (show lifecycle state)
            - Debugging shutdown behavior

        Returns:
            True if currently in drain period during graceful shutdown, False otherwise.
        """
        return self._is_draining

    @property
    def pending_message_count(self) -> int:
        """Return the current count of in-flight messages being processed.

        This property provides visibility into how many messages are currently
        being processed by the runtime host. Used for graceful shutdown to
        determine when it's safe to complete the shutdown process.

        Atomicity Guarantees:
            This property returns the raw counter value WITHOUT acquiring the
            async lock (_pending_lock). This is safe because:

            1. Single int read is atomic under CPython's GIL - reading a single
               integer value cannot be interrupted mid-operation
            2. The value is only used for observability/monitoring purposes
               where exact precision is not required
            3. The slight possibility of reading a stale value during concurrent
               increment/decrement is acceptable for monitoring use cases

        Thread Safety Considerations:
            While the read itself is atomic, the value may be approximate if
            read occurs during concurrent message processing:
            - Another coroutine may be in the middle of incrementing/decrementing
            - The value represents a point-in-time snapshot, not a synchronized view
            - For observability, this approximation is acceptable and avoids
              lock contention that would impact performance

        Use Cases (appropriate for this property):
            - Logging current message count for debugging
            - Metrics/observability dashboards
            - Approximate health status reporting
            - Monitoring drain progress during shutdown

        When to use shutdown_ready() instead:
            For shutdown decisions requiring precise count, use the async
            shutdown_ready() method which acquires the lock to ensure no
            race condition with in-flight message processing. The stop()
            method uses shutdown_ready() internally for this reason.

        Returns:
            Current count of messages being processed. May be approximate
            if reads occur during concurrent increment/decrement operations.
        """
        return self._pending_message_count

    async def shutdown_ready(self) -> bool:
        """Check if process is ready for shutdown (no pending messages).

        This method acquires the pending message lock to ensure an accurate
        count of in-flight messages. Use this method during graceful shutdown
        to determine when all pending messages have been processed.

        Returns:
            True if no messages are currently being processed, False otherwise.
        """
        async with self._pending_lock:
            return self._pending_message_count == 0

    async def start(self) -> None:
        """Start the runtime host.

        Performs the following steps:
        1. Validate architecture compliance (if rules configured) - OMN-1138
        2. Start event bus (if not already started)
        3. Discover/wire handlers:
           - If contract_paths provided: Auto-discover handlers from contracts (OMN-1133)
           - Otherwise: Wire default handlers via wiring module
        4. Populate self._handlers from singleton registry (instantiate and initialize)
        5. Subscribe to input topic

        Architecture Validation (OMN-1138):
            If architecture_rules were provided at init, validation runs FIRST
            before any other startup logic. This ensures:
            - Violations are caught before resources are allocated
            - Fast feedback for CI/CD pipelines
            - Clean startup/failure without partial state

            ERROR severity violations block startup by raising
            ArchitectureViolationError. WARNING/INFO violations are logged
            but don't block startup.

        Contract-Based Handler Discovery (OMN-1133):
            If contract_paths were provided at init, the runtime will auto-discover
            handlers from these paths instead of using wire_default_handlers().

            Discovery errors are logged but do not block startup, enabling
            graceful degradation where some handlers can be registered even
            if others fail to load.

        This method is idempotent - calling start() on an already started
        process is safe and has no effect.

        Raises:
            ArchitectureViolationError: If architecture validation fails with
                blocking violations (ERROR severity).
        """
        if self._is_running:
            logger.debug("RuntimeHostProcess already started, skipping")
            return

        logger.info(
            "Starting RuntimeHostProcess",
            extra={
                "input_topic": self._input_topic,
                "output_topic": self._output_topic,
                "group_id": self._group_id,
                "has_contract_paths": len(self._contract_paths) > 0,
            },
        )

        # Step 1: Validate architecture compliance FIRST (OMN-1138)
        # This runs before event bus starts or handlers are wired to ensure
        # clean failure without partial state if validation fails
        await self._validate_architecture()

        # Step 2: Start event bus
        await self._event_bus.start()

        # Step 3: Discover/wire handlers (OMN-1133)
        # If contract_paths provided, use ContractHandlerDiscovery to auto-discover
        # handlers from contract files. Otherwise, fall back to wire_default_handlers().
        await self._discover_or_wire_handlers()

        # Step 4: Populate self._handlers from singleton registry
        # The wiring/discovery step registers handler classes, so we need to:
        # - Get each registered handler class from the singleton registry
        # - Instantiate the handler class
        # - Call initialize() on each handler instance with config
        # - Store the handler instance in self._handlers for routing
        await self._populate_handlers_from_registry()

        # Step 4.1: FAIL-FAST validation - runtime MUST have at least one handler
        # A runtime with no handlers cannot process any events and is misconfigured.
        # This catches configuration issues early rather than silently starting a
        # runtime that cannot do anything useful.
        if not self._handlers:
            correlation_id = uuid4()
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="validate_handlers",
                target_name="runtime_host_process",
                correlation_id=correlation_id,
            )

            # Build informative error message with context about what was attempted
            contract_paths_info = (
                f"  * contract_paths provided: {[str(p) for p in self._contract_paths]}\n"
                if self._contract_paths
                else "  * contract_paths: NOT PROVIDED (using ONEX_CONTRACTS_DIR env var)\n"
            )

            # Get registry count for additional context
            handler_registry = await self._get_handler_registry()
            registry_protocol_count = len(handler_registry.list_protocols())

            # Build additional diagnostic info
            failed_handlers_detail = ""
            if self._failed_handlers:
                failed_handlers_detail = "FAILED HANDLERS (check these first):\n"
                for handler_type, error_msg in self._failed_handlers.items():
                    failed_handlers_detail += f"  * {handler_type}: {error_msg}\n"
                failed_handlers_detail += "\n"

            raise ProtocolConfigurationError(
                "No handlers registered. The runtime cannot start without at least one handler.\n\n"
                "CURRENT CONFIGURATION:\n"
                f"{contract_paths_info}"
                f"  * Registry protocol count: {registry_protocol_count}\n"
                f"  * Failed handlers: {len(self._failed_handlers)}\n"
                f"  * Correlation ID: {correlation_id}\n\n"
                f"{failed_handlers_detail}"
                "TROUBLESHOOTING STEPS:\n"
                "  1. Verify ONEX_CONTRACTS_DIR points to a valid contracts directory:\n"
                "     - Run: echo $ONEX_CONTRACTS_DIR && ls -la $ONEX_CONTRACTS_DIR\n"
                "     - Expected: Directory containing handler_contract.yaml or contract.yaml files\n\n"
                "  2. Check for handler contract files:\n"
                "     - Run: find $ONEX_CONTRACTS_DIR -name 'handler_contract.yaml' -o -name 'contract.yaml'\n"
                "     - If empty: No contracts found - create handler contracts or set correct path\n\n"
                "  3. Verify handler contracts have required fields:\n"
                "     - Required: handler_name, handler_class, handler_type\n"
                "     - Example:\n"
                "         handler_name: my_handler\n"
                "         handler_class: mymodule.handlers.MyHandler\n"
                "         handler_type: http\n\n"
                "  4. Verify handler modules are importable:\n"
                "     - Run: python -c 'from mymodule.handlers import MyHandler; print(MyHandler)'\n"
                "     - Check PYTHONPATH includes your handler module paths\n\n"
                "  5. Check application logs for loader errors:\n"
                "     - Look for: MODULE_NOT_FOUND (HANDLER_LOADER_010)\n"
                "     - Look for: CLASS_NOT_FOUND (HANDLER_LOADER_011)\n"
                "     - Look for: IMPORT_ERROR (HANDLER_LOADER_012)\n"
                "     - Look for: AMBIGUOUS_CONTRACT (HANDLER_LOADER_040)\n\n"
                "  6. If using wire_handlers() manually:\n"
                "     - Ensure wire_handlers() is called before RuntimeHostProcess.start()\n"
                "     - Check that handlers implement ProtocolContainerAware interface\n\n"
                "  7. Docker/container environment:\n"
                "     - Verify volume mounts include handler contract directories\n"
                "     - Check ONEX_CONTRACTS_DIR is set in docker-compose.yml/Dockerfile\n"
                "     - Run: docker exec <container> ls $ONEX_CONTRACTS_DIR\n\n"
                "For verbose handler discovery logging, set LOG_LEVEL=DEBUG.",
                context=context,
                registered_handler_count=0,
                failed_handler_count=len(self._failed_handlers),
                failed_handlers=list(self._failed_handlers.keys()),
                contract_paths=[str(p) for p in self._contract_paths],
                registry_protocol_count=registry_protocol_count,
            )

        # Step 4.5: Initialize idempotency store if configured (OMN-945)
        await self._initialize_idempotency_store()

        # Step 5: Subscribe to input topic
        self._subscription = await self._event_bus.subscribe(
            topic=self._input_topic,
            group_id=self._group_id,
            on_message=self._on_message,
        )

        self._is_running = True

        logger.info(
            "RuntimeHostProcess started successfully",
            extra={
                "input_topic": self._input_topic,
                "output_topic": self._output_topic,
                "group_id": self._group_id,
                "registered_handlers": list(self._handlers.keys()),
            },
        )

    async def stop(self) -> None:
        """Stop the runtime host with graceful drain period.

        Performs the following steps:
        1. Unsubscribe from topics (stop receiving new messages)
        2. Wait for in-flight messages to drain (up to drain_timeout_seconds)
        3. Shutdown all registered handlers by priority (release resources)
        4. Close event bus

        This method is idempotent - calling stop() on an already stopped
        process is safe and has no effect.

        Drain Period:
            After unsubscribing from topics, the process waits for in-flight
            messages to complete processing. The drain period is controlled by
            the drain_timeout_seconds configuration parameter (default: 30.0
            seconds, valid range: 1-300).

            During the drain period:
            - No new messages are received (unsubscribed from topics)
            - Messages currently being processed are allowed to complete
            - shutdown_ready() is polled every 100ms to check completion
            - If timeout is exceeded, shutdown proceeds with a warning

        Handler Shutdown Order:
            Handlers are shutdown in priority order, with higher priority handlers
            shutting down first. Within the same priority level, handlers are
            shutdown in parallel for performance.

            Priority is determined by the handler's shutdown_priority() method:
            - Higher values = shutdown first
            - Handlers without shutdown_priority() get default priority of 0

            Recommended Priority Scheme:
            - 100: Consumers (stop receiving before stopping producers)
            - 80: Active connections (close before closing pools)
            - 50: Producers (stop producing before closing pools)
            - 40: Connection pools (close last)
            - 0: Default for handlers without explicit priority

            This ensures dependency-based ordering:
            - Consumers shutdown before producers
            - Connections shutdown before connection pools
            - Downstream resources shutdown before upstream resources
        """
        if not self._is_running:
            logger.debug("RuntimeHostProcess already stopped, skipping")
            return

        logger.info("Stopping RuntimeHostProcess")

        # Step 1: Unsubscribe from topics (stop receiving new messages)
        if self._subscription is not None:
            await self._subscription()
            self._subscription = None

        # Step 1.5: Wait for in-flight messages to drain (OMN-756)
        # This allows messages currently being processed to complete
        loop = asyncio.get_running_loop()
        drain_start = loop.time()
        drain_deadline = drain_start + self._drain_timeout_seconds
        last_progress_log = drain_start

        # Mark drain state for health check visibility (OMN-756)
        self._is_draining = True

        # Log drain start for observability
        logger.info(
            "Starting drain period",
            extra={
                "pending_messages": self._pending_message_count,
                "drain_timeout_seconds": self._drain_timeout_seconds,
            },
        )

        while not await self.shutdown_ready():
            remaining = drain_deadline - loop.time()
            if remaining <= 0:
                logger.warning(
                    "Drain timeout exceeded, forcing shutdown",
                    extra={
                        "pending_messages": self._pending_message_count,
                        "drain_timeout_seconds": self._drain_timeout_seconds,
                        "metric.drain_timeout_exceeded": True,
                        "metric.pending_at_timeout": self._pending_message_count,
                    },
                )
                break

            # Wait a short interval before checking again
            await asyncio.sleep(min(0.1, remaining))

            # Log progress every 5 seconds during long drains for observability
            elapsed = loop.time() - drain_start
            if elapsed - (last_progress_log - drain_start) >= 5.0:
                logger.info(
                    "Drain in progress",
                    extra={
                        "pending_messages": self._pending_message_count,
                        "elapsed_seconds": round(elapsed, 2),
                        "remaining_seconds": round(remaining, 2),
                    },
                )
                last_progress_log = loop.time()

        # Clear drain state after drain period completes
        self._is_draining = False

        logger.info(
            "Drain period completed",
            extra={
                "drain_duration_seconds": loop.time() - drain_start,
                "pending_messages": self._pending_message_count,
                "metric.drain_duration": loop.time() - drain_start,
                "metric.forced_shutdown": self._pending_message_count > 0,
            },
        )

        # Step 2: Shutdown all handlers by priority (release resources like DB/Kafka connections)
        # Delegates to ProtocolLifecycleExecutor which handles:
        # - Grouping handlers by priority (higher priority first)
        # - Parallel shutdown within priority groups for performance
        if self._handlers:
            shutdown_result = (
                await self._lifecycle_executor.shutdown_handlers_by_priority(
                    self._handlers
                )
            )

            # Log summary (ProtocolLifecycleExecutor already logs detailed info)
            logger.info(
                "Handler shutdown completed",
                extra={
                    "succeeded_handlers": shutdown_result.succeeded_handlers,
                    "failed_handlers": [
                        f.handler_type for f in shutdown_result.failed_handlers
                    ],
                    "total_handlers": shutdown_result.total_count,
                    "success_count": shutdown_result.success_count,
                    "failure_count": shutdown_result.failure_count,
                },
            )

        # Step 2.5: Cleanup idempotency store if initialized (OMN-945)
        await self._cleanup_idempotency_store()

        # Step 3: Close event bus
        await self._event_bus.close()

        self._is_running = False

        logger.info("RuntimeHostProcess stopped successfully")

    def _load_handler_source_config(self) -> ModelHandlerSourceConfig:
        """Load handler source configuration from runtime config.

        Loads the handler source mode configuration that controls how handlers
        are discovered (BOOTSTRAP, CONTRACT, or HYBRID mode).

        Config Keys:
            handler_source_mode: "bootstrap" | "contract" | "hybrid" (default: "hybrid")
            bootstrap_expires_at: ISO-8601 datetime string (optional, UTC required)

        Returns:
            ModelHandlerSourceConfig with validated settings.

        Note:
            If no configuration is provided, defaults to HYBRID mode with no
            bootstrap expiry (bootstrap handlers always available as fallback).

        .. versionadded:: 0.7.0
            Part of OMN-1095 handler source mode integration.
        """
        from datetime import datetime

        from omnibase_infra.models.handlers import ModelHandlerSourceConfig

        config = self._config or {}
        handler_source_config = config.get("handler_source", {})

        if isinstance(handler_source_config, dict):
            mode_str = handler_source_config.get("mode", "hybrid")
            expires_at_str = handler_source_config.get("bootstrap_expires_at")

            # Parse mode
            try:
                mode = EnumHandlerSourceMode(mode_str)
            except ValueError:
                logger.warning(
                    "Invalid handler_source_mode, defaulting to HYBRID",
                    extra={"invalid_value": mode_str},
                )
                mode = EnumHandlerSourceMode.HYBRID

            # Parse expiry datetime
            expires_at = None
            if expires_at_str:
                try:
                    expires_at = datetime.fromisoformat(str(expires_at_str))
                except ValueError:
                    logger.warning(
                        "Invalid bootstrap_expires_at format, ignoring",
                        extra={"invalid_value": expires_at_str},
                    )

            return ModelHandlerSourceConfig(
                handler_source_mode=mode,
                bootstrap_expires_at=expires_at,
            )

        # Default: HYBRID mode with no expiry
        return ModelHandlerSourceConfig(
            handler_source_mode=EnumHandlerSourceMode.HYBRID
        )

    async def _resolve_handler_descriptors(self) -> list[ModelHandlerDescriptor]:
        """Resolve handler descriptors using the configured source mode.

        Uses HandlerSourceResolver to discover handlers based on the configured
        mode (BOOTSTRAP, CONTRACT, or HYBRID). This replaces the previous
        sequential discovery logic with a unified, mode-driven approach.

        Resolution Modes:
            - BOOTSTRAP: Only hardcoded bootstrap handlers
            - CONTRACT: Only filesystem contract-discovered handlers
            - HYBRID: Contract handlers win per-identity, bootstrap as fallback

        Returns:
            List of resolved handler descriptors.

        Raises:
            RuntimeHostError: If validation errors occur and fail-fast is enabled.

        .. versionadded:: 0.7.0
            Part of OMN-1095 handler source mode integration.
        """
        from pathlib import Path

        from omnibase_core.models.primitives import ModelSemVer
        from omnibase_infra.models.handlers import (
            ModelContractDiscoveryResult,
            ModelHandlerDescriptor,
        )
        from omnibase_infra.runtime.handler_bootstrap_source import (
            HandlerBootstrapSource,
        )
        from omnibase_infra.runtime.handler_plugin_loader import HandlerPluginLoader
        from omnibase_infra.runtime.handler_source_resolver import HandlerSourceResolver
        from omnibase_infra.runtime.protocol_contract_source import (
            ProtocolContractSource,
        )

        # Local adapter class that wraps HandlerPluginLoader to implement
        # ProtocolContractSource. This uses the simpler contract schema
        # (handler_name, handler_class, handler_type) from
        # omnibase_infra.models.runtime.model_handler_contract.ModelHandlerContract
        # which is compatible with existing test contracts.
        class PluginLoaderContractSource(ProtocolContractSource):
            """Adapter that uses HandlerPluginLoader for contract discovery.

            This adapter implements ProtocolContractSource using HandlerPluginLoader,
            which uses the simpler contract schema (handler_name, handler_class,
            handler_type, capability_tags) rather than the full ONEX contract schema.
            """

            def __init__(self, contract_paths: list[Path]) -> None:
                self._contract_paths = contract_paths
                self._plugin_loader = HandlerPluginLoader()

            @property
            def source_type(self) -> str:
                return "CONTRACT"

            async def discover_handlers(self) -> ModelContractDiscoveryResult:
                """Discover handlers using HandlerPluginLoader."""
                from omnibase_infra.models.errors import ModelHandlerValidationError

                # Ensure model is rebuilt for validation_errors field
                ModelContractDiscoveryResult.model_rebuild()

                descriptors: list[ModelHandlerDescriptor] = []
                validation_errors: list[ModelHandlerValidationError] = []

                for path in self._contract_paths:
                    path_obj = Path(path) if isinstance(path, str) else path
                    if not path_obj.exists():
                        logger.warning(
                            "Contract path does not exist, skipping: %s",
                            path_obj,
                        )
                        continue

                    try:
                        # Use plugin loader to discover handlers with simpler schema
                        loaded_handlers = self._plugin_loader.load_from_directory(
                            directory=path_obj,
                        )

                        # Convert ModelLoadedHandler to ModelHandlerDescriptor
                        for loaded in loaded_handlers:
                            descriptor = ModelHandlerDescriptor(
                                # Use protocol_type as handler_id for compatibility
                                # with HandlerSourceResolver's per-identity resolution
                                handler_id=loaded.protocol_type,
                                name=loaded.handler_name,
                                version=ModelSemVer(major=1, minor=0, patch=0),
                                handler_kind="effect",  # Default for infra handlers
                                input_model="omnibase_infra.models.types.JsonDict",
                                output_model="omnibase_core.models.dispatch.ModelHandlerOutput",
                                description=f"Handler: {loaded.handler_name}",
                                handler_class=loaded.handler_class,
                                contract_path=str(loaded.contract_path),
                            )
                            descriptors.append(descriptor)

                    except Exception as e:
                        logger.warning(
                            "Failed to load handlers from path %s: %s",
                            path_obj,
                            e,
                        )
                        # Continue with other paths (graceful degradation)

                return ModelContractDiscoveryResult(
                    descriptors=descriptors,
                    validation_errors=validation_errors,
                )

        source_config = self._load_handler_source_config()

        logger.info(
            "Resolving handlers with source mode",
            extra={
                "mode": source_config.handler_source_mode.value,
                "effective_mode": source_config.effective_mode.value,
                "bootstrap_expires_at": str(source_config.bootstrap_expires_at)
                if source_config.bootstrap_expires_at
                else None,
                "is_bootstrap_expired": source_config.is_bootstrap_expired,
            },
        )

        # Create bootstrap source
        bootstrap_source = HandlerBootstrapSource()

        # Contract source needs paths - use configured paths or default
        # If no contract_paths provided, reuse bootstrap_source as placeholder
        if self._contract_paths:
            # Use PluginLoaderContractSource which uses the simpler contract schema
            # compatible with test contracts (handler_name, handler_class, handler_type)
            contract_source: ProtocolContractSource = PluginLoaderContractSource(
                contract_paths=[Path(p) for p in self._contract_paths],
            )
        else:
            # No contract paths - reuse bootstrap_source as placeholder for contract source
            # The resolver will handle this based on the mode
            # IMPORTANT: Reuse bootstrap_source instance, don't create a new one
            if source_config.effective_mode == EnumHandlerSourceMode.CONTRACT:
                logger.warning(
                    "CONTRACT mode enabled but no contract_paths provided; "
                    "bootstrap handlers will be loaded as 'contract source' fallback. "
                    "This may produce unexpected results. Consider providing "
                    "contract_paths or switching to HYBRID mode.",
                    extra={
                        "mode": source_config.handler_source_mode.value,
                        "effective_mode": source_config.effective_mode.value,
                    },
                )
            contract_source = bootstrap_source  # Reuse, don't create new instance

        # Create resolver with the effective mode (handles expiry enforcement)
        resolver = HandlerSourceResolver(
            bootstrap_source=bootstrap_source,
            contract_source=contract_source,
            mode=source_config.effective_mode,
        )

        # Resolve handlers
        result = await resolver.resolve_handlers()

        # Log resolution results
        logger.info(
            "Handler resolution completed",
            extra={
                "descriptor_count": len(result.descriptors),
                "validation_error_count": len(result.validation_errors),
                "mode": source_config.effective_mode.value,
            },
        )

        # Log validation errors but continue with valid descriptors (graceful degradation)
        # This allows the runtime to start with bootstrap handlers even if some contracts fail
        if result.validation_errors:
            error_summary = "; ".join(
                f"{e.handler_identity.handler_id or 'unknown'}: {e.message}"
                for e in result.validation_errors[:5]  # Show first 5
            )
            if len(result.validation_errors) > 5:
                error_summary += f" ... and {len(result.validation_errors) - 5} more"

            logger.warning(
                "Handler resolution completed with validation errors (continuing with valid handlers)",
                extra={
                    "error_count": len(result.validation_errors),
                    "valid_descriptor_count": len(result.descriptors),
                    "error_summary": error_summary,
                },
            )

        return list(result.descriptors)

    async def _discover_or_wire_handlers(self) -> None:
        """Discover and register handlers for the runtime.

        This method implements the handler discovery/wiring step (Step 3) of the
        start() sequence. It uses HandlerSourceResolver to discover handlers
        based on the configured source mode.

        Handler Source Modes (OMN-1095):
            - BOOTSTRAP: Only hardcoded bootstrap handlers (fast, no filesystem I/O)
            - CONTRACT: Only filesystem contract-discovered handlers
            - HYBRID: Contract handlers win per-identity, bootstrap as fallback

        The mode is configured via runtime config:
            handler_source:
                mode: "hybrid"  # bootstrap|contract|hybrid
                bootstrap_expires_at: "2026-02-01T00:00:00Z"  # Optional, UTC

        The discovery/wiring step registers handler CLASSES with the handler registry.
        The subsequent _populate_handlers_from_registry() step instantiates and
        initializes these handler classes.

        .. versionchanged:: 0.7.0
            Replaced sequential bootstrap+contract discovery with unified
            HandlerSourceResolver-based resolution (OMN-1095).
        """
        # Resolve handlers using configured source mode
        descriptors = await self._resolve_handler_descriptors()

        # Get handler registry for registration
        handler_registry = await self._get_handler_registry()

        registered_count = 0
        error_count = 0

        for descriptor in descriptors:
            try:
                # Extract protocol type from handler_id
                # Bootstrap handler IDs are prefixed (e.g., "bootstrap.consul" -> "consul")
                # Contract handlers use their handler_name directly as protocol_type
                protocol_type = descriptor.handler_id
                if protocol_type.startswith("bootstrap."):
                    # Strip the "bootstrap." prefix for bootstrap handlers
                    protocol_type = protocol_type.split(".", 1)[-1]
                # For contract handlers, preserve the full handler_name as protocol_type

                # Import the handler class from fully qualified path
                handler_class_path = descriptor.handler_class
                if handler_class_path is None:
                    logger.warning(
                        "Handler descriptor missing handler_class, skipping",
                        extra={
                            "handler_id": descriptor.handler_id,
                            "handler_name": descriptor.name,
                        },
                    )
                    error_count += 1
                    continue

                # Import class using rsplit pattern
                if "." not in handler_class_path:
                    logger.error(
                        "Invalid handler class path (must be fully qualified): %s",
                        handler_class_path,
                        extra={"handler_id": descriptor.handler_id},
                    )
                    error_count += 1
                    continue

                module_path, class_name = handler_class_path.rsplit(".", 1)
                module = importlib.import_module(module_path)
                handler_cls = getattr(module, class_name)

                # Verify handler_cls is actually a class before registration
                if not isinstance(handler_cls, type):
                    logger.error(
                        "Handler class path does not resolve to a class type",
                        extra={
                            "handler_id": descriptor.handler_id,
                            "handler_class_path": handler_class_path,
                            "resolved_type": type(handler_cls).__name__,
                        },
                    )
                    error_count += 1
                    continue

                # Register with handler registry
                handler_registry.register(protocol_type, handler_cls)

                # Store descriptor for later use during initialization
                self._handler_descriptors[protocol_type] = descriptor

                registered_count += 1
                logger.debug(
                    "Registered handler from descriptor",
                    extra={
                        "handler_id": descriptor.handler_id,
                        "protocol_type": protocol_type,
                        "handler_class": handler_class_path,
                    },
                )

            except Exception:
                logger.exception(
                    "Failed to register handler from descriptor",
                    extra={
                        "handler_id": descriptor.handler_id,
                    },
                )
                error_count += 1

        logger.info(
            "Handler discovery completed",
            extra={
                "registered_count": registered_count,
                "error_count": error_count,
                "total_descriptors": len(descriptors),
            },
        )

    async def _populate_handlers_from_registry(self) -> None:
        """Populate self._handlers from handler registry (container or singleton).

        This method bridges the gap between the wiring module (which registers
        handler CLASSES to the registry) and the RuntimeHostProcess
        (which needs handler INSTANCES in self._handlers for routing).

        Registry Resolution:
            - If handler_registry provided: Uses pre-resolved registry
            - If no handler_registry: Falls back to singleton get_handler_registry()

        For each registered handler type in the registry:
        1. Skip if handler type is already registered (e.g., by tests)
        2. Get the handler class from the registry
        3. Instantiate the handler class
        4. Call initialize() on the handler instance with self._config
        5. Store the handler instance in self._handlers

        This ensures that after start() is called, self._handlers contains
        fully initialized handler instances ready for envelope routing.

        Note: Handlers already in self._handlers (e.g., injected by tests via
        register_handler() or patch.object()) are preserved and not overwritten.
        """
        # Get handler registry (pre-resolved, container, or singleton)
        handler_registry = await self._get_handler_registry()
        registered_types = handler_registry.list_protocols()

        logger.debug(
            "Populating handlers from registry",
            extra={
                "registered_types": registered_types,
                "existing_handlers": list(self._handlers.keys()),
            },
        )

        # Get or create container once for all handlers to share
        # This ensures all handlers have access to the same DI container
        container = self._get_or_create_container()

        for handler_type in registered_types:
            # Skip if handler is already registered (e.g., by tests or explicit registration)
            if handler_type in self._handlers:
                logger.debug(
                    "Handler already registered, skipping",
                    extra={
                        "handler_type": handler_type,
                        "existing_handler_class": type(
                            self._handlers[handler_type]
                        ).__name__,
                    },
                )
                continue

            try:
                # Get handler class from singleton registry
                handler_cls: type[ProtocolContainerAware] = handler_registry.get(
                    handler_type
                )

                # Instantiate the handler with container for dependency injection
                # ProtocolContainerAware defines __init__(container: ModelONEXContainer)
                handler_instance: ProtocolContainerAware = handler_cls(
                    container=container
                )

                # Call initialize() if the handler has this method
                # Handlers may require async initialization with config
                if hasattr(handler_instance, "initialize"):
                    # Build effective config: contract config as base, runtime overrides on top
                    # This enables contracts to provide handler-specific defaults while
                    # allowing runtime/deploy-time customization without touching contracts
                    effective_config: dict[str, object] = {}
                    config_source = "runtime_only"

                    # Layer 1: Contract config as baseline (if descriptor exists with config)
                    descriptor = self._handler_descriptors.get(handler_type)
                    if descriptor and descriptor.contract_config:
                        effective_config.update(descriptor.contract_config)
                        config_source = "contract_only"

                    # Layer 2: Runtime config overrides
                    # Runtime config takes precedence, enabling deploy-time customization
                    if self._config:
                        effective_config.update(self._config)
                        if descriptor and descriptor.contract_config:
                            config_source = "contract+runtime_override"

                    # Pass empty dict if no config, not None
                    # Handlers expect dict interface (e.g., config.get("key"))
                    await handler_instance.initialize(effective_config)

                    logger.debug(
                        "Handler initialized with effective config",
                        extra={
                            "handler_type": handler_type,
                            "config_source": config_source,
                            "effective_config_keys": list(effective_config.keys()),
                            "has_contract_config": bool(
                                descriptor and descriptor.contract_config
                            ),
                            "has_runtime_config": bool(self._config),
                        },
                    )

                # Store the handler instance for routing
                self._handlers[handler_type] = handler_instance

                logger.debug(
                    "Handler instantiated and initialized",
                    extra={
                        "handler_type": handler_type,
                        "handler_class": handler_cls.__name__,
                    },
                )

            except Exception as e:
                # Track the failure for health_check() reporting
                self._failed_handlers[handler_type] = str(e)

                # Log error but continue with other handlers
                # This allows partial handler availability
                correlation_id = uuid4()
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="populate_handlers",
                    target_name=handler_type,
                    correlation_id=correlation_id,
                )
                infra_error = RuntimeHostError(
                    f"Failed to instantiate handler for type {handler_type}: {e}",
                    context=context,
                )
                infra_error.__cause__ = e

                logger.warning(
                    "Failed to instantiate handler, skipping",
                    extra={
                        "handler_type": handler_type,
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )

        logger.info(
            "Handlers populated from registry",
            extra={
                "populated_handlers": list(self._handlers.keys()),
                "total_count": len(self._handlers),
            },
        )

    async def _get_handler_registry(self) -> RegistryProtocolBinding:
        """Get handler registry (pre-resolved, container, or singleton).

        Resolution order:
            1. If handler_registry was provided to __init__, uses it (cached)
            2. If container was provided and has RegistryProtocolBinding, resolves from container
            3. Falls back to singleton via get_handler_registry()

        Caching Behavior:
            The resolved registry is cached after the first successful resolution.
            Subsequent calls return the cached instance without re-resolving from
            the container or re-fetching the singleton. This ensures consistent
            registry usage throughout the runtime's lifecycle and avoids redundant
            resolution operations.

        Returns:
            RegistryProtocolBinding instance.
        """
        if self._handler_registry is not None:
            # Use pre-resolved registry from constructor
            return self._handler_registry

        # Try to resolve from container if provided
        if self._container is not None and self._container.service_registry is not None:
            try:
                resolved_registry: RegistryProtocolBinding = (
                    await self._container.service_registry.resolve_service(
                        RegistryProtocolBinding
                    )
                )
                # Cache the resolved registry for subsequent calls
                self._handler_registry = resolved_registry
                logger.debug(
                    "Handler registry resolved from container",
                    extra={"registry_type": type(resolved_registry).__name__},
                )
                return resolved_registry
            except (
                RuntimeError,
                ValueError,
                KeyError,
                AttributeError,
                LookupError,
            ) as e:
                # Container resolution failed, fall through to singleton
                logger.debug(
                    "Container registry resolution failed, falling back to singleton",
                    extra={"error": str(e)},
                )

        # Graceful degradation: fall back to singleton pattern when container unavailable
        from omnibase_infra.runtime.handler_registry import get_handler_registry

        singleton_registry = get_handler_registry()
        # Cache for consistency with container resolution path
        self._handler_registry = singleton_registry
        logger.debug(
            "Handler registry resolved from singleton",
            extra={"registry_type": type(singleton_registry).__name__},
        )
        return singleton_registry

    async def _on_message(self, message: ModelEventMessage) -> None:
        """Handle incoming message from event bus subscription.

        This is the callback invoked by the event bus when a message arrives
        on the input topic. It deserializes the envelope and routes it.

        The method tracks pending messages for graceful shutdown support (OMN-756).
        The pending message count is incremented at the start of processing and
        decremented when processing completes (success or failure).

        Args:
            message: The event message containing the envelope payload.
        """
        # Increment pending message count (OMN-756: graceful shutdown tracking)
        async with self._pending_lock:
            self._pending_message_count += 1

        try:
            # Deserialize envelope from message value
            envelope = json.loads(message.value.decode("utf-8"))
            await self._handle_envelope(envelope)
        except json.JSONDecodeError as e:
            # Create infrastructure error context for tracing
            correlation_id = uuid4()
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="decode_envelope",
                target_name=message.topic,
                correlation_id=correlation_id,
            )
            # Chain the error with infrastructure context
            infra_error = RuntimeHostError(
                f"Failed to decode JSON envelope from message: {e}",
                context=context,
            )
            infra_error.__cause__ = e  # Proper error chaining

            logger.exception(
                "Failed to decode envelope from message",
                extra={
                    "error": str(e),
                    "topic": message.topic,
                    "offset": message.offset,
                    "correlation_id": str(correlation_id),
                },
            )
            # Publish error response for malformed messages
            error_response = self._create_error_response(
                error=f"Invalid JSON in message: {e}",
                correlation_id=correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)
        finally:
            # Decrement pending message count (OMN-756: graceful shutdown tracking)
            async with self._pending_lock:
                self._pending_message_count -= 1

    async def _handle_envelope(self, envelope: dict[str, object]) -> None:
        """Route envelope to appropriate handler.

        Validates envelope before dispatch and routes it to the appropriate
        registered handler. Publishes the response to the output topic.

        Validation (performed before dispatch):
        1. Operation presence and type validation
        2. Handler prefix validation against registry
        3. Payload requirement validation for specific operations
        4. Correlation ID normalization to UUID

        Args:
            envelope: Dict with 'operation', 'payload', optional 'correlation_id',
                and 'handler_type'.
        """
        # Pre-validation: Get correlation_id for error responses if validation fails
        # This handles the case where validation itself throws before normalizing
        pre_validation_correlation_id = normalize_correlation_id(
            envelope.get("correlation_id")
        )

        # Step 1: Validate envelope BEFORE dispatch
        # This validates operation, prefix, payload requirements, and normalizes correlation_id
        try:
            validate_envelope(envelope, await self._get_handler_registry())
        except EnvelopeValidationError as e:
            # Validation failed - missing operation or payload
            error_response = self._create_error_response(
                error=str(e),
                correlation_id=pre_validation_correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)
            logger.warning(
                "Envelope validation failed",
                extra={
                    "error": str(e),
                    "correlation_id": str(pre_validation_correlation_id),
                    "error_type": "EnvelopeValidationError",
                },
            )
            return
        except UnknownHandlerTypeError as e:
            # Unknown handler prefix - hard failure
            error_response = self._create_error_response(
                error=str(e),
                correlation_id=pre_validation_correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)
            logger.warning(
                "Unknown handler type in envelope",
                extra={
                    "error": str(e),
                    "correlation_id": str(pre_validation_correlation_id),
                    "error_type": "UnknownHandlerTypeError",
                },
            )
            return

        # After validation, correlation_id is guaranteed to be a UUID
        correlation_id = envelope.get("correlation_id")
        if not isinstance(correlation_id, UUID):
            correlation_id = pre_validation_correlation_id

        # Step 2: Check idempotency before handler dispatch (OMN-945)
        # This prevents duplicate processing under at-least-once delivery
        if not await self._check_idempotency(envelope, correlation_id):
            # Duplicate detected - response already published, return early
            return

        # Extract operation (validated to exist and be a string)
        operation = str(envelope.get("operation"))

        # Determine handler_type from envelope
        # If handler_type not explicit, extract from operation (e.g., "http.get" -> "http")
        handler_type = envelope.get("handler_type")
        if handler_type is None:
            handler_type = operation.split(".")[0]

        # Get handler from registry
        handler = self._handlers.get(str(handler_type))

        if handler is None:
            # Handler not instantiated (different from unknown prefix - validation already passed)
            # This can happen if handler registration failed during start()
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation=str(operation),
                target_name=str(handler_type),
                correlation_id=correlation_id,
            )

            # Create structured error for logging and tracking
            routing_error = RuntimeHostError(
                f"Handler type {handler_type!r} is registered but not instantiated",
                context=context,
            )

            # Publish error response for envelope-based error handling
            error_response = self._create_error_response(
                error=str(routing_error),
                correlation_id=correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)

            # Log with structured error
            logger.warning(
                "Handler registered but not instantiated",
                extra={
                    "handler_type": handler_type,
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                    "registered_handlers": list(self._handlers.keys()),
                    "error": str(routing_error),
                },
            )
            return

        # Execute handler
        try:
            # Handler expected to have async execute(envelope) method
            # NOTE: MVP adapters use legacy execute(envelope: dict) signature.
            # TODO(OMN-40): Migrate handlers to new protocol signature execute(request, operation_config)
            response = await handler.execute(envelope)  # type: ignore[call-arg]  # NOTE: legacy signature

            # Ensure response has correlation_id
            # Make a copy to avoid mutating handler's internal state
            if isinstance(response, dict):
                response = dict(response)
                if "correlation_id" not in response:
                    response["correlation_id"] = correlation_id

            await self._publish_envelope_safe(response, self._output_topic)

            logger.debug(
                "Handler executed successfully",
                extra={
                    "handler_type": handler_type,
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                },
            )

        except Exception as e:
            # Create infrastructure error context for handler execution failure
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="handler_execution",
                target_name=str(handler_type),
                correlation_id=correlation_id,
            )
            # Chain the error with infrastructure context
            infra_error = RuntimeHostError(
                f"Handler execution failed for {handler_type}: {e}",
                context=context,
            )
            infra_error.__cause__ = e  # Proper error chaining

            # Handler execution failed - produce failure envelope
            error_response = self._create_error_response(
                error=str(e),
                correlation_id=correlation_id,
            )
            await self._publish_envelope_safe(error_response, self._output_topic)

            logger.exception(
                "Handler execution failed",
                extra={
                    "handler_type": handler_type,
                    "correlation_id": str(correlation_id),
                    "operation": operation,
                    "error": str(e),
                    "infra_error": str(infra_error),
                },
            )

    def _create_error_response(
        self,
        error: str,
        correlation_id: UUID | None,
    ) -> dict[str, object]:
        """Create a standardized error response envelope.

        Args:
            error: Error message to include.
            correlation_id: Correlation ID to preserve for tracking.

        Returns:
            Error response dict with success=False and error details.
        """
        # Use correlation_id or generate a new one, keeping as UUID for internal use
        final_correlation_id = correlation_id or uuid4()
        return {
            "success": False,
            "status": "error",
            "error": error,
            "correlation_id": final_correlation_id,
        }

    def _serialize_envelope(
        self, envelope: dict[str, object] | BaseModel
    ) -> dict[str, object]:
        """Recursively convert UUID objects to strings for JSON serialization.

        Handles both dict envelopes and Pydantic models (e.g., ModelDuplicateResponse).

        Args:
            envelope: Envelope dict or Pydantic model that may contain UUID objects.

        Returns:
            New dict with all UUIDs converted to strings.
        """
        # Convert Pydantic models to dict first, ensuring type safety
        envelope_dict: JsonDict = (
            envelope.model_dump() if isinstance(envelope, BaseModel) else envelope
        )

        def convert_value(value: object) -> object:
            if isinstance(value, UUID):
                return str(value)
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            return value

        return {k: convert_value(v) for k, v in envelope_dict.items()}

    async def _publish_envelope_safe(
        self, envelope: dict[str, object] | BaseModel, topic: str
    ) -> None:
        """Publish envelope with UUID serialization support.

        Converts any UUID objects to strings before publishing to ensure
        JSON serialization works correctly.

        Args:
            envelope: Envelope dict or Pydantic model (may contain UUID objects).
            topic: Target topic to publish to.
        """
        # Always serialize UUIDs upfront - single code path
        json_safe_envelope = self._serialize_envelope(envelope)
        await self._event_bus.publish_envelope(json_safe_envelope, topic)

    async def health_check(self) -> dict[str, object]:
        """Return health check status.

        Returns:
            Dictionary with health status information:
                - healthy: Overall health status (True only if running,
                  event bus healthy, no handlers failed to instantiate,
                  all registered handlers are healthy, AND at least one
                  handler is registered - a runtime without handlers is useless)
                - degraded: True when process is running but some handlers
                  failed to instantiate. Indicates partial functionality -
                  the system is operational but not at full capacity.
                - is_running: Whether the process is running
                - is_draining: Whether the process is in graceful shutdown drain
                  period, waiting for in-flight messages to complete (OMN-756).
                  Load balancers can use this to remove the service from rotation
                  before the container becomes unhealthy.
                - pending_message_count: Number of messages currently being
                  processed. Useful for monitoring drain progress and determining
                  when the service is ready for shutdown.
                - event_bus: Event bus health status (if running)
                - event_bus_healthy: Boolean indicating event bus health
                - failed_handlers: Dict of handler_type -> error message for
                  handlers that failed to instantiate during start()
                - registered_handlers: List of successfully registered handler types
                - handlers: Dict of handler_type -> health status for each
                  registered handler
                - no_handlers_registered: True if no handlers are registered.
                  This indicates a critical configuration issue - the runtime
                  cannot process any events without handlers (OMN-1317).

        Health State Matrix:
            - healthy=True, degraded=False: Fully operational
            - healthy=False, degraded=True: Running with reduced functionality
            - healthy=False, degraded=False: Not running, event bus unhealthy,
              or no handlers registered (critical configuration issue)
            - healthy=False, no_handlers_registered=True: Configuration error,
              runtime cannot process events

        Drain State:
            When is_draining=True, the service is shutting down gracefully:
            - New messages are no longer being accepted
            - In-flight messages are being allowed to complete
            - Health status may still show healthy during drain
            - Load balancers should remove the service from rotation

        Note:
            Handler health checks are performed concurrently using asyncio.gather()
            with individual timeouts (configurable via health_check_timeout_seconds
            config, default: 5.0 seconds) to prevent slow handlers from blocking.
        """
        # Get event bus health if available
        event_bus_health: dict[str, object] = {}
        event_bus_healthy = False

        try:
            event_bus_health = await self._event_bus.health_check()
            # Explicit type guard (not assert) for production safety
            # health_check() returns dict per contract
            if not isinstance(event_bus_health, dict):
                context = ModelInfraErrorContext(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="health_check",
                )
                raise ProtocolConfigurationError(
                    f"health_check() must return dict, got {type(event_bus_health).__name__}",
                    context=context,
                )
            event_bus_healthy = bool(event_bus_health.get("healthy", False))
        except Exception as e:
            # Create infrastructure error context for health check failure
            correlation_id = uuid4()
            context = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="health_check",
                target_name="event_bus",
                correlation_id=correlation_id,
            )
            # Chain the error with infrastructure context
            infra_error = RuntimeHostError(
                f"Event bus health check failed: {e}",
                context=context,
            )
            infra_error.__cause__ = e  # Proper error chaining

            logger.warning(
                "Event bus health check failed",
                extra={
                    "error": str(e),
                    "correlation_id": str(correlation_id),
                    "infra_error": str(infra_error),
                },
                exc_info=True,
            )
            event_bus_health = {"error": str(e), "correlation_id": str(correlation_id)}
            event_bus_healthy = False

        # Check handler health for all registered handlers concurrently
        # Delegates to ProtocolLifecycleExecutor with configured timeout to prevent blocking
        handler_health_results: dict[str, object] = {}
        handlers_all_healthy = True

        if self._handlers:
            # Run all handler health checks concurrently using asyncio.gather()
            health_check_tasks = [
                self._lifecycle_executor.check_handler_health(handler_type, handler)
                for handler_type, handler in self._handlers.items()
            ]
            results = await asyncio.gather(*health_check_tasks)

            # Process results and build the results dict
            for health_result in results:
                handler_health_results[health_result.handler_type] = (
                    health_result.details
                )
                if not health_result.healthy:
                    handlers_all_healthy = False

        # Check for failed handlers - any failures indicate degraded state
        has_failed_handlers = len(self._failed_handlers) > 0

        # Check for no handlers registered - critical configuration issue
        # A runtime with no handlers cannot process any events and should be unhealthy
        no_handlers_registered = len(self._handlers) == 0

        # Degraded state: process is running but some handlers failed to instantiate
        # This means the system is operational but with reduced functionality
        degraded = self._is_running and has_failed_handlers

        # Overall health is True only if running, event bus is healthy,
        # no handlers failed to instantiate, all registered handlers are healthy,
        # AND at least one handler is registered (runtime without handlers is useless)
        healthy = (
            self._is_running
            and event_bus_healthy
            and not has_failed_handlers
            and handlers_all_healthy
            and not no_handlers_registered
        )

        return {
            "healthy": healthy,
            "degraded": degraded,
            "is_running": self._is_running,
            "is_draining": self._is_draining,
            "pending_message_count": self._pending_message_count,
            "event_bus": event_bus_health,
            "event_bus_healthy": event_bus_healthy,
            "failed_handlers": self._failed_handlers,
            "registered_handlers": list(self._handlers.keys()),
            "handlers": handler_health_results,
            "no_handlers_registered": no_handlers_registered,
        }

    def register_handler(
        self, handler_type: str, handler: ProtocolContainerAware
    ) -> None:
        """Register a handler for a specific type.

        Args:
            handler_type: Protocol type identifier (e.g., "http", "db").
            handler: Handler instance implementing the ProtocolContainerAware protocol.
        """
        self._handlers[handler_type] = handler
        logger.debug(
            "Handler registered",
            extra={
                "handler_type": handler_type,
                "handler_class": type(handler).__name__,
            },
        )

    def get_handler(self, handler_type: str) -> ProtocolContainerAware | None:
        """Get handler for type, returns None if not registered.

        Args:
            handler_type: Protocol type identifier.

        Returns:
            Handler instance if registered, None otherwise.
        """
        return self._handlers.get(handler_type)

    # =========================================================================
    # Architecture Validation Methods (OMN-1138)
    # =========================================================================

    async def _validate_architecture(self) -> None:
        """Validate architecture compliance before starting runtime.

        This method is called at the beginning of start() to validate nodes
        and handlers against registered architecture rules. If any violations
        with ERROR severity are detected, startup is blocked.

        Validation occurs BEFORE:
        - Event bus starts
        - Handlers are wired
        - Subscription begins

        Validation Behavior:
            - ERROR severity violations: Block startup, raise ArchitectureViolationError
            - WARNING severity violations: Log warning, continue startup
            - INFO severity violations: Log info, continue startup

        Raises:
            ArchitectureViolationError: If blocking violations (ERROR severity)
                are detected. Contains all blocking violations for inspection.

        Example:
            >>> # Validation is automatic in start()
            >>> try:
            ...     await runtime.start()
            ... except ArchitectureViolationError as e:
            ...     print(f"Startup blocked: {len(e.violations)} violations")
            ...     for v in e.violations:
            ...         print(v.format_for_logging())

        Note:
            Validation only runs if architecture_rules were provided at init.
            If no rules are configured, this method returns immediately.

        Related:
            - OMN-1138: Architecture Validator for omnibase_infra
            - OMN-1099: Validators implementing ProtocolArchitectureRule
        """
        # Skip validation if no rules configured
        if not self._architecture_rules:
            logger.debug("No architecture rules configured, skipping validation")
            return

        logger.info(
            "Validating architecture compliance",
            extra={
                "rule_count": len(self._architecture_rules),
                "rule_ids": tuple(r.rule_id for r in self._architecture_rules),
            },
        )

        # Import architecture validator components
        from omnibase_infra.errors import ArchitectureViolationError
        from omnibase_infra.nodes.architecture_validator import (
            ModelArchitectureValidationRequest,
            NodeArchitectureValidatorCompute,
        )

        # Create or get container
        container = self._get_or_create_container()

        # Instantiate validator with rules
        validator = NodeArchitectureValidatorCompute(
            container=container,
            rules=self._architecture_rules,
        )

        # Build validation request
        # Note: At this point, handlers haven't been instantiated yet (that happens
        # after validation in _populate_handlers_from_registry). We validate the
        # handler CLASSES from the registry, not handler instances.
        handler_registry = await self._get_handler_registry()
        handler_classes: list[type[ProtocolContainerAware]] = []
        for handler_type in handler_registry.list_protocols():
            try:
                handler_cls = handler_registry.get(handler_type)
                handler_classes.append(handler_cls)
            except Exception as e:
                # If a handler class can't be retrieved, skip it for validation
                # (it will fail later during instantiation anyway)
                logger.debug(
                    "Skipping handler class for architecture validation",
                    extra={
                        "handler_type": handler_type,
                        "error": str(e),
                    },
                )

        request = ModelArchitectureValidationRequest(
            nodes=(),  # Nodes not yet available at this point
            handlers=tuple(handler_classes),
        )

        # Execute validation
        result = validator.compute(request)

        # Separate blocking and non-blocking violations
        blocking_violations = tuple(v for v in result.violations if v.blocks_startup())
        warning_violations = tuple(
            v for v in result.violations if not v.blocks_startup()
        )

        # Log warnings but don't block
        for violation in warning_violations:
            # Note: We can't use to_structured_dict() directly because 'message'
            # is a reserved key in Python logging's extra parameter.
            # We use format_for_logging() instead for the log message.
            logger.warning(
                "Architecture warning: %s",
                violation.format_for_logging(),
                extra={
                    "rule_id": violation.rule_id,
                    "severity": violation.severity.value,
                    "target_type": violation.target_type,
                    "target_name": violation.target_name,
                },
            )

        # Block startup on ERROR violations
        if blocking_violations:
            logger.error(
                "Architecture validation failed",
                extra={
                    "blocking_violation_count": len(blocking_violations),
                    "warning_violation_count": len(warning_violations),
                    "blocking_rule_ids": tuple(v.rule_id for v in blocking_violations),
                },
            )
            raise ArchitectureViolationError(
                message=f"Architecture validation failed with {len(blocking_violations)} blocking violations",
                violations=blocking_violations,
            )

        logger.info(
            "Architecture validation passed",
            extra={
                "rules_checked": result.rules_checked,
                "handlers_checked": result.handlers_checked,
                "warning_count": len(warning_violations),
            },
        )

    def _get_or_create_container(self) -> ModelONEXContainer:
        """Get the injected container or create and cache a new one.

        Returns:
            ModelONEXContainer instance for dependency injection.

        Note:
            If no container was provided at init, a new container is created
            and cached in self._container. This ensures all handlers share
            the same container instance. The container provides basic
            infrastructure for node execution but may not have all services wired.
        """
        if self._container is not None:
            return self._container

        # Create container and cache it for reuse
        from omnibase_core.models.container.model_onex_container import (
            ModelONEXContainer,
        )

        logger.debug("Creating and caching container (no container provided at init)")
        self._container = ModelONEXContainer()
        return self._container

    # =========================================================================
    # Idempotency Guard Methods (OMN-945)
    # =========================================================================

    async def _initialize_idempotency_store(self) -> None:
        """Initialize idempotency store from configuration.

        Reads idempotency configuration from the runtime config and wires
        the appropriate store implementation. If not configured or disabled,
        idempotency checking is skipped.

        Supported store types:
            - "postgres": PostgreSQL-backed durable store (production)
            - "memory": In-memory store (testing only)

        Configuration keys:
            - idempotency.enabled: bool (default: False)
            - idempotency.store_type: "postgres" | "memory" (default: "postgres")
            - idempotency.domain_from_operation: bool (default: True)
            - idempotency.skip_operations: list[str] (default: [])
            - idempotency_database: dict (PostgreSQL connection config)
        """
        # Check if config exists
        if self._config is None:
            logger.debug("No runtime config provided, skipping idempotency setup")
            return

        # Check if config has idempotency section
        idempotency_raw = self._config.get("idempotency")
        if idempotency_raw is None:
            logger.debug("Idempotency guard not configured, skipping")
            return

        try:
            from omnibase_infra.idempotency import ModelIdempotencyGuardConfig

            if isinstance(idempotency_raw, dict):
                self._idempotency_config = ModelIdempotencyGuardConfig.model_validate(
                    idempotency_raw
                )
            elif isinstance(idempotency_raw, ModelIdempotencyGuardConfig):
                self._idempotency_config = idempotency_raw
            else:
                logger.warning(
                    "Invalid idempotency config type",
                    extra={"type": type(idempotency_raw).__name__},
                )
                return

            if not self._idempotency_config.enabled:
                logger.debug("Idempotency guard disabled in config")
                return

            # Create store based on store_type
            if self._idempotency_config.store_type == "postgres":
                from omnibase_infra.idempotency import (
                    ModelPostgresIdempotencyStoreConfig,
                    StoreIdempotencyPostgres,
                )

                # Get database config from container or config
                db_config_raw = self._config.get("idempotency_database", {})
                if isinstance(db_config_raw, dict):
                    db_config = ModelPostgresIdempotencyStoreConfig.model_validate(
                        db_config_raw
                    )
                elif isinstance(db_config_raw, ModelPostgresIdempotencyStoreConfig):
                    db_config = db_config_raw
                else:
                    logger.warning(
                        "Invalid idempotency_database config type",
                        extra={"type": type(db_config_raw).__name__},
                    )
                    return

                self._idempotency_store = StoreIdempotencyPostgres(config=db_config)
                await self._idempotency_store.initialize()

            elif self._idempotency_config.store_type == "memory":
                from omnibase_infra.idempotency import StoreIdempotencyInmemory

                self._idempotency_store = StoreIdempotencyInmemory()

            else:
                logger.warning(
                    "Unknown idempotency store type",
                    extra={"store_type": self._idempotency_config.store_type},
                )
                return

            logger.info(
                "Idempotency guard initialized",
                extra={
                    "store_type": self._idempotency_config.store_type,
                    "domain_from_operation": self._idempotency_config.domain_from_operation,
                    "skip_operations": self._idempotency_config.skip_operations,
                },
            )

        except Exception as e:
            logger.warning(
                "Failed to initialize idempotency store, proceeding without",
                extra={"error": str(e)},
            )
            self._idempotency_store = None
            self._idempotency_config = None

    # =========================================================================
    # WARNING: FAIL-OPEN BEHAVIOR
    # =========================================================================
    # This method implements FAIL-OPEN semantics: if the idempotency store
    # is unavailable or errors, messages are ALLOWED THROUGH for processing.
    #
    # This is an intentional design decision prioritizing availability over
    # exactly-once guarantees. See docstring below for full trade-off analysis.
    #
    # IMPORTANT: Downstream handlers MUST be designed for at-least-once delivery
    # and implement their own idempotency for critical operations.
    # =========================================================================
    async def _check_idempotency(
        self,
        envelope: dict[str, object],
        correlation_id: UUID,
    ) -> bool:
        """Check if envelope should be processed (idempotency guard).

        Extracts message_id from envelope headers and checks against the
        idempotency store. If duplicate detected, publishes a duplicate
        response and returns False.

        Fail-Open Semantics:
            This method implements **fail-open** error handling: if the
            idempotency store is unavailable or throws an error, the message
            is allowed through for processing (with a warning log).

            **Design Rationale**: In distributed event-driven systems, the
            idempotency store (e.g., Redis/Valkey) is a supporting service,
            not a critical path dependency. A temporary store outage should
            not halt message processing entirely, as this would cascade into
            broader system unavailability.

            **Trade-offs**:
            - Pro: High availability - processing continues during store outages
            - Pro: Graceful degradation - system remains functional
            - Con: May result in duplicate message processing during outages
            - Con: Downstream handlers must be designed for at-least-once delivery

            **Mitigation**: Handlers consuming messages should implement their
            own idempotency logic for critical operations (e.g., using database
            constraints or transaction guards) to ensure correctness even when
            duplicates slip through.

        Args:
            envelope: Validated envelope dict.
            correlation_id: Normalized correlation ID (UUID).

        Returns:
            True if message should be processed (new message).
            False if message is duplicate (skip processing).
        """
        # Skip check if idempotency not configured
        if self._idempotency_store is None or self._idempotency_config is None:
            return True

        if not self._idempotency_config.enabled:
            return True

        # Check if operation is in skip list
        operation = envelope.get("operation")
        if isinstance(operation, str):
            if not self._idempotency_config.should_check_idempotency(operation):
                logger.debug(
                    "Skipping idempotency check for operation",
                    extra={
                        "operation": operation,
                        "correlation_id": str(correlation_id),
                    },
                )
                return True

        # Extract message_id from envelope
        message_id = self._extract_message_id(envelope, correlation_id)

        # Extract domain from operation if configured
        domain = self._extract_idempotency_domain(envelope)

        # Check and record in store
        try:
            is_new = await self._idempotency_store.check_and_record(
                message_id=message_id,
                domain=domain,
                correlation_id=correlation_id,
            )

            if not is_new:
                # Duplicate detected - publish duplicate response (NOT an error)
                logger.info(
                    "Duplicate message detected, skipping processing",
                    extra={
                        "message_id": str(message_id),
                        "domain": domain,
                        "correlation_id": str(correlation_id),
                    },
                )

                duplicate_response = self._create_duplicate_response(
                    message_id=message_id,
                    correlation_id=correlation_id,
                )
                # duplicate_response is already a dict from _create_duplicate_response
                await self._publish_envelope_safe(
                    duplicate_response, self._output_topic
                )
                return False

            return True

        except Exception as e:
            # FAIL-OPEN: Allow message through on idempotency store errors.
            # Rationale: Availability over exactly-once. Store outages should not
            # halt processing. Downstream handlers must tolerate duplicates.
            # See docstring for full trade-off analysis.
            logger.warning(
                "Idempotency check failed, allowing message through (fail-open)",
                extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "message_id": str(message_id),
                    "domain": domain,
                    "correlation_id": str(correlation_id),
                },
            )
            return True

    def _extract_message_id(
        self,
        envelope: dict[str, object],
        correlation_id: UUID,
    ) -> UUID:
        """Extract message_id from envelope, falling back to correlation_id.

        Priority:
            1. envelope["headers"]["message_id"]
            2. envelope["message_id"]
            3. Use correlation_id as message_id (fallback)

        Args:
            envelope: Envelope dict to extract message_id from.
            correlation_id: Fallback UUID if message_id not found.

        Returns:
            UUID representing the message_id.
        """
        # Try headers first
        headers = envelope.get("headers")
        if isinstance(headers, dict):
            header_msg_id = headers.get("message_id")
            if header_msg_id is not None:
                if isinstance(header_msg_id, UUID):
                    return header_msg_id
                if isinstance(header_msg_id, str):
                    try:
                        return UUID(header_msg_id)
                    except ValueError:
                        pass

        # Try top-level message_id
        top_level_msg_id = envelope.get("message_id")
        if top_level_msg_id is not None:
            if isinstance(top_level_msg_id, UUID):
                return top_level_msg_id
            if isinstance(top_level_msg_id, str):
                try:
                    return UUID(top_level_msg_id)
                except ValueError:
                    pass

        # Fallback: use correlation_id as message_id
        return correlation_id

    def _extract_idempotency_domain(
        self,
        envelope: dict[str, object],
    ) -> str | None:
        """Extract domain for idempotency key from envelope.

        If domain_from_operation is enabled in config, extracts domain
        from the operation prefix (e.g., "db.query" -> "db").

        Args:
            envelope: Envelope dict to extract domain from.

        Returns:
            Domain string if found and configured, None otherwise.
        """
        if self._idempotency_config is None:
            return None

        if not self._idempotency_config.domain_from_operation:
            return None

        operation = envelope.get("operation")
        if isinstance(operation, str):
            return self._idempotency_config.extract_domain(operation)

        return None

    def _create_duplicate_response(
        self,
        message_id: UUID,
        correlation_id: UUID,
    ) -> dict[str, object]:
        """Create response for duplicate message detection.

        This is NOT an error response - duplicates are expected under
        at-least-once delivery. The response indicates successful
        deduplication.

        Args:
            message_id: UUID of the duplicate message.
            correlation_id: Correlation ID for tracing.

        Returns:
            Dict representation of ModelDuplicateResponse for envelope publishing.
        """
        return ModelDuplicateResponse(
            message_id=message_id,
            correlation_id=correlation_id,
        ).model_dump()

    async def _cleanup_idempotency_store(self) -> None:
        """Cleanup idempotency store during shutdown.

        Closes the idempotency store connection if initialized.
        Called during stop() to release resources.
        """
        if self._idempotency_store is None:
            return

        try:
            if hasattr(self._idempotency_store, "shutdown"):
                await self._idempotency_store.shutdown()
            elif hasattr(self._idempotency_store, "close"):
                await self._idempotency_store.close()
            logger.debug("Idempotency store shutdown complete")
        except Exception as e:
            logger.warning(
                "Failed to shutdown idempotency store",
                extra={"error": str(e)},
            )
        finally:
            self._idempotency_store = None


__all__: list[str] = [
    "RuntimeHostProcess",
    "wire_handlers",
]
