# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
# ruff: noqa: G201
# G201 disabled: Logging extra dict is intentional for structured logging with correlation IDs
"""Node introspection mixin providing automatic capability discovery.

This module provides a reusable mixin for ONEX nodes to implement automatic
capability discovery, endpoint reporting, and periodic heartbeat broadcasting.
It uses reflection to discover node capabilities and integrates with the event
bus for distributed service discovery.

Features:
    - Automatic capability discovery via reflection
    - Endpoint URL discovery (health, api, metrics)
    - FSM state reporting if applicable
    - Cached introspection data with configurable TTL
    - Background heartbeat task for periodic health broadcasts
    - Registry listener for REQUEST_INTROSPECTION events
    - Graceful degradation when event bus is unavailable

Note:
    - active_operations_count in heartbeats is tracked via ``track_operation()``
      context manager. Nodes should wrap their operations with this context
      manager to accurately report concurrent operation counts.

    - **track_operation() Usage Guidelines**:

      Within MixinNodeIntrospection itself, only ``publish_introspection()`` uses
      ``track_operation()``. This is intentional for the following reasons:

      1. **_publish_heartbeat()**: Explicitly excluded because it's an internal
         background task. Tracking it would cause self-referential counting
         (heartbeat counting itself as active) and would report infrastructure
         overhead rather than business load.

      2. **get_introspection_data()**: Called by ``publish_introspection()``, which
         already wraps the entire operation. Adding tracking here would cause
         double-counting. Additionally, this is metadata gathering, not a business
         operation that represents node load.

      3. **start/stop_introspection_tasks()**: One-time lifecycle operations that
         complete quickly. They spawn/cancel background tasks but don't represent
         ongoing load. The counter would increment and immediately decrement.

      4. **get_capabilities(), get_endpoints(), get_current_state()**: Internal
         metadata operations that are part of introspection data gathering, not
         independent business operations.

      **For consuming nodes**: Use ``track_operation()`` in your business methods
      (e.g., ``execute_query()``, ``process_request()``, ``handle_event()``) to
      accurately report concurrent operation counts in heartbeats. See the
      ``track_operation()`` docstring for usage examples.

Security Considerations:
    This mixin uses Python reflection (via the ``inspect`` module) to automatically
    discover node capabilities. While this enables powerful service discovery, it
    has security implications that developers must understand.

    **Threat Model**:

    Introspection data could be valuable to an attacker for:

    - **Reconnaissance**: Learning what operations a node supports to identify
      attack vectors (e.g., discovering ``decrypt_*``, ``admin_*`` methods).
    - **Architecture mapping**: Understanding system topology through protocol
      and mixin discovery (e.g., which nodes implement ``ProtocolDatabaseAdapter``).
    - **Version fingerprinting**: Identifying outdated versions with known
      vulnerabilities via the ``version`` field.
    - **State inference**: Deducing system state or health from FSM state values.

    **What Gets Exposed via Introspection**:

    - **Public method names**: Method names that may reveal operations
      (e.g., ``execute_query``, ``process_payment``).
    - **Method signatures**: Full signatures including parameter names and type
      annotations. Parameter names like ``api_key``, ``user_password``, or
      ``decrypt_key`` reveal sensitive parameter purposes.
    - **Protocol implementations**: Class names from inheritance hierarchy that
      start with ``Protocol`` or ``Mixin`` (e.g., ``ProtocolDatabaseAdapter``,
      ``MixinAsyncCircuitBreaker``).
    - **FSM state information**: Current state value if FSM attributes exist
      (e.g., ``connected``, ``authenticated``, ``processing``).
    - **Endpoint URLs**: Health, API, and metrics endpoint paths.
    - **Node metadata**: Node ID (UUID), type via ``EnumNodeKind`` (EFFECT/COMPUTE/REDUCER/ORCHESTRATOR), and version.

    **What is NOT Exposed**:

    - Private methods (prefixed with ``_``) - completely excluded from discovery.
    - Method implementations or source code - only signatures, not logic.
    - Internal state variables - only FSM state if present.
    - Configuration values - secrets, connection strings, etc. are not exposed.
    - Environment variables or runtime parameters.
    - Request/response payloads or historical data.

    **Built-in Protections**:

    The mixin includes filtering mechanisms to limit exposure:

    - **Private method exclusion**: Methods prefixed with ``_`` are excluded from
      capability discovery.
    - **Utility method filtering**: Common utility prefixes (``get_*``, ``set_*``,
      ``initialize*``, ``start_*``, ``stop_*``) are filtered out by default.
    - **Operation keyword matching**: Only methods containing operation keywords
      (``execute``, ``handle``, ``process``, ``run``, ``invoke``, ``call``) are
      reported as capabilities in the operations list.
    - **Configurable exclusions**: The ``exclude_prefixes`` parameter in
      ``initialize_introspection()`` allows additional filtering.
    - **Caching with TTL**: Introspection data is cached to reduce reflection
      frequency, with configurable TTL for freshness.

    **Best Practices for Node Developers**:

    - Prefix internal/sensitive methods with ``_`` to exclude them from introspection.
    - Avoid exposing sensitive business logic in public method names (e.g., use
      ``process_request`` instead of ``decrypt_and_forward_to_payment_gateway``).
    - Use generic parameter names for public methods (e.g., ``data`` instead of
      ``user_credentials``, ``payload`` instead of ``encrypted_secret``).
    - Review exposed capabilities before deploying to production environments.
    - Consider network segmentation for introspection event topics in multi-tenant
      environments.
    - Use the ``exclude_prefixes`` parameter to filter additional method patterns
      if needed.

    **Network Security Considerations**:

    - Introspection data is published to Kafka topics (``node.introspection``,
      ``node.heartbeat``, ``node.request_introspection``).
    - In multi-tenant environments, ensure proper topic ACLs are configured.
    - Consider whether introspection topics should be accessible outside the cluster.
    - Monitor introspection topic consumers for unauthorized access.
    - The registry listener responds to ANY request on the request topic without
      authentication - secure the topic with Kafka ACLs.

    **Production Deployment Checklist**:

    1. Review ``get_capabilities()`` output for each node before deployment.
    2. Verify no sensitive method names or parameter names are exposed.
    3. Configure Kafka topic ACLs to restrict introspection topic access.
    4. Consider disabling ``enable_registry_listener`` if not needed.
    5. Monitor introspection topic consumer groups for unexpected consumers.
    6. Use network segmentation to isolate introspection traffic if required.

    For more details, see the "Node Introspection Security Considerations" section
    in ``CLAUDE.md``.

Usage:
    ```python
    from omnibase_core.enums import EnumNodeKind
    from omnibase_infra.mixins import MixinNodeIntrospection
    from omnibase_infra.models.discovery import ModelIntrospectionConfig

    class MyNode(MixinNodeIntrospection):
        def __init__(self, node_config, event_bus=None):
            config = ModelIntrospectionConfig(
                node_id=node_config.node_id,
                node_type=EnumNodeKind.EFFECT,
                event_bus=event_bus,
            )
            self.initialize_introspection(config)

        async def startup(self):
            # Publish initial introspection on startup
            await self.publish_introspection(reason="startup")

            # Start background tasks
            await self.start_introspection_tasks(
                enable_heartbeat=True,
                heartbeat_interval_seconds=30.0,
                enable_registry_listener=True,
            )

        async def shutdown(self):
            # Publish shutdown introspection
            await self.publish_introspection(reason="shutdown")

            # Stop background tasks
            await self.stop_introspection_tasks()
    ```

Integration Requirements:
    Classes using this mixin must:
    1. Call `initialize_introspection(config)` during initialization with a
       ModelIntrospectionConfig instance
    2. Optionally call `start_introspection_tasks()` for background operations
    3. Call `stop_introspection_tasks()` during shutdown
    4. Ensure event_bus has `publish_envelope()` method if provided

See Also:
    - ModelIntrospectionConfig for configuration options
    - MixinAsyncCircuitBreaker for circuit breaker pattern
    - ModelNodeIntrospectionEvent for event model
    - ModelNodeHeartbeatEvent for heartbeat model
    - CLAUDE.md "Node Introspection Security Considerations" section
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, ClassVar, TypedDict, cast
from uuid import UUID, uuid4

from omnibase_core.enums import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.capabilities import ContractCapabilityExtractor
from omnibase_infra.constants_topic_patterns import TOPIC_NAME_PATTERN
from omnibase_infra.enums import EnumInfraTransportType, EnumIntrospectionReason
from omnibase_infra.errors import ModelInfraErrorContext, ProtocolConfigurationError
from omnibase_infra.models.discovery import (
    ModelDiscoveredCapabilities,
    ModelIntrospectionConfig,
    ModelIntrospectionTaskConfig,
)
from omnibase_infra.models.discovery.model_introspection_performance_metrics import (
    ModelIntrospectionPerformanceMetrics,
)
from omnibase_infra.models.registration import (
    ModelNodeCapabilities,
    ModelNodeHeartbeatEvent,
)
from omnibase_infra.models.registration.model_node_event_bus_config import (
    ModelEventBusTopicEntry,
    ModelNodeEventBusConfig,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)

if TYPE_CHECKING:
    from omnibase_core.models.contracts import ModelContractBase
    from omnibase_core.protocols.event_bus.protocol_event_bus import ProtocolEventBus
    from omnibase_core.protocols.event_bus.protocol_event_message import (
        ProtocolEventMessage,
    )

logger = logging.getLogger(__name__)

# Performance threshold constants (in milliseconds)
PERF_THRESHOLD_GET_CAPABILITIES_MS = 50.0
PERF_THRESHOLD_DISCOVER_CAPABILITIES_MS = 30.0
PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS = 50.0
PERF_THRESHOLD_CACHE_HIT_MS = 1.0

# Module-level capability extractor instance (stateless, can be shared)
_CAPABILITY_EXTRACTOR = ContractCapabilityExtractor()


class PerformanceMetricsCacheDict(TypedDict, total=False):
    """TypedDict for JSON-serialized ModelIntrospectionPerformanceMetrics.

    This type matches the output of ModelIntrospectionPerformanceMetrics.model_dump(mode="json"),
    enabling proper type checking for cached performance metrics.

    Attributes:
        get_capabilities_ms: Time taken by get_capabilities() in milliseconds.
        discover_capabilities_ms: Time taken by _discover_capabilities() in ms.
        get_endpoints_ms: Time taken by get_endpoints() in milliseconds.
        get_current_state_ms: Time taken by get_current_state() in milliseconds.
        total_introspection_ms: Total time for get_introspection_data() in ms.
        cache_hit: Whether the result was served from cache.
        method_count: Number of methods discovered during reflection.
        threshold_exceeded: Whether any operation exceeded performance thresholds.
        slow_operations: List of operation names that exceeded their thresholds.
        captured_at: UTC timestamp when metrics were captured (ISO string).
    """

    get_capabilities_ms: float
    discover_capabilities_ms: float
    get_endpoints_ms: float
    get_current_state_ms: float
    total_introspection_ms: float
    cache_hit: bool
    method_count: int
    threshold_exceeded: bool
    slow_operations: list[str]
    captured_at: str  # datetime serializes to ISO string in JSON mode


class DiscoveredCapabilitiesCacheDict(TypedDict, total=False):
    """TypedDict for JSON-serialized ModelDiscoveredCapabilities.

    Attributes:
        operations: List of method names matching operation keywords.
        has_fsm: Whether the node has FSM state management.
        method_signatures: Mapping of method names to signature strings.
        attributes: Additional discovered attributes.
    """

    operations: list[str]
    has_fsm: bool
    method_signatures: dict[str, str]
    attributes: dict[str, object]


class IntrospectionCacheDict(TypedDict):
    """TypedDict representing the JSON-serialized ModelNodeIntrospectionEvent.

    This type matches the output of ModelNodeIntrospectionEvent.model_dump(mode="json"),
    enabling proper type checking for cache operations without requiring type: ignore comments.

    Note:
        The capabilities are split into declared_capabilities (from contract) and
        discovered_capabilities (from reflection). This reflects the fundamental
        difference between what a node declares and what introspection discovers.
    """

    node_id: str
    node_type: str
    node_version: dict[str, int]  # ModelSemVer serializes to {major, minor, patch}
    declared_capabilities: dict[str, object]  # ModelNodeCapabilities (flexible schema)
    discovered_capabilities: DiscoveredCapabilitiesCacheDict
    endpoints: dict[str, str]
    current_state: str | None
    reason: str  # EnumIntrospectionReason serializes to string
    correlation_id: str  # UUID serializes to string in JSON mode (required field)
    timestamp: str  # datetime serializes to ISO string in JSON mode
    # Optional fields
    node_role: str | None
    metadata: dict[str, object]  # ModelNodeMetadata serializes to dict
    network_id: str | None
    deployment_id: str | None
    epoch: int | None
    # Performance metrics from introspection operation (may be None)
    performance_metrics: PerformanceMetricsCacheDict | None


class MixinNodeIntrospection:
    """Mixin providing node introspection capabilities.

    Provides automatic capability discovery using reflection, endpoint
    reporting, and periodic heartbeat broadcasting for ONEX nodes.

    State Variables:
        _introspection_cache: Cached introspection data
        _introspection_cache_ttl: Cache time-to-live in seconds
        _introspection_cached_at: Timestamp when cache was populated

    Background Task Variables:
        _heartbeat_task: Background heartbeat task
        _registry_listener_task: Background registry listener task
        _introspection_stop_event: Event to signal task shutdown

    Configuration Variables:
        _introspection_node_id: Node identifier
        _introspection_node_type: Node type classification
        _introspection_event_bus: Optional event bus for publishing
        _introspection_version: Node version string
        _introspection_start_time: Node startup timestamp

    Security Considerations:
        This mixin uses Python reflection (via the ``inspect`` module) to
        automatically discover node capabilities. While this enables powerful
        service discovery, it has security implications:

        **Threat Model**:

        - **Reconnaissance**: Method names may reveal attack vectors
        - **Architecture mapping**: Protocol discovery exposes topology
        - **Version fingerprinting**: Version field enables vulnerability scanning
        - **State inference**: FSM state reveals system status

        **Exposed Information**:

        - Public method names (potential operations a node can perform)
        - Method signatures (parameter names and type annotations)
        - Protocol and mixin implementations (discovered capabilities)
        - FSM state information (if state attributes are present)
        - Endpoint URLs (health, API, metrics paths)
        - Node metadata (name, version, type)

        **What is NOT Exposed**:

        - Private methods (``_`` prefix) - excluded from discovery
        - Method implementations or source code
        - Configuration values, secrets, or connection strings
        - Environment variables or runtime parameters
        - Request/response payloads or historical data

        **Built-in Protections**:

        - Private methods (prefixed with ``_``) are excluded by default
        - Utility method prefixes (``get_*``, ``set_*``, etc.) are filtered
        - Only methods containing operation keywords are reported as operations
        - Configure ``exclude_prefixes`` in ``initialize_introspection()`` for
          additional filtering
        - Caching with TTL reduces reflection frequency

        **Recommendations for Production**:

        - Prefix internal/sensitive methods with ``_`` to exclude them
        - Use generic operation names that don't reveal implementation details
        - Use generic parameter names (``data`` instead of ``user_credentials``)
        - Review ``get_capabilities()`` output before production deployment
        - In multi-tenant environments, configure Kafka topic ACLs for
          introspection events (``node.introspection``, ``node.heartbeat``,
          ``node.request_introspection``)
        - Monitor introspection topic consumers for unauthorized access
        - Consider network segmentation for introspection event topics
        - Consider disabling ``enable_registry_listener`` if not needed

    See Also:
        - Module docstring for detailed security documentation and threat model
        - CLAUDE.md "Node Introspection Security Considerations" section
        - ``get_capabilities()`` for filtering logic details

    Example:
        ```python
        from uuid import UUID
        from omnibase_core.enums import EnumNodeKind
        from omnibase_infra.models.discovery import ModelIntrospectionConfig

        class PostgresAdapter(MixinNodeIntrospection):
            def __init__(self, node_id: UUID, adapter_config):
                config = ModelIntrospectionConfig(
                    node_id=node_id,
                    node_type=EnumNodeKind.EFFECT,
                    event_bus=adapter_config.event_bus,
                )
                self.initialize_introspection(config)

            async def execute(self, query: str) -> list[dict]:
                # Node operation - WILL be exposed via introspection
                ...

            def _internal_helper(self, data: dict) -> dict:
                # Private method - will NOT be exposed
                ...
        ```
    """

    # Class-level cache for method signatures (populated once per class)
    # Maps class -> {method_name: signature_string}
    # This avoids expensive reflection on each introspection call since
    # method signatures don't change after class definition.
    # NOTE: ClassVar is intentionally shared across all instances - this is correct
    # behavior for a per-class cache of immutable method signatures.
    _class_method_cache: ClassVar[dict[type, dict[str, str]]] = {}

    # Type annotations for instance attributes (no default values to avoid shared state)
    # All of these are initialized in initialize_introspection()
    #
    # Caching attributes
    _introspection_cache: IntrospectionCacheDict | None
    _introspection_cache_ttl: float
    _introspection_cached_at: float | None

    # Background task attributes
    _heartbeat_task: asyncio.Task[None] | None
    _registry_listener_task: asyncio.Task[None] | None
    _introspection_stop_event: asyncio.Event | None
    _registry_unsubscribe: Callable[[], None] | Callable[[], Awaitable[None]] | None

    # Configuration attributes
    _introspection_node_id: UUID | None
    _introspection_node_type: EnumNodeKind | None
    _introspection_event_bus: ProtocolEventBus | None
    _introspection_version: str
    _introspection_start_time: float | None
    _introspection_contract: ModelContractBase | None

    # Capability discovery configuration
    _introspection_operation_keywords: frozenset[str]
    _introspection_exclude_prefixes: frozenset[str]

    # Registry listener callback error tracking (instance-level)
    # Used for rate-limiting error logging to prevent log spam during
    # sustained failures. These are initialized in initialize_introspection().
    _registry_callback_consecutive_failures: int
    _registry_callback_last_failure_time: float
    _registry_callback_failure_log_threshold: int

    # Performance metrics tracking (instance-level)
    # Stores the most recent performance metrics from introspection operations
    _introspection_last_metrics: ModelIntrospectionPerformanceMetrics | None

    # Active operations tracking (instance-level)
    # Thread-safe counter for tracking concurrent operations
    # Used by heartbeat to report active_operations_count
    _active_operations: int
    _operations_lock: asyncio.Lock

    # Default operation keywords for capability discovery
    DEFAULT_OPERATION_KEYWORDS: ClassVar[frozenset[str]] = frozenset(
        {
            "execute",
            "handle",
            "process",
            "run",
            "invoke",
            "call",
        }
    )

    # Default prefixes to exclude from capability discovery
    DEFAULT_EXCLUDE_PREFIXES: ClassVar[frozenset[str]] = frozenset(
        {
            "_",
            "get_",
            "set_",
            "initialize",
            "start_",
            "stop_",
        }
    )

    # Node-type-specific operation keyword suggestions
    # Uses EnumNodeKind as keys to ensure type safety when accessing with node_type.
    # Example: keywords = NODE_TYPE_OPERATION_KEYWORDS.get(node_type, set())
    NODE_TYPE_OPERATION_KEYWORDS: ClassVar[dict[EnumNodeKind, set[str]]] = {
        EnumNodeKind.EFFECT: {
            "execute",
            "handle",
            "process",
            "run",
            "invoke",
            "call",
            "fetch",
            "send",
            "query",
            "connect",
        },
        EnumNodeKind.COMPUTE: {
            "execute",
            "handle",
            "process",
            "run",
            "compute",
            "transform",
            "calculate",
            "convert",
            "parse",
        },
        EnumNodeKind.REDUCER: {
            "execute",
            "handle",
            "process",
            "run",
            "aggregate",
            "reduce",
            "merge",
            "combine",
            "accumulate",
        },
        EnumNodeKind.ORCHESTRATOR: {
            "execute",
            "handle",
            "process",
            "run",
            "orchestrate",
            "coordinate",
            "schedule",
            "dispatch",
        },
    }

    def initialize_introspection(
        self,
        config: ModelIntrospectionConfig,
    ) -> None:
        """Initialize introspection from a configuration model.

        This method accepts a typed configuration model for all introspection
        settings. Must be called during class initialization before any
        introspection operations are performed.

        Args:
            config: Configuration model containing all introspection settings.
                See ModelIntrospectionConfig for available options.

        Raises:
            ValueError: If config.node_id is not a valid UUID or config.node_type
                is not a valid EnumNodeKind member.
            TypeError: If node_type is neither EnumNodeKind nor str.

        Example:
            ```python
            from omnibase_core.enums import EnumNodeKind
            from omnibase_infra.models.discovery import ModelIntrospectionConfig

            class MyNode(MixinNodeIntrospection):
                def __init__(self, node_config):
                    config = ModelIntrospectionConfig(
                        node_id=node_config.node_id,
                        node_type=EnumNodeKind.EFFECT,
                        event_bus=node_config.event_bus,
                        version="1.2.0",
                    )
                    self.initialize_introspection(config)

            # With custom operation keywords
            class MyEffectNode(MixinNodeIntrospection):
                def __init__(self, node_config):
                    config = ModelIntrospectionConfig(
                        node_id=node_config.node_id,
                        node_type=EnumNodeKind.EFFECT,
                        event_bus=node_config.event_bus,
                        operation_keywords=frozenset({"fetch", "upload", "download"}),
                    )
                    self.initialize_introspection(config)
            ```

        See Also:
            ModelIntrospectionConfig: Configuration model with all available options.
        """
        # Note: Pydantic validates node_id is a valid UUID and node_type is EnumNodeKind

        # Configuration - extract from config model
        self._introspection_node_id = config.node_id

        # Defensive type handling for node_type: accept both EnumNodeKind and string.
        # While ModelIntrospectionConfig's validator ensures EnumNodeKind, this defensive
        # check handles edge cases like mocked configs or direct attribute access patterns.
        if isinstance(config.node_type, EnumNodeKind):
            self._introspection_node_type = config.node_type
        elif isinstance(config.node_type, str):
            # Coerce string to EnumNodeKind (handles both "effect" and "EFFECT")
            self._introspection_node_type = EnumNodeKind(config.node_type.lower())
        else:
            # Should never happen with proper ModelIntrospectionConfig, but handle gracefully
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="initialize_introspection",
                target_name=str(config.node_id),
            )
            raise ProtocolConfigurationError(
                f"node_type must be EnumNodeKind or str, got {type(config.node_type).__name__}",
                context=context,
                parameter="node_type",
                actual_type=type(config.node_type).__name__,
            )
        self._introspection_event_bus = config.event_bus
        self._introspection_version = config.version
        self._introspection_cache_ttl = config.cache_ttl

        # Capability discovery configuration - frozensets are immutable, no copy needed
        self._introspection_operation_keywords = (
            config.operation_keywords
            if config.operation_keywords is not None
            else self.DEFAULT_OPERATION_KEYWORDS
        )
        self._introspection_exclude_prefixes = (
            config.exclude_prefixes
            if config.exclude_prefixes is not None
            else self.DEFAULT_EXCLUDE_PREFIXES
        )

        # Topic configuration - extract from config model
        self._introspection_topic = config.introspection_topic
        self._heartbeat_topic = config.heartbeat_topic
        self._request_introspection_topic = config.request_introspection_topic

        # Contract for capability extraction (may be None for legacy nodes)
        self._introspection_contract = config.contract

        # State
        self._introspection_cache = None
        self._introspection_cached_at = None
        self._introspection_start_time = time.time()

        # Background tasks
        self._heartbeat_task = None
        self._registry_listener_task = None
        self._introspection_stop_event = asyncio.Event()
        self._registry_unsubscribe = None

        # Registry listener callback error tracking
        # Used for rate-limiting error logging to prevent log spam
        self._registry_callback_consecutive_failures = 0
        self._registry_callback_last_failure_time = 0.0
        # Only log every Nth consecutive failure to prevent log spam
        self._registry_callback_failure_log_threshold = 5

        # Performance metrics tracking
        self._introspection_last_metrics = None

        # Active operations tracking
        # Thread-safe counter for tracking concurrent operations
        self._active_operations = 0
        self._operations_lock = asyncio.Lock()

        if config.event_bus is None:
            logger.warning(
                f"Introspection initialized without event bus for {config.node_id}",
                extra={
                    "node_id": config.node_id,
                    "node_type": config.node_type.value
                    if hasattr(config.node_type, "value")
                    else str(config.node_type),
                },
            )

        logger.debug(
            f"Introspection initialized for {config.node_id}",
            extra={
                "node_id": config.node_id,
                "node_type": config.node_type.value
                if hasattr(config.node_type, "value")
                else str(config.node_type),
                "version": config.version,
                "cache_ttl": config.cache_ttl,
                "has_event_bus": config.event_bus is not None,
                "operation_keywords_count": len(self._introspection_operation_keywords),
                "exclude_prefixes_count": len(self._introspection_exclude_prefixes),
                "introspection_topic": self._introspection_topic,
                "heartbeat_topic": self._heartbeat_topic,
                "request_introspection_topic": self._request_introspection_topic,
            },
        )

    def _ensure_initialized(self) -> None:
        """Ensure introspection has been initialized.

        This method validates that `initialize_introspection()` was called
        before using introspection methods. It should be called at the start
        of public entry point methods.

        Raises:
            ProtocolConfigurationError: If initialize_introspection() was not called.

        Example:
            ```python
            async def get_introspection_data(self) -> ModelNodeIntrospectionEvent:
                self._ensure_initialized()
                # ... rest of method
            ```
        """
        # Use getattr with sentinel to avoid AttributeError if initialize_introspection()
        # was never called. This ensures we always raise structured error, not AttributeError.
        _not_set = object()
        node_id = getattr(self, "_introspection_node_id", _not_set)
        if node_id is _not_set or node_id is None:
            ctx = ModelInfraErrorContext(
                transport_type=EnumInfraTransportType.KAFKA,
                operation="_ensure_initialized",
                target_name="node_introspection_mixin",
            )
            raise ProtocolConfigurationError(
                "MixinNodeIntrospection not initialized. "
                "Call initialize_introspection() before using introspection methods.",
                context=ctx,
            )

    def _get_class_method_signatures(self) -> dict[str, str]:
        """Get method signatures from class-level cache.

        This method returns cached method signatures for the current class,
        populating the cache on first access. The cache is shared across all
        instances of the same class, avoiding expensive reflection operations
        on each introspection call.

        Security Note:
            This method uses Python's ``inspect`` module to extract method
            signatures, which exposes detailed type information:

            - Parameter names may reveal business logic (e.g., ``user_id``,
              ``payment_token``, ``decrypt_key``)
            - Type annotations expose internal data structures
            - Return types reveal output formats

            **Filtering Applied**:

            - Only public methods (not starting with ``_``) are included
            - Methods without inspectable signatures get ``(...)`` placeholder

            **Mitigation**:

            - Use generic parameter names for public methods
            - Prefix sensitive helper methods with ``_``

        Returns:
            Dictionary mapping public method names to signature strings.

        Note:
            The cache is populated lazily on first access and persists for
            the lifetime of the class. Use `_invalidate_class_method_cache()`
            if methods are added dynamically at runtime.

        Example:
            ```python
            # First call populates cache
            signatures = self._get_class_method_signatures()
            # {"execute": "(query: str) -> list[dict]", ...}

            # Subsequent calls return cached data
            signatures = self._get_class_method_signatures()
            ```
        """
        cls = type(self)
        if cls not in MixinNodeIntrospection._class_method_cache:
            # Populate cache for this class
            signatures: dict[str, str] = {}
            for name in dir(self):
                if name.startswith("_"):
                    continue
                attr = getattr(self, name, None)
                if callable(attr) and inspect.ismethod(attr):
                    try:
                        sig = inspect.signature(attr)
                        signatures[name] = str(sig)
                    except (ValueError, TypeError):
                        # Some methods don't have inspectable signatures
                        signatures[name] = "(...)"
            MixinNodeIntrospection._class_method_cache[cls] = signatures
        return MixinNodeIntrospection._class_method_cache[cls]

    @classmethod
    def _invalidate_class_method_cache(cls, target_class: type | None = None) -> None:
        """Invalidate the class-level method signature cache.

        Call this method when methods are dynamically added or removed from
        a class at runtime. For most use cases, this is not necessary as
        class methods are defined at class creation time.

        Args:
            target_class: Specific class to invalidate cache for.
                If None, clears cache for all classes.

        Example:
            ```python
            # Invalidate cache for a specific class
            MixinNodeIntrospection._invalidate_class_method_cache(MyNodeClass)

            # Invalidate cache for all classes
            MixinNodeIntrospection._invalidate_class_method_cache()
            ```

        Note:
            This is typically only needed in testing scenarios or when
            using dynamic method registration patterns.
        """
        if target_class is not None:
            cls._class_method_cache.pop(target_class, None)
        else:
            cls._class_method_cache.clear()

    def _should_skip_method(self, method_name: str) -> bool:
        """Check if method should be excluded from capability discovery.

        Uses the configured exclude_prefixes set for efficient prefix matching.

        Order-Dependent Matching:
            This method uses ``any()`` with a generator expression, which
            short-circuits on the first matching prefix. This means:

            - **Performance**: Prefixes earlier in the set that match common
              patterns will provide faster filtering. However, since frozenset
              has no guaranteed iteration order, this is not controllable.
            - **Correctness**: The result is deterministic regardless of order.
              A method is skipped if ANY prefix matches, so iteration order
              does not affect the outcome.

            The default exclude prefixes are: ``_``, ``get_``, ``set_``,
            ``initialize``, ``start_``, ``stop_``.

        Args:
            method_name: Name of the method to check

        Returns:
            True if method should be skipped, False otherwise
        """
        return any(
            method_name.startswith(prefix)
            for prefix in self._introspection_exclude_prefixes
        )

    def _is_operation_method(self, method_name: str) -> bool:
        """Check if method name indicates an operation.

        Uses the configured operation_keywords set to identify methods
        that represent node operations.

        Order-Dependent Matching:
            This method uses ``any()`` with a generator expression, which
            short-circuits on the first matching keyword. This means:

            - **Performance**: Keywords earlier in the set that appear more
              frequently in method names will provide faster matching. However,
              since frozenset has no guaranteed iteration order, this is not
              directly controllable.
            - **Correctness**: The result is deterministic regardless of order.
              A method is classified as an operation if ANY keyword is found
              in its lowercase name, so iteration order does not affect the
              classification outcome.

            The default operation keywords are: ``execute``, ``handle``,
            ``process``, ``run``, ``invoke``, ``call``.

            Node-type-specific keywords are available via
            ``NODE_TYPE_OPERATION_KEYWORDS`` for specialized filtering.

        Args:
            method_name: Name of the method to check

        Returns:
            True if method appears to be an operation, False otherwise
        """
        name_lower = method_name.lower()
        return any(
            keyword in name_lower for keyword in self._introspection_operation_keywords
        )

    def _has_fsm_state(self) -> bool:
        """Check if this class has FSM state management.

        Looks for common FSM state attribute patterns.

        Returns:
            True if FSM state attributes are found, False otherwise
        """
        fsm_indicators = {"_state", "current_state", "_current_state", "state"}
        return any(hasattr(self, indicator) for indicator in fsm_indicators)

    def _extract_state_value(self, state: object) -> str:
        """Extract string value from a state object.

        Handles both enum states (with .value attribute) and plain values.

        Args:
            state: The state object to extract value from

        Returns:
            String representation of the state value
        """
        if hasattr(state, "value"):
            return str(state.value)
        return str(state)

    def _get_state_from_attribute(self, attr_name: str) -> str | None:
        """Try to get state value from a named attribute.

        Args:
            attr_name: Name of the attribute to check

        Returns:
            State value as string if found and not None, None otherwise
        """
        if not hasattr(self, attr_name):
            return None
        state = getattr(self, attr_name)
        if state is None:
            return None
        return self._extract_state_value(state)

    async def _get_state_from_method(self) -> str | None:
        """Try to get state value from get_state method.

        Handles both sync and async get_state methods.

        Returns:
            State value as string if method exists and returns non-None, None otherwise
        """
        if not hasattr(self, "get_state"):
            return None

        method = self.get_state
        if not callable(method):
            return None

        try:
            result = method()
            if asyncio.iscoroutine(result):
                result = await result
            if result is None:
                return None
            return self._extract_state_value(result)
        except Exception as e:
            logger.debug(
                f"Failed to get state from get_state method: {e}",
                extra={"error": str(e)},
            )
            return None

    def _extract_event_bus_config(
        self,
        env_prefix: str,
    ) -> ModelNodeEventBusConfig | None:
        """Extract and resolve event_bus config from contract.

        Extracts topic suffixes from the contract's event_bus subcontract and
        resolves them to full environment-qualified topic strings.

        Topic Resolution:
            Contract topics are suffixes (e.g., "onex.evt.intent-classified.v1").
            This method prepends the environment prefix to create full topics
            (e.g., "dev.onex.evt.intent-classified.v1").

        Args:
            env_prefix: Environment prefix (e.g., "dev", "prod", "staging").
                Must be a valid identifier without dots or special characters.

        Returns:
            Resolved event bus config with full topic strings, or None if:
            - No contract is configured (_introspection_contract is None)
            - Contract has no event_bus subcontract
            - event_bus subcontract has no publish_topics or subscribe_topics

        Raises:
            ValueError: If topic resolution fails due to unresolved placeholders
                (e.g., "{env}" or "{namespace}" remaining in the resolved topic).
                This is a fail-fast mechanism to prevent misconfigured topics
                from being published to the registry.

        Example:
            >>> config = self._extract_event_bus_config("dev")
            >>> config.publish_topic_strings
            ['dev.onex.evt.node-registered.v1']

        See Also:
            - ModelEventBusSubcontract: Contract model with topic suffixes
            - ModelNodeEventBusConfig: Registry storage model with full topics
        """
        if self._introspection_contract is None:
            return None

        # Get event_bus subcontract if present
        event_bus_sub = getattr(self._introspection_contract, "event_bus", None)
        if event_bus_sub is None:
            return None

        # Get topic suffix lists from the subcontract
        publish_suffixes: list[str] = (
            getattr(event_bus_sub, "publish_topics", None) or []
        )
        subscribe_suffixes: list[str] = (
            getattr(event_bus_sub, "subscribe_topics", None) or []
        )

        if not publish_suffixes and not subscribe_suffixes:
            return None

        def resolve_topic(suffix: str) -> str:
            """Resolve topic suffix to full topic with env prefix."""
            # Full topic format: {env}.{suffix}
            # Strip whitespace from suffix to handle YAML formatting artifacts
            suffix = suffix.strip()

            # Fail-fast: check for unresolved placeholders BEFORE format validation
            # This provides more helpful error messages when placeholders aren't resolved
            if "{" in suffix or "}" in suffix:
                raise ValueError(
                    f"Unresolved placeholder in topic: '{suffix}'. "
                    "Ensure all placeholders like {env} or {namespace} are resolved "
                    "before topic validation."
                )

            # Validate suffix format (alphanumeric, dots, hyphens, underscores)
            if not TOPIC_NAME_PATTERN.match(suffix):
                context = ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.RUNTIME,
                    operation="_extract_event_bus_config",
                    target_name=suffix,
                )
                raise ProtocolConfigurationError(
                    f"Invalid topic suffix format: '{suffix}'. "
                    "Topics must contain only alphanumeric characters, dots, hyphens, and underscores.",
                    context=context,
                    parameter="topic_suffix",
                )

            full_topic = f"{env_prefix}.{suffix}"

            return full_topic

        def build_entry(suffix: str) -> ModelEventBusTopicEntry:
            """Build topic entry from suffix."""
            return ModelEventBusTopicEntry(
                topic=resolve_topic(suffix),
                # Metadata fields left as defaults (tooling-only)
            )

        return ModelNodeEventBusConfig(
            publish_topics=[build_entry(s) for s in publish_suffixes],
            subscribe_topics=[build_entry(s) for s in subscribe_suffixes],
        )

    async def get_capabilities(self) -> ModelDiscoveredCapabilities:
        """Extract node capabilities via reflection.

        Uses the inspect module to discover:
        - Public methods (potential operations)
        - FSM state attributes

        Method signatures are cached at the class level for performance
        optimization, as they don't change after class definition.

        Security Note:
            This method exposes information about the node's public interface.
            The returned data includes method names, parameter signatures, and
            type annotations which may reveal implementation details.

            **What Gets Exposed**:

            - Method names matching operation keywords (execute, handle, etc.)
            - Full method signatures including parameter names and types
            - Whether FSM state management is present

            **Filtering Applied**:

            - Private methods (``_`` prefix) are excluded
            - Utility methods (``get_*``, ``set_*``, ``initialize*``, etc.) are
              filtered based on ``exclude_prefixes`` configuration
            - Only methods containing configured ``operation_keywords`` are
              listed in the ``operations`` field

            **Best Practices**:

            - Review this output before production deployment
            - Use generic operation names (e.g., ``process_request`` instead of
              ``decrypt_and_forward_to_payment_gateway``)
            - Prefix sensitive internal methods with ``_``
            - Configure additional ``exclude_prefixes`` if needed

        Returns:
            ModelDiscoveredCapabilities containing:
            - operations: Tuple of public method names that may be operations
            - has_fsm: Boolean indicating if node has FSM state management
            - method_signatures: Dict of method names to signature strings

        Raises:
            ProtocolConfigurationError: If initialize_introspection() was not called.

        Example:
            ```python
            capabilities = await node.get_capabilities()
            # ModelDiscoveredCapabilities(
            #     operations=("execute", "query", "batch_execute"),
            #     has_fsm=True,
            #     method_signatures={
            #         "execute": "(query: str) -> list[dict]",
            #         ...
            #     }
            # )

            # Review exposed capabilities before production
            for op in capabilities.operations:
                print(f"Exposed operation: {op}")
            ```
        """
        self._ensure_initialized()
        start_time = time.perf_counter()

        # Get cached method signatures (class-level, computed once per class)
        # Track discovery time separately for performance analysis
        discover_start = time.perf_counter()
        cached_signatures = self._get_class_method_signatures()
        discover_elapsed_ms = (time.perf_counter() - discover_start) * 1000

        # Filter signatures and identify operations
        operations: list[str] = []
        method_signatures: dict[str, str] = {}

        for name, sig in cached_signatures.items():
            # Skip utility methods based on configured prefixes
            if self._should_skip_method(name):
                continue

            # Add method signature to filtered results
            method_signatures[name] = sig

            # Add methods that look like operations
            if self._is_operation_method(name):
                operations.append(name)

        # Build capabilities model
        capabilities = ModelDiscoveredCapabilities(
            operations=tuple(operations),
            has_fsm=self._has_fsm_state(),
            method_signatures=method_signatures,
        )

        # Performance instrumentation
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > PERF_THRESHOLD_GET_CAPABILITIES_MS:
            logger.warning(
                "Capability discovery exceeded 50ms target",
                extra={
                    "node_id": self._introspection_node_id,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "discover_elapsed_ms": round(discover_elapsed_ms, 2),
                    "method_count": len(cached_signatures),
                    "operation_count": len(operations),
                    "threshold_ms": PERF_THRESHOLD_GET_CAPABILITIES_MS,
                },
            )

        return capabilities

    async def get_endpoints(self) -> dict[str, str]:
        """Discover endpoint URLs for this node.

        Looks for common endpoint attributes and methods to build
        a dictionary of available endpoints.

        Returns:
            Dictionary mapping endpoint names to URLs.
            Common keys: health, api, metrics, readiness, liveness

        Raises:
            ProtocolConfigurationError: If initialize_introspection() was not called.

        Example:
            ```python
            endpoints = await node.get_endpoints()
            # {
            #     "health": "http://localhost:8080/health",
            #     "metrics": "http://localhost:8080/metrics",
            # }
            ```
        """
        self._ensure_initialized()
        endpoints: dict[str, str] = {}

        # Check for endpoint attributes
        endpoint_attrs = [
            ("health_url", "health"),
            ("health_endpoint", "health"),
            ("api_url", "api"),
            ("api_endpoint", "api"),
            ("metrics_url", "metrics"),
            ("metrics_endpoint", "metrics"),
            ("readiness_url", "readiness"),
            ("readiness_endpoint", "readiness"),
            ("liveness_url", "liveness"),
            ("liveness_endpoint", "liveness"),
        ]

        for attr_name, endpoint_name in endpoint_attrs:
            if hasattr(self, attr_name):
                value = getattr(self, attr_name)
                if value and isinstance(value, str):
                    endpoints[endpoint_name] = value

        # Check for endpoint methods
        endpoint_methods = [
            ("get_health_url", "health"),
            ("get_api_url", "api"),
            ("get_metrics_url", "metrics"),
        ]

        for method_name, endpoint_name in endpoint_methods:
            if hasattr(self, method_name) and endpoint_name not in endpoints:
                method = getattr(self, method_name)
                if callable(method):
                    try:
                        # Handle both sync and async methods
                        result = method()
                        if asyncio.iscoroutine(result):
                            result = await result
                        if result and isinstance(result, str):
                            endpoints[endpoint_name] = result
                    except Exception as e:
                        logger.debug(
                            f"Failed to get endpoint from {method_name}: {e}",
                            extra={"method": method_name, "error": str(e)},
                        )

        return endpoints

    async def get_current_state(self) -> str | None:
        """Get the current FSM state if applicable.

        Checks common FSM state attribute patterns and returns
        the current state value if found.

        Returns:
            Current state string if FSM state is found, None otherwise.

        Raises:
            ProtocolConfigurationError: If initialize_introspection() was not called.

        Example:
            ```python
            state = await node.get_current_state()
            # "connected" or None
            ```
        """
        self._ensure_initialized()

        # Check for state attributes in order of preference
        state_attrs = ["_state", "current_state", "_current_state", "state"]
        for attr_name in state_attrs:
            state_value = self._get_state_from_attribute(attr_name)
            if state_value is not None:
                return state_value

        # Fall back to get_state method
        return await self._get_state_from_method()

    async def get_introspection_data(self) -> ModelNodeIntrospectionEvent:
        """Get introspection data with caching support.

        Returns cached data if available and not expired, otherwise
        builds fresh introspection data and caches it.

        Performance metrics are captured for each call and stored in
        ``_introspection_last_metrics``. Use ``get_performance_metrics()``
        to retrieve the most recent metrics.

        Returns:
            ModelNodeIntrospectionEvent containing full introspection data.

        Raises:
            ProtocolConfigurationError: If initialize_introspection() was not called.

        Example:
            ```python
            data = await node.get_introspection_data()
            print(f"Node {data.node_id} has capabilities: {data.discovered_capabilities}")
            ```
        """
        self._ensure_initialized()
        total_start = time.perf_counter()
        current_time = time.time()

        # Collect metrics values in local variables (model is frozen)
        get_capabilities_ms = 0.0
        get_endpoints_ms = 0.0
        get_current_state_ms = 0.0
        method_count = 0
        slow_operations: list[str] = []

        # Check cache validity
        if (
            self._introspection_cache is not None
            and self._introspection_cached_at is not None
            and current_time - self._introspection_cached_at
            < self._introspection_cache_ttl
        ):
            # Return cached data (timestamp reflects when cache was populated, not current time)
            cached_event = ModelNodeIntrospectionEvent(**self._introspection_cache)

            # Record cache hit metrics
            elapsed_ms = (time.perf_counter() - total_start) * 1000
            threshold_exceeded = elapsed_ms > PERF_THRESHOLD_CACHE_HIT_MS
            if threshold_exceeded:
                slow_operations.append("cache_hit")

            # Create frozen metrics object with final values
            metrics = ModelIntrospectionPerformanceMetrics(
                total_introspection_ms=elapsed_ms,
                cache_hit=True,
                threshold_exceeded=threshold_exceeded,
                slow_operations=slow_operations,
            )
            self._introspection_last_metrics = metrics
            return cached_event

        # Build fresh introspection data with timing for each component
        # First, measure the class method signature discovery time separately.
        # This is cached at the class level, so subsequent calls are instant.
        discover_start = time.perf_counter()
        self._get_class_method_signatures()  # Force cache population if not already done
        discover_capabilities_ms = (time.perf_counter() - discover_start) * 1000

        cap_start = time.perf_counter()
        discovered_capabilities = await self.get_capabilities()
        get_capabilities_ms = (time.perf_counter() - cap_start) * 1000

        # Extract method count from capabilities (now a Pydantic model)
        method_count = len(discovered_capabilities.method_signatures)

        endpoints_start = time.perf_counter()
        endpoints = await self.get_endpoints()
        get_endpoints_ms = (time.perf_counter() - endpoints_start) * 1000

        state_start = time.perf_counter()
        current_state = await self.get_current_state()
        get_current_state_ms = (time.perf_counter() - state_start) * 1000

        # Get node_id and node_type with fallback logging
        # The nil UUID fallback indicates a potential initialization issue
        node_id_uuid = self._introspection_node_id
        if node_id_uuid is None:
            logger.warning(
                "Node ID not initialized, using nil UUID - "
                "ensure initialize_introspection() was called correctly",
                extra={"operation": "get_introspection_data"},
            )
            # Use nil UUID (all zeros) as sentinel for uninitialized node
            node_id_uuid = UUID("00000000-0000-0000-0000-000000000000")

        node_type = self._introspection_node_type
        if node_type is None:
            # Design Note: EnumNodeKind.EFFECT is the intended sentinel/default value
            # when node_type is uninitialized. EFFECT is chosen because:
            # 1. It's the most common node type in the ONEX ecosystem
            # 2. Effect nodes have the broadest capability expectations
            # 3. Fallback to EFFECT is safer than ORCHESTRATOR (avoids privilege escalation)
            logger.warning(
                "Node type not initialized, using EFFECT as fallback - "
                "ensure initialize_introspection() was called correctly",
                extra={
                    "node_id": str(node_id_uuid),
                    "operation": "get_introspection_data",
                },
            )
            node_type = EnumNodeKind.EFFECT

        # Extract operations count from discovered capabilities
        operations_count = len(discovered_capabilities.operations)

        # Finalize metrics calculations
        total_introspection_ms = (time.perf_counter() - total_start) * 1000
        threshold_exceeded = False

        # Check thresholds and identify slow operations
        if get_capabilities_ms > PERF_THRESHOLD_GET_CAPABILITIES_MS:
            threshold_exceeded = True
            slow_operations.append("get_capabilities")

        if total_introspection_ms > PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS:
            threshold_exceeded = True
            if "total_introspection" not in slow_operations:
                slow_operations.append("total_introspection")

        # Create frozen metrics object with final values
        metrics = ModelIntrospectionPerformanceMetrics(
            get_capabilities_ms=get_capabilities_ms,
            discover_capabilities_ms=discover_capabilities_ms,
            get_endpoints_ms=get_endpoints_ms,
            get_current_state_ms=get_current_state_ms,
            total_introspection_ms=total_introspection_ms,
            cache_hit=False,
            method_count=method_count,
            threshold_exceeded=threshold_exceeded,
            slow_operations=slow_operations,
        )

        # Store metrics for later retrieval
        self._introspection_last_metrics = metrics

        # Parse version string into ModelSemVer
        try:
            version_parts = self._introspection_version.split(".")
            node_version = ModelSemVer(
                major=int(version_parts[0]) if len(version_parts) > 0 else 1,
                minor=int(version_parts[1]) if len(version_parts) > 1 else 0,
                patch=int(version_parts[2].split("-")[0])
                if len(version_parts) > 2
                else 0,
            )
        except (ValueError, IndexError):
            # Fallback to 1.0.0 if version parsing fails
            node_version = ModelSemVer(major=1, minor=0, patch=0)

        # Extract contract capabilities if contract is available
        # This is automatic and non-skippable when contract is provided
        contract_capabilities = None
        if self._introspection_contract is not None:
            contract_capabilities = _CAPABILITY_EXTRACTOR.extract(
                self._introspection_contract
            )

        # Extract event_bus config from contract (OMN-1613)
        # Resolves topic suffixes to full environment-qualified topics
        # ValueError from _extract_event_bus_config (unresolved placeholders) is
        # wrapped in ProtocolConfigurationError for consistent error handling.
        # ProtocolConfigurationError (invalid format) propagates directly.
        event_bus_config: ModelNodeEventBusConfig | None = None
        env_prefix = os.getenv("ONEX_ENV", "dev")
        try:
            event_bus_config = self._extract_event_bus_config(env_prefix)
        except ValueError as e:
            # Wrap ValueError in ProtocolConfigurationError for fail-fast behavior
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="get_introspection_data",
                target_name="event_bus",
            )
            raise ProtocolConfigurationError(
                f"Event bus extraction failed: {e}",
                context=context,
                parameter="event_bus",
            ) from e

        # Create event with performance metrics (metrics is already Pydantic model)
        event = ModelNodeIntrospectionEvent(
            node_id=node_id_uuid,
            node_type=node_type,
            node_version=node_version,
            declared_capabilities=ModelNodeCapabilities(),
            discovered_capabilities=discovered_capabilities,
            contract_capabilities=contract_capabilities,
            endpoints=endpoints,
            current_state=current_state,
            reason=EnumIntrospectionReason.HEARTBEAT,  # cache_refresh maps to heartbeat
            correlation_id=uuid4(),
            timestamp=datetime.now(UTC),
            performance_metrics=metrics,
            event_bus=event_bus_config,
        )

        # Update cache - cast the model_dump output to our typed dict since we know
        # the structure matches (model_dump returns dict[str, Any] by default)
        self._introspection_cache = cast(
            "IntrospectionCacheDict", event.model_dump(mode="json")
        )
        self._introspection_cached_at = current_time

        # Log if any threshold was exceeded
        if metrics.threshold_exceeded:
            logger.warning(
                "Introspection exceeded performance threshold",
                extra={
                    "node_id": self._introspection_node_id,
                    "total_ms": round(metrics.total_introspection_ms, 2),
                    "get_capabilities_ms": round(metrics.get_capabilities_ms, 2),
                    "get_endpoints_ms": round(metrics.get_endpoints_ms, 2),
                    "get_current_state_ms": round(metrics.get_current_state_ms, 2),
                    "method_count": metrics.method_count,
                    "slow_operations": metrics.slow_operations,
                    "threshold_ms": PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS,
                },
            )

        logger.debug(
            f"Introspection data refreshed for {self._introspection_node_id}",
            extra={
                "node_id": self._introspection_node_id,
                "capabilities_count": operations_count,
                "endpoints_count": len(endpoints),
                "total_ms": round(metrics.total_introspection_ms, 2),
            },
        )

        return event

    async def publish_introspection(
        self,
        reason: str | EnumIntrospectionReason = EnumIntrospectionReason.STARTUP,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Publish introspection event to the event bus.

        Gracefully degrades if event bus is unavailable - logs warning
        and returns False instead of raising an exception.

        This method uses ``track_operation()`` to track active operations
        for heartbeat reporting, demonstrating the recommended pattern
        for integrating operation tracking into node operations.

        Args:
            reason: Reason for the introspection event. Can be an
                EnumIntrospectionReason or a string matching enum values
                (startup, shutdown, request, heartbeat, health_change,
                capability_change). Invalid strings default to HEARTBEAT.
            correlation_id: Optional correlation ID for tracing

        Returns:
            True if published successfully, False otherwise

        Raises:
            ProtocolConfigurationError: If initialize_introspection() was not called.

        Example:
            ```python
            # On startup (using enum - preferred)
            success = await node.publish_introspection(
                reason=EnumIntrospectionReason.STARTUP
            )

            # On shutdown (using string - backwards compatible)
            success = await node.publish_introspection(reason="shutdown")
            ```
        """
        self._ensure_initialized()

        # Convert reason to enum - check Enum first since EnumIntrospectionReason
        # inherits from str, so isinstance(..., str) would match both types.
        # Normalize string inputs with strip().lower() for robust matching.
        reason_enum: EnumIntrospectionReason
        if isinstance(reason, EnumIntrospectionReason):
            reason_enum = reason
        elif isinstance(reason, str):
            try:
                # Normalize: strip whitespace, lowercase for case-insensitive match
                reason_enum = EnumIntrospectionReason(reason.strip().lower())
            except ValueError:
                logger.warning(
                    f"Unknown introspection reason '{reason}', defaulting to HEARTBEAT",
                    extra={
                        "node_id": self._introspection_node_id,
                        "provided_reason": reason,
                    },
                )
                reason_enum = EnumIntrospectionReason.HEARTBEAT
        else:
            context = ModelInfraErrorContext.with_correlation(
                transport_type=EnumInfraTransportType.RUNTIME,
                operation="publish_introspection",
                target_name=str(self._introspection_node_id),
            )
            raise ProtocolConfigurationError(
                f"reason must be str or EnumIntrospectionReason, got {type(reason).__name__}",
                context=context,
                parameter="reason",
                actual_type=type(reason).__name__,
            )

        if self._introspection_event_bus is None:
            logger.warning(
                f"Cannot publish introspection - no event bus configured for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                    "reason": reason_enum.value,
                },
            )
            return False

        # Track this operation for heartbeat reporting
        async with self.track_operation("publish_introspection"):
            try:
                # Get introspection data
                event = await self.get_introspection_data()

                # Create publish event with updated reason and correlation_id
                # Use model_copy for clean field updates (Pydantic v2)
                final_correlation_id = correlation_id or uuid4()
                publish_event = event.model_copy(
                    update={
                        "reason": reason_enum,
                        "correlation_id": final_correlation_id,
                    }
                )

                # Publish to event bus using configured topic
                # Type narrowing: we've already checked _introspection_event_bus is not None above
                event_bus = self._introspection_event_bus
                assert event_bus is not None  # Redundant but helps mypy
                topic = self._introspection_topic
                if hasattr(event_bus, "publish_envelope"):
                    # Wrap event in ModelEventEnvelope for protocol compliance
                    envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
                        payload=publish_event,
                        correlation_id=final_correlation_id,
                    )
                    await event_bus.publish_envelope(
                        envelope=envelope,  # type: ignore[arg-type]
                        topic=topic,
                    )
                else:
                    # Fallback to publish method with raw bytes
                    event_data = publish_event.model_dump(mode="json")
                    value = json.dumps(event_data).encode("utf-8")
                    await event_bus.publish(
                        topic=topic,
                        key=str(self._introspection_node_id).encode("utf-8")
                        if self._introspection_node_id is not None
                        else None,
                        value=value,
                    )

                logger.info(
                    f"Published introspection event for {self._introspection_node_id}",
                    extra={
                        "node_id": self._introspection_node_id,
                        "reason": reason_enum.value,
                        "correlation_id": str(final_correlation_id),
                    },
                )
                return True

            except Exception as e:
                # Use error() with exc_info=True instead of exception() to include
                # structured error_type and error_message fields for log aggregation
                logger.error(
                    f"Failed to publish introspection for {self._introspection_node_id}",
                    extra={
                        "node_id": self._introspection_node_id,
                        "reason": reason_enum.value,
                        "correlation_id": str(final_correlation_id),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    exc_info=True,
                )
                return False

    async def _publish_heartbeat(self) -> bool:
        """Publish heartbeat event to the event bus.

        Internal method for heartbeat broadcasting. Calculates uptime
        and publishes heartbeat event.

        Note:
            This method intentionally does NOT use ``track_operation()``
            because:
            1. It's an internal background task, not a business operation
            2. Tracking it would cause self-referential counting (the
               heartbeat would count itself as an active operation)
            3. The purpose of operation tracking is to report business
               load, not infrastructure overhead

            For business operations, use ``track_operation()`` as
            demonstrated in ``publish_introspection()``.

        Returns:
            True if published successfully, False otherwise
        """
        if self._introspection_event_bus is None:
            return False

        # Generate correlation_id early for reliable exception logging
        heartbeat_correlation_id = uuid4()

        try:
            # Calculate uptime
            uptime_seconds = 0.0
            if self._introspection_start_time is not None:
                uptime_seconds = time.time() - self._introspection_start_time

            # Get node_id and node_type with fallback logging
            # The nil UUID fallback indicates a potential initialization issue
            node_id = self._introspection_node_id
            if node_id is None:
                logger.warning(
                    "Node ID not initialized, using nil UUID in heartbeat - "
                    "ensure initialize_introspection() was called correctly",
                    extra={
                        "operation": "_publish_heartbeat",
                        "correlation_id": str(heartbeat_correlation_id),
                    },
                )
                # Use nil UUID (all zeros) as sentinel for uninitialized node
                node_id = UUID("00000000-0000-0000-0000-000000000000")

            node_type = self._introspection_node_type
            if node_type is None:
                # Design Note: EnumNodeKind.EFFECT is the intended sentinel/default value.
                # See get_introspection_data() for detailed rationale.
                logger.warning(
                    "Node type not initialized, using EFFECT in heartbeat - "
                    "ensure initialize_introspection() was called correctly",
                    extra={
                        "node_id": str(node_id),
                        "operation": "_publish_heartbeat",
                        "correlation_id": str(heartbeat_correlation_id),
                    },
                )
                node_type = EnumNodeKind.EFFECT

            # Get current active operations count (coroutine-safe)
            async with self._operations_lock:
                active_ops_count = self._active_operations

            # Create heartbeat event
            now = datetime.now(UTC)
            heartbeat = ModelNodeHeartbeatEvent(
                node_id=node_id,
                node_type=node_type,
                uptime_seconds=uptime_seconds,
                active_operations_count=active_ops_count,
                correlation_id=heartbeat_correlation_id,
                timestamp=now,  # Required: time injection pattern
            )

            # Publish to event bus using configured topic
            # Type narrowing: we've already checked _introspection_event_bus is not None above
            event_bus = self._introspection_event_bus
            assert event_bus is not None  # Redundant but helps mypy
            topic = self._heartbeat_topic
            if hasattr(event_bus, "publish_envelope"):
                # Wrap event in ModelEventEnvelope for protocol compliance
                envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
                    payload=heartbeat,
                    correlation_id=heartbeat.correlation_id,
                )
                await event_bus.publish_envelope(
                    envelope=envelope,  # type: ignore[arg-type]
                    topic=topic,
                )
            else:
                value = json.dumps(heartbeat.model_dump(mode="json")).encode("utf-8")
                await event_bus.publish(
                    topic=topic,
                    key=str(self._introspection_node_id).encode("utf-8")
                    if self._introspection_node_id is not None
                    else None,
                    value=value,
                )

            logger.debug(
                f"Published heartbeat for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                    "correlation_id": str(heartbeat_correlation_id),
                    "uptime_seconds": uptime_seconds,
                    "active_operations": active_ops_count,
                    "topic": topic,
                },
            )
            return True

        except Exception as e:
            # Use error() with exc_info=True instead of exception() to include
            # structured error_type and error_message fields for log aggregation
            logger.error(
                f"Failed to publish heartbeat for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                    "correlation_id": str(heartbeat_correlation_id),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                },
                exc_info=True,
            )
            return False

    async def _heartbeat_loop(self, interval: float) -> None:
        """Background loop for periodic heartbeat publishing.

        Runs until stop event is set, publishing heartbeats at the
        specified interval.

        Args:
            interval: Time between heartbeats in seconds
        """
        # Ensure stop event is initialized
        if self._introspection_stop_event is None:
            self._introspection_stop_event = asyncio.Event()

        logger.info(
            f"Starting heartbeat loop for {self._introspection_node_id}",
            extra={
                "node_id": self._introspection_node_id,
                "interval_seconds": interval,
            },
        )

        while not self._introspection_stop_event.is_set():
            # Generate correlation_id for this loop iteration for traceability
            loop_correlation_id = uuid4()
            try:
                await self._publish_heartbeat()
            except asyncio.CancelledError:
                logger.debug(
                    f"Heartbeat loop cancelled for {self._introspection_node_id}",
                    extra={
                        "node_id": self._introspection_node_id,
                        "correlation_id": str(loop_correlation_id),
                    },
                )
                break
            except Exception as e:
                # Use error() with exc_info=True instead of exception() to include
                # structured error_type and error_message fields for log aggregation
                logger.error(
                    f"Error in heartbeat loop for {self._introspection_node_id}",
                    extra={
                        "node_id": self._introspection_node_id,
                        "correlation_id": str(loop_correlation_id),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                    exc_info=True,
                )

            # Wait for next interval or stop event
            try:
                await asyncio.wait_for(
                    self._introspection_stop_event.wait(),
                    timeout=interval,
                )
                # Stop event was set
                break
            except TimeoutError:
                # Normal timeout, continue loop
                pass

        logger.info(
            f"Heartbeat loop stopped for {self._introspection_node_id}",
            extra={"node_id": self._introspection_node_id},
        )

    def _parse_correlation_id(self, raw_value: str | None) -> UUID | None:
        """Parse correlation ID from request data with graceful fallback.

        Args:
            raw_value: Raw correlation_id value from request JSON

        Returns:
            Parsed UUID or None if parsing fails or value is empty
        """
        if not raw_value:
            return None

        try:
            # UUID() raises ValueError for malformed strings,
            # TypeError for non-string inputs (e.g., int, list).
            # Convert to string first for safer handling of unexpected types.
            return UUID(str(raw_value))
        except (ValueError, TypeError) as e:
            # Log warning with structured fields for monitoring.
            # Truncate received value preview to avoid log bloat
            # from potentially malicious oversized input.
            logger.warning(
                "Invalid correlation_id format in introspection "
                "request, generating new correlation_id",
                extra={
                    "node_id": self._introspection_node_id,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "received_value_type": type(raw_value).__name__,
                    "received_value_preview": str(raw_value)[:50],
                },
            )
            return None

    @staticmethod
    def _should_log_failure(consecutive_failures: int, threshold: int) -> bool:
        """Determine if failure should be logged based on rate limiting.

        Logs first failure and every Nth consecutive failure to prevent log spam.

        Args:
            consecutive_failures: Current consecutive failure count
            threshold: Log every Nth failure

        Returns:
            True if this failure should be logged at error level
        """
        return consecutive_failures == 1 or consecutive_failures % threshold == 0

    async def _cleanup_registry_subscription(
        self, correlation_id: UUID | None = None
    ) -> None:
        """Clean up the current registry subscription.

        Args:
            correlation_id: Optional correlation ID for traceability in logs.
                If not provided, a new one will be generated.
        """
        if self._registry_unsubscribe is not None:
            cleanup_correlation_id = correlation_id or uuid4()
            try:
                result = self._registry_unsubscribe()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as cleanup_error:
                logger.debug(
                    "Error unsubscribing registry listener for "
                    f"{self._introspection_node_id}",
                    extra={
                        "node_id": self._introspection_node_id,
                        "correlation_id": str(cleanup_correlation_id),
                        "error_type": type(cleanup_error).__name__,
                        "error_message": str(cleanup_error),
                    },
                )
            self._registry_unsubscribe = None

    async def _handle_introspection_request(
        self, message: ProtocolEventMessage
    ) -> None:
        """Handle incoming introspection request.

        Includes error recovery with rate-limited logging to prevent
        log spam during sustained failures. Continues processing on
        non-fatal errors to maintain graceful degradation.

        Args:
            message: The incoming event message (implements ProtocolEventMessage protocol)
        """
        # Generate correlation_id for this request for traceability
        request_correlation_id = uuid4()
        try:
            await self._process_introspection_request(message)
            # Reset failure counter on success
            self._registry_callback_consecutive_failures = 0
        except Exception as e:
            self._handle_request_error(e, request_correlation_id)

    async def _process_introspection_request(
        self, message: ProtocolEventMessage
    ) -> None:
        """Process the introspection request message.

        Args:
            message: The incoming event message

        Raises:
            Exception: If processing fails (will be caught by caller)
        """
        # Early exit if message has no parseable value
        if not hasattr(message, "value") or not message.value:
            await self.publish_introspection(
                reason="request",
                correlation_id=uuid4(),
            )
            return

        # Parse request data
        request_data = json.loads(message.value.decode("utf-8"))

        # Check if request targets a specific node (early exit if not us)
        # Note: Compare as strings since target_node_id from JSON is a string
        # while _introspection_node_id is a UUID object
        target_node_id = request_data.get("target_node_id")
        if target_node_id and str(target_node_id) != str(self._introspection_node_id):
            return

        # Parse correlation ID with graceful fallback
        correlation_id = self._parse_correlation_id(request_data.get("correlation_id"))

        # Respond with introspection data
        await self.publish_introspection(
            reason="request",
            correlation_id=correlation_id,
        )

    def _handle_request_error(self, error: Exception, correlation_id: UUID) -> None:
        """Handle error during introspection request processing.

        Tracks consecutive failures and rate-limits error logging.

        Args:
            error: The exception that occurred
            correlation_id: Correlation ID for traceability in logs
        """
        # Track consecutive failures for rate-limited logging
        self._registry_callback_consecutive_failures += 1
        self._registry_callback_last_failure_time = time.time()

        # Rate-limit error logging to prevent log spam during sustained failures
        if self._should_log_failure(
            self._registry_callback_consecutive_failures,
            self._registry_callback_failure_log_threshold,
        ):
            logger.error(
                f"Error handling introspection request for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "consecutive_failures": self._registry_callback_consecutive_failures,
                    "log_rate_limited": self._registry_callback_consecutive_failures
                    > 1,
                },
                exc_info=True,
            )
        else:
            # Log at debug level for rate-limited failures
            logger.debug(
                f"Suppressed error log for introspection request "
                f"(failure {self._registry_callback_consecutive_failures})",
                extra={
                    "node_id": self._introspection_node_id,
                    "correlation_id": str(correlation_id),
                    "error_type": type(error).__name__,
                    "consecutive_failures": self._registry_callback_consecutive_failures,
                },
            )

    async def _attempt_subscription(self) -> bool:
        """Attempt to subscribe to the request introspection topic.

        Returns:
            True if subscribed successfully and should wait for stop signal,
            False if subscription not supported or failed

        Note:
            This method should only be called when event bus is verified to exist.
            The caller (_registry_listener_loop) checks for None before calling.
        """
        event_bus = self._introspection_event_bus
        if event_bus is None or not hasattr(event_bus, "subscribe"):
            logger.warning(
                "Event bus does not support subscribe for "
                f"{self._introspection_node_id}",
                extra={"node_id": self._introspection_node_id},
            )
            return False

        request_topic = self._request_introspection_topic
        unsubscribe = await event_bus.subscribe(
            topic=request_topic,
            group_id=f"introspection-{self._introspection_node_id}",
            on_message=self._handle_introspection_request,
        )
        self._registry_unsubscribe = unsubscribe

        logger.info(
            f"Registry listener subscribed for {self._introspection_node_id}",
            extra={
                "node_id": self._introspection_node_id,
                "topic": request_topic,
            },
        )
        return True

    async def _wait_for_backoff_or_stop(self, backoff_seconds: float) -> bool:
        """Wait for backoff period or stop signal.

        Args:
            backoff_seconds: Time to wait in seconds

        Returns:
            True if stop signal received, False if timeout (should retry)

        Note:
            This method should only be called when stop_event is verified to exist.
            The caller (_registry_listener_loop) initializes the event before calling.
        """
        stop_event = self._introspection_stop_event
        if stop_event is None:
            # Should not happen if called correctly, but handle gracefully
            return False

        try:
            await asyncio.wait_for(
                stop_event.wait(),
                timeout=backoff_seconds,
            )
            # Stop signal received during backoff
            return True
        except TimeoutError:
            # Normal timeout, continue to retry
            return False

    async def _registry_listener_loop(
        self,
        max_retries: int = 3,
        base_backoff_seconds: float = 1.0,
    ) -> None:
        """Background loop listening for REQUEST_INTROSPECTION events.

        Subscribes to the request_introspection topic and responds
        with introspection data when requests are received. Includes
        retry logic with exponential backoff for subscription failures.

        Security Note:
            This method subscribes to the ``node.request_introspection`` Kafka
            topic and responds with full introspection data to any request.
            This creates a network-accessible endpoint for capability discovery.

            **Network Exposure**:

            - Any consumer on the Kafka cluster can request introspection data
            - Responses are published to ``node.introspection`` topic
            - No authentication is performed on incoming requests

            **Multi-tenant Considerations**:

            - Configure Kafka topic ACLs to restrict access to introspection
              topics in multi-tenant environments
            - Consider whether introspection topics should be accessible
              outside the cluster boundary
            - Monitor topic consumers for unauthorized access patterns
            - Use separate Kafka clusters for different security domains

            **Request Validation**:

            - The ``target_node_id`` field allows filtering requests to
              specific nodes - only matching requests are processed
            - Malformed requests are handled gracefully without crashing
            - Correlation IDs are validated but invalid IDs don't block
              processing

        Args:
            max_retries: Maximum subscription retry attempts (default: 3)
            base_backoff_seconds: Base backoff time for exponential retry
        """
        if self._introspection_event_bus is None:
            logger.warning(
                f"Cannot start registry listener - no event bus for {self._introspection_node_id}",
                extra={"node_id": self._introspection_node_id},
            )
            return

        # Ensure stop event is initialized
        if self._introspection_stop_event is None:
            self._introspection_stop_event = asyncio.Event()

        logger.info(
            f"Starting registry listener for {self._introspection_node_id}",
            extra={"node_id": self._introspection_node_id},
        )

        # Retry loop with exponential backoff for subscription failures
        retry_count = 0
        while not self._introspection_stop_event.is_set():
            # Generate correlation_id for this subscription attempt for traceability
            subscription_correlation_id = uuid4()
            try:
                if await self._attempt_subscription():
                    # Wait for stop signal
                    await self._introspection_stop_event.wait()
                # Exit loop after subscription ends or not supported
                break

            except asyncio.CancelledError:
                logger.debug(
                    f"Registry listener cancelled for {self._introspection_node_id}",
                    extra={
                        "node_id": self._introspection_node_id,
                        "correlation_id": str(subscription_correlation_id),
                    },
                )
                break
            except Exception as e:
                retry_count += 1
                if not await self._handle_subscription_error(
                    e,
                    retry_count,
                    max_retries,
                    base_backoff_seconds,
                    subscription_correlation_id,
                ):
                    break

        # Final cleanup
        await self._cleanup_registry_subscription()

        logger.info(
            f"Registry listener stopped for {self._introspection_node_id}",
            extra={"node_id": self._introspection_node_id},
        )

    async def _handle_subscription_error(
        self,
        error: Exception,
        retry_count: int,
        max_retries: int,
        base_backoff_seconds: float,
        correlation_id: UUID,
    ) -> bool:
        """Handle subscription error with retry logic.

        Args:
            error: The exception that occurred
            retry_count: Current retry attempt number
            max_retries: Maximum retry attempts
            base_backoff_seconds: Base backoff time for exponential retry
            correlation_id: Correlation ID for traceability in logs

        Returns:
            True if should continue retrying, False if should stop
        """
        logger.error(
            f"Error in registry listener for {self._introspection_node_id}",
            extra={
                "node_id": self._introspection_node_id,
                "correlation_id": str(correlation_id),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "retry_count": retry_count,
                "max_retries": max_retries,
            },
            exc_info=True,
        )

        # Clean up any partial subscription before retry
        await self._cleanup_registry_subscription(correlation_id)

        # Check if we should retry
        if retry_count >= max_retries:
            logger.error(
                "Registry listener exhausted retries",
                extra={
                    "node_id": self._introspection_node_id,
                    "correlation_id": str(correlation_id),
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                exc_info=True,
            )
            return False

        # Exponential backoff before retry
        backoff = base_backoff_seconds * (2 ** (retry_count - 1))
        logger.info(
            f"Registry listener retrying in {backoff}s for "
            f"{self._introspection_node_id}",
            extra={
                "node_id": self._introspection_node_id,
                "backoff_seconds": backoff,
                "retry_count": retry_count,
            },
        )

        # Wait for backoff period or stop signal
        if await self._wait_for_backoff_or_stop(backoff):
            return False  # Stop signal received

        return True  # Continue retrying

    async def start_introspection_tasks(
        self,
        enable_heartbeat: bool = True,
        heartbeat_interval_seconds: float = 30.0,
        enable_registry_listener: bool = True,
    ) -> None:
        """Start background introspection tasks.

        Starts the heartbeat loop and/or registry listener as background
        tasks. Safe to call multiple times - won't start duplicate tasks.

        Args:
            enable_heartbeat: Whether to start the heartbeat loop
            heartbeat_interval_seconds: Interval between heartbeats in seconds
            enable_registry_listener: Whether to start the registry listener

        Raises:
            ProtocolConfigurationError: If initialize_introspection() was not called.

        Example:
            ```python
            await node.start_introspection_tasks(
                enable_heartbeat=True,
                heartbeat_interval_seconds=30.0,
                enable_registry_listener=True,
            )
            ```
        """
        self._ensure_initialized()
        # Reset stop event if previously set
        if self._introspection_stop_event is None:
            self._introspection_stop_event = asyncio.Event()
        elif self._introspection_stop_event.is_set():
            self._introspection_stop_event.clear()

        # Start heartbeat task if enabled and not running
        if enable_heartbeat and self._heartbeat_task is None:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(heartbeat_interval_seconds),
                name=f"heartbeat-{self._introspection_node_id}",
            )
            logger.debug(
                f"Started heartbeat task for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                    "interval": heartbeat_interval_seconds,
                },
            )

        # Start registry listener if enabled and not running
        if enable_registry_listener and self._registry_listener_task is None:
            self._registry_listener_task = asyncio.create_task(
                self._registry_listener_loop(),
                name=f"registry-listener-{self._introspection_node_id}",
            )
            logger.debug(
                f"Started registry listener task for {self._introspection_node_id}",
                extra={"node_id": self._introspection_node_id},
            )

    async def start_introspection_tasks_from_config(
        self,
        config: ModelIntrospectionTaskConfig,
    ) -> None:
        """Start background introspection tasks from a configuration model.

        This method provides an alternative to ``start_introspection_tasks()``
        using a configuration model instead of individual parameters. This
        reduces union types in calling code and follows ONEX patterns.

        Args:
            config: Configuration model containing task settings.
                See ModelIntrospectionTaskConfig for available options.

        Raises:
            ProtocolConfigurationError: If initialize_introspection() was not called.

        Example:
            ```python
            from omnibase_infra.models.discovery import ModelIntrospectionTaskConfig

            class MyNode(MixinNodeIntrospection):
                async def startup(self):
                    config = ModelIntrospectionTaskConfig(
                        enable_heartbeat=True,
                        heartbeat_interval_seconds=15.0,
                        enable_registry_listener=True,
                    )
                    await self.start_introspection_tasks_from_config(config)

            # Using defaults
            class SimpleNode(MixinNodeIntrospection):
                async def startup(self):
                    config = ModelIntrospectionTaskConfig()
                    await self.start_introspection_tasks_from_config(config)
            ```

        See Also:
            start_introspection_tasks: Original method with parameters.
            ModelIntrospectionTaskConfig: Configuration model with all options.
        """
        await self.start_introspection_tasks(
            enable_heartbeat=config.enable_heartbeat,
            heartbeat_interval_seconds=config.heartbeat_interval_seconds,
            enable_registry_listener=config.enable_registry_listener,
        )

    async def stop_introspection_tasks(self) -> None:
        """Stop all background introspection tasks.

        Signals tasks to stop and waits for clean shutdown.
        Safe to call multiple times.

        Example:
            ```python
            await node.stop_introspection_tasks()
            ```
        """
        logger.info(
            f"Stopping introspection tasks for {self._introspection_node_id}",
            extra={"node_id": self._introspection_node_id},
        )

        # Signal tasks to stop
        if self._introspection_stop_event is not None:
            self._introspection_stop_event.set()

        # Cancel and wait for heartbeat task
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Cancel and wait for registry listener task
        if self._registry_listener_task is not None:
            self._registry_listener_task.cancel()
            try:
                await self._registry_listener_task
            except asyncio.CancelledError:
                pass
            self._registry_listener_task = None

        logger.info(
            f"Introspection tasks stopped for {self._introspection_node_id}",
            extra={"node_id": self._introspection_node_id},
        )

    def invalidate_introspection_cache(self) -> None:
        """Invalidate the introspection cache.

        Call this when node capabilities change to ensure fresh
        data is reported on next introspection request.

        Example:
            ```python
            node.register_new_handler(handler)
            node.invalidate_introspection_cache()
            ```
        """
        self._introspection_cache = None
        self._introspection_cached_at = None
        logger.debug(
            f"Introspection cache invalidated for {self._introspection_node_id}",
            extra={"node_id": self._introspection_node_id},
        )

    def get_performance_metrics(self) -> ModelIntrospectionPerformanceMetrics | None:
        """Get the most recent performance metrics from introspection operations.

        Returns the performance metrics captured during the last call to
        ``get_introspection_data()``. Use this to monitor introspection
        performance and detect when operations exceed the <50ms threshold.

        Returns:
            ModelIntrospectionPerformanceMetrics if introspection has been called,
            None if no introspection has been performed yet.

        Example:
            ```python
            # After calling introspection
            await node.get_introspection_data()

            # Check performance metrics
            metrics = node.get_performance_metrics()
            if metrics and metrics.threshold_exceeded:
                logger.warning(
                    "Slow introspection detected",
                    extra={
                        "slow_operations": metrics.slow_operations,
                        "total_ms": metrics.total_introspection_ms,
                    }
                )

            # Access individual timings
            if metrics:
                print(f"Total time: {metrics.total_introspection_ms:.2f}ms")
                print(f"Cache hit: {metrics.cache_hit}")
                print(f"Methods discovered: {metrics.method_count}")
            ```
        """
        return self._introspection_last_metrics

    @asynccontextmanager
    async def track_operation(
        self,
        operation_name: str | None = None,
    ) -> AsyncIterator[None]:
        """Context manager for tracking active operations.

        Provides coroutine-safe tracking of concurrent operations for
        heartbeat reporting. Increments the active operations counter
        on entry and decrements it on exit (whether successful or not).

        Concurrency Safety:
            Uses asyncio.Lock for coroutine-safe counter updates.
            The lock is held only during counter updates, not during
            the operation itself. Logging occurs AFTER lock release
            to prevent blocking during I/O.

        Error Handling:
            Counter updates are protected with try/except to ensure
            operation tracking failures don't affect the main operation.
            The counter will never go negative due to atomic operations.

        Args:
            operation_name: Optional name for logging/debugging.
                Not used for counter logic but useful for diagnostics.

        Yields:
            None. The context manager is used purely for side effects.

        Example:
            ```python
            class MyNode(MixinNodeIntrospection):
                async def execute_query(self, query: str) -> Result:
                    async with self.track_operation("execute_query"):
                        # This operation is now tracked in heartbeats
                        return await self._database.execute(query)

                async def process_batch(self, items: list[Item]) -> None:
                    # Track multiple concurrent operations
                    async with asyncio.TaskGroup() as tg:
                        for item in items:
                            tg.create_task(self._process_with_tracking(item))

                async def _process_with_tracking(self, item: Item) -> None:
                    async with self.track_operation("process_item"):
                        await self._process_single(item)
            ```

        Note:
            The counter is read by ``_publish_heartbeat()`` to report
            the current number of active operations. This provides
            visibility into node load for monitoring and scaling.
        """
        # Increment counter on entry - capture count inside lock, log outside
        count_after_increment = 0
        increment_succeeded = False
        try:
            async with self._operations_lock:
                self._active_operations += 1
                count_after_increment = self._active_operations
            increment_succeeded = True
        except Exception as e:
            # Log but don't fail the operation
            logger.warning(
                f"Failed to increment operation counter: {e}",
                extra={
                    "node_id": self._introspection_node_id,
                    "operation": operation_name,
                    "error_type": type(e).__name__,
                },
            )

        # Log AFTER releasing lock to prevent blocking during I/O
        if increment_succeeded and operation_name:
            logger.debug(
                f"Operation started: {operation_name}",
                extra={
                    "node_id": self._introspection_node_id,
                    "operation": operation_name,
                    "active_operations": count_after_increment,
                },
            )

        try:
            yield
        finally:
            # Decrement counter on exit - capture state inside lock, log outside
            count_after_decrement = 0
            decrement_succeeded = False
            counter_was_zero = False
            try:
                async with self._operations_lock:
                    # Prevent negative counter (defensive check)
                    if self._active_operations > 0:
                        self._active_operations -= 1
                    else:
                        counter_was_zero = True
                    count_after_decrement = self._active_operations
                decrement_succeeded = True
            except Exception as e:
                # Log but don't fail the operation
                logger.warning(
                    f"Failed to decrement operation counter: {e}",
                    extra={
                        "node_id": self._introspection_node_id,
                        "operation": operation_name,
                        "error_type": type(e).__name__,
                    },
                )

            # Log AFTER releasing lock to prevent blocking during I/O
            if decrement_succeeded:
                if counter_was_zero:
                    # This should never happen, but log if it does
                    logger.warning(
                        "Active operations counter already at zero during decrement",
                        extra={
                            "node_id": self._introspection_node_id,
                            "operation": operation_name,
                        },
                    )
                elif operation_name:
                    logger.debug(
                        f"Operation completed: {operation_name}",
                        extra={
                            "node_id": self._introspection_node_id,
                            "operation": operation_name,
                            "active_operations": count_after_decrement,
                        },
                    )

    async def get_active_operations_count(self) -> int:
        """Get the current count of active operations.

        Returns the number of operations currently being tracked via
        ``track_operation()``. This is the same value reported in
        heartbeat events.

        Concurrency Safety:
            Uses asyncio.Lock for coroutine-safe counter access.
            The returned value is a snapshot; concurrent operations
            may change the count immediately after reading.

        Returns:
            Current number of active operations (>= 0).

        Example:
            ```python
            count = await node.get_active_operations_count()
            if count > threshold:
                logger.warning(f"High operation load: {count} active")
            ```
        """
        async with self._operations_lock:
            return self._active_operations


__all__ = [
    "PERF_THRESHOLD_CACHE_HIT_MS",
    "PERF_THRESHOLD_DISCOVER_CAPABILITIES_MS",
    "PERF_THRESHOLD_GET_CAPABILITIES_MS",
    "PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS",
    "DiscoveredCapabilitiesCacheDict",  # TypedDict for cached discovered capabilities
    "IntrospectionCacheDict",
    "MixinNodeIntrospection",
    "ModelIntrospectionPerformanceMetrics",
    "PerformanceMetricsCacheDict",  # TypedDict for cached performance metrics
]
