# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
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
    - active_operations_count in heartbeats is currently hardcoded to 0.
      Full implementation deferred - see TODO in _publish_heartbeat().

Security Considerations:
    This mixin uses Python reflection (via the ``inspect`` module) to automatically
    discover node capabilities. While this enables powerful service discovery, it
    has security implications that developers must understand.

    **What Gets Exposed via Introspection**:

    - Public method names (potential operations a node can perform)
    - Method signatures (parameter names and type annotations)
    - Protocol and mixin implementations (discovered capabilities)
    - FSM state information (if ``MixinFSM`` or similar state attributes are present)
    - Endpoint URLs (health, API, metrics paths)
    - Node metadata (name, version, type)

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

    **Best Practices for Node Developers**:

    - Prefix internal/sensitive methods with ``_`` to exclude them from introspection.
    - Avoid exposing sensitive business logic in public method names (e.g., use
      ``process_request`` instead of ``decrypt_and_forward_to_payment_gateway``).
    - Review exposed capabilities before deploying to production environments.
    - Consider network segmentation for introspection event topics in multi-tenant
      environments.
    - Use the ``exclude_prefixes`` parameter to filter additional method patterns
      if needed.

    **Network Security Considerations**:

    - Introspection data is published to Kafka topics (``node.introspection``).
    - In multi-tenant environments, ensure proper topic ACLs are configured.
    - Consider whether introspection topics should be accessible outside the cluster.
    - Monitor introspection topic consumers for unauthorized access.

    For more details, see the "Node Introspection Security Considerations" section
    in ``CLAUDE.md``.

Usage:
    ```python
    from omnibase_infra.mixins import MixinNodeIntrospection

    class MyNode(MixinNodeIntrospection):
        def __init__(self, config, event_bus=None):
            self.initialize_introspection(
                node_id=config.node_id,
                node_type="EFFECT",
                event_bus=event_bus,
            )

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
    1. Call `initialize_introspection()` during initialization
    2. Optionally call `start_introspection_tasks()` for background operations
    3. Call `stop_introspection_tasks()` during shutdown
    4. Ensure event_bus has `publish_envelope()` method if provided

See Also:
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
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, TypedDict, cast
from uuid import UUID, uuid4

from omnibase_infra.mixins.model_introspection_config import ModelIntrospectionConfig
from omnibase_infra.models.discovery import ModelNodeIntrospectionEvent
from omnibase_infra.models.registration import ModelNodeHeartbeatEvent

if TYPE_CHECKING:
    from omnibase_core.protocols.event_bus import ProtocolEventBus

    from omnibase_infra.event_bus.models import ModelEventMessage

logger = logging.getLogger(__name__)

# Event topic constants
INTROSPECTION_TOPIC = "node.introspection"
HEARTBEAT_TOPIC = "node.heartbeat"
REQUEST_INTROSPECTION_TOPIC = "node.request_introspection"

# Type alias for capabilities dictionary structure
# operations: list of method names, protocols: list of protocol names
# has_fsm: boolean, method_signatures: dict of method name to signature string
CapabilitiesDict = dict[str, list[str] | bool | dict[str, str]]

# Performance threshold constants (in milliseconds)
PERF_THRESHOLD_GET_CAPABILITIES_MS = 50.0
PERF_THRESHOLD_DISCOVER_CAPABILITIES_MS = 30.0
PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS = 50.0
PERF_THRESHOLD_CACHE_HIT_MS = 1.0


@dataclass
class IntrospectionPerformanceMetrics:
    """Performance metrics for introspection operations.

    This dataclass captures timing information for introspection operations,
    enabling performance monitoring and alerting when operations exceed
    the <50ms target threshold.

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

    Example:
        ```python
        metrics = node.get_performance_metrics()
        if metrics.threshold_exceeded:
            logger.warning(
                "Introspection performance degraded",
                extra={
                    "slow_operations": metrics.slow_operations,
                    "total_ms": metrics.total_introspection_ms,
                }
            )
        ```
    """

    get_capabilities_ms: float = 0.0
    discover_capabilities_ms: float = 0.0
    get_endpoints_ms: float = 0.0
    get_current_state_ms: float = 0.0
    total_introspection_ms: float = 0.0
    cache_hit: bool = False
    method_count: int = 0
    threshold_exceeded: bool = False
    slow_operations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | bool | int | list[str]]:
        """Convert metrics to dictionary for logging/serialization.

        Returns:
            Dictionary with all metric fields.
        """
        return {
            "get_capabilities_ms": self.get_capabilities_ms,
            "discover_capabilities_ms": self.discover_capabilities_ms,
            "get_endpoints_ms": self.get_endpoints_ms,
            "get_current_state_ms": self.get_current_state_ms,
            "total_introspection_ms": self.total_introspection_ms,
            "cache_hit": self.cache_hit,
            "method_count": self.method_count,
            "threshold_exceeded": self.threshold_exceeded,
            "slow_operations": self.slow_operations,
        }


class IntrospectionCacheDict(TypedDict):
    """TypedDict representing the JSON-serialized ModelNodeIntrospectionEvent.

    This type matches the output of ModelNodeIntrospectionEvent.model_dump(mode="json"),
    enabling proper type checking for cache operations without requiring type: ignore comments.

    Note:
        The capabilities field uses a permissive type to accommodate all possible
        values from model_dump(), including nested structures and dynamic values.
        The actual structure is: operations (list[str]), protocols (list[str]),
        has_fsm (bool), method_signatures (dict[str, str]).
    """

    node_id: str
    node_type: str
    # Permissive type for capabilities to handle all model_dump() output variations
    # Actual structure: {"operations": list, "protocols": list, "has_fsm": bool, "method_signatures": dict}
    capabilities: dict[str, list[str] | bool | dict[str, str] | int | float | None]
    endpoints: dict[str, str]
    current_state: str | None
    version: str
    reason: str
    correlation_id: str | None  # UUID serializes to string in JSON mode
    timestamp: str  # datetime serializes to ISO string in JSON mode


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

        **Exposed Information**:

        - Public method names (potential operations a node can perform)
        - Method signatures (parameter names and type annotations)
        - Protocol and mixin implementations (discovered capabilities)
        - FSM state information (if state attributes are present)
        - Endpoint URLs (health, API, metrics paths)
        - Node metadata (name, version, type)

        **Built-in Protections**:

        - Private methods (prefixed with ``_``) are excluded by default
        - Utility method prefixes (``get_*``, ``set_*``, etc.) are filtered
        - Only methods containing operation keywords are reported as operations
        - Configure ``exclude_prefixes`` in ``initialize_introspection()`` for
          additional filtering

        **Recommendations for Production**:

        - Prefix internal/sensitive methods with ``_`` to exclude them
        - Use generic operation names that don't reveal implementation details
        - Review ``get_capabilities()`` output before production deployment
        - In multi-tenant environments, configure Kafka topic ACLs for
          introspection events (``node.introspection``, ``node.heartbeat``,
          ``node.request_introspection``)
        - Monitor introspection topic consumers for unauthorized access
        - Consider network segmentation for introspection event topics

    See Also:
        - Module docstring for detailed security documentation
        - CLAUDE.md "Node Introspection Security Considerations" section
        - ``get_capabilities()`` for filtering logic details

    Example:
        ```python
        class PostgresAdapter(MixinNodeIntrospection):
            def __init__(self, config):
                self.initialize_introspection(
                    node_id="postgres-adapter-001",
                    node_type="EFFECT",
                    event_bus=config.event_bus,
                )

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
    _introspection_node_type: str | None
    _introspection_event_bus: ProtocolEventBus | None
    _introspection_version: str
    _introspection_start_time: float | None

    # Capability discovery configuration
    _introspection_operation_keywords: set[str]
    _introspection_exclude_prefixes: set[str]

    # Registry listener callback error tracking (instance-level)
    # Used for rate-limiting error logging to prevent log spam during
    # sustained failures. These are initialized in initialize_introspection().
    _registry_callback_consecutive_failures: int
    _registry_callback_last_failure_time: float
    _registry_callback_failure_log_threshold: int

    # Performance metrics tracking (instance-level)
    # Stores the most recent performance metrics from introspection operations
    _introspection_last_metrics: IntrospectionPerformanceMetrics | None

    # Default operation keywords for capability discovery
    DEFAULT_OPERATION_KEYWORDS: ClassVar[set[str]] = {
        "execute",
        "handle",
        "process",
        "run",
        "invoke",
        "call",
    }

    # Default prefixes to exclude from capability discovery
    DEFAULT_EXCLUDE_PREFIXES: ClassVar[set[str]] = {
        "_",
        "get_",
        "set_",
        "initialize",
        "start_",
        "stop_",
    }

    # Node-type-specific operation keyword suggestions
    NODE_TYPE_OPERATION_KEYWORDS: ClassVar[dict[str, set[str]]] = {
        "EFFECT": {
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
        "COMPUTE": {
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
        "REDUCER": {
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
        "ORCHESTRATOR": {
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

    def initialize_introspection_from_config(
        self,
        config: ModelIntrospectionConfig,
    ) -> None:
        """Initialize introspection from a configuration model.

        This is the preferred initialization method that accepts a typed
        configuration model for all introspection settings.

        Must be called during class initialization before any introspection
        operations are performed.

        Args:
            config: Introspection configuration model containing all settings.

        Raises:
            ValueError: If node_id or node_type is empty (validated by model).

        Example:
            ```python
            from uuid import uuid4

            from omnibase_infra.mixins import (
                MixinNodeIntrospection,
                ModelIntrospectionConfig,
            )

            class MyNode(MixinNodeIntrospection):
                def __init__(self, node_config, event_bus=None):
                    introspection_config = ModelIntrospectionConfig(
                        node_id=uuid4(),
                        node_type="EFFECT",
                        event_bus=event_bus,
                        version="1.2.0",
                    )
                    self.initialize_introspection_from_config(introspection_config)

            # With custom operation keywords
            class MyEffectNode(MixinNodeIntrospection):
                def __init__(self, node_config, event_bus=None):
                    introspection_config = ModelIntrospectionConfig(
                        node_id=uuid4(),
                        node_type="EFFECT",
                        event_bus=event_bus,
                        operation_keywords={"fetch", "upload", "download"},
                    )
                    self.initialize_introspection_from_config(introspection_config)
            ```
        """
        # Configuration from model
        self._introspection_node_id = config.node_id
        self._introspection_node_type = config.node_type
        self._introspection_event_bus = config.event_bus
        self._introspection_version = config.version
        self._introspection_cache_ttl = config.cache_ttl

        # Capability discovery configuration - use copies to avoid mutation
        self._introspection_operation_keywords = (
            config.operation_keywords.copy()
            if config.operation_keywords is not None
            else self.DEFAULT_OPERATION_KEYWORDS.copy()
        )
        self._introspection_exclude_prefixes = (
            config.exclude_prefixes.copy()
            if config.exclude_prefixes is not None
            else self.DEFAULT_EXCLUDE_PREFIXES.copy()
        )

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

        if config.event_bus is None:
            logger.warning(
                f"Introspection initialized without event bus for {config.node_id}",
                extra={
                    "node_id": str(config.node_id),
                    "node_type": config.node_type,
                },
            )

        logger.debug(
            f"Introspection initialized for {config.node_id}",
            extra={
                "node_id": str(config.node_id),
                "node_type": config.node_type,
                "version": config.version,
                "cache_ttl": config.cache_ttl,
                "has_event_bus": config.event_bus is not None,
                "operation_keywords_count": len(self._introspection_operation_keywords),
                "exclude_prefixes_count": len(self._introspection_exclude_prefixes),
            },
        )

    def initialize_introspection(
        self,
        node_id: UUID,
        node_type: str,
        event_bus: ProtocolEventBus | None = None,
        version: str = "1.0.0",
        cache_ttl: float = 300.0,
        operation_keywords: set[str] | None = None,
        exclude_prefixes: set[str] | None = None,
    ) -> None:
        """Initialize introspection configuration (legacy interface).

        This method provides backward compatibility. For new code, prefer using
        :meth:`initialize_introspection_from_config` with a
        :class:`ModelIntrospectionConfig` instance.

        Must be called during class initialization before any introspection
        operations are performed.

        Args:
            node_id: Unique identifier for this node instance (UUID)
            node_type: Node type classification (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
            event_bus: Optional event bus for publishing introspection events.
                Must have ``publish_envelope()`` method if provided.
            version: Node version string (default: "1.0.0")
            cache_ttl: Cache time-to-live in seconds (default: 300.0)
            operation_keywords: Optional set of keywords to identify operation methods.
                Methods containing these keywords are reported as operations.
                If None, uses DEFAULT_OPERATION_KEYWORDS.
            exclude_prefixes: Optional set of prefixes to exclude from capability
                discovery. Methods starting with these prefixes are filtered out.
                If None, uses DEFAULT_EXCLUDE_PREFIXES.

        Raises:
            ValueError: If node_id or node_type is empty

        Example:
            ```python
            from uuid import uuid4

            class MyNode(MixinNodeIntrospection):
                def __init__(self, config):
                    self.initialize_introspection(
                        node_id=uuid4(),
                        node_type="EFFECT",
                        event_bus=config.event_bus,
                        version="1.2.0",
                    )

            # With custom operation keywords
            class MyEffectNode(MixinNodeIntrospection):
                def __init__(self, config):
                    self.initialize_introspection(
                        node_id=uuid4(),
                        node_type="EFFECT",
                        event_bus=config.event_bus,
                        operation_keywords={"fetch", "upload", "download"},
                    )
            ```

        See Also:
            :meth:`initialize_introspection_from_config`: Preferred config-based initialization.
        """
        # Validate required fields (matching model validation)
        if not node_id:
            raise ValueError("node_id cannot be empty")
        if not node_type:
            raise ValueError("node_type cannot be empty")

        # Create config model and delegate to config-based initialization
        config = ModelIntrospectionConfig(
            node_id=node_id,
            node_type=node_type,
            event_bus=event_bus,
            version=version,
            cache_ttl=cache_ttl,
            operation_keywords=operation_keywords,
            exclude_prefixes=exclude_prefixes,
        )
        self.initialize_introspection_from_config(config)

    def _ensure_initialized(self) -> None:
        """Ensure introspection has been initialized.

        This method validates that `initialize_introspection()` was called
        before using introspection methods. It should be called at the start
        of public entry point methods.

        Raises:
            RuntimeError: If initialize_introspection() was not called.

        Example:
            ```python
            async def get_introspection_data(self) -> ModelNodeIntrospectionEvent:
                self._ensure_initialized()
                # ... rest of method
            ```
        """
        if self._introspection_node_id is None:
            raise RuntimeError(
                "MixinNodeIntrospection not initialized. "
                "Call initialize_introspection() before using introspection methods."
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

        Args:
            method_name: Name of the method to check

        Returns:
            True if method appears to be an operation, False otherwise
        """
        name_lower = method_name.lower()
        return any(
            keyword in name_lower for keyword in self._introspection_operation_keywords
        )

    def _discover_protocols(self) -> list[str]:
        """Discover protocol and mixin implementations from class hierarchy.

        Security Note:
            This method exposes the inheritance hierarchy by returning class
            names that start with ``Protocol`` or ``Mixin``. This reveals what
            capabilities the node implements (e.g., ``ProtocolDatabaseAdapter``,
            ``MixinAsyncCircuitBreaker``). This information is generally safe
            to expose as it describes the node's public interface contracts,
            but be aware that it may reveal architectural decisions.

        Returns:
            List of protocol and mixin class names implemented by this class
        """
        protocols: list[str] = []
        for base in type(self).__mro__:
            base_name = base.__name__
            if base_name.startswith(("Protocol", "Mixin")):
                protocols.append(base_name)
        return protocols

    def _has_fsm_state(self) -> bool:
        """Check if this class has FSM state management.

        Looks for common FSM state attribute patterns.

        Returns:
            True if FSM state attributes are found, False otherwise
        """
        fsm_indicators = {"_state", "current_state", "_current_state", "state"}
        return any(hasattr(self, indicator) for indicator in fsm_indicators)

    async def get_capabilities(self) -> CapabilitiesDict:
        """Extract node capabilities via reflection.

        Uses the inspect module to discover:
        - Public methods (potential operations)
        - Protocol implementations
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
            - Protocol/mixin class names from the inheritance hierarchy
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
            Dictionary containing:
            - operations: List of public method names that may be operations
            - protocols: List of protocol/interface names implemented
            - has_fsm: Boolean indicating if node has FSM state management
            - method_signatures: Dict of method names to signature strings

        Raises:
            RuntimeError: If initialize_introspection() was not called.

        Example:
            ```python
            capabilities = await node.get_capabilities()
            # {
            #     "operations": ["execute", "query", "batch_execute"],
            #     "protocols": ["ProtocolDatabaseAdapter"],
            #     "has_fsm": True,
            #     "method_signatures": {
            #         "execute": "(query: str) -> list[dict]",
            #         ...
            #     }
            # }

            # Review exposed capabilities before production
            for op in capabilities["operations"]:
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

        # Build capabilities dict
        capabilities: CapabilitiesDict = {
            "operations": operations,
            "protocols": self._discover_protocols(),
            "has_fsm": self._has_fsm_state(),
            "method_signatures": method_signatures,
        }

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
            RuntimeError: If initialize_introspection() was not called.

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
            RuntimeError: If initialize_introspection() was not called.

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
            if hasattr(self, attr_name):
                state = getattr(self, attr_name)
                if state is not None:
                    # Handle enum states
                    if hasattr(state, "value"):
                        return str(state.value)
                    return str(state)

        # Check for get_state method
        if hasattr(self, "get_state"):
            method = self.get_state  # type: ignore[attr-defined]
            if callable(method):
                try:
                    result = method()
                    if asyncio.iscoroutine(result):
                        result = await result
                    if result is not None:
                        if hasattr(result, "value"):
                            return str(result.value)
                        return str(result)
                except Exception as e:
                    logger.debug(
                        f"Failed to get state from get_state method: {e}",
                        extra={"error": str(e)},
                    )

        return None

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
            RuntimeError: If initialize_introspection() was not called.

        Example:
            ```python
            data = await node.get_introspection_data()
            print(f"Node {data.node_id} has capabilities: {data.capabilities}")
            ```
        """
        self._ensure_initialized()
        total_start = time.perf_counter()
        current_time = time.time()

        # Initialize metrics for this call
        metrics = IntrospectionPerformanceMetrics()

        # Check cache validity
        if (
            self._introspection_cache is not None
            and self._introspection_cached_at is not None
            and current_time - self._introspection_cached_at
            < self._introspection_cache_ttl
        ):
            # Return cached data with updated timestamp
            cached_event = ModelNodeIntrospectionEvent(**self._introspection_cache)

            # Record cache hit metrics
            elapsed_ms = (time.perf_counter() - total_start) * 1000
            metrics.total_introspection_ms = elapsed_ms
            metrics.cache_hit = True

            # Check cache hit threshold
            if elapsed_ms > PERF_THRESHOLD_CACHE_HIT_MS:
                metrics.threshold_exceeded = True
                metrics.slow_operations.append("cache_hit")

            self._introspection_last_metrics = metrics
            return cached_event

        # Build fresh introspection data with timing for each component
        cap_start = time.perf_counter()
        capabilities = await self.get_capabilities()
        metrics.get_capabilities_ms = (time.perf_counter() - cap_start) * 1000

        # Extract method count from capabilities
        method_sigs = capabilities.get("method_signatures", {})
        metrics.method_count = len(method_sigs) if isinstance(method_sigs, dict) else 0

        endpoints_start = time.perf_counter()
        endpoints = await self.get_endpoints()
        metrics.get_endpoints_ms = (time.perf_counter() - endpoints_start) * 1000

        state_start = time.perf_counter()
        current_state = await self.get_current_state()
        metrics.get_current_state_ms = (time.perf_counter() - state_start) * 1000

        # Get node_id and node_type with fallback logging
        # The nil UUID fallback indicates a potential initialization issue
        node_id = self._introspection_node_id
        if node_id is None:
            logger.warning(
                "Node ID not initialized, using nil UUID - "
                "ensure initialize_introspection() was called correctly",
                extra={"operation": "get_introspection_data"},
            )
            # Use nil UUID (all zeros) as sentinel for uninitialized node
            node_id = UUID("00000000-0000-0000-0000-000000000000")

        node_type = self._introspection_node_type
        if node_type is None:
            logger.warning(
                "Node type not initialized, using 'unknown' - "
                "ensure initialize_introspection() was called correctly",
                extra={"node_id": node_id, "operation": "get_introspection_data"},
            )
            node_type = "unknown"

        event = ModelNodeIntrospectionEvent(
            node_id=node_id,
            node_type=node_type,
            capabilities=capabilities,
            endpoints=endpoints,
            current_state=current_state,
            version=self._introspection_version,
            reason="cache_refresh",
            correlation_id=uuid4(),
        )

        # Update cache - cast the model_dump output to our typed dict since we know
        # the structure matches (model_dump returns dict[str, Any] by default)
        self._introspection_cache = cast(
            IntrospectionCacheDict, event.model_dump(mode="json")
        )
        self._introspection_cached_at = current_time

        # Extract operations list with proper type narrowing
        operations_value = capabilities.get("operations", [])
        operations_count = (
            len(operations_value) if isinstance(operations_value, list) else 0
        )

        # Finalize metrics
        metrics.total_introspection_ms = (time.perf_counter() - total_start) * 1000
        metrics.cache_hit = False

        # Check thresholds and identify slow operations
        if metrics.get_capabilities_ms > PERF_THRESHOLD_GET_CAPABILITIES_MS:
            metrics.threshold_exceeded = True
            metrics.slow_operations.append("get_capabilities")

        if metrics.total_introspection_ms > PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS:
            metrics.threshold_exceeded = True
            if "total_introspection" not in metrics.slow_operations:
                metrics.slow_operations.append("total_introspection")

        # Store metrics for later retrieval
        self._introspection_last_metrics = metrics

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
        reason: str = "startup",
        correlation_id: UUID | None = None,
    ) -> bool:
        """Publish introspection event to the event bus.

        Gracefully degrades if event bus is unavailable - logs warning
        and returns False instead of raising an exception.

        Args:
            reason: Reason for the introspection event
                (startup, shutdown, request, heartbeat)
            correlation_id: Optional correlation ID for tracing

        Returns:
            True if published successfully, False otherwise

        Raises:
            RuntimeError: If initialize_introspection() was not called.

        Example:
            ```python
            # On startup
            success = await node.publish_introspection(reason="startup")

            # On shutdown
            success = await node.publish_introspection(reason="shutdown")
            ```
        """
        self._ensure_initialized()
        if self._introspection_event_bus is None:
            logger.warning(
                f"Cannot publish introspection - no event bus configured for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                    "reason": reason,
                },
            )
            return False

        try:
            # Get introspection data
            event = await self.get_introspection_data()

            # Create publish event with updated reason and correlation_id
            # Use model_copy for clean field updates (Pydantic v2)
            final_correlation_id = correlation_id or uuid4()
            publish_event = event.model_copy(
                update={
                    "reason": reason,
                    "correlation_id": final_correlation_id,
                }
            )

            # Publish to event bus
            if hasattr(self._introspection_event_bus, "publish_envelope"):
                await self._introspection_event_bus.publish_envelope(  # type: ignore[union-attr]
                    envelope=publish_event,
                    topic=INTROSPECTION_TOPIC,
                )
            else:
                # Fallback to publish method with raw bytes
                event_data = publish_event.model_dump(mode="json")
                value = json.dumps(event_data).encode("utf-8")
                await self._introspection_event_bus.publish(
                    topic=INTROSPECTION_TOPIC,
                    key=str(self._introspection_node_id).encode("utf-8")
                    if self._introspection_node_id
                    else None,
                    value=value,
                )

            logger.info(
                f"Published introspection event for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                    "reason": reason,
                    "correlation_id": str(final_correlation_id),
                },
            )
            return True

        except Exception as e:
            # Use error() with exc_info=True instead of exception() to include
            # structured error_type and error_message fields for log aggregation
            logger.error(  # noqa: G201
                f"Failed to publish introspection for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                    "reason": reason,
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

        Returns:
            True if published successfully, False otherwise
        """
        if self._introspection_event_bus is None:
            return False

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
                    extra={"operation": "_publish_heartbeat"},
                )
                # Use nil UUID (all zeros) as sentinel for uninitialized node
                node_id = UUID("00000000-0000-0000-0000-000000000000")

            node_type = self._introspection_node_type
            if node_type is None:
                logger.warning(
                    "Node type not initialized, using 'unknown' in heartbeat - "
                    "ensure initialize_introspection() was called correctly",
                    extra={"node_id": str(node_id), "operation": "_publish_heartbeat"},
                )
                node_type = "unknown"

            # Create heartbeat event
            heartbeat = ModelNodeHeartbeatEvent(
                node_id=node_id,
                node_type=node_type,
                uptime_seconds=uptime_seconds,
                # TODO(OMN-XXX): Implement active operation tracking
                # Currently hardcoded to 0. Full implementation requires:
                # - Operation counter increment/decrement around async operations
                # - Thread-safe counter for concurrent operations
                # - Integration with node's actual execution context
                # This is intentionally left as 0 until the tracking infrastructure
                # is implemented. See MixinNodeIntrospection docstring note.
                active_operations_count=0,
                correlation_id=uuid4(),
            )

            # Publish to event bus
            if hasattr(self._introspection_event_bus, "publish_envelope"):
                await self._introspection_event_bus.publish_envelope(  # type: ignore[union-attr]
                    envelope=heartbeat,
                    topic=HEARTBEAT_TOPIC,
                )
            else:
                value = json.dumps(heartbeat.model_dump(mode="json")).encode("utf-8")
                await self._introspection_event_bus.publish(
                    topic=HEARTBEAT_TOPIC,
                    key=str(self._introspection_node_id).encode("utf-8")
                    if self._introspection_node_id
                    else None,
                    value=value,
                )

            logger.debug(
                f"Published heartbeat for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                    "uptime_seconds": uptime_seconds,
                },
            )
            return True

        except Exception as e:
            # Use error() with exc_info=True instead of exception() to include
            # structured error_type and error_message fields for log aggregation
            logger.error(  # noqa: G201
                f"Failed to publish heartbeat for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
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
            try:
                await self._publish_heartbeat()
            except asyncio.CancelledError:
                logger.debug(
                    f"Heartbeat loop cancelled for {self._introspection_node_id}",
                    extra={"node_id": self._introspection_node_id},
                )
                break
            except Exception as e:
                # Use error() with exc_info=True instead of exception() to include
                # structured error_type and error_message fields for log aggregation
                logger.error(  # noqa: G201
                    f"Error in heartbeat loop for {self._introspection_node_id}",
                    extra={
                        "node_id": self._introspection_node_id,
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

        def _parse_correlation_id(raw_value: str | None) -> UUID | None:
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

        async def on_request(message: ModelEventMessage) -> None:
            """Handle incoming introspection request.

            Includes error recovery with rate-limited logging to prevent
            log spam during sustained failures. Continues processing on
            non-fatal errors to maintain graceful degradation.
            """
            try:
                # Early exit if message has no parseable value
                if not hasattr(message, "value") or not message.value:
                    await self.publish_introspection(
                        reason="request",
                        correlation_id=None,
                    )
                    self._registry_callback_consecutive_failures = 0
                    return

                # Parse request data
                request_data = json.loads(message.value.decode("utf-8"))

                # Check if request targets a specific node (early exit if not us)
                target_node_id = request_data.get("target_node_id")
                if target_node_id and target_node_id != self._introspection_node_id:
                    return

                # Parse correlation ID with graceful fallback
                correlation_id = _parse_correlation_id(
                    request_data.get("correlation_id")
                )

                # Respond with introspection data
                await self.publish_introspection(
                    reason="request",
                    correlation_id=correlation_id,
                )

                # Reset failure counter on success
                self._registry_callback_consecutive_failures = 0

            except Exception as e:
                # Track consecutive failures for rate-limited logging
                self._registry_callback_consecutive_failures += 1
                self._registry_callback_last_failure_time = time.time()

                # Rate-limit error logging to prevent log spam during sustained failures
                if _should_log_failure(
                    self._registry_callback_consecutive_failures,
                    self._registry_callback_failure_log_threshold,
                ):
                    logger.error(  # noqa: G201
                        f"Error handling introspection request for {self._introspection_node_id}",
                        extra={
                            "node_id": self._introspection_node_id,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "consecutive_failures": (
                                self._registry_callback_consecutive_failures
                            ),
                            "log_rate_limited": (
                                self._registry_callback_consecutive_failures > 1
                            ),
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
                            "error_type": type(e).__name__,
                            "consecutive_failures": (
                                self._registry_callback_consecutive_failures
                            ),
                        },
                    )

                # Continue processing - graceful degradation
                # The callback should not raise exceptions that would disrupt the listener

        # Helper function to clean up subscription
        async def cleanup_subscription() -> None:
            """Clean up the current subscription."""
            if self._registry_unsubscribe is not None:
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
                            "error_type": type(cleanup_error).__name__,
                            "error_message": str(cleanup_error),
                        },
                    )
                self._registry_unsubscribe = None

        # Retry loop with exponential backoff for subscription failures
        retry_count = 0
        while not self._introspection_stop_event.is_set():
            try:
                # Subscribe to request topic
                if hasattr(self._introspection_event_bus, "subscribe"):
                    unsubscribe = await self._introspection_event_bus.subscribe(
                        topic=REQUEST_INTROSPECTION_TOPIC,
                        group_id=f"introspection-{self._introspection_node_id}",
                        on_message=on_request,
                    )
                    self._registry_unsubscribe = unsubscribe

                    # Reset retry count on successful subscription
                    retry_count = 0

                    logger.info(
                        f"Registry listener subscribed for {self._introspection_node_id}",
                        extra={
                            "node_id": self._introspection_node_id,
                            "topic": REQUEST_INTROSPECTION_TOPIC,
                        },
                    )

                    # Wait for stop signal
                    await self._introspection_stop_event.wait()
                    # Stop signal received, exit loop
                    break

                logger.warning(
                    "Event bus does not support subscribe for "
                    f"{self._introspection_node_id}",
                    extra={"node_id": self._introspection_node_id},
                )
                break

            except asyncio.CancelledError:
                logger.debug(
                    f"Registry listener cancelled for {self._introspection_node_id}",
                    extra={"node_id": self._introspection_node_id},
                )
                break
            except Exception as e:
                retry_count += 1
                logger.error(  # noqa: G201
                    f"Error in registry listener for {self._introspection_node_id}",
                    extra={
                        "node_id": self._introspection_node_id,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "retry_count": retry_count,
                        "max_retries": max_retries,
                    },
                    exc_info=True,
                )

                # Clean up any partial subscription before retry
                await cleanup_subscription()

                # Check if we should retry
                if retry_count >= max_retries:
                    # Use error() with exc_info=True instead of exception()
                    # to include structured error_type and error_message fields
                    # for log aggregation
                    logger.error(  # noqa: G201
                        "Registry listener exhausted retries",
                        extra={
                            "node_id": self._introspection_node_id,
                            "retry_count": retry_count,
                            "max_retries": max_retries,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        },
                        exc_info=True,
                    )
                    break

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
                try:
                    await asyncio.wait_for(
                        self._introspection_stop_event.wait(),
                        timeout=backoff,
                    )
                    # Stop signal received during backoff
                    break
                except TimeoutError:
                    # Normal timeout, continue to retry
                    pass

        # Final cleanup
        await cleanup_subscription()

        logger.info(
            f"Registry listener stopped for {self._introspection_node_id}",
            extra={"node_id": self._introspection_node_id},
        )

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
            RuntimeError: If initialize_introspection() was not called.

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

    def get_performance_metrics(self) -> IntrospectionPerformanceMetrics | None:
        """Get the most recent performance metrics from introspection operations.

        Returns the performance metrics captured during the last call to
        ``get_introspection_data()``. Use this to monitor introspection
        performance and detect when operations exceed the <50ms threshold.

        Returns:
            IntrospectionPerformanceMetrics if introspection has been called,
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


__all__ = [
    "MixinNodeIntrospection",
    "INTROSPECTION_TOPIC",
    "HEARTBEAT_TOPIC",
    "REQUEST_INTROSPECTION_TOPIC",
    "CapabilitiesDict",
    "IntrospectionCacheDict",
    "IntrospectionPerformanceMetrics",
    "PERF_THRESHOLD_GET_CAPABILITIES_MS",
    "PERF_THRESHOLD_DISCOVER_CAPABILITIES_MS",
    "PERF_THRESHOLD_GET_INTROSPECTION_DATA_MS",
    "PERF_THRESHOLD_CACHE_HIT_MS",
]
