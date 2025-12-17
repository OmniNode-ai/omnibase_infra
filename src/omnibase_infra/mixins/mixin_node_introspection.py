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
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

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

# Type alias for introspection cache structure
# The cache stores JSON-serializable data from ModelNodeIntrospectionEvent.model_dump()
IntrospectionCacheValue = str | int | float | bool | list[str] | dict[str, str]

# Type alias for capabilities dictionary structure
# operations: list of method names, protocols: list of protocol names
# has_fsm: boolean, method_signatures: dict of method name to signature string
CapabilitiesDict = dict[str, list[str] | bool | dict[str, str]]


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
                # Node operation
                ...
        ```
    """

    # Caching attributes (class-level defaults, instance overrides in initialize)
    _introspection_cache: dict[str, IntrospectionCacheValue] | None = None
    _introspection_cache_ttl: float = 300.0  # 5 minutes
    _introspection_cached_at: float | None = None

    # Background task attributes
    _heartbeat_task: asyncio.Task[None] | None = None
    _registry_listener_task: asyncio.Task[None] | None = None
    _introspection_stop_event: asyncio.Event | None = None
    _registry_unsubscribe: Callable[[], None] | Callable[[], Awaitable[None]] | None = (
        None
    )

    # Configuration attributes
    _introspection_node_id: str | None = None
    _introspection_node_type: str | None = None
    _introspection_event_bus: ProtocolEventBus | None = None
    _introspection_version: str = "1.0.0"
    _introspection_start_time: float | None = None

    def initialize_introspection(
        self,
        node_id: str,
        node_type: str,
        event_bus: ProtocolEventBus | None = None,
        version: str = "1.0.0",
        cache_ttl: float = 300.0,
    ) -> None:
        """Initialize introspection configuration.

        Must be called during class initialization before any introspection
        operations are performed.

        Args:
            node_id: Unique identifier for this node instance
            node_type: Node type classification (EFFECT, COMPUTE, REDUCER, ORCHESTRATOR)
            event_bus: Optional event bus for publishing introspection events.
                Must have `publish_envelope()` method if provided.
            version: Node version string (default: "1.0.0")
            cache_ttl: Cache time-to-live in seconds (default: 300.0)

        Raises:
            ValueError: If node_id or node_type is empty

        Example:
            ```python
            class MyNode(MixinNodeIntrospection):
                def __init__(self, config):
                    self.initialize_introspection(
                        node_id=config.node_id,
                        node_type="EFFECT",
                        event_bus=config.event_bus,
                        version="1.2.0",
                    )
            ```
        """
        if not node_id:
            raise ValueError("node_id cannot be empty")
        if not node_type:
            raise ValueError("node_type cannot be empty")

        # Configuration
        self._introspection_node_id = node_id
        self._introspection_node_type = node_type
        self._introspection_event_bus = event_bus
        self._introspection_version = version
        self._introspection_cache_ttl = cache_ttl

        # State
        self._introspection_cache = None
        self._introspection_cached_at = None
        self._introspection_start_time = time.time()

        # Background tasks
        self._heartbeat_task = None
        self._registry_listener_task = None
        self._introspection_stop_event = asyncio.Event()
        self._registry_unsubscribe = None

        if event_bus is None:
            logger.warning(
                f"Introspection initialized without event bus for {node_id}",
                extra={
                    "node_id": node_id,
                    "node_type": node_type,
                },
            )

        logger.debug(
            f"Introspection initialized for {node_id}",
            extra={
                "node_id": node_id,
                "node_type": node_type,
                "version": version,
                "cache_ttl": cache_ttl,
                "has_event_bus": event_bus is not None,
            },
        )

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

    async def get_capabilities(self) -> CapabilitiesDict:
        """Extract node capabilities via reflection.

        Uses the inspect module to discover:
        - Public methods (potential operations)
        - Protocol implementations
        - FSM state attributes

        Returns:
            Dictionary containing:
            - operations: List of public method names that may be operations
            - protocols: List of protocol/interface names implemented
            - has_fsm: Boolean indicating if node has FSM state management
            - method_signatures: Dict of method names to signature strings

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
            ```
        """
        capabilities: CapabilitiesDict = {
            "operations": [],
            "protocols": [],
            "has_fsm": False,
            "method_signatures": {},
        }

        # Discover operations from public methods
        operation_keywords = {"execute", "handle", "process", "run", "invoke", "call"}
        exclude_prefixes = {"_", "get_", "set_", "initialize", "start_", "stop_"}

        # Get the operations and method_signatures as mutable lists/dicts
        operations: list[str] = []
        method_signatures: dict[str, str] = {}

        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            # Skip private/special methods
            if name.startswith("_"):
                continue

            # Skip common utility methods
            skip = False
            for prefix in exclude_prefixes:
                if name.startswith(prefix):
                    skip = True
                    break
            if skip:
                continue

            # Add methods that look like operations
            is_operation = any(
                keyword in name.lower() for keyword in operation_keywords
            )
            if is_operation or name in {"execute", "handle", "process"}:
                operations.append(name)

            # Capture method signature
            try:
                sig = inspect.signature(method)
                method_signatures[name] = str(sig)
            except (ValueError, TypeError):
                # Some methods don't have inspectable signatures
                method_signatures[name] = "(...)"

        # Discover protocols from base classes
        protocols: list[str] = []
        for base in type(self).__mro__:
            base_name = base.__name__
            if base_name.startswith(("Protocol", "Mixin")):
                protocols.append(base_name)

        # Check for FSM state attributes
        has_fsm = False
        fsm_indicators = {"_state", "current_state", "_current_state", "state"}
        for indicator in fsm_indicators:
            if hasattr(self, indicator):
                has_fsm = True
                break

        # Assign to capabilities dict
        capabilities["operations"] = operations
        capabilities["protocols"] = protocols
        capabilities["has_fsm"] = has_fsm
        capabilities["method_signatures"] = method_signatures

        return capabilities

    async def get_endpoints(self) -> dict[str, str]:
        """Discover endpoint URLs for this node.

        Looks for common endpoint attributes and methods to build
        a dictionary of available endpoints.

        Returns:
            Dictionary mapping endpoint names to URLs.
            Common keys: health, api, metrics, readiness, liveness

        Example:
            ```python
            endpoints = await node.get_endpoints()
            # {
            #     "health": "http://localhost:8080/health",
            #     "metrics": "http://localhost:8080/metrics",
            # }
            ```
        """
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

        Example:
            ```python
            state = await node.get_current_state()
            # "connected" or None
            ```
        """
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
        current_time = time.time()

        # Check cache validity
        if (
            self._introspection_cache is not None
            and self._introspection_cached_at is not None
            and current_time - self._introspection_cached_at
            < self._introspection_cache_ttl
        ):
            # Return cached data with updated timestamp
            cached_event = ModelNodeIntrospectionEvent(**self._introspection_cache)
            return cached_event

        # Build fresh introspection data
        capabilities = await self.get_capabilities()
        endpoints = await self.get_endpoints()
        current_state = await self.get_current_state()

        event = ModelNodeIntrospectionEvent(
            node_id=self._introspection_node_id or "unknown",
            node_type=self._introspection_node_type or "unknown",
            capabilities=capabilities,
            endpoints=endpoints,
            current_state=current_state,
            version=self._introspection_version,
            reason="cache_refresh",
            correlation_id=uuid4(),
        )

        # Update cache
        self._introspection_cache = event.model_dump(mode="json")  # type: ignore[assignment]
        self._introspection_cached_at = current_time

        logger.debug(
            f"Introspection data refreshed for {self._introspection_node_id}",
            extra={
                "node_id": self._introspection_node_id,
                "capabilities_count": len(capabilities.get("operations", [])),  # type: ignore[arg-type]
                "endpoints_count": len(endpoints),
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
                    key=self._introspection_node_id.encode("utf-8")
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

        except Exception:
            logger.exception(
                f"Failed to publish introspection for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                    "reason": reason,
                },
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

            # Create heartbeat event
            heartbeat = ModelNodeHeartbeatEvent(
                node_id=self._introspection_node_id or "unknown",
                node_type=self._introspection_node_type or "unknown",
                uptime_seconds=uptime_seconds,
                active_operations_count=0,  # Could be extended to track actual operations
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
                    key=self._introspection_node_id.encode("utf-8")
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

        except Exception:
            logger.exception(
                f"Failed to publish heartbeat for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                },
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
            except Exception:
                logger.exception(
                    f"Error in heartbeat loop for {self._introspection_node_id}",
                    extra={
                        "node_id": self._introspection_node_id,
                    },
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

    async def _registry_listener_loop(self) -> None:
        """Background loop listening for REQUEST_INTROSPECTION events.

        Subscribes to the request_introspection topic and responds
        with introspection data when requests are received.
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

        async def on_request(message: ModelEventMessage) -> None:
            """Handle incoming introspection request."""
            try:
                # Parse request to check if it targets this node
                if hasattr(message, "value") and message.value:
                    request_data = json.loads(message.value.decode("utf-8"))
                    target_node_id = request_data.get("target_node_id")

                    # If request has a target and it's not us, ignore
                    if target_node_id and target_node_id != self._introspection_node_id:
                        return

                    correlation_id_str = request_data.get("correlation_id")
                    correlation_id: UUID | None = None
                    if correlation_id_str:
                        try:
                            correlation_id = UUID(correlation_id_str)
                        except (ValueError, AttributeError):
                            logger.warning(
                                "Invalid correlation_id format in introspection request",
                                extra={
                                    "node_id": self._introspection_node_id,
                                    "received_correlation_id": str(correlation_id_str)[
                                        :50
                                    ],  # Truncate for safety
                                },
                            )
                            correlation_id = None
                else:
                    correlation_id = None

                # Respond with introspection data
                await self.publish_introspection(
                    reason="request",
                    correlation_id=correlation_id,
                )

            except Exception:
                logger.exception(
                    f"Error handling introspection request for {self._introspection_node_id}",
                    extra={
                        "node_id": self._introspection_node_id,
                    },
                )

        try:
            # Subscribe to request topic
            if hasattr(self._introspection_event_bus, "subscribe"):
                unsubscribe = await self._introspection_event_bus.subscribe(
                    topic=REQUEST_INTROSPECTION_TOPIC,
                    group_id=f"introspection-{self._introspection_node_id}",
                    on_message=on_request,
                )
                self._registry_unsubscribe = unsubscribe

                # Wait for stop signal
                await self._introspection_stop_event.wait()

        except asyncio.CancelledError:
            logger.debug(
                f"Registry listener cancelled for {self._introspection_node_id}",
                extra={"node_id": self._introspection_node_id},
            )
        except Exception:
            logger.exception(
                f"Error in registry listener for {self._introspection_node_id}",
                extra={
                    "node_id": self._introspection_node_id,
                },
            )
        finally:
            # Clean up subscription
            if self._registry_unsubscribe is not None:
                try:
                    result = self._registry_unsubscribe()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.debug(
                        f"Error unsubscribing registry listener: {e}",
                        extra={"error": str(e)},
                    )
                self._registry_unsubscribe = None

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


__all__ = [
    "MixinNodeIntrospection",
    "INTROSPECTION_TOPIC",
    "HEARTBEAT_TOPIC",
    "REQUEST_INTROSPECTION_TOPIC",
]
