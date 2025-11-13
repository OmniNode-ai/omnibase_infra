"""
Node Introspection Mixin (Performance Optimized)

Provides introspection capabilities for ONEX nodes with caching for production performance.
Fixed to prevent dynamic imports in hot path and optimize FSM state discovery.

Performance Optimizations:
- Module-level caching for expensive imports
- Lazy loading with memoization
- Optimized reflection with caching
- Reduced object creation overhead
"""

import asyncio
import logging
import os
import time
from abc import ABCMeta
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

# Constants for introspection timing and caching
CACHE_TTL_SECONDS = 60  # General cache TTL
CURRENT_STATE_CACHE_TTL_SECONDS = 30  # Cache TTL for current state
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 30  # Default heartbeat interval
REGISTRY_LISTENER_INTERVAL_SECONDS = 60  # Registry listener check interval


class IntrospectionEnvelope:
    """Simple envelope object for introspection data."""

    def __init__(
        self,
        node_id: str,
        node_type: str,
        timestamp: str,
        reason: str,
        data: dict[str, Any],
        correlation_id: Optional[str] = None,
        network_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        epoch: Optional[int] = None,
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.timestamp = timestamp
        self.reason = reason
        self.data = data
        self.correlation_id = correlation_id
        # Network topology metadata (Phase 1a MVP)
        self.network_id = network_id
        self.deployment_id = deployment_id
        self.epoch = epoch


# Performance optimization: Cache psutil import and availability
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None

# Performance optimization: Module-level cache for enum imports
_ENUM_CACHE: dict[str, Any] = {}
_SUPPORTED_STATES_CACHE: dict[str, list[str]] = {}


def _get_enum_cached(enum_path: str) -> Optional[type]:
    """
    Get enum class with module-level caching to avoid repeated imports.

    Args:
        enum_path: Full module path to enum class

    Returns:
        Enum class or None if import fails
    """
    if enum_path in _ENUM_CACHE:
        return _ENUM_CACHE[enum_path]

    try:
        # Parse module path and class name
        module_path, class_name = enum_path.rsplit(".", 1)

        # Import module
        import importlib

        module = importlib.import_module(module_path)

        # Get enum class
        enum_class = getattr(module, class_name)

        # Cache the result
        _ENUM_CACHE[enum_path] = enum_class

        return enum_class

    except (ImportError, AttributeError) as e:
        logger.debug(f"Failed to import enum {enum_path}: {e}")
        _ENUM_CACHE[enum_path] = None
        return None


def _get_supported_states_cached(node_type: str) -> list[str]:
    """
    Get supported states with caching to avoid repeated enum access.

    Args:
        node_type: Type of node (orchestrator, reducer, etc.)

    Returns:
        List of supported state values
    """
    cache_key = f"supported_states_{node_type}"

    if cache_key in _SUPPORTED_STATES_CACHE:
        return _SUPPORTED_STATES_CACHE[cache_key]

    supported_states = []

    if node_type == "orchestrator":
        enum_class = _get_enum_cached(
            "omninode_bridge.nodes.orchestrator.v1_0_0.models.enum_workflow_state.EnumWorkflowState"
        )
        if enum_class:
            supported_states = [state.value for state in enum_class]
    elif node_type == "reducer":
        enum_class = _get_enum_cached(
            "omninode_bridge.nodes.reducer.v1_0_0.models.enum_aggregation_state.EnumAggregationState"
        )
        if enum_class:
            supported_states = [state.value for state in enum_class]

    # Cache the result
    _SUPPORTED_STATES_CACHE[cache_key] = supported_states
    return supported_states


class IntrospectionMixin(metaclass=ABCMeta):
    """
    Mixin providing introspection capabilities for ONEX nodes (Performance Optimized).

    This mixin allows nodes to automatically discover and report their capabilities,
    configuration, and operational state. It's designed to be mixed into ONEX node
    classes to provide comprehensive introspection functionality.

    Performance Optimizations:
    ===========================
    1. **Cached Imports**: Enum imports are cached at module level to prevent repeated imports
    2. **Lazy Loading**: Heavy operations are only performed when needed
    3. **Memoization**: Results of expensive operations are cached
    4. **Optimized Reflection**: Reduced attribute access overhead
    5. **Efficient State Discovery**: FSM state discovery is optimized with caching

    Key Features:
    - Automatic capability detection (endpoints, operations, resources)
    - FSM state introspection for workflow nodes
    - Performance metrics and resource monitoring
    - Dynamic configuration discovery
    - Health check integration
    """

    _startup_time: datetime | float

    def __init__(self, *args, **kwargs):
        """
        Initialize introspection mixin.
        """
        super().__init__(*args, **kwargs)

        # Performance optimization: Cache for expensive computations
        self._introspection_cache: dict[str, Any] = {}
        self._cache_timestamps: dict[str, float] = {}
        self._cache_ttl_seconds = 300  # 5 minutes cache TTL

        # Cache node type to avoid repeated detection
        self._cached_node_type: Optional[str] = None
        # Track last introspection broadcast time
        self._last_introspection_broadcast = None
        # Cache capabilities to avoid repeated extraction
        self._cached_capabilities: Optional[dict[str, Any]] = None

    async def get_introspection_data(self) -> dict[str, Any]:
        """
        Get comprehensive introspection data for this node.

        This method collects all relevant information about the node's capabilities,
        configuration, and current state. It's optimized with caching to avoid
        repeated expensive operations.

        Returns:
            Dictionary containing comprehensive introspection data
        """
        # Check cache first
        cache_key = "introspection_data"
        current_time = datetime.now(UTC).timestamp()

        if (
            cache_key in self._introspection_cache
            and cache_key in self._cache_timestamps
            and current_time - self._cache_timestamps[cache_key]
            < self._cache_ttl_seconds
        ):
            # Return cached data without updating timestamp (allows TTL expiration)
            return self._introspection_cache[cache_key]

        # Generate fresh introspection data
        logger.debug("Building introspection data dict...")
        introspection_data = {
            # Node identification
            "node_id": self.node_id,
            "node_name": getattr(self, "node_name", self.__class__.__name__),
            "node_type": self._get_node_type_cached(),
            "node_version": getattr(self, "node_version", "1.0.0"),
        }
        logger.debug("Getting capabilities for introspection...")
        introspection_data["capabilities"] = await self.get_capabilities()
        logger.debug("Getting endpoints for introspection...")
        introspection_data["endpoints"] = await self.get_endpoints()
        logger.debug("Getting configuration for introspection...")
        introspection_data["configuration"] = await self.get_configuration()
        logger.debug("Getting current state for introspection...")
        introspection_data["current_state"] = await self.get_current_state()
        logger.debug("Adding metadata to introspection...")
        introspection_data["introspection_timestamp"] = datetime.now(UTC).isoformat()
        introspection_data["introspection_version"] = "2.0.0-optimized"
        logger.debug("Introspection data dict complete")

        # Cache the result
        self._introspection_cache[cache_key] = introspection_data
        self._cache_timestamps[cache_key] = current_time

        return introspection_data

    def _get_node_type_cached(self) -> str:
        """
        Get node type with caching to avoid repeated introspection.

        Returns:
            Node type string
        """
        if self._cached_node_type is None:
            self._cached_node_type = self._get_node_type()
        return self._cached_node_type

    def _get_node_type(self) -> str:
        """
        Get the type of this node based on class name and attributes.

        Returns:
            Node type string (effect, compute, reducer, or orchestrator)
        """
        class_name = self.__class__.__name__.lower()

        # Check class name patterns (ONEX 4-node architecture)
        if "orchestrator" in class_name:
            return "orchestrator"
        elif "reducer" in class_name:
            return "reducer"
        elif "registry" in class_name:
            # Registry is an effect node with specialized role
            return "effect"
        elif "effect" in class_name:
            return "effect"
        elif "compute" in class_name:
            return "compute"

        # Check for specific attributes
        if hasattr(self, "workflow_fsm_states"):
            return "orchestrator"  # Orchestrators manage workflows
        elif hasattr(self, "execute_effect"):
            return "effect"
        elif hasattr(self, "execute_reduction"):
            return "reducer"
        elif hasattr(self, "execute_compute"):
            return "compute"
        else:
            return "effect"  # Default to effect for unknown types

    async def get_capabilities(self) -> dict[str, Any]:
        """
        Get node capabilities with performance optimizations.

        Returns:
            Dictionary describing node capabilities
        """
        capabilities = {}

        # Get node type (cached)
        node_type = self._get_node_type_cached()
        capabilities["node_type"] = node_type

        # Workflow and FSM capabilities (optimized)
        workflow_dict = getattr(self, "workflow_fsm_states", None)
        # For orchestrator/reducer nodes, always include fsm_states even if empty
        if workflow_dict is not None and node_type in ("orchestrator", "reducer"):
            # Performance optimization: Use cached active states
            active_states = (
                self._get_active_states_cached(workflow_dict)
                if workflow_dict
                else set()
            )

            # Performance optimization: Use cached supported states
            supported_states = _get_supported_states_cached(node_type)

            capabilities["fsm_states"] = {
                "supported_states": supported_states,
                "active_workflows": len(workflow_dict) if workflow_dict else 0,
                "active_states": sorted(active_states) if active_states else [],
            }

        # Performance metrics (if available)
        if hasattr(self, "stamping_metrics") or hasattr(self, "get_stamping_metrics"):
            metrics = (
                getattr(self, "stamping_metrics", {})
                if hasattr(self, "stamping_metrics")
                else getattr(self, "get_stamping_metrics", lambda: {})()
            )
            if metrics:
                capabilities["performance"] = {
                    "operations_tracked": len(metrics),
                    "metrics_available": True,
                }

        # Supported operations (cached)
        capabilities["supported_operations"] = self._get_supported_operations_cached(
            node_type
        )

        # Resource characteristics (cached)
        if PSUTIL_AVAILABLE:
            resource_info = await self._get_resource_info_cached()
            if resource_info:
                capabilities["resource_limits"] = resource_info

        # Service URLs (if configured)
        self._add_service_urls(capabilities)

        return capabilities

    def _get_active_states_cached(self, workflow_dict: dict[str, Any]) -> set[str]:
        """
        Get active states with caching to avoid repeated introspection.

        Args:
            workflow_dict: Workflow FSM states dictionary

        Returns:
            Set of active state values
        """
        # Create cache key based on workflow states to detect changes
        try:
            # Extract state values for hashing
            state_tuples = []
            for wf_id, wf_state in sorted(workflow_dict.items()):
                state = getattr(wf_state, "current_state", None)
                state_val = getattr(state, "value", None) if state else None
                state_tuples.append((wf_id, state_val))

            # Create hash of state values
            state_hash = hash(tuple(state_tuples))
            cache_key = f"active_states_{state_hash}"
        except (TypeError, AttributeError):
            # Fall back to id if states aren't hashable
            cache_key = f"active_states_{id(workflow_dict)}"

        current_time = datetime.now(UTC).timestamp()

        if (
            cache_key in self._introspection_cache
            and cache_key in self._cache_timestamps
            and current_time - self._cache_timestamps[cache_key]
            < self._cache_ttl_seconds
        ):

            return self._introspection_cache[cache_key]

        # Compute active states
        active_states = set()
        for workflow_id, workflow_state in workflow_dict.items():
            state = getattr(workflow_state, "current_state", None)
            if state and hasattr(state, "value"):
                active_states.add(state.value)

        # Cache the result
        self._introspection_cache[cache_key] = active_states
        self._cache_timestamps[cache_key] = current_time

        return active_states

    def _get_supported_operations_cached(self, node_type: str) -> list[str]:
        """
        Get supported operations with caching.

        Args:
            node_type: Node type

        Returns:
            List of supported operations
        """
        cache_key = f"supported_operations_{node_type}"
        current_time = datetime.now(UTC).timestamp()

        if (
            cache_key in self._introspection_cache
            and cache_key in self._cache_timestamps
            and current_time - self._cache_timestamps[cache_key]
            < self._cache_ttl_seconds
        ):

            return self._introspection_cache[cache_key]

        # Compute supported operations
        operations = []

        # Check for standard ONEX operations
        if hasattr(self, "execute_effect"):
            operations.append("execute_effect")
        if hasattr(self, "execute_reduction"):
            operations.append("execute_reduction")
        if hasattr(self, "execute_orchestration"):
            operations.append("execute_orchestration")
        if hasattr(self, "execute_compute"):
            operations.append("execute_compute")

        # Add node-specific operations
        if node_type == "orchestrator":
            operations.extend(
                ["coordinate_workflow", "manage_state", "route_operations"]
            )
        elif node_type == "reducer":
            operations.extend(["aggregate_data", "reduce_results", "manage_state"])
        elif node_type == "registry":
            operations.extend(
                ["register_nodes", "discover_services", "manage_registrations"]
            )
        elif node_type == "effect":
            operations.extend(["perform_effects", "handle_side_effects"])

        # Cache the result
        self._introspection_cache[cache_key] = operations
        self._cache_timestamps[cache_key] = current_time

        return operations

    async def _get_resource_info_cached(self) -> Optional[dict[str, Any]]:
        """
        Get resource information with caching.

        Returns:
            Resource information dictionary or None
        """
        cache_key = "resource_info"
        current_time = datetime.now(UTC).timestamp()

        if (
            cache_key in self._introspection_cache
            and cache_key in self._cache_timestamps
            and current_time - self._cache_timestamps[cache_key] < CACHE_TTL_SECONDS
        ):  # 1 minute cache for resource info

            return self._introspection_cache[cache_key]

        # Compute resource info (wrap blocking psutil calls in asyncio.to_thread)
        try:
            # Run blocking psutil calls in thread pool to avoid blocking event loop
            process = await asyncio.to_thread(psutil.Process)
            cpu_cores = await asyncio.to_thread(psutil.cpu_count, True)
            virtual_mem = await asyncio.to_thread(psutil.virtual_memory)
            mem_info = await asyncio.to_thread(process.memory_info)

            resource_info = {
                "cpu_cores_available": cpu_cores,
                "memory_available_gb": round(virtual_mem.available / (1024**3), 2),
                "current_memory_usage_mb": round(mem_info.rss / (1024**2), 2),
            }

            # Cache the result
            self._introspection_cache[cache_key] = resource_info
            self._cache_timestamps[cache_key] = current_time

            return resource_info

        except Exception as e:
            logger.debug("Failed to get resource info - error=%s", str(e))
            return None

    def _add_service_urls(self, capabilities: dict[str, Any]) -> None:
        """
        Add service URLs to capabilities if configured.

        Args:
            capabilities: Capabilities dictionary to modify
        """
        if hasattr(self, "metadata_stamping_service_url"):
            capabilities["service_integration"] = {
                "metadata_stamping": getattr(
                    self, "metadata_stamping_service_url", None
                ),
            }
        if hasattr(self, "onextree_service_url"):
            capabilities.setdefault("service_integration", {})["onextree"] = getattr(
                self, "onextree_service_url", None
            )

    async def get_endpoints(self) -> dict[str, str]:
        """
        Get all node endpoints with caching.

        Returns:
            Dictionary of endpoint URLs by type
        """
        cache_key = "endpoints"
        current_time = datetime.now(UTC).timestamp()

        if (
            cache_key in self._introspection_cache
            and cache_key in self._cache_timestamps
            and current_time - self._cache_timestamps[cache_key]
            < self._cache_ttl_seconds
        ):

            return self._introspection_cache[cache_key]

        endpoints = {}

        # Get configuration from container
        container = getattr(self, "container", None)
        if container:
            config = getattr(container, "config", None)

            # Helper function to get config value (supports both dict and object)
            def _get_config_value(key: str, default=None):
                if isinstance(config, dict):
                    return config.get(key, default)
                elif hasattr(config, key):
                    value = getattr(config, key)
                    # If it's callable, call it
                    return value() if callable(value) else value
                return default

            # Explicit configuration takes precedence
            if config:
                health_endpoint = _get_config_value("health_endpoint")
                if health_endpoint:
                    endpoints["health"] = str(health_endpoint)
                api_endpoint = _get_config_value("api_endpoint")
                if api_endpoint:
                    endpoints["api"] = str(api_endpoint)
                metrics_endpoint = _get_config_value("metrics_endpoint")
                if metrics_endpoint:
                    endpoints["metrics"] = str(metrics_endpoint)

            # Try to construct from host/port settings
            if not endpoints and config:
                host = _get_config_value("host", "localhost")
                # Try port first, fallback to api_port
                api_port = (
                    _get_config_value("port") or _get_config_value("api_port") or 8053
                )
                metrics_port = _get_config_value("metrics_port", 9090)

                # Determine if running in HTTPS
                use_https_config = _get_config_value("use_https")
                if use_https_config is not None:
                    use_https = bool(use_https_config)
                else:
                    use_https = str(os.getenv("USE_HTTPS", "false")).lower() == "true"
                protocol = "https" if use_https else "http"

                # Build endpoint URLs
                endpoints["api"] = f"{protocol}://{host}:{api_port}"
                endpoints["metrics"] = f"{protocol}://{host}:{metrics_port}"

                # Health endpoint (usually same as API)
                health_path = _get_config_value("health_path", "/health")
                endpoints["health"] = f"{endpoints['api']}{health_path}"

        # Use HealthCheckMixin if available
        if hasattr(self, "get_health_endpoint_url"):
            try:
                health_url = self.get_health_endpoint_url()
                if health_url:
                    endpoints["health"] = health_url
            except Exception as e:
                logger.debug(
                    "Failed to get health endpoint from HealthCheckMixin - error=%s",
                    str(e),
                )

        # Add node-specific endpoints based on node type
        node_type = self._get_node_type_cached()
        if node_type == "orchestrator" and "api" in endpoints:
            # Orchestrator has orchestration endpoint (same as API)
            endpoints["orchestration"] = endpoints["api"]
        elif node_type == "reducer" and "api" in endpoints:
            # Reducer has aggregation endpoint (same as API)
            endpoints["aggregation"] = endpoints["api"]

        # Cache the result
        self._introspection_cache[cache_key] = endpoints
        self._cache_timestamps[cache_key] = current_time

        return endpoints

    async def get_configuration(self) -> dict[str, Any]:
        """
        Get node configuration with filtering for sensitive data.

        Returns:
            Dictionary containing non-sensitive configuration
        """
        cache_key = "configuration"
        current_time = datetime.now(UTC).timestamp()

        if (
            cache_key in self._introspection_cache
            and cache_key in self._cache_timestamps
            and current_time - self._cache_timestamps[cache_key]
            < self._cache_ttl_seconds
        ):

            return self._introspection_cache[cache_key]

        config = {}
        container = getattr(self, "container", None)

        if container:
            container_config = getattr(container, "config", {})
            # Handle case where config exists but is None
            if container_config is None:
                container_config = {}

            # Only include non-sensitive configuration
            sensitive_keys = {
                "password",
                "pwd",
                "secret",
                "token",
                "api_key",
                "private_key",
                "certificate",
                "credential",
                "auth",
                "authorization",
                "bearer",
                "session",
                "cookie",
                "csrf",
                "jwt",
                "oauth",
                "connection_string",
            }

            try:
                # Use get_children() instead of items() for dependency_injector Configuration
                # items() returns None, get_children() returns actual config dict
                config_items = None
                if hasattr(container_config, "get_children"):
                    # Configuration provider - use get_children()
                    children = container_config.get_children()
                    if children:
                        # Call each provider to get actual values
                        config_items = {
                            key: provider()
                            for key, provider in children.items()
                            if callable(provider)
                        }
                elif hasattr(container_config, "items") and callable(
                    container_config.items
                ):
                    # Regular dict - use items()
                    config_items = container_config.items()

                if config_items:
                    for key, value in config_items.items():
                        # Unwrap _ConfigValue objects if present (for stub compatibility)
                        if hasattr(value, "_value"):
                            value = value._value

                        if isinstance(key, str) and key.lower() not in sensitive_keys:
                            # Only include simple serializable values
                            # Skip complex objects like BridgeMetricsCollector
                            if isinstance(
                                value,
                                str | int | float | bool | list | dict | type(None),
                            ):
                                if isinstance(value, dict):
                                    # Filter nested dictionaries
                                    filtered_value = {}
                                    for nested_key, nested_value in value.items():
                                        # Unwrap nested _ConfigValue objects
                                        if hasattr(nested_value, "_value"):
                                            nested_value = nested_value._value
                                        if (
                                            isinstance(nested_key, str)
                                            and nested_key.lower() not in sensitive_keys
                                            and isinstance(
                                                nested_value,
                                                str
                                                | int
                                                | float
                                                | bool
                                                | list
                                                | dict
                                                | type(None),
                                            )
                                        ):
                                            filtered_value[nested_key] = nested_value
                                    config[key] = filtered_value
                                else:
                                    config[key] = value
            except Exception as e:
                import traceback

                logger.error(f"Error iterating config: {e}\n{traceback.format_exc()}")

        # Add node-specific configuration
        if hasattr(self, "node_ttl_hours"):
            config["node_ttl_hours"] = self.node_ttl_hours
        if hasattr(self, "cleanup_interval_hours"):
            config["cleanup_interval_hours"] = self.cleanup_interval_hours

        # Cache the result
        self._introspection_cache[cache_key] = config
        self._cache_timestamps[cache_key] = current_time

        return config

    async def get_current_state(self) -> dict[str, Any]:
        """
        Get current operational state of the node.

        Returns:
            Dictionary describing current state
        """
        cache_key = "current_state"
        current_time = datetime.now(UTC).timestamp()

        # Use shorter cache for current state
        if (
            cache_key in self._introspection_cache
            and cache_key in self._cache_timestamps
            and current_time - self._cache_timestamps[cache_key]
            < CURRENT_STATE_CACHE_TTL_SECONDS
        ):

            return self._introspection_cache[cache_key]

        state = {
            "status": "active",  # Could be enhanced with actual status tracking
            "uptime_seconds": 0,  # Could be enhanced with actual uptime tracking
            "last_activity": datetime.now(UTC).isoformat(),
        }

        # Add FSM state information if available
        workflow_dict = getattr(self, "workflow_fsm_states", None)
        if workflow_dict:
            active_states = self._get_active_states_cached(workflow_dict)
            state["workflow_states"] = {
                "active_workflows": len(workflow_dict),
                "active_states": sorted(active_states) if active_states else [],
            }

        # Add health status if available
        if hasattr(self, "health_status"):
            state["health_status"] = self.health_status.value
        if hasattr(self, "last_health_check"):
            state["last_health_check"] = self.last_health_check.isoformat()

        # Cache the result
        self._introspection_cache[cache_key] = state
        self._cache_timestamps[cache_key] = current_time

        return state

    def clear_introspection_cache(self) -> None:
        """
        Clear the introspection cache.

        This method can be called to force refresh of introspection data.
        """
        self._introspection_cache.clear()
        self._cache_timestamps.clear()
        logger.debug("Introspection cache cleared")

    def get_cache_stats(self) -> dict[str, Any]:
        """
        Get statistics about the introspection cache.

        Returns:
            Cache statistics dictionary
        """
        current_time = datetime.now(UTC).timestamp()

        return {
            "cache_entries": len(self._introspection_cache),
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "cached_keys": list(self._introspection_cache.keys()),
            "cache_timestamps": {
                key: {
                    "timestamp": timestamp,
                    "age_seconds": current_time - timestamp,
                    "expired": current_time - timestamp > self._cache_ttl_seconds,
                }
                for key, timestamp in self._cache_timestamps.items()
            },
        }

    async def publish_introspection(
        self,
        reason: str = "periodic",
        correlation_id: Optional[UUID] = None,
        force_refresh: bool = False,
    ) -> bool:
        """
        Publish introspection data to Kafka with network topology metadata.

        This method collects node introspection data and publishes it as a
        NODE_INTROSPECTION event. As of Phase 1a MVP, it includes network
        topology metadata read from environment variables for multi-network
        support.

        Environment Variables (Optional):
            NETWORK_ID: Logical network identifier (e.g., "omninode-network-1")
            DEPLOYMENT_ID: Deployment instance identifier (default: "dev-001")
            EPOCH: Deployment epoch for blue-green deployments (default: "1")

        Args:
            reason: Reason for publishing introspection (e.g., "startup", "periodic")
            correlation_id: Optional correlation ID for tracking related events
            force_refresh: Force refresh of cached data

        Returns:
            True if successful, False otherwise

        Example:
            >>> # With environment variables set:
            >>> # export NETWORK_ID="prod-network-1"
            >>> # export DEPLOYMENT_ID="prod-us-west-2-001"
            >>> # export EPOCH="3"
            >>> success = await node.publish_introspection(reason="startup")
        """
        try:
            # Get capabilities (with caching)
            logger.debug("Getting capabilities...")
            if force_refresh or self._cached_capabilities is None:
                self._cached_capabilities = await self.get_capabilities()
            logger.debug("Capabilities retrieved successfully")

            # Get introspection data
            logger.debug("Getting introspection data...")
            introspection_data = await self.get_introspection_data()
            logger.debug("Introspection data retrieved successfully")

            # Read network topology metadata from environment variables (Phase 1a MVP)
            network_id = os.getenv("NETWORK_ID")
            deployment_id = os.getenv("DEPLOYMENT_ID", "dev-001")
            epoch_str = os.getenv("EPOCH", "1")
            try:
                epoch = int(epoch_str)
            except ValueError:
                logger.warning("Invalid EPOCH value '%s', defaulting to 1", epoch_str)
                epoch = 1

            # Create envelope
            logger.debug("Creating introspection envelope...")
            envelope = IntrospectionEnvelope(
                node_id=str(getattr(self, "node_id", uuid4())),
                node_type=self._get_node_type_cached(),
                timestamp=datetime.now(UTC).isoformat(),
                reason=reason,
                data=introspection_data,
                correlation_id=str(correlation_id) if correlation_id else None,
                network_id=network_id,
                deployment_id=deployment_id,
                epoch=epoch,
            )
            logger.debug("Envelope created successfully")

            # Publish to Kafka
            logger.debug("Publishing to Kafka...")
            kafka_success = await self._publish_to_kafka(envelope)
            logger.debug(f"Kafka publish result: {kafka_success}")

            # Update last broadcast time
            self._last_introspection_broadcast = datetime.now(UTC)

            if kafka_success:
                logger.info(
                    "Node introspection broadcast successful - node_id=%s, node_type=%s, reason=%s",
                    self.node_id,
                    self._get_node_type_cached(),
                    reason,
                )
            else:
                logger.warning(
                    "Node introspection completed but Kafka publish failed (degraded mode) - node_id=%s, node_type=%s, reason=%s",
                    self.node_id,
                    self._get_node_type_cached(),
                    reason,
                )

            return kafka_success

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to publish introspection: {e}\n{traceback.format_exc()}"
            )
            return False

    async def _publish_to_kafka(self, envelope: Any) -> bool:
        """
        Publish envelope to Kafka.

        Args:
            envelope: Envelope data to publish

        Returns:
            True if publish successful, False otherwise
        """
        try:
            # Check if kafka client is available (could be kafka_producer or kafka_client)
            kafka_producer = getattr(self, "kafka_producer", None)
            kafka_client = getattr(self, "kafka_client", None)

            if kafka_producer is None and kafka_client is None:
                logger.warning("Kafka client not available - graceful degradation")
                return False

            # This is a mock implementation - in real implementation this would publish to Kafka
            logger.debug(f"Publishing to Kafka: {envelope}")

            # For testing purposes, we'll just store the envelope
            if not hasattr(self, "_kafka_messages"):
                self._kafka_messages = []
            self._kafka_messages.append(envelope)

            return True
        except Exception as e:
            logger.warning(f"Failed to publish to Kafka: {e}")
            return False

    def initialize_introspection(
        self,
        enable_heartbeat: bool = True,
        heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        enable_registry_listener: bool = True,
    ) -> None:
        """
        Initialize node introspection system.

        Should be called in node __init__ after super().__init__().

        Args:
            enable_heartbeat: Whether to start periodic heartbeat broadcasting
            heartbeat_interval_seconds: Interval between heartbeat broadcasts
            enable_registry_listener: Whether to listen for registry requests
        """
        # Initialize attributes if they don't exist (in case __init__ wasn't called via MRO)
        if not hasattr(self, "_introspection_initialized"):
            self._introspection_initialized = False
            self._startup_time = time.time()
            self._last_introspection_broadcast = None
            self._heartbeat_task = None
            self._registry_listener_task = None
            self._cached_capabilities = None
            self._cached_endpoints = None

        if self._introspection_initialized:
            logger.warning(
                "Introspection already initialized - node_id=%s",
                getattr(self, "node_id", "unknown"),
            )
            return

        self._introspection_initialized = True

        logger.info(
            "Node introspection initialized - node_id=%s, enable_heartbeat=%s, heartbeat_interval=%s, enable_registry_listener=%s",
            getattr(self, "node_id", "unknown"),
            enable_heartbeat,
            heartbeat_interval_seconds,
            enable_registry_listener,
        )

        # Start background tasks (in async context)
        # Note: These will be started via start_introspection_tasks()

    async def start_introspection_tasks(
        self,
        enable_heartbeat: bool = True,
        heartbeat_interval_seconds: int = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
        enable_registry_listener: bool = True,
    ) -> None:
        """
        Start background introspection tasks (heartbeat, registry listener).

        Should be called in an async context (e.g., node startup).

        Args:
            enable_heartbeat: Whether to start periodic heartbeat broadcasting
            heartbeat_interval_seconds: Interval between heartbeat broadcasts
            enable_registry_listener: Whether to listen for registry requests
        """
        if enable_heartbeat:
            self._heartbeat_task = asyncio.create_task(
                self._heartbeat_loop(heartbeat_interval_seconds)
            )
            logger.info(
                "Heartbeat broadcasting started - node_id=%s, interval_seconds=%s",
                getattr(self, "node_id", "unknown"),
                heartbeat_interval_seconds,
            )

        if enable_registry_listener:
            self._registry_listener_task = asyncio.create_task(
                self._registry_listener_loop()
            )
            logger.info(
                "Registry request listener started - node_id=%s",
                getattr(self, "node_id", "unknown"),
            )

    async def stop_introspection_tasks(self, timeout_seconds: float = 5.0) -> None:
        """
        Stop background introspection tasks with timeout.

        Should be called during node shutdown to ensure clean cleanup of
        background tasks without hanging indefinitely.

        Args:
            timeout_seconds: Maximum time to wait for task cancellation (default: 5.0)
        """
        node_id = getattr(self, "node_id", "unknown")

        # Stop heartbeat task
        heartbeat_task = getattr(self, "_heartbeat_task", None)
        if heartbeat_task and not heartbeat_task.done():
            heartbeat_task.cancel()
            try:
                # Issue #3 fix: Await task after cancellation to ensure complete cleanup
                # Add timeout to prevent indefinite hangs during shutdown
                await asyncio.wait_for(heartbeat_task, timeout=timeout_seconds)
                logger.info("Heartbeat broadcasting stopped - node_id=%s", node_id)
            except TimeoutError:
                logger.warning(
                    "Heartbeat task cancellation timed out after %s seconds - node_id=%s",
                    timeout_seconds,
                    node_id,
                )
            except asyncio.CancelledError:
                # Expected during cancellation - this is normal
                logger.info("Heartbeat task cancelled - node_id=%s", node_id)
            except Exception as e:
                logger.error(
                    "Error stopping heartbeat task - node_id=%s, error=%s, error_type=%s",
                    node_id,
                    str(e),
                    type(e).__name__,
                )

        # Stop registry listener task
        registry_listener_task = getattr(self, "_registry_listener_task", None)
        if registry_listener_task and not registry_listener_task.done():
            registry_listener_task.cancel()
            try:
                # Issue #3 fix: Await task after cancellation to ensure complete cleanup
                # Add timeout to prevent indefinite hangs during shutdown
                await asyncio.wait_for(registry_listener_task, timeout=timeout_seconds)
                logger.info("Registry request listener stopped - node_id=%s", node_id)
            except TimeoutError:
                logger.warning(
                    "Registry listener task cancellation timed out after %s seconds - node_id=%s",
                    timeout_seconds,
                    node_id,
                )
            except asyncio.CancelledError:
                # Expected during cancellation - this is normal
                logger.info("Registry listener task cancelled - node_id=%s", node_id)
            except Exception as e:
                logger.error(
                    "Error stopping registry listener task - node_id=%s, error=%s, error_type=%s",
                    node_id,
                    str(e),
                    type(e).__name__,
                )

    async def listen_for_registry_requests(self) -> None:
        """
        Listen for REGISTRY_REQUEST events and re-broadcast introspection.

        This method should be called in an async context (e.g., background task).
        Listens to Kafka topic for registry requests and auto-responds with
        introspection data.

        Implementation:
        - Consumes from registry-request-introspection.v1 topic
        - Deserializes OnexEnvelopeV1 messages
        - Extracts ModelRegistryRequestEvent payload
        - Responds with NODE_INTROSPECTION event using correlation_id
        - Filters based on target_node_types if specified
        - Handles errors gracefully with circuit breaker pattern
        """
        # Get kafka_client from container
        container = getattr(self, "container", None)
        if container is None:
            logger.warning(
                "Container not available - registry request listener disabled for node_id=%s",
                getattr(self, "node_id", "unknown"),
            )
            return

        kafka_client = container.get_service("kafka_client")
        if kafka_client is None:
            logger.warning(
                "Kafka client not available - registry request listener disabled for node_id=%s",
                getattr(self, "node_id", "unknown"),
            )
            return

        # Get environment for topic naming
        env = self._get_environment()
        request_topic = (
            f"{env}.omninode_bridge.onex.evt.registry-request-introspection.v1"
        )
        consumer_group = f"introspection-listener-{getattr(self, 'node_id', uuid4())}"

        logger.info(
            "Starting registry request listener for node_id=%s, topic=%s, group=%s",
            getattr(self, "node_id", "unknown"),
            request_topic,
            consumer_group,
        )

        # Get node type for filtering
        node_type = self._get_node_type_cached()

        # Track metrics
        requests_received = 0
        responses_published = 0
        errors_encountered = 0

        try:
            # Consume messages from Kafka
            while True:
                try:
                    # Check if Kafka is connected before consuming
                    if not kafka_client.is_connected:
                        logger.warning(
                            "Kafka client not connected - registry listener in degraded mode - node_id=%s",
                            getattr(self, "node_id", "unknown"),
                        )
                        # Sleep to avoid tight loop when Kafka is unavailable
                        await asyncio.sleep(5.0)
                        continue

                    # Poll for messages with timeout
                    messages = await asyncio.wait_for(
                        kafka_client.consume_messages(
                            topic=request_topic,
                            group_id=consumer_group,
                            max_messages=10,  # Process in small batches
                            timeout_ms=5000,  # 5 second timeout
                        ),
                        timeout=10.0,  # Overall timeout for consume operation
                    )

                    if not messages:
                        continue

                    logger.debug(
                        f"Received {len(messages)} registry request messages for node_id={getattr(self, 'node_id', 'unknown')}"
                    )

                    # Process each message
                    for message in messages:
                        try:
                            # Import models here to avoid circular imports
                            from ..orchestrator.v1_0_0.models.model_registry_request_event import (
                                ModelRegistryRequestEvent,
                            )
                            from ..registry.v1_0_0.models.model_onex_envelope_v1 import (
                                ModelOnexEnvelopeV1,
                            )

                            # Deserialize envelope
                            envelope = ModelOnexEnvelopeV1.from_bytes(message.value)

                            # Extract correlation_id from envelope
                            correlation_id = envelope.correlation_id

                            # Parse registry request event
                            request_event = ModelRegistryRequestEvent.model_validate(
                                envelope.payload
                            )

                            requests_received += 1

                            # Check if this node should respond
                            if not request_event.should_node_respond(node_type):
                                logger.debug(
                                    "Skipping registry request - node_type=%s not in target_node_types=%s",
                                    node_type,
                                    request_event.target_node_types,
                                )
                                continue

                            logger.info(
                                "Received registry request - registry_id=%s, reason=%s, correlation_id=%s, node_id=%s",
                                request_event.registry_id,
                                request_event.reason.value,
                                correlation_id,
                                getattr(self, "node_id", "unknown"),
                            )

                            # Respond with introspection
                            response_success = await self.publish_introspection(
                                reason="registry_request",
                                correlation_id=correlation_id,
                                force_refresh=True,  # Force refresh to get latest data
                            )

                            if response_success:
                                responses_published += 1
                                logger.debug(
                                    "Responded to registry request - correlation_id=%s, node_id=%s",
                                    correlation_id,
                                    getattr(self, "node_id", "unknown"),
                                )
                            else:
                                logger.warning(
                                    "Failed to respond to registry request - correlation_id=%s, node_id=%s",
                                    correlation_id,
                                    getattr(self, "node_id", "unknown"),
                                )

                        except Exception as e:
                            errors_encountered += 1
                            logger.error(
                                "Failed to process registry request message - node_id=%s, error=%s, error_type=%s",
                                getattr(self, "node_id", "unknown"),
                                str(e),
                                type(e).__name__,
                                exc_info=True,
                            )
                            # Continue processing other messages

                    # Commit offsets after successful batch processing
                    if messages:
                        try:
                            await kafka_client.commit_offsets(messages)
                            logger.debug(
                                f"Committed offsets for {len(messages)} messages - node_id={getattr(self, 'node_id', 'unknown')}"
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to commit offsets - node_id=%s, error=%s",
                                getattr(self, "node_id", "unknown"),
                                str(e),
                            )

                except asyncio.TimeoutError:
                    # Timeout is normal - just means no messages available
                    logger.debug(
                        "Registry request consumer timeout (no messages) - node_id=%s",
                        getattr(self, "node_id", "unknown"),
                    )
                    continue
                except asyncio.CancelledError:
                    logger.info(
                        "Registry request listener cancelled - node_id=%s, requests_received=%d, responses_published=%d, errors=%d",
                        getattr(self, "node_id", "unknown"),
                        requests_received,
                        responses_published,
                        errors_encountered,
                    )
                    raise
                except Exception as e:
                    errors_encountered += 1
                    logger.error(
                        "Error in registry request consumer loop - node_id=%s, error=%s, error_type=%s",
                        getattr(self, "node_id", "unknown"),
                        str(e),
                        type(e).__name__,
                        exc_info=True,
                    )
                    # Continue after error with exponential backoff
                    await asyncio.sleep(min(5.0, errors_encountered * 0.5))

        except asyncio.CancelledError:
            logger.info(
                "Registry request listener shutdown - node_id=%s, total_requests_received=%d, total_responses_published=%d, total_errors=%d",
                getattr(self, "node_id", "unknown"),
                requests_received,
                responses_published,
                errors_encountered,
            )
            raise
        except Exception as e:
            logger.error(
                "Fatal error in registry request listener - node_id=%s, error=%s, error_type=%s",
                getattr(self, "node_id", "unknown"),
                str(e),
                type(e).__name__,
                exc_info=True,
            )
            raise

    async def _heartbeat_loop(self, interval_seconds: int) -> None:
        """
        Background task for periodic heartbeat broadcasting.

        Args:
            interval_seconds: Interval between heartbeat broadcasts
        """
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                await self._publish_heartbeat()
        except asyncio.CancelledError:
            logger.debug(
                "Heartbeat loop cancelled - node_id=%s",
                getattr(self, "node_id", "unknown"),
            )
            raise

    async def _registry_listener_loop(self) -> None:
        """
        Background task for listening to registry requests.

        This wraps the main listener coroutine and handles lifecycle management.
        The actual Kafka consumption happens in listen_for_registry_requests().
        """
        try:
            await self.listen_for_registry_requests()
        except asyncio.CancelledError:
            logger.debug(
                "Registry listener loop cancelled - node_id=%s",
                getattr(self, "node_id", "unknown"),
            )
            raise

    async def _publish_heartbeat(self) -> bool:
        """
        Publish NODE_HEARTBEAT event to Kafka.

        Returns:
            True if heartbeat published successfully, False otherwise
        """
        try:
            # Import event models and envelope helpers
            from ..orchestrator.v1_0_0.models.introspection_event_helpers import (
                create_heartbeat_envelope,
            )
            from ..orchestrator.v1_0_0.models.model_node_heartbeat_event import (
                EnumNodeHealthStatus,
                ModelNodeHeartbeatEvent,
            )

            # Get node metadata
            node_id = str(getattr(self, "node_id", uuid4()))
            node_type = self._get_node_type()

            # Calculate uptime
            uptime_seconds = int(time.time() - self._startup_time)

            # Get health status (if HealthCheckMixin is available)
            health_status = EnumNodeHealthStatus.HEALTHY
            if hasattr(self, "get_health_status"):
                try:
                    # get_health_status() is synchronous and returns a dict
                    health_result = self.get_health_status()
                    health_status_str = health_result.get("overall_status", "healthy")
                    if health_status_str == "unhealthy":
                        health_status = EnumNodeHealthStatus.UNHEALTHY
                    elif health_status_str == "degraded":
                        health_status = EnumNodeHealthStatus.DEGRADED
                except Exception as e:
                    logger.debug("Failed to get health status - error=%s", str(e))

            # Get resource usage (wrap blocking psutil calls in asyncio.to_thread)
            resource_usage = {}
            if PSUTIL_AVAILABLE:
                try:
                    # Run blocking psutil calls in thread pool to avoid blocking event loop
                    process = await asyncio.to_thread(psutil.Process)
                    cpu_percent = await asyncio.to_thread(process.cpu_percent, None)
                    mem_info = await asyncio.to_thread(process.memory_info)
                    mem_percent = await asyncio.to_thread(process.memory_percent)

                    resource_usage = {
                        "cpu_percent": round(cpu_percent, 2),
                        "memory_mb": round(mem_info.rss / (1024**2), 2),
                        "memory_percent": round(mem_percent, 2),
                    }
                except Exception as e:
                    logger.debug("Failed to get resource usage - error=%s", str(e))

            # Get active operations count (if available)
            active_operations = 0
            if hasattr(self, "workflow_fsm_states"):
                active_operations = len(getattr(self, "workflow_fsm_states", {}))

            # Create heartbeat event
            heartbeat_event = ModelNodeHeartbeatEvent.create(
                node_id=node_id,
                node_type=node_type,
                health_status=health_status,
                uptime_seconds=uptime_seconds,
                last_activity_timestamp=datetime.now(UTC),
                active_operations=active_operations,
                resource_usage=resource_usage,
                metadata={
                    "version": (
                        getattr(self.container, "version", "1.0.0")
                        if hasattr(self, "container")
                        else "1.0.0"
                    ),
                    "environment": self._get_environment(),
                },
            )

            # Wrap in OnexEnvelopeV1
            envelope = create_heartbeat_envelope(
                heartbeat_data=heartbeat_event,
                source_instance=node_id,
                environment=self._get_environment(),
            )

            # Publish to Kafka
            kafka_success = await self._publish_to_kafka(envelope)

            if kafka_success:
                logger.debug(
                    "Heartbeat published - node_id=%s, uptime_seconds=%s, health_status=%s",
                    node_id,
                    uptime_seconds,
                    health_status.value,
                )
            else:
                logger.debug(
                    "Heartbeat failed to publish (Kafka unavailable) - node_id=%s, uptime_seconds=%s",
                    node_id,
                    uptime_seconds,
                )

            return kafka_success

        except Exception as e:
            logger.error(
                "Failed to publish heartbeat - node_id=%s, error=%s, error_type=%s",
                getattr(self, "node_id", "unknown"),
                str(e),
                type(e).__name__,
            )
            return False

    def _get_environment(self) -> str:
        """
        Get environment name from container or default to development.

        Returns:
            Environment name (development, staging, production)
        """
        if hasattr(self, "container"):
            container = self.container
            if hasattr(container, "config"):
                # Try to get environment from config object
                try:
                    return container.config.get("environment", "development")
                except Exception:
                    # Fallback for dependency_injector ConfigurationOption providers
                    try:
                        env_attr = getattr(container.config, "environment", None)
                        if env_attr is not None:
                            # Call it if it's a callable (dependency_injector provider)
                            result = env_attr() if callable(env_attr) else str(env_attr)
                            # Ensure we don't return None - fallback to development
                            return result if result else "development"
                        return "development"
                    except Exception:
                        # If all fails, return development
                        return "development"
        return "development"


# Factory function to create optimized introspection mixin
def create_introspection_mixin(cache_ttl_seconds: int = 300) -> type:
    """
    Create an optimized introspection mixin class with custom cache TTL.

    Args:
        cache_ttl_seconds: Cache time-to-live in seconds

    Returns:
        Optimized introspection mixin class
    """

    class OptimizedIntrospectionMixin(IntrospectionMixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._cache_ttl_seconds = cache_ttl_seconds

    return OptimizedIntrospectionMixin


# Utility function to clear all global caches
def clear_global_caches() -> None:
    """
    Clear all global caches for enum imports and supported states.

    This function can be used to reset caches if needed (e.g., after code reloads).
    """
    global _ENUM_CACHE, _SUPPORTED_STATES_CACHE
    _ENUM_CACHE.clear()
    _SUPPORTED_STATES_CACHE.clear()
    logger.info("Global introspection caches cleared")


# Utility function to get cache statistics
def get_global_cache_stats() -> dict[str, Any]:
    """
    Get statistics about global caches.

    Returns:
        Global cache statistics dictionary
    """
    return {
        "enum_cache": {
            "entries": len(_ENUM_CACHE),
            "keys": list(_ENUM_CACHE.keys()),
        },
        "supported_states_cache": {
            "entries": len(_SUPPORTED_STATES_CACHE),
            "keys": list(_SUPPORTED_STATES_CACHE.keys()),
        },
    }
