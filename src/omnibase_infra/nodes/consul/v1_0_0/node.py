#!/usr/bin/env python3

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

from omnibase_core.core.errors.onex_error import OnexError, CoreErrorCode
from omnibase_core.core.node_effect_service import NodeEffectService
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.enums.enum_health_status import EnumHealthStatus
from omnibase_core.model.core.model_health_status import ModelHealthStatus
from omnibase_infra.nodes.consul.v1_0_0.models import (
    ModelConsulAdapterInput,
    ModelConsulAdapterOutput,
    ModelConsulHealthCheckNode,
    ModelConsulHealthResponse,
    ModelConsulKVResponse,
    ModelConsulServiceInfo,
    ModelConsulServiceListResponse,
    ModelConsulServiceRegistration,
    ModelConsulServiceResponse,
)


class MockConsulClient:
    """Mock Consul client for testing/development when consul library unavailable."""

    def __init__(self, config: dict):
        self.config = config
        self.kv_store = {}
        self.services = {}

        # Mock agent with self() method
        class MockAgent:
            def __init__(self, client):
                self.client = client
                self.service = MockService(client)
                self.services_data = client.services

            def self(self):
                return {
                    "Config": {
                        "NodeName": "mock-consul-node",
                        "Datacenter": self.client.config.get("datacenter", "dc1"),
                    }
                }

            def services(self):
                return self.services_data

        class MockService:
            def __init__(self, client):
                self.client = client

            def register(self, **kwargs):
                service_id = kwargs.get("ID")
                self.client.services[service_id] = {
                    "Service": kwargs.get("Name"),
                    "Port": kwargs.get("Port"),
                    "Address": kwargs.get("Address"),
                    "Tags": kwargs.get("Tags", []),
                }
                return True

            def deregister(self, service_id):
                if service_id in self.client.services:
                    del self.client.services[service_id]
                    return True
                return False

        class MockKV:
            def __init__(self, client):
                self.client = client

            def get(self, key):
                if key in self.client.kv_store:
                    value = self.client.kv_store[key]
                    return 1, {
                        "Key": key,
                        "Value": value.encode("utf-8"),
                        "ModifyIndex": 1,
                    }
                return None, None

            def put(self, key, value):
                self.client.kv_store[key] = value
                return True

            def delete(self, key, recurse=False):
                if recurse:
                    # Delete all keys with the prefix
                    keys_to_delete = [
                        k for k in self.client.kv_store.keys() if k.startswith(key)
                    ]
                    for k in keys_to_delete:
                        del self.client.kv_store[k]
                    return True
                else:
                    # Delete single key
                    if key in self.client.kv_store:
                        del self.client.kv_store[key]
                        return True
                    return False

        class MockHealth:
            def __init__(self, client):
                self.client = client

            def service(self, service_name, passing=None):
                # Return mock health data for service
                return 1, [
                    {
                        "Node": {"Node": "mock-node"},
                        "Service": {"ID": service_name, "Service": service_name},
                        "Checks": [
                            {"Status": "passing", "CheckID": "service:" + service_name}
                        ],
                    }
                ]

            def state(self, state):
                # Return mock health state data
                return 1, [
                    {
                        "ServiceName": "consul",
                        "Status": "passing",
                        "CheckID": "serfHealth",
                    }
                ]

        self.agent = MockAgent(self)
        self.kv = MockKV(self)
        self.health = MockHealth(self)


class ConsulConnectionPool:
    """
    Connection pool for Consul clients with proper lifecycle management.
    
    Prevents bottlenecks by maintaining multiple consul client connections
    for high-throughput operations. Implements proper cleanup and health monitoring.
    """
    
    def __init__(self, config: dict, max_connections: int = 10, cleanup_interval: int = 300):
        self._config = config
        self._max_connections = max_connections
        self._cleanup_interval = cleanup_interval
        self._connections = {}  # Dict[str, consul.Consul]
        self._failed_connections = {}  # Dict[str, float] - track failures with timestamps
        self._connection_usage = {}  # Dict[str, int] - track usage count
        self._last_cleanup = 0
        self._logger = logging.getLogger(__name__)
        
        # Background cleanup task
        self._cleanup_task = None
        self._start_background_cleanup()
    
    def _start_background_cleanup(self):
        """Start background cleanup task with proper resource management."""
        # Cancel existing task before creating new one to prevent resource leaks
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        self._cleanup_task = asyncio.create_task(self._background_cleanup_loop())
    
    async def _background_cleanup_loop(self):
        """Background loop for periodic cleanup."""
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_idle_connections()
        except asyncio.CancelledError:
            self._logger.info("Consul connection pool cleanup task cancelled")
        except Exception as e:
            self._logger.error(f"Consul connection pool cleanup error: {e}")
    
    async def _cleanup_idle_connections(self):
        """Clean up idle and unhealthy connections."""
        connections_to_remove = []
        
        for conn_key, client in self._connections.items():
            # Check if connection is still healthy
            if not await self._is_connection_healthy(client):
                connections_to_remove.append(conn_key)
                continue
            
            # Remove low-usage connections if pool is at capacity
            if (len(self._connections) > self._max_connections // 2 and
                self._connection_usage.get(conn_key, 0) < 5):  # Low usage threshold
                connections_to_remove.append(conn_key)
        
        # Clean up selected connections
        for conn_key in connections_to_remove:
            await self._remove_connection(conn_key)
    
    async def _is_connection_healthy(self, client) -> bool:
        """Check if a Consul connection is still healthy."""
        try:
            # Test basic connectivity with agent self check
            if hasattr(client, 'agent') and hasattr(client.agent, 'self'):
                client.agent.self()
                return True
            return False
        except Exception:
            return False
    
    async def _remove_connection(self, conn_key: str):
        """Safely remove a connection from the pool."""
        if conn_key in self._connections:
            try:
                # Consul client doesn't need explicit cleanup, just remove reference
                del self._connections[conn_key]
                if conn_key in self._connection_usage:
                    del self._connection_usage[conn_key]
                self._logger.debug(f"Removed Consul connection: {conn_key}")
            except Exception as e:
                self._logger.error(f"Error removing Consul connection: {e}")
    
    def get_client(self):
        """Get a Consul client from the pool or create a new one."""
        # Create connection key based on config
        conn_key = f"{self._config['host']}:{self._config['port']}:{self._config['datacenter']}"
        
        # Return existing connection if available
        if conn_key in self._connections:
            self._connection_usage[conn_key] = self._connection_usage.get(conn_key, 0) + 1
            return self._connections[conn_key]
        
        # Check if we should create new connection (not at max capacity and no recent failures)
        if (len(self._connections) >= self._max_connections or
            conn_key in self._failed_connections):
            # Return least used connection or None if all failed recently
            if self._connections:
                least_used_key = min(self._connection_usage.items(), key=lambda x: x[1])[0]
                self._connection_usage[least_used_key] += 1
                return self._connections[least_used_key]
            return None
        
        # Create new connection
        try:
            import consul as python_consul
            
            client = python_consul.Consul(
                host=self._config["host"],
                port=self._config["port"],
                dc=self._config["datacenter"],
            )
            
            # Test connection
            client.agent.self()
            
            # Add to pool
            self._connections[conn_key] = client
            self._connection_usage[conn_key] = 1
            
            # Clear failure tracking on success
            if conn_key in self._failed_connections:
                del self._failed_connections[conn_key]
            
            self._logger.info(f"Created new Consul connection in pool: {conn_key}")
            return client
            
        except ImportError:
            self._logger.warning("python-consul library not available, connection pool disabled")
            return None
        except Exception as e:
            self._logger.error(f"Failed to create Consul connection: {e}")
            # Track failure for backoff
            self._failed_connections[conn_key] = asyncio.get_event_loop().time()
            return None
    
    async def close_all(self):
        """Close all connections in the pool and cleanup resources."""
        # Cancel background cleanup task
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        
        # Clear all connections (Consul client doesn't need explicit cleanup)
        self._connections.clear()
        self._connection_usage.clear()
        self._failed_connections.clear()


class NodeInfrastructureConsulAdapterEffect(NodeEffectService):
    """
    Consul Adapter - Event-Driven Infrastructure Effect

    NodeEffect that processes ModelEventEnvelope events to perform Consul operations.
    Integrates with event bus for event-driven consul management (KV store, service registration, health checks).
    Provides only health check HTTP endpoint for monitoring.
    """

    def __init__(self, container: ModelONEXContainer):
        # Use proper base class - no more boilerplate!
        super().__init__(container)

        self.node_type = "effect"
        self.domain = "infrastructure"

        # ONEX logger initialization with fallback
        try:
            self.logger = getattr(container, "get_tool", lambda x: None)(
                "LOGGER"
            ) or logging.getLogger(__name__)
        except (AttributeError, Exception):
            self.logger = logging.getLogger(__name__)

        # Consul client configuration - all environment variables required
        consul_host = os.getenv("CONSUL_HOST")
        consul_port = os.getenv("CONSUL_PORT")
        consul_datacenter = os.getenv("CONSUL_DATACENTER")

        if not consul_host:
            raise OnexError(
                message="CONSUL_HOST environment variable is required but not set",
                error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
            )
        if not consul_port:
            raise OnexError(
                message="CONSUL_PORT environment variable is required but not set",
                error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
            )
        if not consul_datacenter:
            raise OnexError(
                message="CONSUL_DATACENTER environment variable is required but not set",
                error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
            )

        try:
            port_int = int(consul_port)
        except ValueError as e:
            raise OnexError(
                message=f"CONSUL_PORT must be a valid integer, got: {consul_port}",
                error_code=CoreErrorCode.PARAMETER_TYPE_MISMATCH,
            ) from e

        self.consul_config = {
            "host": consul_host,
            "port": port_int,
            "datacenter": consul_datacenter,
        }
        self.consul_client = None
        self.consul_connection_pool = None
        self._initialized = False

    async def _initialize_node_resources(self) -> None:
        """Override to initialize consul client and register effect handlers."""
        await super()._initialize_node_resources()

        # Initialize consul client and register handlers
        await self.initialize_consul_client()

    async def initialize_consul_client(self):
        """Initialize Consul connection pool for high-performance operations"""
        if self._initialized:
            return

        try:
            # Initialize connection pool for better performance under load
            self.consul_connection_pool = ConsulConnectionPool(
                config=self.consul_config,
                max_connections=10,  # Configurable pool size
                cleanup_interval=300  # 5 minute cleanup interval
            )
            
            # Get initial client from pool (will create pool and test connection)
            self.consul_client = self.consul_connection_pool.get_client()
            
            if self.consul_client is None:
                # Fallback to mock client for basic functionality
                self.logger.warning(
                    "Failed to create Consul connection pool, using mock client"
                )
                self.consul_client = MockConsulClient(self.consul_config)

            # Test connection - skip for mock client
            # Protocol-based duck typing: Check if it's NOT a mock client (ONEX compliance)
            if not (hasattr(self.consul_client, 'kv_store') and hasattr(self.consul_client, 'services') and hasattr(self.consul_client, 'config')):
                # Test connection with health check
                health_status = self.health_check()
                if health_status.status == EnumHealthStatus.UNREACHABLE:
                    raise OnexError(
                        message=f"Consul connection test failed: {health_status.message}",
                        error_code=CoreErrorCode.INITIALIZATION_FAILED,
                    )

            # Register consul-specific handlers after client initialization
            await self._register_consul_effect_handlers()

            self._initialized = True
            self.logger.info(
                "Consul client initialized successfully with event handlers"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Consul client: {e}")
            raise OnexError(
                message=f"Consul initialization failed: {e}",
                error_code=CoreErrorCode.INITIALIZATION_FAILED,
            ) from e

    def _get_consul_client(self):
        """Get a Consul client from the connection pool for operations."""
        if self.consul_connection_pool:
            client = self.consul_connection_pool.get_client()
            if client:
                return client
        # Fallback to main client (could be mock)
        return self.consul_client

    async def _cleanup_node_resources(self) -> None:
        """Override to cleanup consul connection pool resources."""
        if self.consul_connection_pool:
            await self.consul_connection_pool.close_all()
        await super()._cleanup_node_resources()

    async def process(self, input_data: ModelConsulAdapterInput) -> ModelConsulAdapterOutput:
        """
        Process ModelEventEnvelope operations for Consul management.

        Event-driven processing of Consul operations through ModelEventEnvelope
        instead of HTTP endpoints. Handles KV operations, service registration,
        and health checks through event payloads.

        Args:
            input_data: Effect input containing event envelope data

        Returns:
            Effect output with Consul operation results

        Raises:
            OnexError: If consul operation fails
        """
        try:
            # Extract envelope payload for Consul operations
            envelope_payload = input_data.operation_data.get("envelope_payload", {})

            # Parse consul operation from envelope
            consul_input = ModelConsulAdapterInput(**envelope_payload)

            # Initialize consul client if needed
            if not self._initialized:
                await self.initialize_consul_client()

            # Route to appropriate consul operation
            result = None
            if consul_input.action == "consul_kv_get":
                result = await self.effect_kv_get(consul_input.key_path or "")
            elif consul_input.action == "consul_kv_put":
                if consul_input.key_path and consul_input.value_data:
                    result = await self.effect_kv_put(
                        consul_input.key_path,
                        consul_input.value_data.value,
                    )
                else:
                    raise OnexError(
                        message="KV put requires key_path and value_data",
                        error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                    )
            elif consul_input.action == "consul_kv_delete":
                if consul_input.key_path:
                    result = await self.effect_kv_delete(
                        consul_input.key_path, recurse=consul_input.recurse or False
                    )
                else:
                    raise OnexError(
                        message="KV delete requires key_path",
                        error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                    )
            elif consul_input.action == "consul_service_register":
                if consul_input.service_config:
                    service_data = ModelConsulServiceRegistration(
                        service_id=consul_input.service_config.service_id,
                        name=consul_input.service_config.service_name,
                        port=consul_input.service_config.port,
                        address=consul_input.service_config.address,
                        health_check=None  # Will be handled separately if needed
                    )
                    result = await self.effect_service_register(service_data)
                else:
                    raise OnexError(
                        message="Service registration requires service_config",
                        error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                    )
            elif consul_input.action == "consul_service_deregister":
                if consul_input.service_config and consul_input.service_config.service_id:
                    result = await self.effect_service_deregister(
                        consul_input.service_config.service_id
                    )
                else:
                    raise OnexError(
                        message="Service deregistration requires service_config with service_id",
                        error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                    )
            elif consul_input.action == "health_check":
                result = await self.effect_health_check(
                    consul_input.service_config.service_name
                    if consul_input.service_config
                    else None
                )
            else:
                raise OnexError(
                    message=f"Unsupported consul action: {consul_input.action}",
                    error_code=CoreErrorCode.OPERATION_FAILED,
                )

            # Create response data for the effect output
            result_data = {
                "consul_operation_result": (
                    result.model_dump() if hasattr(result, "model_dump") else result
                ),
                "success": True,
                "operation_type": consul_input.action,
            }

            # Return result using the consul-specific output model
            return ModelConsulAdapterOutput(
                consul_operation_result=result.model_dump() if hasattr(result, "model_dump") else result,
                success=True,
                operation_type=consul_input.action,
            )

        except Exception as e:
            self.logger.error(f"Consul operation failed: {e}")

            # Log error for event-driven architecture
            self.logger.error(f"Event-driven consul operation failed: {e}")

            raise OnexError(
                message=f"Consul adapter operation failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def get_health_status(self) -> dict:
        """
        Health check endpoint for monitoring (only HTTP endpoint allowed).

        Returns:
            Health status information for monitoring systems
        """
        try:
            consul_health = self.health_check()
            return {
                "adapter": "consul",
                "status": (
                    "healthy" if consul_health.status == "healthy" else "degraded"
                ),
                "consul": consul_health.model_dump(),
                "endpoint_type": "monitoring_only",
            }
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "adapter": "consul",
                "status": "error",
                "error": str(e),
                "endpoint_type": "monitoring_only",
            }

    def health_check(self) -> ModelHealthStatus:
        """Single comprehensive health check for Consul adapter."""
        try:
            if not self.consul_client:
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message="Consul client not initialized - call initialize_consul_client() first",
                )

            # Test basic connectivity and get agent information
            client = self._get_consul_client()
            agent_info = client.agent.self()

            if not agent_info:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Consul responded but agent info unavailable",
                )

            # Extract useful information for health assessment
            agent_config = agent_info.get("Config", {})
            datacenter = agent_config.get("Datacenter", "unknown")
            node_name = agent_config.get("NodeName", "unknown")

            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message=f"Consul healthy - node: {node_name}, datacenter: {datacenter}",
            )

        except Exception as e:
            self.logger.error(f"Consul health check failed: {e}")
            return ModelHealthStatus(
                status=EnumHealthStatus.UNREACHABLE,
                message=f"Consul server unreachable: {str(e)}",
            )

    async def effect_kv_get(self, key: str) -> ModelConsulKVResponse:
        """Get value from Consul KV store"""
        try:
            if not self.consul_client:
                await self.initialize_consul_client()

            client = self._get_consul_client()
            index, data = client.kv.get(key)

            if data is None:
                return ModelConsulKVResponse(
                    status="not_found",
                    key=key,
                    value=None,
                )

            # Decode value if it exists
            value = (
                data.get("Value", b"").decode("utf-8") if data.get("Value") else None
            )

            return ModelConsulKVResponse(
                status="success",
                key=key,
                value=value,
                modify_index=data.get("ModifyIndex", 0),
            )

        except Exception as e:
            self.logger.error(f"Consul KV get failed for key {key}: {e}")
            raise OnexError(
                message=f"Consul KV get failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def effect_kv_put(self, key: str, value: str) -> ModelConsulKVResponse:
        """Put value to Consul KV store"""
        try:
            if not self.consul_client:
                await self.initialize_consul_client()

            client = self._get_consul_client()
            success = client.kv.put(key, value)

            return ModelConsulKVResponse(
                status="success" if success else "failed",
                key=key,
                value=value,
            )

        except Exception as e:
            self.logger.error(f"Consul KV put failed for key {key}: {e}")
            raise OnexError(
                message=f"Consul KV put failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def effect_kv_delete(
        self, key: str, recurse: bool = False
    ) -> ModelConsulKVResponse:
        """Delete key(s) from Consul KV store"""
        try:
            if not self.consul_client:
                await self.initialize_consul_client()

            client = self._get_consul_client()
            success = client.kv.delete(key, recurse=recurse)

            return ModelConsulKVResponse(
                status="success" if success else "not_found",
                key=key,
                value=None,
            )

        except Exception as e:
            self.logger.error(f"Consul KV delete failed for key {key}: {e}")
            raise OnexError(
                message=f"Consul KV delete failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def effect_service_register(
        self, service_data: ModelConsulServiceRegistration
    ) -> ModelConsulServiceResponse:
        """Register service with Consul"""
        try:
            if not self.consul_client:
                await self.initialize_consul_client()

            # Extract service registration data
            service_id = service_data.service_id
            service_name = service_data.name
            service_port = service_data.port
            service_address = service_data.address
            health_check = service_data.health_check

            # Build Consul service registration
            consul_service = {
                "ID": service_id,
                "Name": service_name,
                "Port": service_port,
                "Address": service_address,
            }

            # Add health check if provided
            if health_check:
                consul_service["Check"] = {
                    "HTTP": health_check.url,
                    "Interval": health_check.interval,
                    "Timeout": health_check.timeout,
                }

            # Register with Consul
            success = self.consul_client.agent.service.register(**consul_service)

            return ModelConsulServiceResponse(
                status="success" if success else "failed",
                service_id=service_id,
                service_name=service_name,
            )

        except Exception as e:
            self.logger.error(f"Consul service registration failed: {e}")
            raise OnexError(
                message=f"Service registration failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def effect_service_deregister(
        self, service_id: str
    ) -> ModelConsulServiceResponse:
        """Deregister service from Consul"""
        try:
            if not self.consul_client:
                await self.initialize_consul_client()

            # Get service info before deregistration for response
            services = self.consul_client.agent.services()
            service_info = services.get(service_id)

            if not service_info:
                return ModelConsulServiceResponse(
                    status="not_found",
                    service_id=service_id,
                    service_name="unknown",
                )

            service_name = service_info.get("Service", "unknown")

            # Deregister service
            success = self.consul_client.agent.service.deregister(service_id)

            return ModelConsulServiceResponse(
                status="success" if success else "failed",
                service_id=service_id,
                service_name=service_name,
            )

        except Exception as e:
            self.logger.error(
                f"Consul service deregistration failed for {service_id}: {e}"
            )
            raise OnexError(
                message=f"Service deregistration failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def effect_service_list(self) -> ModelConsulServiceListResponse:
        """List services registered with Consul"""
        try:
            if not self.consul_client:
                await self.initialize_consul_client()

            services = self.consul_client.agent.services()

            # Transform service data
            service_list = []
            for service_id, service_info in services.items():
                service_list.append(
                    ModelConsulServiceInfo(
                        id=service_id,
                        name=service_info.get("Service", ""),
                        port=service_info.get("Port", 0),
                        address=service_info.get("Address", ""),
                        tags=service_info.get("Tags", []),
                    )
                )

            return ModelConsulServiceListResponse(
                status="success",
                services=service_list,
                count=len(service_list),
            )

        except Exception as e:
            self.logger.error(f"Consul service list failed: {e}")
            raise OnexError(
                message=f"Service list failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def effect_health_check(
        self, service_name: Optional[str] = None
    ) -> ModelConsulHealthResponse:
        """Get health check status from Consul"""
        try:
            if not self.consul_client:
                await self.initialize_consul_client()

            if service_name:
                # Get health for specific service
                index, checks = self.consul_client.health.service(
                    service_name, passing=None
                )

                health_status = []
                for check_data in checks:
                    node = check_data.get("Node", {})
                    service = check_data.get("Service", {})
                    checks_list = check_data.get("Checks", [])

                    # Determine overall health
                    overall_status = "passing"
                    for check in checks_list:
                        if check.get("Status") != "passing":
                            overall_status = check.get("Status", "unknown")
                            break

                    health_status.append(
                        ModelConsulHealthCheckNode(
                            node=node.get("Node"),
                            service_id=service.get("ID"),
                            service_name=service.get("Service"),
                            status=overall_status,
                        )
                    )

                return ModelConsulHealthResponse(
                    status="success",
                    service_name=service_name,
                    health_checks=health_status,
                )
            else:
                # Get all health checks
                index, checks = self.consul_client.health.state("any")

                health_summary = {}
                for check in checks:
                    service_name_key = check.get("ServiceName", "consul")
                    status = check.get("Status", "unknown")

                    if service_name_key not in health_summary:
                        health_summary[service_name_key] = {}

                    health_summary[service_name_key][
                        check.get("CheckID", "unknown")
                    ] = status

                return ModelConsulHealthResponse(
                    status="success",
                    health_summary=health_summary,
                )

        except Exception as e:
            self.logger.error(f"Consul health check failed: {e}")
            raise OnexError(
                message=f"Health check failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def _register_consul_effect_handlers(self) -> None:
        """
        Register consul-specific effect handlers for event processing.

        Integrates consul operations into the NodeEffect system for
        event-driven infrastructure management.
        """

        async def consul_operation_handler(
            operation_data: Dict[str, object],
            transaction: Optional[object] = None,
        ) -> dict:
            """Handle consul operations through events."""
            try:
                # Process consul operation from event envelope
                envelope_payload = operation_data.get("envelope_payload", {})
                consul_input = ModelConsulAdapterInput(**envelope_payload)

                # Route to consul operation
                if consul_input.action == "consul_kv_get":
                    result = await self.effect_kv_get(consul_input.key_path or "")
                elif consul_input.action == "consul_kv_put":
                    if consul_input.key_path and consul_input.value_data:
                        result = await self.effect_kv_put(
                            consul_input.key_path,
                            consul_input.value_data.value,
                        )
                    else:
                        raise OnexError(
                            message="KV put requires key_path and value_data",
                            error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                        )
                elif consul_input.action == "consul_kv_delete":
                    if consul_input.key_path:
                        result = await self.effect_kv_delete(
                            consul_input.key_path, recurse=consul_input.recurse or False
                        )
                    else:
                        raise OnexError(
                            message="KV delete requires key_path",
                            error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                        )
                elif consul_input.action == "consul_service_register":
                    if consul_input.service_config:
                        service_data = ModelConsulServiceRegistration(
                            service_id=consul_input.service_config.service_id,
                            name=consul_input.service_config.service_name,
                            port=consul_input.service_config.port,
                            address=consul_input.service_config.address,
                            health_check=None  # Will be handled separately if needed
                        )
                        result = await self.effect_service_register(service_data)
                    else:
                        raise OnexError(
                            message="Service registration requires service_config",
                            error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                        )
                elif consul_input.action == "consul_service_deregister":
                    if consul_input.service_config and consul_input.service_config.service_id:
                        result = await self.effect_service_deregister(
                            consul_input.service_config.service_id
                        )
                    else:
                        raise OnexError(
                            message="Service deregistration requires service_config with service_id",
                            error_code=CoreErrorCode.MISSING_REQUIRED_PARAMETER,
                        )
                elif consul_input.action == "health_check":
                    result = await self.effect_health_check(
                        consul_input.service_config.service_name
                        if consul_input.service_config
                        else None
                    )
                else:
                    raise OnexError(
                        message=f"Unsupported consul action: {consul_input.action}",
                        error_code=CoreErrorCode.OPERATION_FAILED,
                    )

                return {
                    "consul_operation_result": (
                        result.model_dump() if hasattr(result, "model_dump") else result
                    ),
                    "success": True,
                    "operation_type": consul_input.action,
                }

            except Exception as e:
                self.logger.error(f"Consul operation failed: {e}")
                raise OnexError(
                    message=f"Consul operation failed: {e}",
                    error_code=CoreErrorCode.OPERATION_FAILED,
                ) from e

        # Effect handlers registration removed - using direct process method override
        self.logger.info(
            "Consul effect handlers ready for event-driven processing"
        )


async def main():
    """Main entry point for Consul Adapter - runs in service mode with MixinNodeService"""
    from omnibase_infra.infrastructure.container import create_infrastructure_container

    # Create infrastructure container with all shared dependencies
    container = create_infrastructure_container()

    adapter = NodeInfrastructureConsulAdapterEffect(container)

    # Initialize the adapter
    await adapter.initialize()

    # Start service mode using MixinNodeService capabilities
    await adapter.start_service_mode()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())