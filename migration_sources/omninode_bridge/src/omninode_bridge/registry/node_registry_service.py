#!/usr/bin/env python3
"""
Node Registry Service - Node Discovery and Dual Registration.

Manages node registration with dual-backend strategy:
- Consul: Service discovery and health monitoring
- PostgreSQL: Tool orchestration and queryable registry

Features:
- Kafka event-driven registration
- Dual registration with fallback
- Health monitoring and heartbeats
- Capability tracking
- Search and discovery API

ONEX v2.0 Compliance:
- Suffix-based naming: NodeRegistryService
- Contract-driven from node_registry.yaml
- Event-driven architecture with Kafka
- Effect node pattern (I/O operations)
- Graceful degradation on backend failures

Performance Targets:
- <100ms registration operations
- 50+ registrations per second
- 99% dual-registration consistency
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# === Enums ===


class EnumNodeType(str, Enum):
    """ONEX node types."""

    EFFECT = "effect"
    COMPUTE = "compute"
    REDUCER = "reducer"
    ORCHESTRATOR = "orchestrator"


class EnumHealthStatus(str, Enum):
    """Node health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class EnumRegistryOperation(str, Enum):
    """Registry operation types."""

    REGISTER_NODE = "register_node"
    DEREGISTER_NODE = "deregister_node"
    UPDATE_NODE_HEALTH = "update_node_health"
    QUERY_NODES = "query_nodes"
    REQUEST_INTROSPECTION = "request_introspection"


# === Models ===


class ModelCapability(BaseModel):
    """Node capability model."""

    name: str
    description: Optional[str] = None


class ModelEndpoint(BaseModel):
    """Node endpoint configuration."""

    protocol: str = "http"  # http, grpc, kafka
    host: str
    port: int
    path: Optional[str] = None


class ModelHealthCheck(BaseModel):
    """Health check configuration."""

    endpoint: str
    interval_seconds: int = 30
    timeout_seconds: int = 5


class ModelNodeIntrospection(BaseModel):
    """Node introspection data."""

    node_id: str
    node_name: str
    node_type: EnumNodeType
    version: str
    capabilities: list[ModelCapability] = Field(default_factory=list)
    endpoints: Optional[dict[str, ModelEndpoint]] = None
    health_check: Optional[ModelHealthCheck] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelNodeRegistrationInput(BaseModel):
    """Input for node registration."""

    operation_type: EnumRegistryOperation = EnumRegistryOperation.REGISTER_NODE
    node_introspection: ModelNodeIntrospection
    correlation_id: Optional[UUID] = Field(default_factory=uuid4)


class ModelNodeQueryInput(BaseModel):
    """Input for node query."""

    operation_type: EnumRegistryOperation = EnumRegistryOperation.QUERY_NODES
    query_filters: dict[str, Any] = Field(default_factory=dict)
    correlation_id: Optional[UUID] = Field(default_factory=uuid4)


class ModelRegistrationResult(BaseModel):
    """Registration result."""

    node_id: str
    consul_registered: bool = False
    postgres_registered: bool = False
    registration_timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ModelNodeRegistrationOutput(BaseModel):
    """Output from node registration."""

    success: bool
    operation_type: EnumRegistryOperation
    execution_time_ms: int
    registration_result: Optional[ModelRegistrationResult] = None
    kafka_event_published: bool = False
    error_message: Optional[str] = None
    error_details: Optional[dict[str, Any]] = None


class ModelNodeQueryOutput(BaseModel):
    """Output from node query."""

    success: bool
    operation_type: EnumRegistryOperation = EnumRegistryOperation.QUERY_NODES
    execution_time_ms: int
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    total_count: int = 0
    kafka_event_published: bool = False


@dataclass
class RegisteredNode:
    """Internal representation of registered node."""

    node_id: str
    node_name: str
    node_type: EnumNodeType
    version: str
    capabilities: list[ModelCapability]
    endpoints: dict[str, ModelEndpoint] = field(default_factory=dict)
    health_status: EnumHealthStatus = EnumHealthStatus.UNKNOWN
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(UTC))
    registration_timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)
    consul_registered: bool = False
    postgres_registered: bool = False


# === Node Registry Service ===


class NodeRegistryService:
    """
    Node Registry Service - EXPERIMENTAL.

    **Implementation Status:** Core orchestration complete, backend integrations pending.
    Currently operates in cache-only mode. Consul and PostgreSQL registration methods
    are defined but return False (not implemented).

    **Roadmap:**
    - Phase 1: ✅ Architecture and models (COMPLETE)
    - Phase 2: ✅ Cache-based registration (COMPLETE)
    - Phase 3: ⏳ Consul integration (PENDING)
    - Phase 4: ⏳ PostgreSQL integration (PENDING)
    - Phase 5: ⏳ Kafka event publishing (PENDING)

    Manages node registration with dual-backend strategy:
    - Consul: Service discovery and health monitoring (Phase 2)
    - PostgreSQL: Tool orchestration and query API (Phase 2)

    Current Features (Cache-Only):
    - Event-driven registration orchestration
    - In-memory cache registration
    - Search and discovery API
    - Health monitoring
    - Capability tracking

    Performance Targets (when backends implemented):
    - <100ms registration operations
    - 50+ registrations per second
    - 99% dual-registration consistency
    """

    def __init__(
        self,
        enable_consul: bool = False,
        enable_postgres: bool = False,
        enable_kafka: bool = True,
    ) -> None:
        """
        Initialize node registry service.

        Args:
            enable_consul: Enable Consul registration
            enable_postgres: Enable PostgreSQL registration
            enable_kafka: Enable Kafka event publishing
        """
        self.enable_consul = enable_consul
        self.enable_postgres = enable_postgres
        self.enable_kafka = enable_kafka

        # In-memory registry (cache)
        self._registered_nodes: dict[str, RegisteredNode] = {}

        # Metrics
        self._total_registrations = 0
        self._successful_registrations = 0
        self._failed_registrations = 0
        self._consul_failures = 0
        self._postgres_failures = 0
        self._kafka_events_published = 0

        logger.info(
            f"NodeRegistryService initialized: "
            f"consul={enable_consul}, postgres={enable_postgres}, kafka={enable_kafka}"
        )

    async def register_node(
        self, input_data: ModelNodeRegistrationInput
    ) -> ModelNodeRegistrationOutput:
        """
        Register a node in the registry.

        Performs dual registration:
        1. Consul (service discovery)
        2. PostgreSQL (tool orchestration)

        Falls back gracefully if either backend is unavailable.

        Args:
            input_data: Node registration input

        Returns:
            Registration result
        """
        start_time = datetime.now(UTC)
        self._total_registrations += 1

        try:
            node = input_data.node_introspection

            # Validate node data
            if not node.node_id or not node.node_name:
                return ModelNodeRegistrationOutput(
                    success=False,
                    operation_type=EnumRegistryOperation.REGISTER_NODE,
                    execution_time_ms=0,
                    error_message="Invalid node data: missing node_id or node_name",
                )

            # Check if already registered
            if node.node_id in self._registered_nodes:
                logger.info(f"Node {node.node_id} already registered - updating")

            # Register in Consul
            consul_success = False
            if self.enable_consul:
                consul_success = await self._register_in_consul(node)
                if not consul_success:
                    self._consul_failures += 1
                    logger.warning(f"Consul registration failed for {node.node_id}")

            # Register in PostgreSQL
            postgres_success = False
            if self.enable_postgres:
                postgres_success = await self._register_in_postgres(node)
                if not postgres_success:
                    self._postgres_failures += 1
                    logger.warning(f"PostgreSQL registration failed for {node.node_id}")

            # Update in-memory cache
            registered_node = RegisteredNode(
                node_id=node.node_id,
                node_name=node.node_name,
                node_type=node.node_type,
                version=node.version,
                capabilities=node.capabilities,
                endpoints=node.endpoints or {},
                health_status=EnumHealthStatus.HEALTHY,
                metadata=node.metadata,
                consul_registered=consul_success,
                postgres_registered=postgres_success,
            )
            self._registered_nodes[node.node_id] = registered_node

            # Publish to Kafka
            kafka_published = False
            if self.enable_kafka:
                kafka_published = await self._publish_registration_event(node)
                if kafka_published:
                    self._kafka_events_published += 1

            # Compute execution time
            execution_time_ms = int(
                (datetime.now(UTC) - start_time).total_seconds() * 1000
            )

            # Check if registration was successful
            # Success requires at least one backend to succeed
            # In-memory cache is always updated regardless (lines 290-302)
            overall_success = consul_success or postgres_success

            if overall_success:
                self._successful_registrations += 1
            else:
                self._failed_registrations += 1

            logger.info(
                f"Node {node.node_id} registered: "
                f"consul={consul_success}, postgres={postgres_success}, "
                f"time={execution_time_ms}ms"
            )

            return ModelNodeRegistrationOutput(
                success=overall_success,
                operation_type=EnumRegistryOperation.REGISTER_NODE,
                execution_time_ms=execution_time_ms,
                registration_result=ModelRegistrationResult(
                    node_id=node.node_id,
                    consul_registered=consul_success,
                    postgres_registered=postgres_success,
                ),
                kafka_event_published=kafka_published,
            )

        except Exception as e:
            self._failed_registrations += 1
            execution_time_ms = int(
                (datetime.now(UTC) - start_time).total_seconds() * 1000
            )
            logger.error(f"Error registering node: {e}", exc_info=True)

            return ModelNodeRegistrationOutput(
                success=False,
                operation_type=EnumRegistryOperation.REGISTER_NODE,
                execution_time_ms=execution_time_ms,
                error_message=str(e),
                error_details={"exception_type": type(e).__name__},
            )

    async def deregister_node(self, node_id: str) -> bool:
        """
        Deregister a node from the registry.

        Args:
            node_id: Node identifier

        Returns:
            True if deregistration successful
        """
        if node_id not in self._registered_nodes:
            logger.warning(f"Node {node_id} not found in registry")
            return False

        # Deregister from Consul
        if self.enable_consul:
            await self._deregister_from_consul(node_id)

        # Deregister from PostgreSQL
        if self.enable_postgres:
            await self._deregister_from_postgres(node_id)

        # Remove from cache
        del self._registered_nodes[node_id]

        # Publish to Kafka
        if self.enable_kafka:
            await self._publish_deregistration_event(node_id)

        logger.info(f"Node {node_id} deregistered")
        return True

    async def update_node_health(
        self, node_id: str, health_status: EnumHealthStatus
    ) -> bool:
        """
        Update node health status.

        Args:
            node_id: Node identifier
            health_status: New health status

        Returns:
            True if update successful
        """
        if node_id not in self._registered_nodes:
            logger.warning(f"Node {node_id} not found in registry")
            return False

        node = self._registered_nodes[node_id]
        node.health_status = health_status
        node.last_heartbeat = datetime.now(UTC)

        # Update in backends
        if self.enable_consul:
            await self._update_consul_health(node_id, health_status)

        if self.enable_postgres:
            await self._update_postgres_health(node_id, health_status)

        logger.debug(f"Node {node_id} health updated to {health_status}")
        return True

    async def query_nodes(
        self, input_data: ModelNodeQueryInput
    ) -> ModelNodeQueryOutput:
        """
        Query nodes with filters.

        Supports filtering by:
        - node_type: Filter by ONEX node type
        - capability: Filter by capability name
        - health_status: Filter by health status
        - version: Filter by version
        - name: Filter by node name (partial match)

        Args:
            input_data: Query input with filters

        Returns:
            Query results
        """
        start_time = datetime.now(UTC)

        try:
            filters = input_data.query_filters
            results = []

            for node in self._registered_nodes.values():
                # Apply filters
                if (
                    filters.get("node_type")
                    and node.node_type.value != filters["node_type"]
                ):
                    continue

                if (
                    filters.get("health_status")
                    and node.health_status.value != filters["health_status"]
                ):
                    continue

                if filters.get("version") and node.version != filters["version"]:
                    continue

                if filters.get("name"):
                    if filters["name"].lower() not in node.node_name.lower():
                        continue

                if filters.get("capability"):
                    capability_match = any(
                        cap.name == filters["capability"] for cap in node.capabilities
                    )
                    if not capability_match:
                        continue

                # Node matches all filters
                results.append(
                    {
                        "node_id": node.node_id,
                        "node_name": node.node_name,
                        "node_type": node.node_type.value,
                        "version": node.version,
                        "health_status": node.health_status.value,
                        "capabilities": [
                            {"name": cap.name, "description": cap.description}
                            for cap in node.capabilities
                        ],
                        "last_heartbeat": node.last_heartbeat.isoformat(),
                        "registration_timestamp": node.registration_timestamp.isoformat(),
                        "consul_registered": node.consul_registered,
                        "postgres_registered": node.postgres_registered,
                    }
                )

            execution_time_ms = int(
                (datetime.now(UTC) - start_time).total_seconds() * 1000
            )

            logger.debug(
                f"Query completed: {len(results)} nodes matched "
                f"(filters={filters}, time={execution_time_ms}ms)"
            )

            return ModelNodeQueryOutput(
                success=True,
                execution_time_ms=execution_time_ms,
                nodes=results,
                total_count=len(results),
            )

        except Exception as e:
            execution_time_ms = int(
                (datetime.now(UTC) - start_time).total_seconds() * 1000
            )
            logger.error(f"Error querying nodes: {e}", exc_info=True)

            return ModelNodeQueryOutput(
                success=False,
                execution_time_ms=execution_time_ms,
                nodes=[],
                total_count=0,
            )

    async def get_all_nodes(self) -> list[RegisteredNode]:
        """Get all registered nodes."""
        return list(self._registered_nodes.values())

    async def get_node_by_id(self, node_id: str) -> Optional[RegisteredNode]:
        """Get node by ID."""
        return self._registered_nodes.get(node_id)

    async def get_nodes_by_type(self, node_type: EnumNodeType) -> list[RegisteredNode]:
        """Get all nodes of a specific type."""
        return [
            node
            for node in self._registered_nodes.values()
            if node.node_type == node_type
        ]

    async def get_nodes_by_capability(
        self, capability_name: str
    ) -> list[RegisteredNode]:
        """Get all nodes with a specific capability."""
        return [
            node
            for node in self._registered_nodes.values()
            if any(cap.name == capability_name for cap in node.capabilities)
        ]

    async def get_healthy_nodes(self) -> list[RegisteredNode]:
        """Get all healthy nodes."""
        return [
            node
            for node in self._registered_nodes.values()
            if node.health_status == EnumHealthStatus.HEALTHY
        ]

    async def _register_in_consul(self, node: ModelNodeIntrospection) -> bool:
        """Register node in Consul service discovery."""
        # Consul registration (Phase 2 implementation pending)
        # This would use python-consul library to register the service
        logger.debug(f"Would register {node.node_id} in Consul")
        return False  # Not implemented yet

    async def _register_in_postgres(self, node: ModelNodeIntrospection) -> bool:
        """Register node in PostgreSQL tool registry."""
        # PostgreSQL registration (Phase 2 implementation pending)
        # This would insert into registered_nodes table
        logger.debug(f"Would register {node.node_id} in PostgreSQL")
        return False  # Not implemented yet

    async def _deregister_from_consul(self, node_id: str) -> bool:
        """Deregister node from Consul."""
        # Consul deregistration (Phase 2 implementation pending)
        logger.debug(f"Would deregister {node_id} from Consul")
        return False

    async def _deregister_from_postgres(self, node_id: str) -> bool:
        """Deregister node from PostgreSQL."""
        # PostgreSQL deregistration (Phase 2 implementation pending)
        logger.debug(f"Would deregister {node_id} from PostgreSQL")
        return False

    async def _update_consul_health(
        self, node_id: str, health_status: EnumHealthStatus
    ) -> bool:
        """Update health status in Consul."""
        # Consul health update (Phase 2 implementation pending)
        logger.debug(f"Would update {node_id} health to {health_status} in Consul")
        return False

    async def _update_postgres_health(
        self, node_id: str, health_status: EnumHealthStatus
    ) -> bool:
        """Update health status in PostgreSQL."""
        # PostgreSQL health update (Phase 2 implementation pending)
        logger.debug(f"Would update {node_id} health to {health_status} in PostgreSQL")
        return False

    async def _publish_registration_event(self, node: ModelNodeIntrospection) -> bool:
        """Publish node registration event to Kafka."""
        # Kafka event publishing (Phase 2 implementation pending)
        logger.debug(f"Would publish registration event for {node.node_id}")
        return False

    async def _publish_deregistration_event(self, node_id: str) -> bool:
        """Publish node deregistration event to Kafka."""
        # Kafka event publishing (Phase 2 implementation pending)
        logger.debug(f"Would publish deregistration event for {node_id}")
        return False

    async def get_metrics(self) -> dict[str, Any]:
        """Get registry metrics."""
        return {
            "total_registrations": self._total_registrations,
            "successful_registrations": self._successful_registrations,
            "failed_registrations": self._failed_registrations,
            "consul_failures": self._consul_failures,
            "postgres_failures": self._postgres_failures,
            "kafka_events_published": self._kafka_events_published,
            "registered_nodes_total": len(self._registered_nodes),
            "registered_nodes_by_type": {
                node_type.value: len(
                    [
                        n
                        for n in self._registered_nodes.values()
                        if n.node_type == node_type
                    ]
                )
                for node_type in EnumNodeType
            },
            "registered_nodes_by_health": {
                health.value: len(
                    [
                        n
                        for n in self._registered_nodes.values()
                        if n.health_status == health
                    ]
                )
                for health in EnumHealthStatus
            },
            "dual_registration_consistency": (
                len(
                    [
                        n
                        for n in self._registered_nodes.values()
                        if n.consul_registered and n.postgres_registered
                    ]
                )
                / len(self._registered_nodes)
                if len(self._registered_nodes) > 0
                else 0.0
            ),
        }
