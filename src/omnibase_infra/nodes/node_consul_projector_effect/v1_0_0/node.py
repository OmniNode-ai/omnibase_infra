#!/usr/bin/env python3

import asyncio
import logging
from datetime import UTC, datetime

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.core.node_effect_service import NodeEffectService
from omnibase_core.core.onex_container import ModelONEXContainer
from omnibase_core.enums.enum_health_status import EnumHealthStatus
from omnibase_core.models.core.model_health_status import ModelHealthStatus

# Import node-specific models
from .models import (
    ModelConsulHealthCacheEntry,
    ModelConsulHealthProjection,
    ModelConsulKVCacheEntry,
    ModelConsulKVProjection,
    ModelConsulProjectorInput,
    ModelConsulProjectorOutput,
    ModelConsulServiceCacheEntry,
    ModelConsulServiceProjection,
    ModelConsulTopologyProjection,
)


class NodeConsulProjectorEffect(NodeEffectService):
    """
    Consul Projector - Event-Driven Infrastructure State Projector

    NodeEffect that processes Consul state data to create projected views and aggregations.
    Integrates with event bus for event-driven state projection and monitoring.
    Provides comprehensive state views for service discovery, health monitoring, and topology analysis.
    """

    def __init__(self, container: ModelONEXContainer):
        # Use proper base class - no more boilerplate!
        super().__init__(container)

        self.node_type = "effect"
        self.domain = "infrastructure"

        # ONEX logger initialization with fallback
        try:
            self.logger = getattr(container, "get_tool", lambda x: None)(
                "LOGGER",
            ) or logging.getLogger(__name__)
        except (AttributeError, Exception):
            self.logger = logging.getLogger(__name__)

        # State cache for projection optimization with strong typing
        self._service_cache: dict[str, ModelConsulServiceCacheEntry] = {}
        self._health_cache: dict[str, ModelConsulHealthCacheEntry] = {}
        self._kv_cache: dict[str, ModelConsulKVCacheEntry] = {}
        self._cache_ttl: int = 300  # 5 minutes

        self._initialized = False

    async def _initialize_node_resources(self) -> None:
        """Override to initialize projector resources."""
        await super()._initialize_node_resources()

        # Initialize projector-specific resources
        await self._initialize_projector()

    async def _initialize_projector(self):
        """Initialize Consul projector resources"""
        if self._initialized:
            return

        try:
            # Initialize projection caches
            self._service_cache = {}
            self._health_cache = {}
            self._kv_cache = {}

            self._initialized = True
            self.logger.info(
                "Consul projector initialized successfully",
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Consul projector: {e}")
            raise OnexError(
                message=f"Consul projector initialization failed: {e}",
                error_code=CoreErrorCode.INITIALIZATION_FAILED,
            ) from e

    async def project_service_state(self, input_data: ModelConsulProjectorInput) -> ModelConsulServiceProjection:
        """Project current service state from Consul data."""
        try:
            # TODO: Integration with Consul adapter to get service data
            # For now, return mock projection structure

            services = []
            # Mock service projection - replace with actual Consul adapter integration
            mock_services = [
                {
                    "service_id": "service-1",
                    "service_name": "api-gateway",
                    "instances": 3,
                    "health_status": "healthy",
                    "last_updated": datetime.now(UTC).isoformat(),
                },
                {
                    "service_id": "service-2",
                    "service_name": "user-service",
                    "instances": 2,
                    "health_status": "warning",
                    "last_updated": datetime.now(UTC).isoformat(),
                },
            ]

            # Apply service filtering if specified
            target_services = getattr(input_data, "target_services", [])
            if target_services:
                mock_services = [
                    s for s in mock_services
                    if s["service_name"] in target_services
                ]

            services = mock_services

            return ModelConsulServiceProjection(
                services=services,
                total_services=len(services),
            )

        except Exception as e:
            self.logger.error(f"Service state projection failed: {e}")
            raise OnexError(
                message=f"Service state projection failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def project_health_state(self, input_data: ModelConsulProjectorInput) -> ModelConsulHealthProjection:
        """Project health state aggregation from Consul data."""
        try:
            # TODO: Integration with Consul adapter to get health data
            # For now, return mock projection structure

            health_summary = {
                "healthy": 5,
                "warning": 2,
                "critical": 1,
            }

            service_health = [
                {
                    "service_name": "api-gateway",
                    "status": "passing",
                    "check_count": 3,
                },
                {
                    "service_name": "user-service",
                    "status": "warning",
                    "check_count": 2,
                },
                {
                    "service_name": "data-service",
                    "status": "critical",
                    "check_count": 1,
                },
            ]

            return ModelConsulHealthProjection(
                health_summary=health_summary,
                service_health=service_health,
            )

        except Exception as e:
            self.logger.error(f"Health state projection failed: {e}")
            raise OnexError(
                message=f"Health state projection failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def project_kv_state(self, input_data: ModelConsulProjectorInput) -> ModelConsulKVProjection:
        """Project KV store state changes from Consul data."""
        try:
            # TODO: Integration with Consul adapter to get KV data
            # For now, return mock projection structure

            key_summary = {
                "total_keys": 25,
                "prefixes": ["config/", "services/", "features/"],
            }

            # Optional key details
            key_details = None

            return ModelConsulKVProjection(
                key_summary=key_summary,
                key_details=key_details,
            )

        except Exception as e:
            self.logger.error(f"KV state projection failed: {e}")
            raise OnexError(
                message=f"KV state projection failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    async def project_topology(self, input_data: ModelConsulProjectorInput) -> ModelConsulTopologyProjection:
        """Project service topology view from Consul data."""
        try:
            # TODO: Integration with Consul adapter to build topology
            # For now, return mock topology structure

            topology_graph = {
                "nodes": [
                    {"id": "api-gateway", "name": "API Gateway", "type": "gateway"},
                    {"id": "user-service", "name": "User Service", "type": "service"},
                    {"id": "data-service", "name": "Data Service", "type": "service"},
                ],
                "edges": [
                    {"source": "api-gateway", "target": "user-service", "relationship": "depends_on"},
                    {"source": "user-service", "target": "data-service", "relationship": "depends_on"},
                ],
            }

            metrics = {
                "node_count": len(topology_graph["nodes"]),
                "edge_count": len(topology_graph["edges"]),
                "depth": getattr(input_data, "depth", 2),
            }

            return ModelConsulTopologyProjection(
                topology_graph=topology_graph,
                metrics=metrics,
            )

        except Exception as e:
            self.logger.error(f"Topology projection failed: {e}")
            raise OnexError(
                message=f"Topology projection failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    def health_check(self) -> ModelHealthStatus:
        """Single comprehensive health check for Consul projector."""
        try:
            if not self._initialized:
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message="Consul projector not initialized",
                )

            # Check cache health and projector state
            cache_health = len(self._service_cache) + len(self._health_cache) + len(self._kv_cache)

            if cache_health == 0:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Consul projector operational but caches empty",
                )

            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message=f"Consul projector healthy - cache entries: {cache_health}",
            )

        except Exception as e:
            self.logger.error(f"Consul projector health check failed: {e}")
            return ModelHealthStatus(
                status=EnumHealthStatus.UNREACHABLE,
                message=f"Consul projector health check failed: {e!s}",
            )


# Entry point for running the node
if __name__ == "__main__":
    import sys

    # Create container (simplified for standalone operation)
    container = ModelONEXContainer()

    # Create and run the node
    node = NodeConsulProjectorEffect(container)

    # Run the node with asyncio
    try:
        asyncio.run(node.run())
    except KeyboardInterrupt:
        print("\nConsul projector shutting down...")
        sys.exit(0)
