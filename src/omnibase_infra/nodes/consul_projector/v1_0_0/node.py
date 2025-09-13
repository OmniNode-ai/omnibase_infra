#!/usr/bin/env python3

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from omnibase_core.exceptions.base_onex_error import OnexError
from omnibase_core.enums.enum_core_error_code import CoreErrorCode
from omnibase_core.node_effect import (
    EffectType,
    ModelEffectInput,
    ModelEffectOutput,
)
from omnibase_core.node_effect_service import NodeEffectService
from omnibase_core.onex_container import ONEXContainer
from omnibase_core.enums.enum_health_status import EnumHealthStatus
from omnibase_core.model.core.model_health_status import ModelHealthStatus

# Import shared Consul models
from omnibase_infra.models.consul import (
    ModelConsulServiceListResponse,
    ModelConsulHealthResponse,
    ModelConsulKVResponse,
)

# Import node-specific models
from .models import (
    ModelConsulProjectorInput,
    ModelConsulProjectorOutput,
    ModelConsulServiceProjection,
    ModelConsulHealthProjection,
    ModelConsulKVProjection,
    ModelConsulTopologyProjection,
    ModelConsulServiceCacheEntry,
    ModelConsulHealthCacheEntry,
    ModelConsulKVCacheEntry,
)


class NodeInfrastructureConsulProjectorEffect(NodeEffectService):
    """
    Consul Projector - Event-Driven Infrastructure State Projector

    NodeEffect that processes Consul state data to create projected views and aggregations.
    Integrates with event bus for event-driven state projection and monitoring.
    Provides comprehensive state views for service discovery, health monitoring, and topology analysis.
    """

    def __init__(self, container: ONEXContainer):
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

        # State cache for projection optimization with strong typing
        self._service_cache: Dict[str, ModelConsulServiceCacheEntry] = {}
        self._health_cache: Dict[str, ModelConsulHealthCacheEntry] = {}
        self._kv_cache: Dict[str, ModelConsulKVCacheEntry] = {}
        self._cache_ttl: int = 300  # 5 minutes

        self._initialized = False

    async def _initialize_node_resources(self) -> None:
        """Override to initialize projector resources and register effect handlers."""
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

            # Register projector-specific handlers
            await self._register_projector_effect_handlers()

            self._initialized = True
            self.logger.info(
                "Consul projector initialized successfully with event handlers"
            )

        except Exception as e:
            self.logger.error(f"Failed to initialize Consul projector: {e}")
            raise OnexError(
                message=f"Consul projector initialization failed: {e}",
                error_code=CoreErrorCode.INITIALIZATION_FAILED,
            ) from e

    async def process(self, input_data: ModelEffectInput) -> ModelEffectOutput:
        """
        Process ModelEventEnvelope operations for Consul state projection.

        Event-driven processing of Consul projection operations through ModelEventEnvelope.
        Handles service state projection, health aggregation, KV state tracking,
        and topology analysis.

        Args:
            input_data: Effect input containing event envelope data

        Returns:
            Effect output with projection results

        Raises:
            OnexError: If projection operation fails
        """
        try:
            # Extract envelope payload for projection operations
            envelope_payload = input_data.operation_data.get("envelope_payload", {})

            # Parse projector operation from envelope
            projector_input = ModelConsulProjectorInput(**envelope_payload)

            # Initialize projector if needed
            if not self._initialized:
                await self._initialize_projector()

            # Route to appropriate projection operation
            result = None
            timestamp = datetime.now(timezone.utc).isoformat()

            if projector_input.projection_type == "service_state":
                result = await self._project_service_state(projector_input)
            elif projector_input.projection_type == "health_state":
                result = await self._project_health_state(projector_input)
            elif projector_input.projection_type == "kv_state":
                result = await self._project_kv_state(projector_input)
            elif projector_input.projection_type == "topology":
                result = await self._project_topology(projector_input)
            else:
                raise OnexError(
                    message=f"Unsupported projection type: {projector_input.projection_type}",
                    error_code=CoreErrorCode.OPERATION_FAILED,
                )

            # Create projector output
            projector_output = ModelConsulProjectorOutput(
                projection_result=result.model_dump() if hasattr(result, "model_dump") else result,
                projection_type=projector_input.projection_type,
                timestamp=timestamp,
                metadata={
                    "cache_used": True,  # TODO: Implement cache usage tracking
                    "aggregation_window": projector_input.aggregation_window,
                } if projector_input.include_metadata else None,
            )

            # Return the result directly since we override process completely
            from omnibase_core.node_effect import TransactionState

            return ModelEffectOutput(
                result=projector_output.model_dump(),
                operation_id=input_data.operation_id,
                effect_type=input_data.effect_type,
                transaction_state=TransactionState.COMMITTED,
                processing_time_ms=0,  # Will be calculated by parent
            )

        except Exception as e:
            self.logger.error(f"Consul projection operation failed: {e}")
            raise OnexError(
                message=f"Consul projector operation failed: {e}",
                error_code=CoreErrorCode.OPERATION_FAILED,
            ) from e

    def health_check(self) -> ModelHealthStatus:
        """Single comprehensive health check for Consul projector."""
        try:
            if not self._initialized:
                return ModelHealthStatus(
                    status=EnumHealthStatus.UNHEALTHY,
                    message="Consul projector not initialized - call _initialize_projector() first",
                )

            # Check cache health and projector state
            cache_health = len(self._service_cache) + len(self._health_cache) + len(self._kv_cache)
            
            if cache_health == 0:
                return ModelHealthStatus(
                    status=EnumHealthStatus.DEGRADED,
                    message="Consul projector operational but caches empty - may need initial data",
                )

            return ModelHealthStatus(
                status=EnumHealthStatus.HEALTHY,
                message=f"Consul projector healthy - cache entries: {cache_health}",
            )

        except Exception as e:
            self.logger.error(f"Consul projector health check failed: {e}")
            return ModelHealthStatus(
                status=EnumHealthStatus.UNREACHABLE,
                message=f"Consul projector health check failed: {str(e)}",
            )

    async def _project_service_state(self, input_data: ModelConsulProjectorInput) -> ModelConsulServiceProjection:
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
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                },
                {
                    "service_id": "service-2", 
                    "service_name": "user-service",
                    "instances": 2,
                    "health_status": "warning",
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                },
            ]

            # Apply service filtering if specified
            target_services = getattr(input_data, 'target_services', [])
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

    async def _project_health_state(self, input_data: ModelConsulProjectorInput) -> ModelConsulHealthProjection:
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

    async def _project_kv_state(self, input_data: ModelConsulProjectorInput) -> ModelConsulKVProjection:
        """Project KV store state changes from Consul data."""
        try:
            # TODO: Integration with Consul adapter to get KV data
            # For now, return mock projection structure

            key_summary = {
                "total_keys": 25,
                "prefixes": ["config/", "services/", "features/"],
            }

            # Optional key details based on include_values flag
            key_details = None
            # if input_data.include_values:  # TODO: Add this field to request model
            #     key_details = [
            #         {
            #             "key": "config/database/host",
            #             "value": "localhost",
            #             "modify_index": 1001,
            #         },
            #     ]

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

    async def _project_topology(self, input_data: ModelConsulProjectorInput) -> ModelConsulTopologyProjection:
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
                "depth": getattr(input_data, 'depth', 2),
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

    async def _register_projector_effect_handlers(self) -> None:
        """
        Register projector-specific effect handlers for event processing.

        Integrates projection operations into the NodeEffect system for
        event-driven state projection and monitoring.
        """

        async def projection_handler(
            operation_data: Dict[str, object],
            transaction: Optional[object] = None,
        ) -> Dict[str, object]:
            """Handle projection operations through events."""
            try:
                # Process projection operation from event envelope
                envelope_payload = operation_data.get("envelope_payload", {})
                projector_input = ModelConsulProjectorInput(**envelope_payload)

                # Route to projection operation
                timestamp = datetime.now(timezone.utc).isoformat()
                
                if projector_input.projection_type == "service_state":
                    result = await self._project_service_state(projector_input)
                elif projector_input.projection_type == "health_state":
                    result = await self._project_health_state(projector_input)
                elif projector_input.projection_type == "kv_state":
                    result = await self._project_kv_state(projector_input)
                elif projector_input.projection_type == "topology":
                    result = await self._project_topology(projector_input)
                else:
                    raise OnexError(
                        message=f"Unsupported projection type: {projector_input.projection_type}",
                        error_code=CoreErrorCode.OPERATION_FAILED,
                    )

                return {
                    "projection_result": (
                        result.model_dump() if hasattr(result, "model_dump") else result
                    ),
                    "projection_type": projector_input.projection_type,
                    "timestamp": timestamp,
                    "success": True,
                }

            except Exception as e:
                self.logger.error(f"Projection operation failed: {e}")
                raise OnexError(
                    message=f"Projection operation failed: {e}",
                    error_code=CoreErrorCode.OPERATION_FAILED,
                ) from e

        # Register the projection handler for API calls and monitoring operations
        self.effect_handlers[EffectType.API_CALL] = projection_handler
        self.effect_handlers[EffectType.MONITORING] = projection_handler

        self.logger.info(
            "Consul projector effect handlers registered for event-driven processing"
        )


async def main():
    """Main entry point for Consul Projector - runs in service mode with MixinNodeService"""
    from omnibase_infra.infrastructure.container import create_infrastructure_container

    # Create infrastructure container with all shared dependencies
    container = create_infrastructure_container()

    projector = NodeInfrastructureConsulProjectorEffect(container)

    # Initialize the projector
    await projector.initialize()

    # Start service mode using MixinNodeService capabilities
    await projector.start_service_mode()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())