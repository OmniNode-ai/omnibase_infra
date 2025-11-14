"""NodeOmniInfraReducer - Pure Infrastructure State Aggregation with DB Storage.

This reducer aggregates infrastructure state from all adapters (postgres, kafka, consul)
and stores the consolidated state in PostgreSQL. Following ONEX reducer pattern:
- Pure function - no in-memory state
- All state stored in database
- Emits intents to orchestrator based on state changes
"""

import time
from uuid import UUID, uuid4

from omnibase_core.core.errors.onex_error import CoreErrorCode, OnexError
from omnibase_core.core.node_reducer_service import NodeReducerService
from omnibase_core.core.onex_container import ModelONEXContainer

from omnibase_infra.models.postgres.model_postgres_query_request import (
    ModelPostgresQueryRequest,
)

from .models.model_omni_infra_reducer_input import ModelOmniInfraReducerInput
from .models.model_omni_infra_reducer_output import ModelOmniInfraReducerOutput


class NodeOmniInfraReducer(NodeReducerService):
    """
    Pure Infrastructure State Reducer.

    Responsibilities:
    1. Consume events from all infrastructure adapters
    2. Aggregate state and store in PostgreSQL (via postgres_adapter)
    3. Emit intents to orchestrator based on state changes
    4. NO in-memory state - purely functional

    Communication Pattern:
    Adapters → Events → Reducer → DB Storage → Intents → Orchestrator
    """

    def __init__(self, container: ModelONEXContainer):
        """Initialize pure reducer with container injection."""
        super().__init__(container)
        self.node_type = "reducer"
        self.domain = "infrastructure"

        # Get event bus for intent emission
        self._event_bus = self.container.get_service("ProtocolEventBus")
        if self._event_bus is None:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="ProtocolEventBus required for intent emission",
            )

        # Get postgres adapter for state storage
        self._postgres_adapter = self.container.get_service("postgres_adapter")
        if self._postgres_adapter is None:
            raise OnexError(
                code=CoreErrorCode.DEPENDENCY_RESOLUTION_ERROR,
                message="PostgreSQL adapter required for state storage",
            )

    async def reduce(self, input_data: ModelOmniInfraReducerInput) -> ModelOmniInfraReducerOutput:
        """
        Pure reduce function - aggregates state and stores in DB.

        Process:
        1. Parse incoming infrastructure event
        2. Aggregate state from event payload
        3. Store aggregated state in PostgreSQL
        4. Determine if intents should be emitted
        5. Emit intents to orchestrator if needed

        Args:
            input_data: Infrastructure event from adapter

        Returns:
            Output indicating state storage and emitted intents
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Parse event and extract state
            aggregated_state = self._aggregate_state_from_event(input_data)

            # Step 2: Store state in PostgreSQL (pure - no in-memory state)
            state_id = await self._store_state_in_db(aggregated_state, input_data.correlation_id)

            # Step 3: Determine intents based on state
            intents_to_emit = self._determine_intents(aggregated_state)

            # Step 4: Emit intents to orchestrator
            if intents_to_emit:
                await self._emit_intents(intents_to_emit, state_id, input_data.correlation_id)

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            return ModelOmniInfraReducerOutput(
                state_updated=True,
                state_id=state_id,
                intents_emitted=intents_to_emit,
                correlation_id=input_data.correlation_id,
                timestamp=time.time(),
            )

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            raise OnexError(
                code=CoreErrorCode.PROCESSING_ERROR,
                message=f"Reducer processing failed: {str(e)}",
                details={
                    "event_type": input_data.event_type,
                    "adapter_source": input_data.adapter_source,
                    "execution_time_ms": execution_time_ms,
                },
            ) from e

    def _aggregate_state_from_event(self, input_data: ModelOmniInfraReducerInput) -> dict:
        """
        Extract and aggregate state from incoming event.

        Pure function - transforms event payload into aggregated state.
        """
        event_payload = input_data.event_payload
        adapter_source = input_data.adapter_source

        aggregated_state = {
            "adapter_type": adapter_source,
            "timestamp": input_data.timestamp,
            "event_type": input_data.event_type,
        }

        # Adapter-specific state extraction
        if adapter_source == "postgres":
            aggregated_state.update({
                "health_status": event_payload.get("health_status", "unknown"),
                "connection_pool_size": event_payload.get("pool_size", 0),
                "active_connections": event_payload.get("active_connections", 0),
                "circuit_breaker_state": event_payload.get("circuit_breaker_state", "unknown"),
            })

        elif adapter_source == "kafka":
            aggregated_state.update({
                "health_status": event_payload.get("health_status", "unknown"),
                "producer_pool_size": event_payload.get("producer_count", 0),
                "messages_published": event_payload.get("messages_published", 0),
            })

        elif adapter_source == "consul":
            aggregated_state.update({
                "health_status": event_payload.get("health_status", "unknown"),
                "services_registered": event_payload.get("services_count", 0),
                "kv_operations": event_payload.get("kv_operations", 0),
            })

        # Store full metrics as JSON
        aggregated_state["metrics_json"] = event_payload

        return aggregated_state

    async def _store_state_in_db(self, aggregated_state: dict, correlation_id: UUID) -> UUID:
        """
        Store aggregated state in PostgreSQL via postgres_adapter.

        Pure function - stores state and returns state ID.
        """
        state_id = uuid4()

        # Build INSERT query
        query = """
        INSERT INTO infrastructure_state (
            state_id, timestamp, adapter_type, health_status,
            circuit_breaker_state, connection_pool_size,
            active_connections, metrics_json
        ) VALUES ($1, NOW(), $2, $3, $4, $5, $6, $7::jsonb)
        """

        parameters = [
            str(state_id),
            aggregated_state["adapter_type"],
            aggregated_state.get("health_status", "unknown"),
            aggregated_state.get("circuit_breaker_state"),
            aggregated_state.get("connection_pool_size", 0),
            aggregated_state.get("active_connections", 0),
            str(aggregated_state["metrics_json"]),
        ]

        # Execute via postgres adapter
        query_request = ModelPostgresQueryRequest(
            query=query,
            parameters=parameters,
            timeout=10.0,
            correlation_id=correlation_id,
        )

        await self._postgres_adapter.process(query_request)

        return state_id

    def _determine_intents(self, aggregated_state: dict) -> list[str]:
        """
        Determine intents to emit based on aggregated state.

        Pure function - analyzes state and returns list of intents.
        """
        intents = []

        health_status = aggregated_state.get("health_status", "unknown")
        circuit_breaker_state = aggregated_state.get("circuit_breaker_state")
        active_connections = aggregated_state.get("active_connections", 0)
        pool_size = aggregated_state.get("connection_pool_size", 0)

        # Health degradation intents
        if health_status == "degraded":
            intents.append("infrastructure_health_degraded")
        elif health_status == "unhealthy":
            intents.append("infrastructure_health_critical")

        # Circuit breaker intents
        if circuit_breaker_state == "open":
            intents.append("circuit_breaker_opened")
            intents.append("failover_required")
        elif circuit_breaker_state == "half_open":
            intents.append("recovery_initiated")

        # Connection pool intents
        if pool_size > 0 and active_connections >= pool_size * 0.9:
            intents.append("connection_pool_exhausted")

        return intents

    async def _emit_intents(self, intents: list[str], state_id: UUID, correlation_id: UUID):
        """
        Emit intents to orchestrator via event bus.

        Intents are stored in DB and published as events.
        """
        for intent_type in intents:
            intent_id = uuid4()

            # Store intent in DB
            intent_query = """
            INSERT INTO infrastructure_intents (
                intent_id, intent_type, timestamp,
                source_adapter, intent_data, processed
            ) VALUES ($1, $2, NOW(), $3, $4::jsonb, false)
            """

            intent_data = {
                "state_id": str(state_id),
                "correlation_id": str(correlation_id),
                "intent_type": intent_type,
            }

            intent_params = [
                str(intent_id),
                intent_type,
                "omni_infra_reducer",
                str(intent_data),
            ]

            intent_request = ModelPostgresQueryRequest(
                query=intent_query,
                parameters=intent_params,
                timeout=5.0,
                correlation_id=correlation_id,
            )

            await self._postgres_adapter.process(intent_request)

            # Publish intent event for orchestrator
            # TODO: Create intent event model and publish via event bus
            # await self._event_bus.publish_async(intent_event)

    async def initialize(self) -> None:
        """Initialize reducer - ensure DB tables exist."""
        # TODO: Create infrastructure_state and infrastructure_intents tables
        pass

    async def cleanup(self) -> None:
        """Cleanup reducer resources."""
        # Pure reducer - no resources to cleanup
        pass
