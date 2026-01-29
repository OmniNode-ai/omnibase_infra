# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pure function reducer for contract registry projection.

This reducer implements the pure function pattern for contract registry workflows:
    reduce(state, event, metadata) -> ModelReducerOutput[state + intents]

The reducer handles four event types:
    1. contract-registered: Upsert contract record, extract topics
    2. contract-deregistered: Mark contract as inactive
    3. node-heartbeat: Update last_seen_at timestamp
    4. runtime-tick: Compute staleness, mark stale contracts inactive

Architecture:
    - Pure function: reduce(state, event) -> new_state + intents
    - No internal state - state passed in and returned
    - No I/O - emits intents for Effect layer (PostgreSQL writes)
    - Deterministic - same inputs produce same outputs

Circuit Breaker Considerations:
    This reducer does NOT require a circuit breaker because:

    1. **Pure Function Pattern**: Reducers are pure functions - they perform
       NO I/O operations. All external interactions are delegated to the
       Effect layer via emitted intents.

    2. **No Transient Failures**: Without I/O, there are no transient failures
       to recover from. Circuit breakers are designed for I/O resilience.

    3. **Deterministic Behavior**: Given the same state and event, the reducer
       always produces the same output. There's no "retry" semantic.

    4. **Effect Layer Responsibility**: Circuit breakers should be implemented
       in the Effect layer nodes (PostgresAdapter) that actually perform the
       external I/O operations.

Staleness Strategy:
    Staleness is computed on runtime-tick events (not on every heartbeat).
    This design:
    - Reduces PostgreSQL write amplification (batch staleness updates)
    - Centralizes TTL logic in one place (the tick handler)
    - Makes heartbeat handling cheap (just update last_seen_at)

    Heartbeats opportunistically update last_seen_at. The staleness check
    on runtime-tick marks contracts as stale if last_seen_at > threshold.

Intent Emission:
    The reducer emits ModelIntent objects for Effect layer execution:
    - postgres.upsert_contract: Upsert contract record
    - postgres.update_topic: Update topic routing entry
    - postgres.mark_stale: Mark contracts as stale (batch operation)
    - postgres.update_heartbeat: Update last_seen_at timestamp
    - postgres.deactivate_contract: Mark contract as inactive

    The payload contains the serialized typed intent for Effect layer execution.

Related:
    - OMN-1653: Contract registry reducer implementation
    - RegistrationReducer: Reference implementation for pure reducer pattern
    - DESIGN_CONTRACT_REGISTRY.md: Architecture design
"""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import yaml

from omnibase_core.enums import EnumReductionType, EnumStreamingMode

# =============================================================================
# Event Models from omnibase_core 0.9.8
# =============================================================================
from omnibase_core.models.events import (
    ModelContractDeregisteredEvent,
    ModelContractRegisteredEvent,
    ModelNodeHeartbeatEvent,
)
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.nodes import ModelReducerOutput
from omnibase_infra.nodes.contract_registry_reducer.models.model_contract_registry_state import (
    ModelContractRegistryState,
)
from omnibase_infra.nodes.contract_registry_reducer.models.model_payload_deactivate_contract import (
    ModelPayloadDeactivateContract,
)
from omnibase_infra.nodes.contract_registry_reducer.models.model_payload_mark_stale import (
    ModelPayloadMarkStale,
)
from omnibase_infra.nodes.contract_registry_reducer.models.model_payload_update_heartbeat import (
    ModelPayloadUpdateHeartbeat,
)
from omnibase_infra.nodes.contract_registry_reducer.models.model_payload_update_topic import (
    ModelPayloadUpdateTopic,
)
from omnibase_infra.nodes.contract_registry_reducer.models.model_payload_upsert_contract import (
    ModelPayloadUpsertContract,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

# Union type for all events this reducer handles
ContractRegistryEvent = (
    ModelContractRegisteredEvent
    | ModelContractDeregisteredEvent
    | ModelNodeHeartbeatEvent
    | ModelRuntimeTick
)

# =============================================================================
# Performance Threshold Constants (in milliseconds)
# =============================================================================

# Target processing time for reduce() method (<300ms per event)
PERF_THRESHOLD_REDUCE_MS: float = float(
    os.getenv("ONEX_PERF_THRESHOLD_CONTRACT_REDUCE_MS", "300.0")
)

# Target processing time for intent building (<50ms per intent)
PERF_THRESHOLD_INTENT_BUILD_MS: float = float(
    os.getenv("ONEX_PERF_THRESHOLD_CONTRACT_INTENT_BUILD_MS", "50.0")
)

# Target processing time for staleness computation (<100ms)
PERF_THRESHOLD_STALENESS_CHECK_MS: float = float(
    os.getenv("ONEX_PERF_THRESHOLD_STALENESS_CHECK_MS", "100.0")
)

# Staleness threshold (contracts without heartbeat for this duration are stale)
STALENESS_THRESHOLD_SECONDS: int = int(
    os.getenv("ONEX_CONTRACT_STALENESS_THRESHOLD_SECONDS", "300")
)
STALENESS_THRESHOLD: timedelta = timedelta(seconds=STALENESS_THRESHOLD_SECONDS)

# Logger for performance warnings
_logger = logging.getLogger(__name__)


class ContractRegistryReducer:
    """Pure reducer for contract registry projection.

    Follows ProtocolReducer pattern:
    - reduce(state, event, metadata) -> ModelReducerOutput
    - Pure function, no side effects
    - Emits intents for PostgreSQL operations

    This is a stateless class - all state is passed in and returned via
    ModelContractRegistryState. The class exists to group related pure functions.

    Event Handling:
        This reducer handles four event types:

        1. reduce_contract_registered(): Processes contract registration events.
           - Upserts contract record to PostgreSQL
           - Extracts topics from contract_yaml and updates topic routing

        2. reduce_contract_deregistered(): Processes deregistration events.
           - Marks contract as inactive in PostgreSQL
           - Preserves contract data for auditing

        3. reduce_heartbeat(): Processes heartbeat events.
           - Updates last_seen_at timestamp
           - Does NOT compute staleness (deferred to runtime-tick)

        4. reduce_runtime_tick(): Processes periodic tick events.
           - Computes staleness across all contracts
           - Marks contracts as stale if last_seen_at > threshold

    Example:
        >>> from uuid import uuid4
        >>> reducer = ContractRegistryReducer()
        >>> state = ModelContractRegistryState()  # Initial state
        >>> # Simulate contract registration (event model stubbed)
        >>> # output = reducer.reduce(state, event, metadata)
        >>> # print(output.result.contracts_processed)  # 1
    """

    def reduce(
        self,
        state: ModelContractRegistryState,
        event: ContractRegistryEvent,
        event_metadata: dict[str, object],
    ) -> ModelReducerOutput[ModelContractRegistryState]:
        """Pure reduce function: state + event -> new_state + intents.

        Routes to the appropriate handler based on event type. All handlers
        follow the same pure function pattern - no I/O, emit intents.

        Args:
            state: Current contract registry state (immutable).
            event: Contract event to process (registration, deregistration,
                heartbeat, or runtime-tick).
            event_metadata: Kafka metadata containing topic, partition, offset.

        Returns:
            ModelReducerOutput containing new_state and intents tuple.
        """
        start_time = time.perf_counter()

        # Extract Kafka metadata for idempotency
        topic = str(event_metadata.get("topic", ""))
        partition_raw = event_metadata.get("partition", 0)
        offset_raw = event_metadata.get("offset", 0)
        partition = int(partition_raw) if isinstance(partition_raw, (int, str)) else 0
        offset = int(offset_raw) if isinstance(offset_raw, (int, str)) else 0

        # Idempotency guard - skip if we've already processed this event
        if state.is_duplicate_event(topic, partition, offset):
            return self._build_output(
                state=state,
                intents=(),
                processing_time_ms=0.0,
                items_processed=0,
            )

        # Route to handler based on event type (using isinstance for type safety)
        if isinstance(event, ModelContractRegisteredEvent):
            return self._on_contract_registered(state, event, topic, partition, offset)
        elif isinstance(event, ModelContractDeregisteredEvent):
            return self._on_contract_deregistered(
                state, event, topic, partition, offset
            )
        elif isinstance(event, ModelNodeHeartbeatEvent):
            return self._on_heartbeat(state, event, topic, partition, offset)
        elif isinstance(event, ModelRuntimeTick):
            return self._on_runtime_tick(state, event, topic, partition, offset)
        else:
            # Unknown event type - skip (should not happen with proper typing)
            _logger.warning(
                "Unknown event type received",
                extra={
                    "event_type": type(event).__name__,
                    "topic": topic,
                    "partition": partition,
                    "offset": offset,
                },
            )
            return self._build_output(
                state=state,
                intents=(),
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                items_processed=0,
            )

    def _on_contract_registered(
        self,
        state: ModelContractRegistryState,
        event: ModelContractRegisteredEvent,
        topic: str,
        partition: int,
        offset: int,
    ) -> ModelReducerOutput[ModelContractRegistryState]:
        """Handle contract registration - upsert contract and extract topics.

        This handler:
        1. Validates the event has required fields (contract_id, node_id)
        2. Builds a postgres.upsert_contract intent
        3. Extracts topics from contract_yaml and builds postgres.update_topic intents
        4. Returns new state with event marked as processed

        Args:
            state: Current state.
            event: Contract registered event.
            topic: Kafka topic.
            partition: Kafka partition.
            offset: Kafka offset.

        Returns:
            ModelReducerOutput with new state and PostgreSQL intents.
        """
        start_time = time.perf_counter()

        # Extract fields from typed event
        event_id = event.event_id
        correlation_id = event.correlation_id or event_id

        # Derive contract identity from node_name + version (natural key)
        version = event.node_version
        contract_id = (
            f"{event.node_name}:{version.major}.{version.minor}.{version.patch}"
        )

        intents: list[ModelIntent] = []

        # Intent 1: Upsert contract record
        upsert_intent = self._build_upsert_contract_intent(event, correlation_id)
        intents.append(upsert_intent)

        # Intent 2+: Extract and update topics from contract_yaml
        topic_intents = self._build_topic_update_intents(event, correlation_id)
        intents.extend(topic_intents)

        # Update state
        new_state = state.with_event_processed(
            event_id=event_id,
            topic=topic,
            partition=partition,
            offset=offset,
        ).with_contract_registered()

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        if processing_time_ms > PERF_THRESHOLD_REDUCE_MS:
            _logger.warning(
                "Contract registration processing exceeded threshold",
                extra={
                    "processing_time_ms": processing_time_ms,
                    "threshold_ms": PERF_THRESHOLD_REDUCE_MS,
                    "contract_id": contract_id,
                    "node_name": event.node_name,
                    "correlation_id": str(correlation_id),
                },
            )

        return self._build_output(
            state=new_state,
            intents=tuple(intents),
            processing_time_ms=processing_time_ms,
            items_processed=1,
        )

    def _on_contract_deregistered(
        self,
        state: ModelContractRegistryState,
        event: ModelContractDeregisteredEvent,
        topic: str,
        partition: int,
        offset: int,
    ) -> ModelReducerOutput[ModelContractRegistryState]:
        """Handle deregistration - mark contract inactive.

        This handler:
        1. Validates the event has required fields (contract_id)
        2. Builds a postgres.deactivate_contract intent
        3. Returns new state with event marked as processed

        Args:
            state: Current state.
            event: Contract deregistered event.
            topic: Kafka topic.
            partition: Kafka partition.
            offset: Kafka offset.

        Returns:
            ModelReducerOutput with new state and deactivation intent.
        """
        start_time = time.perf_counter()

        # Extract fields from typed event
        event_id = event.event_id
        correlation_id = event.correlation_id or event_id

        # Derive contract identity from node_name + version
        version = event.node_version
        contract_id = (
            f"{event.node_name}:{version.major}.{version.minor}.{version.patch}"
        )

        payload = ModelPayloadDeactivateContract(
            correlation_id=correlation_id,
            contract_id=contract_id,
            node_name=event.node_name,
            reason=event.reason.value,
            deactivated_at=event.timestamp,
        )

        intent = ModelIntent(
            intent_type="extension",
            target=f"postgres://contracts/{contract_id}",
            payload=payload,
        )

        new_state = state.with_event_processed(
            event_id=event_id,
            topic=topic,
            partition=partition,
            offset=offset,
        ).with_deregistration_processed()

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return self._build_output(
            state=new_state,
            intents=(intent,),
            processing_time_ms=processing_time_ms,
            items_processed=1,
        )

    def _on_heartbeat(
        self,
        state: ModelContractRegistryState,
        event: ModelNodeHeartbeatEvent,
        topic: str,
        partition: int,
        offset: int,
    ) -> ModelReducerOutput[ModelContractRegistryState]:
        """Handle heartbeat - update last_seen_at.

        Heartbeats opportunistically update last_seen_at. Staleness computation
        is deferred to runtime-tick events to reduce write amplification.

        Args:
            state: Current state.
            event: Node heartbeat event.
            topic: Kafka topic.
            partition: Kafka partition.
            offset: Kafka offset.

        Returns:
            ModelReducerOutput with new state and heartbeat update intent.
        """
        start_time = time.perf_counter()

        # Extract fields from typed event
        event_id = event.event_id
        correlation_id = event.correlation_id or event_id

        # Derive contract identity from node_name + version
        version = event.node_version
        contract_id = (
            f"{event.node_name}:{version.major}.{version.minor}.{version.patch}"
        )

        payload = ModelPayloadUpdateHeartbeat(
            correlation_id=correlation_id,
            contract_id=contract_id,
            node_name=event.node_name,
            source_node_id=str(event.source_node_id) if event.source_node_id else None,
            last_seen_at=event.timestamp,
            uptime_seconds=event.uptime_seconds,
            sequence_number=event.sequence_number,
        )

        intent = ModelIntent(
            intent_type="extension",
            target=f"postgres://contracts/{contract_id}/heartbeat",
            payload=payload,
        )

        new_state = state.with_event_processed(
            event_id=event_id,
            topic=topic,
            partition=partition,
            offset=offset,
        ).with_heartbeat_processed()

        processing_time_ms = (time.perf_counter() - start_time) * 1000

        return self._build_output(
            state=new_state,
            intents=(intent,),
            processing_time_ms=processing_time_ms,
            items_processed=1,
        )

    def _on_runtime_tick(
        self,
        state: ModelContractRegistryState,
        event: ModelRuntimeTick,
        topic: str,
        partition: int,
        offset: int,
    ) -> ModelReducerOutput[ModelContractRegistryState]:
        """Handle runtime tick - compute staleness.

        This handler is called periodically (via runtime-tick events) to:
        1. Compute the staleness cutoff time
        2. Emit a batch intent to mark stale contracts as inactive
        3. Update state with staleness check timestamp

        The actual staleness computation happens in PostgreSQL (via the Effect layer)
        which can efficiently update all contracts with last_seen_at < cutoff.

        Args:
            state: Current state.
            event: Runtime tick event with authoritative timestamp.
            topic: Kafka topic.
            partition: Kafka partition.
            offset: Kafka offset.

        Returns:
            ModelReducerOutput with new state and staleness batch intent.
        """
        start_time = time.perf_counter()

        # Use the tick's authoritative time (not datetime.now())
        now = event.now
        stale_cutoff = now - STALENESS_THRESHOLD

        payload = ModelPayloadMarkStale(
            correlation_id=event.correlation_id,
            stale_cutoff=stale_cutoff,
            checked_at=now,
        )

        intent = ModelIntent(
            intent_type="extension",
            target="postgres://contracts/stale",
            payload=payload,
        )

        new_state = state.with_event_processed(
            event_id=event.tick_id,
            topic=topic,
            partition=partition,
            offset=offset,
        ).with_staleness_check(now)

        processing_time_ms = (time.perf_counter() - start_time) * 1000
        if processing_time_ms > PERF_THRESHOLD_STALENESS_CHECK_MS:
            _logger.warning(
                "Staleness check processing exceeded threshold",
                extra={
                    "processing_time_ms": processing_time_ms,
                    "threshold_ms": PERF_THRESHOLD_STALENESS_CHECK_MS,
                    "stale_cutoff": stale_cutoff.isoformat(),
                },
            )

        return self._build_output(
            state=new_state,
            intents=(intent,),
            processing_time_ms=processing_time_ms,
            items_processed=1,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_upsert_contract_intent(
        self,
        event: ModelContractRegisteredEvent,
        correlation_id: UUID,
    ) -> ModelIntent:
        """Build PostgreSQL upsert intent for contract record.

        Args:
            event: Contract registered event with typed fields.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelIntent for postgres.upsert_contract.
        """
        version = event.node_version
        contract_id = (
            f"{event.node_name}:{version.major}.{version.minor}.{version.patch}"
        )

        payload = ModelPayloadUpsertContract(
            correlation_id=correlation_id,
            contract_id=contract_id,
            node_name=event.node_name,
            version_major=version.major,
            version_minor=version.minor,
            version_patch=version.patch,
            contract_hash=event.contract_hash,
            contract_yaml=event.contract_yaml,
            source_node_id=str(event.source_node_id) if event.source_node_id else None,
            is_active=True,
            registered_at=event.timestamp,
            last_seen_at=event.timestamp,
        )

        return ModelIntent(
            intent_type="extension",
            target=f"postgres://contracts/{contract_id}",
            payload=payload,
        )

    def _build_topic_update_intents(
        self,
        event: ModelContractRegisteredEvent,
        correlation_id: UUID,
    ) -> list[ModelIntent]:
        """Extract topics from contract_yaml and build update intents.

        Parses the contract_yaml for consumed_events and published_events,
        then creates postgres.update_topic intents for each topic suffix.

        Args:
            event: Contract registered event with contract_yaml.
            correlation_id: Correlation ID for tracing.

        Returns:
            List of ModelIntent for postgres.update_topic operations.
        """
        intents: list[ModelIntent] = []
        contract_yaml_raw = event.contract_yaml

        # Parse contract_yaml if it's a string
        contract_yaml: dict
        if isinstance(contract_yaml_raw, str):
            try:
                parsed = yaml.safe_load(contract_yaml_raw)
                if not isinstance(parsed, dict):
                    _logger.debug(
                        "Parsed contract_yaml is not a dict, skipping topic extraction",
                        extra={"correlation_id": str(correlation_id)},
                    )
                    return intents
                contract_yaml = parsed
            except yaml.YAMLError as e:
                _logger.warning(
                    "Failed to parse contract_yaml, skipping topic extraction",
                    extra={"correlation_id": str(correlation_id), "error": str(e)},
                )
                return intents
        elif isinstance(contract_yaml_raw, dict):
            contract_yaml = contract_yaml_raw
        else:
            return intents

        version = event.node_version
        contract_id = (
            f"{event.node_name}:{version.major}.{version.minor}.{version.patch}"
        )

        # Extract consumed_events (subscribe topics)
        consumed_events = contract_yaml.get("consumed_events", [])
        if isinstance(consumed_events, list):
            for consumed in consumed_events:
                if isinstance(consumed, dict):
                    topic_suffix = consumed.get("topic")
                    if topic_suffix:
                        payload = ModelPayloadUpdateTopic(
                            correlation_id=correlation_id,
                            topic_suffix=topic_suffix,
                            direction="subscribe",
                            contract_id=contract_id,
                            node_name=event.node_name,
                            event_type=consumed.get("event_type"),
                            last_seen_at=event.timestamp,
                        )
                        intents.append(
                            ModelIntent(
                                intent_type="extension",
                                target=f"postgres://topics/{topic_suffix}",
                                payload=payload,
                            )
                        )

        # Extract published_events (publish topics)
        published_events = contract_yaml.get("published_events", [])
        if isinstance(published_events, list):
            for published in published_events:
                if isinstance(published, dict):
                    topic_suffix = published.get("topic")
                    if topic_suffix:
                        payload = ModelPayloadUpdateTopic(
                            correlation_id=correlation_id,
                            topic_suffix=topic_suffix,
                            direction="publish",
                            contract_id=contract_id,
                            node_name=event.node_name,
                            event_type=published.get("event_type"),
                            last_seen_at=event.timestamp,
                        )
                        intents.append(
                            ModelIntent(
                                intent_type="extension",
                                target=f"postgres://topics/{topic_suffix}",
                                payload=payload,
                            )
                        )

        return intents

    def _build_output(
        self,
        state: ModelContractRegistryState,
        intents: tuple[ModelIntent, ...],
        processing_time_ms: float,
        items_processed: int,
    ) -> ModelReducerOutput[ModelContractRegistryState]:
        """Build standardized ModelReducerOutput.

        Args:
            state: New contract registry state to return.
            intents: Tuple of ModelIntent objects to emit.
            processing_time_ms: Time taken to process the event.
            items_processed: Number of events processed (0 or 1).

        Returns:
            ModelReducerOutput containing the state and intents.
        """
        return ModelReducerOutput(
            result=state,
            operation_id=uuid4(),
            reduction_type=EnumReductionType.MERGE,
            processing_time_ms=processing_time_ms,
            items_processed=items_processed,
            conflicts_resolved=0,
            streaming_mode=EnumStreamingMode.BATCH,
            batches_processed=1,
            intents=intents,
        )


__all__ = [
    "ContractRegistryReducer",
    "ContractRegistryEvent",
    "PERF_THRESHOLD_INTENT_BUILD_MS",
    "PERF_THRESHOLD_REDUCE_MS",
    "PERF_THRESHOLD_STALENESS_CHECK_MS",
    "STALENESS_THRESHOLD",
    "STALENESS_THRESHOLD_SECONDS",
]
