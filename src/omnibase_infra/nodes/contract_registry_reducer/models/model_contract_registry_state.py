# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Immutable state model for contract registry reducer.

This module provides ModelContractRegistryState, an immutable state model for the
contract registry projection reducer. The state follows the pure reducer pattern
where state is passed in and returned from reduce(), with no internal mutation.

Architecture:
    ModelContractRegistryState is designed for projection to PostgreSQL. The reducer
    processes contract registration events and emits intents for persistence. The state
    tracks:

    - Last processed event (Kafka offset-based idempotency)
    - Staleness tracking (for TTL-based garbage collection)
    - Processing statistics (for observability)

    State transitions are performed via `with_*` methods that return new instances,
    ensuring the reducer remains pure and deterministic.

Idempotency:
    The state uses Kafka-based idempotency (topic, partition, offset) rather than
    event ID-based idempotency. This is more robust for replay scenarios since
    Kafka guarantees ordering within a partition.

    The `is_duplicate_event` method checks if an event was already processed by
    comparing topic, partition, and offset against the last processed values.

Related:
    - NodeContractRegistryReducer: Declarative reducer that uses this state model
    - OMN-1653: Contract registry reducer implementation
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelContractRegistryState(BaseModel):
    """Immutable state for contract registry projection.

    This state tracks the last processed event for idempotency and provides
    statistics for observability. The actual registry data lives in PostgreSQL
    (this reducer projects to it).

    The state is immutable (frozen=True) to enforce the pure reducer pattern.
    All state transitions create new instances via `with_*` methods.

    Persistence Integration:
        This model is persisted to PostgreSQL by the Projector component:

        - **Stored**: By Runtime calling Projector.persist() after reduce() returns
        - **Retrieved**: By Orchestrator via ProtocolProjectionReader before reduce()
        - **Idempotency**: Kafka offset tracking enables duplicate detection

        The reducer does NOT persist state directly - it returns the new state
        in ModelReducerOutput.result. The Runtime handles persistence.

    Immutability:
        This model uses frozen=True to enforce strict immutability:

        - All fields are immutable after construction
        - Transition methods (with_*) return NEW instances
        - Original state is never modified
        - Safe for concurrent access and comparison

    Attributes:
        last_event_id: UUID of last processed event (for correlation).
        last_event_topic: Kafka topic of last processed event.
        last_event_partition: Kafka partition of last processed event.
        last_event_offset: Kafka offset of last processed event.
        last_staleness_check_at: Timestamp of last staleness check run.
        contracts_processed: Count of contract registration events processed.
        heartbeats_processed: Count of heartbeat events processed.
        deregistrations_processed: Count of deregistration events processed.

    Example:
        >>> from uuid import uuid4
        >>> state = ModelContractRegistryState()  # Initial state
        >>> state.contracts_processed
        0
        >>> state = state.with_event_processed(
        ...     uuid4(), "contracts", 0, 1
        ... ).with_contract_registered()
        >>> state.contracts_processed
        1
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # Last processed event (for idempotency/dedupe via Kafka offsets)
    last_event_id: UUID | None = Field(
        default=None,
        description="UUID of the last processed event for correlation",
    )
    last_event_topic: str | None = Field(
        default=None,
        description="Kafka topic of the last processed event",
    )
    last_event_partition: int | None = Field(
        default=None,
        description="Kafka partition of the last processed event",
    )
    last_event_offset: int | None = Field(
        default=None,
        description="Kafka offset of the last processed event",
    )

    # Staleness tracking
    last_staleness_check_at: datetime | None = Field(
        default=None,
        description="Timestamp of the last staleness check run",
    )

    # Statistics (for observability)
    contracts_processed: int = Field(
        default=0,
        description="Count of contract registration events processed",
    )
    heartbeats_processed: int = Field(
        default=0,
        description="Count of heartbeat events processed",
    )
    deregistrations_processed: int = Field(
        default=0,
        description="Count of deregistration events processed",
    )

    def is_duplicate_event(self, topic: str, partition: int, offset: int) -> bool:
        """Check if event was already processed (Kafka-based idempotency).

        Uses Kafka offset comparison for duplicate detection. An event is
        considered a duplicate if:
        - It's from the same topic and partition
        - Its offset is less than or equal to the last processed offset

        Args:
            topic: Kafka topic of the event.
            partition: Kafka partition of the event.
            offset: Kafka offset of the event.

        Returns:
            True if this event was already processed (is a duplicate).
        """
        if self.last_event_topic != topic:
            return False
        if self.last_event_partition != partition:
            return False
        return self.last_event_offset is not None and offset <= self.last_event_offset

    def with_event_processed(
        self,
        event_id: UUID,
        topic: str,
        partition: int,
        offset: int,
    ) -> ModelContractRegistryState:
        """Return new state with event marked as processed.

        Creates a new immutable state instance with the Kafka offset tracking
        updated. Statistics are preserved; use the specific `with_*` methods
        to increment them.

        Args:
            event_id: UUID of the processed event.
            topic: Kafka topic of the event.
            partition: Kafka partition of the event.
            offset: Kafka offset of the event.

        Returns:
            New ModelContractRegistryState with updated offset tracking.
        """
        return ModelContractRegistryState(
            last_event_id=event_id,
            last_event_topic=topic,
            last_event_partition=partition,
            last_event_offset=offset,
            last_staleness_check_at=self.last_staleness_check_at,
            contracts_processed=self.contracts_processed,
            heartbeats_processed=self.heartbeats_processed,
            deregistrations_processed=self.deregistrations_processed,
        )

    def with_contract_registered(self) -> ModelContractRegistryState:
        """Return new state with contract registration count incremented.

        Returns:
            New ModelContractRegistryState with contracts_processed + 1.
        """
        return ModelContractRegistryState(
            last_event_id=self.last_event_id,
            last_event_topic=self.last_event_topic,
            last_event_partition=self.last_event_partition,
            last_event_offset=self.last_event_offset,
            last_staleness_check_at=self.last_staleness_check_at,
            contracts_processed=self.contracts_processed + 1,
            heartbeats_processed=self.heartbeats_processed,
            deregistrations_processed=self.deregistrations_processed,
        )

    def with_heartbeat_processed(self) -> ModelContractRegistryState:
        """Return new state with heartbeat count incremented.

        Returns:
            New ModelContractRegistryState with heartbeats_processed + 1.
        """
        return ModelContractRegistryState(
            last_event_id=self.last_event_id,
            last_event_topic=self.last_event_topic,
            last_event_partition=self.last_event_partition,
            last_event_offset=self.last_event_offset,
            last_staleness_check_at=self.last_staleness_check_at,
            contracts_processed=self.contracts_processed,
            heartbeats_processed=self.heartbeats_processed + 1,
            deregistrations_processed=self.deregistrations_processed,
        )

    def with_deregistration_processed(self) -> ModelContractRegistryState:
        """Return new state with deregistration count incremented.

        Returns:
            New ModelContractRegistryState with deregistrations_processed + 1.
        """
        return ModelContractRegistryState(
            last_event_id=self.last_event_id,
            last_event_topic=self.last_event_topic,
            last_event_partition=self.last_event_partition,
            last_event_offset=self.last_event_offset,
            last_staleness_check_at=self.last_staleness_check_at,
            contracts_processed=self.contracts_processed,
            heartbeats_processed=self.heartbeats_processed,
            deregistrations_processed=self.deregistrations_processed + 1,
        )

    def with_staleness_check(self, check_time: datetime) -> ModelContractRegistryState:
        """Return new state with staleness check timestamp updated.

        Args:
            check_time: Timestamp of the staleness check.

        Returns:
            New ModelContractRegistryState with updated staleness check time.
        """
        return ModelContractRegistryState(
            last_event_id=self.last_event_id,
            last_event_topic=self.last_event_topic,
            last_event_partition=self.last_event_partition,
            last_event_offset=self.last_event_offset,
            last_staleness_check_at=check_time,
            contracts_processed=self.contracts_processed,
            heartbeats_processed=self.heartbeats_processed,
            deregistrations_processed=self.deregistrations_processed,
        )


__all__ = ["ModelContractRegistryState"]
