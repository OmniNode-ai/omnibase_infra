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

    - Processed positions per (topic, partition) for multi-topic idempotency
    - Last processed event ID (for correlation/debugging)
    - Staleness tracking (for TTL-based garbage collection)
    - Processing statistics (for observability)

    State transitions are performed via `with_*` methods that return new instances,
    ensuring the reducer remains pure and deterministic.

Idempotency:
    The state uses Kafka-based idempotency (topic, partition, offset) rather than
    event ID-based idempotency. This is more robust for replay scenarios since
    Kafka guarantees ordering within a partition.

    **Multi-Topic Support**: The reducer consumes from 4 different Kafka topics
    (contract-registered, contract-deregistered, heartbeat, runtime-tick). The state
    tracks the last processed offset **per (topic, partition)** to correctly detect
    duplicates even when events arrive interleaved from different topics.

    The `is_duplicate_event` method checks if an event was already processed by
    looking up the (topic, partition) key in `processed_positions` and comparing
    offsets.

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

    This state tracks processed positions per (topic, partition) for multi-topic
    idempotency and provides statistics for observability. The actual registry
    data lives in PostgreSQL (this reducer projects to it).

    The state is immutable (frozen=True) to enforce the pure reducer pattern.
    All state transitions create new instances via `with_*` methods.

    Multi-Topic Idempotency:
        The reducer consumes from 4 different Kafka topics. A naive single-position
        tracker would fail when events arrive interleaved:

        1. Process topic A, partition 0, offset 100 -> track (A, 0, 100)
        2. Process topic B, partition 0, offset 50 -> track (B, 0, 50)
        3. Replay topic A, partition 0, offset 100 -> NOT detected as duplicate!

        This model uses `processed_positions` to track the last offset per
        (topic, partition) combination, ensuring correct duplicate detection
        across all consumed topics.

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
        last_event_id: UUID of last processed event (for correlation/debugging).
        processed_positions: Dict mapping "topic:partition" to last processed offset.
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
        >>> # Multi-topic: positions are tracked independently
        >>> state = state.with_event_processed(uuid4(), "heartbeats", 0, 50)
        >>> state.is_duplicate_event("contracts", 0, 1)  # Still detected
        True
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    # Last processed event ID (for correlation/debugging only, not idempotency)
    last_event_id: UUID | None = Field(
        default=None,
        description="UUID of the last processed event for correlation",
    )

    # Multi-topic idempotency: track last offset per (topic, partition)
    # Key format: "topic:partition" -> last processed offset
    processed_positions: dict[str, int] = Field(
        default_factory=dict,
        description="Last processed offset per (topic:partition) for multi-topic idempotency",
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

    @staticmethod
    def _position_key(topic: str, partition: int) -> str:
        """Generate dict key for (topic, partition) combination.

        Args:
            topic: Kafka topic name.
            partition: Kafka partition number.

        Returns:
            String key in format "topic:partition".
        """
        return f"{topic}:{partition}"

    def is_duplicate_event(self, topic: str, partition: int, offset: int) -> bool:
        """Check if event was already processed (Kafka-based idempotency).

        Uses per-(topic, partition) offset tracking for duplicate detection.
        An event is considered a duplicate if:
        - The (topic, partition) has been processed before
        - The event's offset is <= the last processed offset for that combination

        This correctly handles multi-topic consumption where events from different
        topics arrive interleaved.

        Args:
            topic: Kafka topic of the event.
            partition: Kafka partition of the event.
            offset: Kafka offset of the event.

        Returns:
            True if this event was already processed (is a duplicate).
        """
        key = self._position_key(topic, partition)
        last_offset = self.processed_positions.get(key)
        if last_offset is None:
            return False
        return offset <= last_offset

    def with_event_processed(
        self,
        event_id: UUID,
        topic: str,
        partition: int,
        offset: int,
    ) -> ModelContractRegistryState:
        """Return new state with event marked as processed.

        Creates a new immutable state instance with the Kafka offset tracking
        updated for the specific (topic, partition) combination. Statistics are
        preserved; use the specific `with_*` methods to increment them.

        Multi-Topic Support:
            Each (topic, partition) combination has its own tracked offset.
            This ensures correct idempotency when the reducer consumes from
            multiple topics (contract-registered, contract-deregistered,
            heartbeat, runtime-tick).

        Args:
            event_id: UUID of the processed event.
            topic: Kafka topic of the event.
            partition: Kafka partition of the event.
            offset: Kafka offset of the event.

        Returns:
            New ModelContractRegistryState with updated offset tracking.
        """
        key = self._position_key(topic, partition)
        # Create new dict with updated position (immutable pattern)
        new_positions = {**self.processed_positions, key: offset}
        return self.model_copy(
            update={
                "last_event_id": event_id,
                "processed_positions": new_positions,
            }
        )

    def with_contract_registered(self) -> ModelContractRegistryState:
        """Return new state with contract registration count incremented.

        Returns:
            New ModelContractRegistryState with contracts_processed + 1.
        """
        return self.model_copy(
            update={"contracts_processed": self.contracts_processed + 1}
        )

    def with_heartbeat_processed(self) -> ModelContractRegistryState:
        """Return new state with heartbeat count incremented.

        Returns:
            New ModelContractRegistryState with heartbeats_processed + 1.
        """
        return self.model_copy(
            update={"heartbeats_processed": self.heartbeats_processed + 1}
        )

    def with_deregistration_processed(self) -> ModelContractRegistryState:
        """Return new state with deregistration count incremented.

        Returns:
            New ModelContractRegistryState with deregistrations_processed + 1.
        """
        return self.model_copy(
            update={"deregistrations_processed": self.deregistrations_processed + 1}
        )

    def with_staleness_check(self, check_time: datetime) -> ModelContractRegistryState:
        """Return new state with staleness check timestamp updated.

        Args:
            check_time: Timestamp of the staleness check.

        Returns:
            New ModelContractRegistryState with updated staleness check time.
        """
        return self.model_copy(update={"last_staleness_check_at": check_time})


__all__ = ["ModelContractRegistryState"]
