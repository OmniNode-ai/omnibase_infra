# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Pytest fixtures for restart-safe timeout integration tests.

Provides in-memory mocks for event bus and projection storage to test
timeout behavior without requiring real PostgreSQL or Kafka.

Fixture Categories:
    Event Bus:
        - mock_event_bus: Captures published events for verification

    Projection Storage:
        - in_memory_projection_store: Dict-based projection storage
        - mock_projection_reader: Mock reader using in-memory store
        - mock_projector: Mock projector using in-memory store

    Factories:
        - runtime_tick_factory: Creates ModelRuntimeTick with controlled time

    Services:
        - timeout_query_service: ServiceTimeoutScanner with mock reader
        - timeout_emission_service: ServiceTimeoutEmitter with mocks

Design Notes:
    These fixtures enable testing restart-safe behavior by allowing:
    1. Full control over time via injected `now` field
    2. Simulating service restart by recreating service instances
    3. Verifying exactly-once emission via event capture
    4. Inspecting projection markers for deduplication

Related Tickets:
    - OMN-932 (C2): Durable Timeout Handling
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict
from uuid import UUID, uuid4

import pytest
from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.primitives.model_semver import ModelSemVer

from omnibase_infra.enums import EnumRegistrationState

if TYPE_CHECKING:
    from pydantic import BaseModel
from omnibase_infra.models.projection import (
    ModelRegistrationProjection,
    ModelSequenceInfo,
)
from omnibase_infra.models.registration.model_node_capabilities import (
    ModelNodeCapabilities,
)
from omnibase_infra.runtime.models.model_runtime_tick import ModelRuntimeTick

# =============================================================================
# Type Aliases
# =============================================================================


class RawEventDict(TypedDict):
    """TypedDict for raw bytes events from publish method."""

    key: bytes | None
    value: bytes


# =============================================================================
# Mock Event Bus
# =============================================================================


@dataclass
class MockEventBus:
    """Mock event bus that captures published events for verification.

    Thread Safety:
        Uses asyncio.Lock for concurrent publish safety.

    Attributes:
        published_events: List of (topic, envelope) tuples for structured events.
            Uses BaseModel for type safety since all event envelopes are Pydantic models.
        published_raw_events: List of (topic, raw_event) tuples for raw bytes events.
        _lock: Async lock for thread-safe access
    """

    published_events: list[tuple[str, BaseModel]] = field(default_factory=list)
    published_raw_events: list[tuple[str, RawEventDict]] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def publish_envelope(
        self,
        envelope: BaseModel,
        topic: str,
    ) -> None:
        """Capture published event for later verification.

        Args:
            envelope: The event envelope/model being published.
                Uses BaseModel for type safety.
            topic: The topic to publish to
        """
        async with self._lock:
            self.published_events.append((topic, envelope))

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
    ) -> None:
        """Capture raw bytes publish (fallback method).

        Args:
            topic: The topic to publish to
            key: Optional message key
            value: Message value as bytes
        """
        async with self._lock:
            raw_event: RawEventDict = {"key": key, "value": value}
            self.published_raw_events.append((topic, raw_event))

    def get_events_for_topic(self, topic_pattern: str) -> list[BaseModel]:
        """Get all structured events published to topics matching pattern.

        Args:
            topic_pattern: Substring to match in topic names

        Returns:
            List of BaseModel envelopes published to matching topics
        """
        return [
            envelope
            for topic, envelope in self.published_events
            if topic_pattern in topic
        ]

    def get_raw_events_for_topic(self, topic_pattern: str) -> list[RawEventDict]:
        """Get all raw bytes events published to topics matching pattern.

        Args:
            topic_pattern: Substring to match in topic names

        Returns:
            List of RawEventDict events published to matching topics
        """
        return [
            raw_event
            for topic, raw_event in self.published_raw_events
            if topic_pattern in topic
        ]

    def count_events(self, topic_pattern: str | None = None) -> int:
        """Count structured events, optionally filtered by topic pattern.

        Args:
            topic_pattern: Optional substring to filter topics

        Returns:
            Number of matching structured events
        """
        if topic_pattern is None:
            return len(self.published_events)
        return len(self.get_events_for_topic(topic_pattern))

    def count_all_events(self, topic_pattern: str | None = None) -> int:
        """Count all events (structured + raw), optionally filtered by topic pattern.

        Args:
            topic_pattern: Optional substring to filter topics

        Returns:
            Number of matching events (both structured and raw)
        """
        if topic_pattern is None:
            return len(self.published_events) + len(self.published_raw_events)
        return len(self.get_events_for_topic(topic_pattern)) + len(
            self.get_raw_events_for_topic(topic_pattern)
        )

    def clear(self) -> None:
        """Clear all captured events (both structured and raw)."""
        self.published_events.clear()
        self.published_raw_events.clear()


# =============================================================================
# In-Memory Projection Store
# =============================================================================


@dataclass
class InMemoryProjectionStore:
    """In-memory projection storage simulating PostgreSQL.

    Provides a dict-based storage for projections keyed by (entity_id, domain).
    Supports all operations needed for timeout testing.

    Thread Safety:
        Uses asyncio.Lock for concurrent access safety.

    Attributes:
        projections: Dict mapping (entity_id, domain) to projection
        _lock: Async lock for thread-safe access
    """

    projections: dict[tuple[UUID, str], ModelRegistrationProjection] = field(
        default_factory=dict
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def persist(
        self,
        projection: ModelRegistrationProjection,
        entity_id: UUID,
        domain: str,
        sequence_info: ModelSequenceInfo,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Persist projection with ordering enforcement.

        Args:
            projection: Projection to persist
            entity_id: Entity identifier
            domain: Domain namespace
            sequence_info: Sequence info for ordering
            correlation_id: Optional correlation ID

        Returns:
            True if persisted, False if stale (rejected)
        """
        async with self._lock:
            key = (entity_id, domain)
            existing = self.projections.get(key)

            if existing is not None:
                # Check for stale update
                existing_offset = existing.last_applied_offset
                incoming_offset = (
                    sequence_info.sequence
                    if sequence_info.offset is None
                    else sequence_info.offset
                )
                if incoming_offset <= existing_offset:
                    return False

            self.projections[key] = projection
            return True

    async def get_entity_state(
        self,
        entity_id: UUID,
        domain: str = "registration",
        correlation_id: UUID | None = None,
    ) -> ModelRegistrationProjection | None:
        """Get projection for entity.

        Args:
            entity_id: Entity identifier
            domain: Domain namespace
            correlation_id: Optional correlation ID

        Returns:
            Projection if exists, None otherwise
        """
        async with self._lock:
            return self.projections.get((entity_id, domain))

    async def get_overdue_ack_registrations(
        self,
        now: datetime,
        domain: str = "registration",
        limit: int = 100,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Get registrations with overdue ack deadlines (not yet emitted).

        Args:
            now: Current time (injected)
            domain: Domain namespace
            limit: Maximum results
            correlation_id: Optional correlation ID

        Returns:
            List of overdue projections
        """
        async with self._lock:
            ack_states = {
                EnumRegistrationState.ACCEPTED,
                EnumRegistrationState.AWAITING_ACK,
            }
            results = []
            for (_, d), proj in self.projections.items():
                if d != domain:
                    continue
                if proj.current_state not in ack_states:
                    continue
                if proj.ack_deadline is None:
                    continue
                if proj.ack_deadline >= now:
                    continue
                if proj.ack_timeout_emitted_at is not None:
                    continue
                results.append(proj)
                if len(results) >= limit:
                    break
            return results

    async def get_overdue_liveness_registrations(
        self,
        now: datetime,
        domain: str = "registration",
        limit: int = 100,
        correlation_id: UUID | None = None,
    ) -> list[ModelRegistrationProjection]:
        """Get registrations with overdue liveness deadlines (not yet emitted).

        Args:
            now: Current time (injected)
            domain: Domain namespace
            limit: Maximum results
            correlation_id: Optional correlation ID

        Returns:
            List of overdue projections
        """
        async with self._lock:
            results = []
            for (_, d), proj in self.projections.items():
                if d != domain:
                    continue
                if proj.current_state != EnumRegistrationState.ACTIVE:
                    continue
                if proj.liveness_deadline is None:
                    continue
                if proj.liveness_deadline >= now:
                    continue
                if proj.liveness_timeout_emitted_at is not None:
                    continue
                results.append(proj)
                if len(results) >= limit:
                    break
            return results

    def clear(self) -> None:
        """Clear all projections."""
        self.projections.clear()


# =============================================================================
# Mock Projector (for marker updates)
# =============================================================================


@dataclass
class MockProjector:
    """Mock projector using in-memory store.

    Wraps InMemoryProjectionStore and provides projector interface.

    Attributes:
        store: The underlying in-memory store
    """

    store: InMemoryProjectionStore

    async def persist(
        self,
        projection: ModelRegistrationProjection,
        entity_id: UUID,
        domain: str,
        sequence_info: ModelSequenceInfo,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Persist projection.

        Args:
            projection: Projection to persist
            entity_id: Entity identifier
            domain: Domain namespace
            sequence_info: Sequence info
            correlation_id: Optional correlation ID

        Returns:
            True if persisted, False if stale
        """
        return await self.store.persist(
            projection=projection,
            entity_id=entity_id,
            domain=domain,
            sequence_info=sequence_info,
            correlation_id=correlation_id,
        )

    async def update_ack_timeout_marker(
        self,
        entity_id: UUID,
        domain: str,
        emitted_at: datetime,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Update ack timeout emission marker.

        Args:
            entity_id: Node UUID to update
            domain: Domain namespace
            emitted_at: Timestamp when the timeout event was emitted
            correlation_id: Optional correlation ID for tracing

        Returns:
            True if marker was updated, False if entity not found
        """
        async with self.store._lock:
            key = (entity_id, domain)
            existing = self.store.projections.get(key)
            if existing is None:
                return False

            # Create updated projection with marker set
            updated = ModelRegistrationProjection(
                entity_id=existing.entity_id,
                domain=existing.domain,
                current_state=existing.current_state,
                node_type=existing.node_type,
                node_version=existing.node_version,
                capabilities=existing.capabilities,
                ack_deadline=existing.ack_deadline,
                liveness_deadline=existing.liveness_deadline,
                ack_timeout_emitted_at=emitted_at,  # Set the marker
                liveness_timeout_emitted_at=existing.liveness_timeout_emitted_at,
                last_applied_event_id=existing.last_applied_event_id,
                last_applied_offset=existing.last_applied_offset,
                last_applied_sequence=existing.last_applied_sequence,
                last_applied_partition=existing.last_applied_partition,
                registered_at=existing.registered_at,
                updated_at=emitted_at,
                correlation_id=correlation_id,
            )
            self.store.projections[key] = updated
            return True

    async def update_liveness_timeout_marker(
        self,
        entity_id: UUID,
        domain: str,
        emitted_at: datetime,
        correlation_id: UUID | None = None,
    ) -> bool:
        """Update liveness timeout emission marker.

        Args:
            entity_id: Node UUID to update
            domain: Domain namespace
            emitted_at: Timestamp when the expiration event was emitted
            correlation_id: Optional correlation ID for tracing

        Returns:
            True if marker was updated, False if entity not found
        """
        async with self.store._lock:
            key = (entity_id, domain)
            existing = self.store.projections.get(key)
            if existing is None:
                return False

            # Create updated projection with marker set
            updated = ModelRegistrationProjection(
                entity_id=existing.entity_id,
                domain=existing.domain,
                current_state=existing.current_state,
                node_type=existing.node_type,
                node_version=existing.node_version,
                capabilities=existing.capabilities,
                ack_deadline=existing.ack_deadline,
                liveness_deadline=existing.liveness_deadline,
                ack_timeout_emitted_at=existing.ack_timeout_emitted_at,
                liveness_timeout_emitted_at=emitted_at,  # Set the marker
                last_applied_event_id=existing.last_applied_event_id,
                last_applied_offset=existing.last_applied_offset,
                last_applied_sequence=existing.last_applied_sequence,
                last_applied_partition=existing.last_applied_partition,
                registered_at=existing.registered_at,
                updated_at=emitted_at,
                correlation_id=correlation_id,
            )
            self.store.projections[key] = updated
            return True

    async def partial_update(
        self,
        aggregate_id: UUID,
        updates: dict[str, object],
        correlation_id: UUID,
    ) -> bool:
        """Perform a partial update on specific columns.

        Mirrors the ProjectorShell.partial_update() interface for testing.

        Args:
            aggregate_id: The entity UUID identifying the projection to update.
            updates: Dictionary mapping column names to their new values.
            correlation_id: Correlation ID for distributed tracing.

        Returns:
            True if a row was updated (entity found and modified).
            False if no row was found matching the aggregate_id.
        """
        async with self.store._lock:
            # Search for the projection by entity_id (aggregate_id)
            key = (aggregate_id, "registration")
            existing = self.store.projections.get(key)
            if existing is None:
                return False

            # Build update dict from existing projection
            projection_dict = existing.model_dump()

            # Apply updates
            for field_name, value in updates.items():
                if field_name in projection_dict:
                    projection_dict[field_name] = value

            # Create updated projection
            updated = ModelRegistrationProjection(**projection_dict)
            self.store.projections[key] = updated
            return True


# =============================================================================
# RuntimeTick Factory
# =============================================================================


def create_runtime_tick(
    now: datetime | None = None,
    sequence_number: int = 1,
    scheduler_id: str = "test-scheduler",
    tick_interval_ms: int = 1000,
) -> ModelRuntimeTick:
    """Create a RuntimeTick with controlled parameters.

    Args:
        now: Current time (defaults to UTC now)
        sequence_number: Tick sequence number
        scheduler_id: Scheduler identifier
        tick_interval_ms: Tick interval in milliseconds

    Returns:
        ModelRuntimeTick configured for testing
    """
    current_time = now or datetime.now(UTC)
    return ModelRuntimeTick(
        now=current_time,
        tick_id=uuid4(),
        sequence_number=sequence_number,
        scheduled_at=current_time,
        correlation_id=uuid4(),
        scheduler_id=scheduler_id,
        tick_interval_ms=tick_interval_ms,
    )


# =============================================================================
# Projection Factory
# =============================================================================


def create_test_projection(
    entity_id: UUID | None = None,
    state: EnumRegistrationState = EnumRegistrationState.AWAITING_ACK,
    node_type: EnumNodeKind = EnumNodeKind.EFFECT,
    ack_deadline: datetime | None = None,
    liveness_deadline: datetime | None = None,
    ack_timeout_emitted_at: datetime | None = None,
    liveness_timeout_emitted_at: datetime | None = None,
    offset: int = 100,
) -> ModelRegistrationProjection:
    """Create a test projection with sensible defaults.

    Args:
        entity_id: Node UUID (generated if not provided)
        state: FSM state
        node_type: ONEX node type (EnumNodeKind)
        ack_deadline: Optional ack deadline
        liveness_deadline: Optional liveness deadline
        ack_timeout_emitted_at: Optional ack timeout emission marker
        liveness_timeout_emitted_at: Optional liveness timeout emission marker
        offset: Kafka offset

    Returns:
        ModelRegistrationProjection configured for testing
    """
    now = datetime.now(UTC)
    return ModelRegistrationProjection(
        entity_id=entity_id or uuid4(),
        domain="registration",
        current_state=state,
        node_type=node_type,
        node_version=ModelSemVer.parse("1.0.0"),
        capabilities=ModelNodeCapabilities(postgres=True, read=True, write=True),
        ack_deadline=ack_deadline,
        liveness_deadline=liveness_deadline,
        ack_timeout_emitted_at=ack_timeout_emitted_at,
        liveness_timeout_emitted_at=liveness_timeout_emitted_at,
        last_applied_event_id=uuid4(),
        last_applied_offset=offset,
        registered_at=now,
        updated_at=now,
    )


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture
def mock_event_bus() -> MockEventBus:
    """Create fresh mock event bus for test."""
    return MockEventBus()


@pytest.fixture
def in_memory_store() -> InMemoryProjectionStore:
    """Create fresh in-memory projection store for test."""
    return InMemoryProjectionStore()


@pytest.fixture
def mock_projector(in_memory_store: InMemoryProjectionStore) -> MockProjector:
    """Create mock projector wrapping in-memory store."""
    return MockProjector(store=in_memory_store)


@pytest.fixture
def runtime_tick_factory() -> Callable[..., ModelRuntimeTick]:
    """Factory for creating RuntimeTicks with controlled parameters."""
    return create_runtime_tick


@pytest.fixture
def projection_factory() -> Callable[..., ModelRegistrationProjection]:
    """Factory for creating test projections."""
    return create_test_projection
