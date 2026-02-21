# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Integration tests for projection ordering guarantee (OMN-2510).

Acceptance criteria from the ticket:

    R3-1: Integration test confirms projection is persisted before any Kafka
          consumer receives the intent.
    R3-2: Test simulates projection failure and verifies no Kafka message is
          published.
    R3-3: Test simulates slow projection (within timeout) and verifies Kafka
          publish is delayed, not skipped.

These tests use in-memory fakes for the event bus and projection store to
avoid infrastructure dependencies while still exercising the full
DispatchResultApplier pipeline.

Related:
    - OMN-2363: Projection ordering guarantee epic
    - OMN-2510: Runtime wires projection before Kafka publish (this ticket)
    - OMN-2508: NodeProjectionEffect stub (omnibase_spi)
    - OMN-2509: Reducer emits ModelProjectionIntent (omnibase_core)
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.errors.error_projection import ProjectionError
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.models.projection.model_projection_intent import (
    ModelProjectionIntent,
)
from omnibase_infra.runtime.models.model_projection_result_local import (
    ModelProjectionResultLocal,
)
from omnibase_infra.runtime.service_dispatch_result_applier import DispatchResultApplier

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# In-memory fakes
# ---------------------------------------------------------------------------


class _InMemoryProjectionStore:
    """In-memory projection store for integration tests.

    Records every projection write so tests can assert persistence before
    Kafka publish.
    """

    def __init__(self) -> None:
        self.writes: list[dict[str, Any]] = []

    def write(self, intent: ModelProjectionIntent) -> ModelProjectionResultLocal:
        """Persist a projection synchronously and record the write."""
        self.writes.append(
            {
                "subject": intent.subject,
                "aggregate_id": intent.aggregate_id,
                "projection_type": intent.projection_type,
                "payload": intent.payload,
                "causation_event_id": intent.causation_event_id,
                "correlation_id": intent.correlation_id,
                "written_at": datetime.now(UTC),
            }
        )
        return ModelProjectionResultLocal.success_result(
            artifact_ref=f"mem:{intent.projection_type}:{intent.aggregate_id}"
        )


class _FailingProjectionStore:
    """In-memory projection store that always fails."""

    def __init__(self, error_message: str = "simulated write failure") -> None:
        self.calls: int = 0
        self._error_message = error_message

    def write(self, intent: ModelProjectionIntent) -> ModelProjectionResultLocal:
        self.calls += 1
        raise RuntimeError(self._error_message)


class _SlowProjectionStore:
    """In-memory projection store with configurable latency."""

    def __init__(self, delay_seconds: float = 0.1) -> None:
        self.writes: list[ModelProjectionIntent] = []
        self._delay = delay_seconds

    def write(self, intent: ModelProjectionIntent) -> ModelProjectionResultLocal:
        time.sleep(self._delay)
        self.writes.append(intent)
        return ModelProjectionResultLocal.success_result()


class _ProjectionEffectAdapter:
    """Adapts an in-memory store to ProtocolProjectionEffect."""

    def __init__(
        self,
        store: _InMemoryProjectionStore
        | _FailingProjectionStore
        | _SlowProjectionStore,
    ) -> None:
        self._store = store

    def execute(self, intent: ModelProjectionIntent) -> ModelProjectionResultLocal:
        return self._store.write(intent)


class _CapturingEventBus:
    """In-memory event bus that captures published envelopes."""

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []

    async def publish_envelope(
        self,
        *,
        envelope: ModelEventEnvelope[Any],
        topic: str,
        key: bytes | None = None,
    ) -> None:
        self.published.append(
            {
                "envelope": envelope,
                "topic": topic,
                "key": key,
                "captured_at": datetime.now(UTC),
            }
        )


class _FakeOutputEvent(BaseModel):
    entity_id: UUID = uuid4()
    value: str = "integration-test-event"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_result(
    projection_intents: tuple[ModelProjectionIntent, ...] = (),
    output_events: list[BaseModel] | None = None,
    **overrides: object,
) -> ModelDispatchResult:
    defaults: dict[str, object] = {
        "status": EnumDispatchStatus.SUCCESS,
        "topic": "test.topic",
        "started_at": datetime.now(UTC),
        "correlation_id": uuid4(),
        "dispatcher_id": "integration-test-dispatcher",
        "projection_intents": projection_intents,
        "output_events": output_events or [],
    }
    defaults.update(overrides)
    return ModelDispatchResult(**defaults)


def _make_intent(**overrides: object) -> ModelProjectionIntent:
    defaults: dict[str, object] = {
        "subject": "NodeRegistration",
        "aggregate_id": uuid4(),
        "projection_type": "NodeRegistration",
        "causation_event_id": uuid4(),
        "correlation_id": uuid4(),
    }
    defaults.update(overrides)
    return ModelProjectionIntent(**defaults)


# ---------------------------------------------------------------------------
# R3-1: Projection persisted before Kafka consumer receives the intent
# ---------------------------------------------------------------------------


class TestProjectionPersistedBeforeKafka:
    """R3-1: Verify projection is persisted before Kafka publish occurs."""

    @pytest.mark.asyncio
    async def test_projection_write_recorded_before_kafka_publish(self) -> None:
        """Assert that when apply() completes, the store write precedes publish.

        Because DispatchResultApplier.apply() executes projection synchronously
        in Phase 0 before Kafka publish in Phase 2, by the time publish_envelope
        is called the store must already contain the write record.
        """
        store = _InMemoryProjectionStore()
        bus = _CapturingEventBus()
        projection_write_timestamps: list[datetime] = []
        kafka_publish_timestamps: list[datetime] = []

        original_write = store.write

        def _timestamped_write(
            intent: ModelProjectionIntent,
        ) -> ModelProjectionResultLocal:
            result = original_write(intent)
            projection_write_timestamps.append(datetime.now(UTC))
            return result

        store.write = _timestamped_write  # type: ignore[method-assign]

        original_publish = bus.publish_envelope

        async def _timestamped_publish(**kwargs: object) -> None:
            await original_publish(**kwargs)
            kafka_publish_timestamps.append(datetime.now(UTC))

        bus.publish_envelope = _timestamped_publish  # type: ignore[method-assign]

        proj_intent = _make_intent()
        result = _make_result(
            projection_intents=(proj_intent,),
            output_events=[_FakeOutputEvent()],
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="onex.evt.test.projection-ordering.v1",
            projection_effect=_ProjectionEffectAdapter(store),
        )
        await applier.apply(result)

        # Assertions: R3-1
        assert len(store.writes) == 1, "Exactly one projection write expected"
        assert len(bus.published) == 1, "Exactly one Kafka publish expected"
        assert len(projection_write_timestamps) == 1
        assert len(kafka_publish_timestamps) == 1

        # Projection timestamp must be strictly before Kafka publish timestamp
        assert projection_write_timestamps[0] <= kafka_publish_timestamps[0], (
            "Projection must be written before Kafka publish"
        )

    @pytest.mark.asyncio
    async def test_projection_payload_persisted_correctly(self) -> None:
        """Verify the correct projection data is written to the store."""
        store = _InMemoryProjectionStore()
        bus = _CapturingEventBus()
        aggregate_id = uuid4()
        causation_event_id = uuid4()
        correlation_id = uuid4()
        payload = {"node_id": str(aggregate_id), "state": "registered"}

        proj_intent = _make_intent(
            subject="NodeRegistration",
            aggregate_id=aggregate_id,
            projection_type="NodeRegistration",
            causation_event_id=causation_event_id,
            correlation_id=correlation_id,
            payload=payload,
        )
        result = _make_result(projection_intents=(proj_intent,))

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="onex.evt.test.v1",
            projection_effect=_ProjectionEffectAdapter(store),
        )
        await applier.apply(result)

        assert len(store.writes) == 1
        write = store.writes[0]
        assert write["aggregate_id"] == aggregate_id
        assert write["projection_type"] == "NodeRegistration"
        assert write["causation_event_id"] == causation_event_id
        assert write["payload"] == payload


# ---------------------------------------------------------------------------
# R3-2: Projection failure → no Kafka message published
# ---------------------------------------------------------------------------


class TestProjectionFailureNoKafka:
    """R3-2: Simulate projection failure, verify no Kafka message is published."""

    @pytest.mark.asyncio
    async def test_projection_failure_no_kafka_messages(self) -> None:
        """When projection raises, no Kafka messages must reach the bus."""
        failing_store = _FailingProjectionStore("DB connection refused")
        bus = _CapturingEventBus()

        proj_intent = _make_intent()
        result = _make_result(
            projection_intents=(proj_intent,),
            output_events=[_FakeOutputEvent()],
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="onex.evt.test.v1",
            projection_effect=_ProjectionEffectAdapter(failing_store),
        )

        with pytest.raises(ProjectionError) as exc_info:
            await applier.apply(result)

        assert len(bus.published) == 0, (
            "Zero Kafka messages must be published when projection fails"
        )
        assert "DB connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_projection_failure_exception_carries_context(self) -> None:
        """ProjectionError must carry projection_type and originating_event_id."""
        aggregate_id = uuid4()
        causation_event_id = uuid4()
        failing_store = _FailingProjectionStore()
        bus = _CapturingEventBus()

        proj_intent = _make_intent(
            projection_type="NodeRegistration",
            aggregate_id=aggregate_id,
            causation_event_id=causation_event_id,
        )
        result = _make_result(projection_intents=(proj_intent,))

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="onex.evt.test.v1",
            projection_effect=_ProjectionEffectAdapter(failing_store),
        )

        with pytest.raises(ProjectionError) as exc_info:
            await applier.apply(result)

        err = exc_info.value
        assert err.projection_type == "NodeRegistration"
        assert err.originating_event_id == causation_event_id

    @pytest.mark.asyncio
    async def test_multiple_intents_failure_on_first_no_kafka(self) -> None:
        """When first of two projection intents fails, zero Kafka messages published."""
        failing_store = _FailingProjectionStore()
        bus = _CapturingEventBus()

        intent_a = _make_intent(projection_type="TypeA")
        intent_b = _make_intent(projection_type="TypeB")
        result = _make_result(
            projection_intents=(intent_a, intent_b),
            output_events=[_FakeOutputEvent()],
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="onex.evt.test.v1",
            projection_effect=_ProjectionEffectAdapter(failing_store),
        )

        with pytest.raises(ProjectionError):
            await applier.apply(result)

        assert failing_store.calls == 1, (
            "Only first projection should have been attempted"
        )
        assert len(bus.published) == 0


# ---------------------------------------------------------------------------
# R3-3: Slow projection defers Kafka publish, does not skip it
# ---------------------------------------------------------------------------


class TestSlowProjectionDeferredKafka:
    """R3-3: Slow projection delays Kafka publish but does not skip it."""

    @pytest.mark.asyncio
    async def test_slow_projection_kafka_publish_still_occurs(self) -> None:
        """A projection taking 50ms must still result in Kafka publish on success."""
        slow_store = _SlowProjectionStore(delay_seconds=0.05)
        bus = _CapturingEventBus()

        proj_intent = _make_intent()
        result = _make_result(
            projection_intents=(proj_intent,),
            output_events=[_FakeOutputEvent()],
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="onex.evt.test.v1",
            projection_effect=_ProjectionEffectAdapter(slow_store),
        )

        start = time.monotonic()
        await applier.apply(result)
        elapsed = time.monotonic() - start

        # Kafka publish happened
        assert len(bus.published) == 1, (
            "Kafka publish must occur despite slow projection"
        )
        # And projection completed
        assert len(slow_store.writes) == 1

        # Elapsed time reflects the projection latency
        assert elapsed >= 0.05, (
            f"Expected at least 50ms latency from slow projection, got {elapsed:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_slow_projection_does_not_publish_before_projection_completes(
        self,
    ) -> None:
        """Verify Kafka publish does not happen until after the slow projection finishes.

        Uses a flag set inside the projection effect and checked during publish
        to confirm sequencing.
        """
        projection_completed = False
        bus_received_before_projection: list[bool] = []

        class _FlagEffect:
            def execute(
                self, intent: ModelProjectionIntent
            ) -> ModelProjectionResultLocal:
                nonlocal projection_completed
                time.sleep(0.05)
                projection_completed = True
                return ModelProjectionResultLocal.success_result()

        class _CheckingBus:
            published: list[dict[str, object]] = []

            async def publish_envelope(
                self,
                *,
                envelope: object,
                topic: str,
                key: object = None,
            ) -> None:
                bus_received_before_projection.append(not projection_completed)
                self.published.append({"envelope": envelope, "topic": topic})

        bus = _CheckingBus()
        proj_intent = _make_intent()
        result = _make_result(
            projection_intents=(proj_intent,),
            output_events=[_FakeOutputEvent()],
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="onex.evt.test.v1",
            projection_effect=_FlagEffect(),
        )
        await applier.apply(result)

        assert len(bus.published) == 1
        # The flag must have been True (projection complete) before publish
        assert not any(bus_received_before_projection), (
            "Kafka publish must NOT happen before projection is complete"
        )
