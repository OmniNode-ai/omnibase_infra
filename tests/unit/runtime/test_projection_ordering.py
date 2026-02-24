# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for projection ordering guarantee (OMN-2510).

Validates that DispatchResultApplier executes NodeProjectionEffect synchronously
BEFORE Kafka publish, and that projection failure blocks all Kafka emit.

Coverage:
    - Projection executes before Kafka publish (ordering contract)
    - Projection failure blocks Kafka publish (zero partial emit)
    - Projection failure raises ProjectionError with context
    - No projection effect configured but intents present → RuntimeHostError
    - No projection intents → no projection call, publish proceeds normally
    - Projection success → intent execution → Kafka publish (full happy path)
    - Slow projection (within timeout) defers publish, does not skip it
    - Multiple projection intents execute sequentially, all must succeed
    - Effect result success=False raises ProjectionError
    - Projection error logged with projector_key, event_type, correlation_id

Related:
    - OMN-2363: Projection ordering guarantee epic
    - OMN-2508: NodeProjectionEffect (omnibase_spi) — stubbed here
    - OMN-2509: Reducer emits ModelProjectionIntent (omnibase_core) — canonical model used (OMN-2718)
    - OMN-2510: Runtime wires projection before Kafka publish (this ticket)
    - OMN-2718: Remove ModelProjectionIntent local stub, use omnibase_core canonical
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel

from omnibase_core.models.projectors.model_projection_intent import (
    ModelProjectionIntent,
)
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.errors.error_projection import ProjectionError
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.models.model_projection_result_local import (
    ModelProjectionResultLocal,
)
from omnibase_infra.runtime.protocol_projection_effect import ProtocolProjectionEffect
from omnibase_infra.runtime.service_dispatch_result_applier import DispatchResultApplier

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubProjectionEffect:
    """Minimal ProtocolProjectionEffect that succeeds by default."""

    def __init__(
        self,
        *,
        should_fail: bool = False,
        fail_with: Exception | None = None,
        return_success_false: bool = False,
        side_effects: list[Exception | None] | None = None,
    ) -> None:
        self.calls: list[ModelProjectionIntent] = []
        self._should_fail = should_fail
        self._fail_with = fail_with
        self._return_success_false = return_success_false
        self._side_effects = side_effects or []

    def execute(self, intent: ModelProjectionIntent) -> ModelProjectionResultLocal:
        self.calls.append(intent)
        call_idx = len(self.calls) - 1
        if self._side_effects and call_idx < len(self._side_effects):
            exc = self._side_effects[call_idx]
            if exc is not None:
                raise exc
        if self._should_fail:
            raise (self._fail_with or RuntimeError("projection write failed"))
        if self._return_success_false:
            return ModelProjectionResultLocal.failure_result("explicit failure")
        return ModelProjectionResultLocal.success_result(artifact_ref="row:1")


class _StubEnvelope(BaseModel):
    """Minimal envelope for test projection intents."""

    value: str = "stub"


def _make_projection_intent(**overrides: object) -> ModelProjectionIntent:
    """Build a ModelProjectionIntent with sensible defaults.

    Uses the canonical omnibase_core model fields:
        projector_key, event_type, envelope, correlation_id.
    """
    defaults: dict[str, object] = {
        "projector_key": "node_registration_projector",
        "event_type": "node.registration.v1",
        "envelope": _StubEnvelope(),
        "correlation_id": uuid4(),
    }
    defaults.update(overrides)
    return ModelProjectionIntent(**defaults)


def _make_result(
    projection_intents: tuple[ModelProjectionIntent, ...] = (),
    **overrides: object,
) -> ModelDispatchResult:
    """Build a ModelDispatchResult with sensible defaults."""
    defaults: dict[str, object] = {
        "status": EnumDispatchStatus.SUCCESS,
        "topic": "test.topic",
        "started_at": datetime.now(UTC),
        "correlation_id": uuid4(),
        "dispatcher_id": "test-dispatcher",
        "projection_intents": projection_intents,
    }
    defaults.update(overrides)
    return ModelDispatchResult(**defaults)


# ---------------------------------------------------------------------------
# ProtocolProjectionEffect protocol satisfaction
# ---------------------------------------------------------------------------


class TestProtocolProjectionEffect:
    """Verify that _StubProjectionEffect satisfies ProtocolProjectionEffect."""

    def test_stub_satisfies_protocol(self) -> None:
        """_StubProjectionEffect must satisfy ProtocolProjectionEffect protocol."""
        stub = _StubProjectionEffect()
        assert isinstance(stub, ProtocolProjectionEffect)


# ---------------------------------------------------------------------------
# Ordering contract: projection before Kafka
# ---------------------------------------------------------------------------


class TestProjectionOrdering:
    """Validate that projection executes synchronously before Kafka publish."""

    @pytest.mark.asyncio
    async def test_projection_executes_before_kafka_publish(self) -> None:
        """Projection must complete before any Kafka publish call.

        Verifies that execute() is called before publish_envelope() using
        a shared call-order tracker.
        """
        call_order: list[str] = []

        class _TrackingEffect:
            def execute(
                self, intent: ModelProjectionIntent
            ) -> ModelProjectionResultLocal:
                call_order.append("projection.execute")
                return ModelProjectionResultLocal.success_result()

        bus = MagicMock()

        async def _track_publish(**kwargs: object) -> None:
            call_order.append("kafka.publish")

        bus.publish_envelope = AsyncMock(side_effect=_track_publish)

        class _FakeEvent(BaseModel):
            value: str = "x"

        proj_intent = _make_projection_intent()
        result = _make_result(
            projection_intents=(proj_intent,),
            output_events=[_FakeEvent()],
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            projection_effect=_TrackingEffect(),
        )
        await applier.apply(result)

        assert call_order == ["projection.execute", "kafka.publish"], (
            f"Expected projection before kafka, got: {call_order}"
        )

    @pytest.mark.asyncio
    async def test_projection_executes_before_intents(self) -> None:
        """Projection must execute before IntentExecutor.execute_all()."""
        call_order: list[str] = []

        class _TrackingEffect:
            def execute(
                self, intent: ModelProjectionIntent
            ) -> ModelProjectionResultLocal:
                call_order.append("projection.execute")
                return ModelProjectionResultLocal.success_result()

        executor = MagicMock()

        async def _track_intents(*args: object, **kwargs: object) -> None:
            call_order.append("intent.execute_all")

        executor.execute_all = AsyncMock(side_effect=_track_intents)

        from omnibase_core.models.reducer.model_intent import ModelIntent

        class _StubPayload(BaseModel):
            intent_type: str = "test.stub"

        intent = ModelIntent(
            intent_type="test.intent",
            target="test://target",
            payload=_StubPayload(),
        )
        proj_intent = _make_projection_intent()
        bus = AsyncMock()

        result = _make_result(
            projection_intents=(proj_intent,),
            output_intents=(intent,),
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            intent_executor=executor,
            projection_effect=_TrackingEffect(),
        )
        await applier.apply(result)

        assert call_order == ["projection.execute", "intent.execute_all"], (
            f"Expected projection before intents, got: {call_order}"
        )


# ---------------------------------------------------------------------------
# Projection failure blocks Kafka publish
# ---------------------------------------------------------------------------


class TestProjectionFailureBlocksKafka:
    """Validate that projection failure prevents any Kafka publish."""

    @pytest.mark.asyncio
    async def test_projection_raise_blocks_kafka(self) -> None:
        """When projection raises, publish_envelope must never be called."""
        bus = AsyncMock()
        effect = _StubProjectionEffect(should_fail=True)
        proj_intent = _make_projection_intent()
        result = _make_result(projection_intents=(proj_intent,))

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            projection_effect=effect,
        )

        with pytest.raises(ProjectionError):
            await applier.apply(result)

        bus.publish_envelope.assert_not_called()

    @pytest.mark.asyncio
    async def test_projection_raise_blocks_intent_execution(self) -> None:
        """When projection raises, IntentExecutor must NOT be called."""
        bus = AsyncMock()
        executor = AsyncMock()
        effect = _StubProjectionEffect(should_fail=True)
        proj_intent = _make_projection_intent()
        result = _make_result(projection_intents=(proj_intent,))

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            intent_executor=executor,
            projection_effect=effect,
        )

        with pytest.raises(ProjectionError):
            await applier.apply(result)

        executor.execute_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_projection_success_false_blocks_kafka(self) -> None:
        """When projection returns success=False, publish must be blocked."""
        bus = AsyncMock()
        effect = _StubProjectionEffect(return_success_false=True)
        proj_intent = _make_projection_intent()

        class _FakeEvent(BaseModel):
            value: str = "x"

        result = _make_result(
            projection_intents=(proj_intent,),
            output_events=[_FakeEvent()],
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            projection_effect=effect,
        )

        with pytest.raises(ProjectionError):
            await applier.apply(result)

        bus.publish_envelope.assert_not_called()

    @pytest.mark.asyncio
    async def test_projection_error_contains_projection_type(self) -> None:
        """ProjectionError must include projector_key for operator diagnostics."""
        bus = AsyncMock()
        effect = _StubProjectionEffect(should_fail=True)
        proj_intent = _make_projection_intent(
            projector_key="node_registration_projector",
        )
        result = _make_result(projection_intents=(proj_intent,))

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            projection_effect=effect,
        )

        with pytest.raises(ProjectionError) as exc_info:
            await applier.apply(result)

        err = exc_info.value
        assert err.projection_type == "node_registration_projector"


# ---------------------------------------------------------------------------
# No projection effect configured
# ---------------------------------------------------------------------------


class TestNoProjectionEffectConfigured:
    """Validate behaviour when projection_effect is None."""

    @pytest.mark.asyncio
    async def test_no_projection_effect_with_intents_raises(self) -> None:
        """When projection_intents present but no effect configured, raise RuntimeHostError."""
        bus = AsyncMock()
        proj_intent = _make_projection_intent()
        result = _make_result(projection_intents=(proj_intent,))

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            # no projection_effect
        )

        with pytest.raises(RuntimeHostError):
            await applier.apply(result)

        bus.publish_envelope.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_projection_intents_no_effect_publish_proceeds(self) -> None:
        """When no projection intents, missing effect is fine — publish proceeds."""

        class _FakeEvent(BaseModel):
            value: str = "x"

        bus = AsyncMock()
        result = _make_result(output_events=[_FakeEvent()])

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            # no projection_effect
        )

        await applier.apply(result)

        bus.publish_envelope.assert_called_once()


# ---------------------------------------------------------------------------
# Multiple projection intents
# ---------------------------------------------------------------------------


class TestMultipleProjectionIntents:
    """Validate sequential execution of multiple projection intents."""

    @pytest.mark.asyncio
    async def test_multiple_intents_execute_sequentially(self) -> None:
        """All projection intents must execute in order before Kafka publish."""
        effect = _StubProjectionEffect()
        bus = AsyncMock()

        intent_a = _make_projection_intent(projector_key="type_a_projector")
        intent_b = _make_projection_intent(projector_key="type_b_projector")
        result = _make_result(projection_intents=(intent_a, intent_b))

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            projection_effect=effect,
        )
        await applier.apply(result)

        assert len(effect.calls) == 2
        assert effect.calls[0].projector_key == "type_a_projector"
        assert effect.calls[1].projector_key == "type_b_projector"

    @pytest.mark.asyncio
    async def test_first_projection_failure_stops_remaining(self) -> None:
        """If the first projection intent fails, the second must NOT execute."""
        effect = _StubProjectionEffect(
            side_effects=[RuntimeError("first fails"), None],
        )
        bus = AsyncMock()

        intent_a = _make_projection_intent(projector_key="type_a_projector")
        intent_b = _make_projection_intent(projector_key="type_b_projector")
        result = _make_result(projection_intents=(intent_a, intent_b))

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            projection_effect=effect,
        )

        with pytest.raises(ProjectionError):
            await applier.apply(result)

        # Only one call — second intent must NOT have been attempted
        assert len(effect.calls) == 1
        bus.publish_envelope.assert_not_called()


# ---------------------------------------------------------------------------
# Slow projection (within timeout) — deferred, not skipped
# ---------------------------------------------------------------------------


class TestSlowProjection:
    """Verify that a slow-but-successful projection still allows Kafka publish."""

    @pytest.mark.asyncio
    async def test_slow_projection_delays_but_does_not_skip_kafka(self) -> None:
        """A projection that takes 50ms must still allow Kafka publish on success."""

        class _SlowEffect:
            def execute(
                self, intent: ModelProjectionIntent
            ) -> ModelProjectionResultLocal:
                # Simulate synchronous blocking work (short, deterministic)
                time.sleep(0.05)
                return ModelProjectionResultLocal.success_result()

        bus = AsyncMock()

        class _FakeEvent(BaseModel):
            value: str = "x"

        proj_intent = _make_projection_intent()
        result = _make_result(
            projection_intents=(proj_intent,),
            output_events=[_FakeEvent()],
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            projection_effect=_SlowEffect(),
        )
        await applier.apply(result)

        bus.publish_envelope.assert_called_once()


# ---------------------------------------------------------------------------
# Happy path: full pipeline
# ---------------------------------------------------------------------------


class TestFullPipelineHappyPath:
    """End-to-end happy path: projection -> intents -> Kafka."""

    @pytest.mark.asyncio
    async def test_full_pipeline_succeeds(self) -> None:
        """Projection, intents, and Kafka publish all execute in correct order."""
        call_order: list[str] = []

        class _TrackingEffect:
            def execute(
                self, intent: ModelProjectionIntent
            ) -> ModelProjectionResultLocal:
                call_order.append(f"projection:{intent.projector_key}")
                return ModelProjectionResultLocal.success_result()

        executor = MagicMock()

        async def _track_intents(*args: object, **kwargs: object) -> None:
            call_order.append("intents")

        executor.execute_all = AsyncMock(side_effect=_track_intents)

        bus = MagicMock()

        async def _track_publish(**kwargs: object) -> None:
            call_order.append("kafka")

        bus.publish_envelope = AsyncMock(side_effect=_track_publish)

        from omnibase_core.models.reducer.model_intent import ModelIntent

        class _StubPayload(BaseModel):
            intent_type: str = "test.stub"

        class _FakeEvent(BaseModel):
            value: str = "x"

        proj_intent = _make_projection_intent(
            projector_key="node_registration_projector"
        )
        intent = ModelIntent(
            intent_type="test.intent",
            target="test://target",
            payload=_StubPayload(),
        )
        result = _make_result(
            projection_intents=(proj_intent,),
            output_intents=(intent,),
            output_events=[_FakeEvent()],
        )

        applier = DispatchResultApplier(
            event_bus=bus,
            output_topic="out.topic",
            intent_executor=executor,
            projection_effect=_TrackingEffect(),
        )
        await applier.apply(result)

        assert call_order == [
            "projection:node_registration_projector",
            "intents",
            "kafka",
        ], f"Unexpected pipeline order: {call_order}"
