# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-then-GREEN proof for OMN-14507 — the auto-wired consume boundary must
not silently discard a handler exception.

Before this fix, ``_make_event_bus_callback``'s ``except Exception`` clause
did exactly one thing: ``logger.error(...)``. The message that triggered the
exception was gone -- no DLQ, no redelivery, no metric -- ACKed by omission
(the callback returned normally, so the consumer's normal offset-commit path
proceeded as if nothing had happened). That is EXISTS-but-WRONG, not
RED-on-absence: the swallow is a real, reachable code path exercised by any
handler that raises.

``test_flag_off_still_swallows_like_before`` reproduces the PRE-fix behavior
verbatim -- it is the RED half of the proof, pinned as a permanent regression
guard (not deleted after the fix) so the default-off historical shape can
never silently regress further. ``test_flag_on_routes_to_dlq_instead_of_vanishing``
is the GREEN half: same raising handler, only the env flag differs, and the
message is now durably preserved in the DLQ instead of vanishing.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _BOUNDARY_DLQ_ENV,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_event_bus_callback as _make_contract_scoped_event_bus_callback,
)


def _make_event_bus_callback(
    topic: str,
    dispatch_engine: object,
    result_applier: object | None = None,
    **kwargs: object,
) -> Callable[..., Awaitable[None]]:
    """Build the boundary under its required synthetic contract scope."""
    return _make_contract_scoped_event_bus_callback(
        topic,
        dispatch_engine,  # type: ignore[arg-type]
        result_applier=result_applier,  # type: ignore[arg-type]
        allowed_dispatcher_ids={"test-dispatcher"},
        **kwargs,  # type: ignore[arg-type]
    )


def _dlq_capable_event_bus() -> MagicMock:
    """A bus mock spec'd to EventBusKafka so only real attributes (including
    _publish_raw_to_dlq, from MixinKafkaDlq) can be set -- satisfies the
    transport-mock-lint gate (OMN-13026, PR #1181's bare-mock incident)."""
    bus = MagicMock(spec=EventBusKafka)
    bus._publish_raw_to_dlq = AsyncMock()
    return bus


def _raising_dispatch_engine(exc: Exception) -> MagicMock:
    engine = MagicMock()
    engine.dispatch_scoped = AsyncMock(side_effect=exc)
    return engine


def _envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object].model_construct(
        event_type="onex.cmd.test.v1",
        payload={},
        correlation_id=uuid4(),
    )


class TestBoundaryDlqFlagOff:
    """flag OFF (default, unset) — the exact pre-OMN-14507 swallow shape."""

    @pytest.mark.asyncio
    async def test_flag_off_still_swallows_like_before(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(_BOUNDARY_DLQ_ENV, raising=False)

        boom = RuntimeError("handler exploded")
        dispatch_engine = _raising_dispatch_engine(boom)
        event_bus = _dlq_capable_event_bus()

        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        # Must not raise -- the boundary contract is "never crash the
        # consumer loop," preserved identically pre- and post-fix.
        await callback(_envelope())

        dispatch_engine.dispatch_scoped.assert_awaited_once()
        # RED premise pinned as a permanent regression guard: with the flag
        # off, the message is still swallowed -- no DLQ, exactly like the
        # pre-fix code. This is the historical shape, not a bug re-introduced
        # by this test; it documents that the staged rollout truly defaults
        # off.
        event_bus._publish_raw_to_dlq.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flag_off_with_no_event_bus_never_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No event_bus supplied at all (legacy call shape) must still work."""
        monkeypatch.delenv(_BOUNDARY_DLQ_ENV, raising=False)

        dispatch_engine = _raising_dispatch_engine(RuntimeError("boom"))
        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
        )

        await callback(_envelope())  # must not raise

        dispatch_engine.dispatch_scoped.assert_awaited_once()


class TestBoundaryDlqFlagOn:
    """flag ON (ONEX_BOUNDARY_DLQ_ENABLED=1) — the OMN-14507 fix."""

    @pytest.mark.asyncio
    async def test_flag_on_routes_to_dlq_instead_of_vanishing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        boom = RuntimeError("handler exploded")
        dispatch_engine = _raising_dispatch_engine(boom)
        event_bus = _dlq_capable_event_bus()

        callback = _make_event_bus_callback(
            "onex.evt.platform.node-heartbeat.v1",
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        envelope = _envelope()
        await callback(envelope)  # must not raise -- still never crashes the consumer

        # GREEN: the same raising handler now lands in the DLQ instead of
        # vanishing. Retried _BOUNDARY_DLQ_MAX_ATTEMPTS times first (bounded,
        # not infinite) -- see test_flag_on_bounded_retry_before_dlq below.
        event_bus._publish_raw_to_dlq.assert_awaited_once()
        call_kwargs = event_bus._publish_raw_to_dlq.call_args.kwargs
        assert call_kwargs["original_topic"] == "onex.evt.platform.node-heartbeat.v1"
        assert call_kwargs["error"] is boom
        assert call_kwargs["correlation_id"] == envelope.correlation_id
        assert call_kwargs["failure_type"] == "handler_exception"

    @pytest.mark.asyncio
    async def test_flag_on_bounded_retry_before_dlq(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Retries are bounded (no infinite redelivery storm) and a transient
        failure that clears within the retry budget must NOT reach DLQ."""
        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        dispatch_engine = MagicMock()
        dispatch_result = MagicMock()
        # Fails once, then succeeds -- must recover without ever touching DLQ.
        dispatch_engine.dispatch_scoped = AsyncMock(
            side_effect=[RuntimeError("transient"), dispatch_result]
        )
        result_applier = MagicMock()
        result_applier.apply = AsyncMock()
        event_bus = _dlq_capable_event_bus()

        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
            result_applier=result_applier,
            event_bus=event_bus,
        )

        await callback(_envelope())

        assert dispatch_engine.dispatch_scoped.await_count == 2
        result_applier.apply.assert_awaited_once()
        event_bus._publish_raw_to_dlq.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_flag_on_retry_is_bounded_not_infinite(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A permanently-failing handler must exhaust a FIXED retry budget,
        never loop forever, before falling through to DLQ."""
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _BOUNDARY_DLQ_MAX_ATTEMPTS,
        )

        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        dispatch_engine = _raising_dispatch_engine(RuntimeError("always fails"))
        event_bus = _dlq_capable_event_bus()

        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        await callback(_envelope())

        assert dispatch_engine.dispatch_scoped.await_count == _BOUNDARY_DLQ_MAX_ATTEMPTS
        event_bus._publish_raw_to_dlq.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_flag_on_without_event_bus_degrades_to_loud_log_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flag on but no event_bus wired (legacy caller) must degrade
        gracefully -- never raise, never crash the consumer."""
        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        dispatch_engine = _raising_dispatch_engine(RuntimeError("boom"))
        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
        )

        await callback(_envelope())  # must not raise

    @pytest.mark.asyncio
    async def test_flag_on_dlq_publish_failure_never_crashes_consumer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DLQ publish that itself raises must not propagate -- the DLQ
        publish is itself a boundary."""
        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        dispatch_engine = _raising_dispatch_engine(RuntimeError("boom"))
        event_bus = _dlq_capable_event_bus()
        event_bus._publish_raw_to_dlq = AsyncMock(
            side_effect=RuntimeError("dlq topic unreachable")
        )

        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        await callback(_envelope())  # must not raise

        event_bus._publish_raw_to_dlq.assert_awaited_once()


def _real_validation_error() -> ValidationError:
    """A genuine pydantic ValidationError -- not a hand-rolled stand-in --
    for the G2 non-retryable-content-error tests. Mirrors the exact shape a
    handler-level wire model raises when ``extra="forbid"`` rejects an
    unknown field (the §7 death signal this boundary must carry)."""

    class _StrictModel(BaseModel):
        model_config = ConfigDict(extra="forbid")
        field: int

    try:
        _StrictModel.model_validate({"field": 1, "unexpected_field": "boom"})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected ValidationError")  # pragma: no cover


class TestBoundaryDlqNonRetryableClassification:
    """G2 (OMN-14507 review): content/config errors must NOT be retried --
    they are deterministic, so retrying only burns the backoff budget before
    an identical failure. This is the exact §7 wire-model-forbid path."""

    @pytest.mark.asyncio
    async def test_content_error_skips_retry_goes_straight_to_dlq(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        validation_error = _real_validation_error()
        dispatch_engine = _raising_dispatch_engine(validation_error)
        event_bus = _dlq_capable_event_bus()

        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        await callback(_envelope())

        # ONE attempt, not _BOUNDARY_DLQ_MAX_ATTEMPTS -- the retry budget is
        # never spent on a guaranteed-repeat content error.
        dispatch_engine.dispatch_scoped.assert_awaited_once()
        event_bus._publish_raw_to_dlq.assert_awaited_once()
        call_kwargs = event_bus._publish_raw_to_dlq.call_args.kwargs
        assert call_kwargs["error"] is validation_error

    @pytest.mark.asyncio
    async def test_protocol_configuration_error_skips_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.errors import ProtocolConfigurationError

        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        config_error = ProtocolConfigurationError("no dispatcher registered")
        dispatch_engine = _raising_dispatch_engine(config_error)
        event_bus = _dlq_capable_event_bus()

        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        await callback(_envelope())

        dispatch_engine.dispatch_scoped.assert_awaited_once()
        event_bus._publish_raw_to_dlq.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_transient_error_still_gets_full_retry_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contrast case: a non-content error (e.g. RuntimeError) is NOT
        classified as non-retryable and still gets the full bounded budget."""
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _BOUNDARY_DLQ_MAX_ATTEMPTS,
        )

        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        dispatch_engine = _raising_dispatch_engine(RuntimeError("transient"))
        event_bus = _dlq_capable_event_bus()

        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        await callback(_envelope())

        assert dispatch_engine.dispatch_scoped.await_count == _BOUNDARY_DLQ_MAX_ATTEMPTS


class TestBoundaryDlqMetricNaming:
    """G3 (OMN-14507 review): only dlq_routed=true is real prevention.
    Everything else must log boundary_swallow_observed, not
    boundary_swallow_prevented, so an operator alerting on the "prevented"
    counter is not misled into believing a lost message survived."""

    @pytest.mark.asyncio
    async def test_flag_off_logs_observed_not_prevented(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.delenv(_BOUNDARY_DLQ_ENV, raising=False)

        dispatch_engine = _raising_dispatch_engine(RuntimeError("boom"))
        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
        )

        with caplog.at_level("ERROR"):
            await callback(_envelope())

        assert "metric_name=boundary_swallow_observed" in caplog.text
        assert "metric_name=boundary_swallow_prevented" not in caplog.text

    @pytest.mark.asyncio
    async def test_flag_on_dlq_success_logs_prevented(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        dispatch_engine = _raising_dispatch_engine(RuntimeError("boom"))
        event_bus = _dlq_capable_event_bus()
        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        with caplog.at_level("ERROR"):
            await callback(_envelope())

        assert "metric_name=boundary_swallow_prevented dlq_routed=true" in caplog.text

    @pytest.mark.asyncio
    async def test_flag_on_dlq_failure_logs_observed_and_message_lost(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        dispatch_engine = _raising_dispatch_engine(RuntimeError("boom"))
        event_bus = _dlq_capable_event_bus()
        event_bus._publish_raw_to_dlq = AsyncMock(
            side_effect=RuntimeError("dlq topic unreachable")
        )
        callback = _make_event_bus_callback(
            "onex.cmd.test.v1",
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        with caplog.at_level("ERROR"):
            await callback(_envelope())

        assert "metric_name=boundary_swallow_prevented" not in caplog.text
        assert "metric_name=boundary_swallow_observed" in caplog.text
        assert "message_lost=true" in caplog.text


# OMN-14551 test-only topic identifiers, shared by the counter tests below
# (single configuration-backed source reused at every site, per CodeRabbit
# review on PR #2424 -- these are the two "unique per test" topics that
# previously duplicated the same literal at each individual test).
_LOSS_COUNTER_TEST_TOPIC = "onex.cmd.omn14551-message-lost-counter-test.v1"
_LOSS_COUNTER_FALSE_RETURN_TEST_TOPIC = (
    "onex.cmd.omn14551-message-lost-false-return-test.v1"
)
_DLQ_SUCCESS_COUNTER_TEST_TOPIC = "onex.cmd.omn14551-dlq-success-counter-test.v1"


class TestBoundaryDlqAlertableMessageLostCounter:
    """OMN-14551 — forbid-verify residual ask: ``message_lost=true`` must
    increment a REAL alertable signal, not just a greppable log line
    ("that MUST page -- a greppable log won't"). RED-then-GREEN proof that
    the increment actually fires on the genuine double-failure path (retry
    budget exhausted AND the best-effort DLQ publish itself also fails) --
    not merely that the counter object exists somewhere unreachable."""

    @pytest.mark.asyncio
    async def test_message_lost_increments_alertable_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _BOUNDARY_MESSAGE_LOST_COUNTER,
        )

        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        # Unique topic label per test so the counter's per-label-set value
        # starts at a known baseline (0 on first touch) regardless of
        # cross-test process-global Counter state.
        topic = _LOSS_COUNTER_TEST_TOPIC
        error_type = "RuntimeError"

        assert _BOUNDARY_MESSAGE_LOST_COUNTER is not None, (
            "prometheus_client Counter failed to initialize -- the "
            "alertable signal is unavailable in this process"
        )
        before = _BOUNDARY_MESSAGE_LOST_COUNTER.labels(
            topic=topic, error_type=error_type
        )._value.get()

        dispatch_engine = _raising_dispatch_engine(RuntimeError("boom"))
        event_bus = _dlq_capable_event_bus()
        event_bus._publish_raw_to_dlq = AsyncMock(
            side_effect=RuntimeError("dlq topic unreachable")
        )
        callback = _make_event_bus_callback(
            topic,
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        await callback(_envelope())  # must not raise -- boundary never crashes

        after = _BOUNDARY_MESSAGE_LOST_COUNTER.labels(
            topic=topic, error_type=error_type
        )._value.get()
        assert after == before + 1, (
            "onex_boundary_message_lost_total did not increment on the "
            "double-failure (retry-exhausted AND DLQ-publish-failed) path -- "
            "this is exactly the EXISTS-but-WRONG class (counter defined "
            "but its call site never reached)"
        )

    @pytest.mark.asyncio
    async def test_dlq_success_path_does_not_increment_message_lost_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Contrast case: DLQ-routed is the SUCCESS path (nothing lost) --
        the loss counter must stay untouched, matching the same
        prevented-vs-observed honesty distinction as the metric_name logs
        (G3, OMN-14507 review)."""
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _BOUNDARY_MESSAGE_LOST_COUNTER,
        )

        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        topic = _DLQ_SUCCESS_COUNTER_TEST_TOPIC
        error_type = "RuntimeError"

        assert _BOUNDARY_MESSAGE_LOST_COUNTER is not None
        before = _BOUNDARY_MESSAGE_LOST_COUNTER.labels(
            topic=topic, error_type=error_type
        )._value.get()

        dispatch_engine = _raising_dispatch_engine(RuntimeError("boom"))
        event_bus = _dlq_capable_event_bus()
        # Explicit True: a genuinely-persisted DLQ publish, per
        # _publish_raw_to_dlq's documented bool-return contract (OMN-14936).
        event_bus._publish_raw_to_dlq = AsyncMock(return_value=True)

        callback = _make_event_bus_callback(
            topic,
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        await callback(_envelope())

        event_bus._publish_raw_to_dlq.assert_awaited_once()
        after = _BOUNDARY_MESSAGE_LOST_COUNTER.labels(
            topic=topic, error_type=error_type
        )._value.get()
        assert after == before, (
            "the DLQ-routed SUCCESS path must never increment "
            "onex_boundary_message_lost_total -- only the genuine "
            "double-failure loss window does"
        )

    @pytest.mark.asyncio
    async def test_dlq_publish_false_return_increments_message_lost_counter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OMN-14936 / CodeRabbit review (PR #2424): ``_publish_raw_to_dlq``
        signals failed persistence via a ``False`` return, not only via a
        raised exception (rejected input, producer unavailable, or the send
        itself failing/timing out WITHOUT raising). Before this test's fix,
        that return value was ignored -- the boundary logged
        ``boundary_swallow_prevented dlq_routed=true`` and left the loss
        counter untouched even though the message was never durably
        persisted. A ``False`` return must be treated exactly like the
        except-branch double-failure: loud log + counter increment."""
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _BOUNDARY_MESSAGE_LOST_COUNTER,
        )

        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        topic = _LOSS_COUNTER_FALSE_RETURN_TEST_TOPIC
        error_type = "RuntimeError"

        assert _BOUNDARY_MESSAGE_LOST_COUNTER is not None
        before = _BOUNDARY_MESSAGE_LOST_COUNTER.labels(
            topic=topic, error_type=error_type
        )._value.get()

        dispatch_engine = _raising_dispatch_engine(RuntimeError("boom"))
        event_bus = _dlq_capable_event_bus()
        # Publish "succeeds" (no exception) but reports non-persistence via
        # its documented bool contract -- the exact case the except-only
        # handling in the original fix missed.
        event_bus._publish_raw_to_dlq = AsyncMock(return_value=False)

        callback = _make_event_bus_callback(
            topic,
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        await callback(_envelope())  # must not raise

        event_bus._publish_raw_to_dlq.assert_awaited_once()
        after = _BOUNDARY_MESSAGE_LOST_COUNTER.labels(
            topic=topic, error_type=error_type
        )._value.get()
        assert after == before + 1, (
            "a False return from _publish_raw_to_dlq (failed persistence "
            "without a raised exception) did not increment "
            "onex_boundary_message_lost_total -- the message is lost here "
            "exactly as if the publish had raised, but the counter treated "
            "it as the success path"
        )

    @pytest.mark.asyncio
    async def test_dlq_publish_false_return_logs_observed_not_prevented(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Companion assertion to the counter test above: the log line for
        a False-return publish must use the same honest
        boundary_swallow_observed/message_lost=true vocabulary as the
        exception path, never boundary_swallow_prevented (G3)."""
        monkeypatch.setenv(_BOUNDARY_DLQ_ENV, "1")

        dispatch_engine = _raising_dispatch_engine(RuntimeError("boom"))
        event_bus = _dlq_capable_event_bus()
        event_bus._publish_raw_to_dlq = AsyncMock(return_value=False)

        callback = _make_event_bus_callback(
            "onex.cmd.omn14551-false-return-log-test.v1",
            dispatch_engine,  # type: ignore[arg-type]
            event_bus=event_bus,
        )

        with caplog.at_level("ERROR"):
            await callback(_envelope())

        assert "metric_name=boundary_swallow_prevented" not in caplog.text
        assert "metric_name=boundary_swallow_observed" in caplog.text
        assert "message_lost=true" in caplog.text
