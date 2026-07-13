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

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _BOUNDARY_DLQ_ENV,
    _make_event_bus_callback,
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
    engine.dispatch = AsyncMock(side_effect=exc)
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

        dispatch_engine.dispatch.assert_awaited_once()
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

        dispatch_engine.dispatch.assert_awaited_once()


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
        dispatch_engine.dispatch = AsyncMock(
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

        assert dispatch_engine.dispatch.await_count == 2
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

        assert dispatch_engine.dispatch.await_count == _BOUNDARY_DLQ_MAX_ATTEMPTS
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
