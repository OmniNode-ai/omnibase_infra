# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-then-GREEN proof for OMN-14498 — the non-outbox result-applier publish
seam must never silently drop, even at the DEFAULT (unset)
``ONEX_BOUNDARY_DLQ_ENABLED`` flag state.

Adversarial verify (Linear comment 3c6da9a0-7d29-4bbc-9342-585aea0e8b5d)
proved the prior "proof rerun" PASS was vacuous: the two OMN-14498 seam
tests that exist (``test_sync_event_publisher_dlq_omn14498.py`` and
``test_seam_correlation_survives_ack_and_dlq_roundtrip_omn14498.py``) either
drive the unrelated ``_make_sync_event_publisher`` fire-and-forget leg, or
force ``ONEX_BOUNDARY_DLQ_ENABLED=1`` — a configuration the runtime does not
ship with by default. Neither exercises the AC's actual named trigger: a
forced ``result_applier.apply()`` publish failure on the auto-wired
NON-outbox (``propagate_publish_failures=False``, i.e. a non-``state_io``
contract) path, at the DEFAULT flag state.

Counter-probe from that comment, run against the real
``_make_event_bus_callback`` with the flag unset and a forced ``apply()``
failure:

    Auto-wiring callback error: ... error=RuntimeError: downstream publish
    failed (forced) ...
    _boundary_dlq_enabled() = False
    result_applier.apply awaited : 1
    callback raised              : None
    _publish_raw_to_dlq awaited  : 0
    OFFSET WOULD ADVANCE (ACK)   : True
    VERDICT: SILENT-DROP

That is the RED this file pins as ``test_apply_publish_failure_no_dlq_route_...``
(before the fix: passes because the drop happens; after the fix: passes
because the callback now raises instead of silently ACKing). The paired
``test_apply_publish_failure_routes_to_dlq_at_default_flag_state`` proves the
alternative accepted outcome — durable DLQ persistence — is ALSO reachable
at the same default flag state, without needing ``ONEX_BOUNDARY_DLQ_ENABLED``
at all: this seam's own publish failure is not the doubtful/unvalidated
payload the staged rollout exists to hold back, it is a handler result that
already dispatched successfully and only failed to land downstream, so the
DLQ route the sync-publisher leg already gets unconditionally (#2436 /
OMN-15029) is mirrored here rather than staged behind the boundary flag.

Fix location: ``_dispatch_with_bounded_retry``'s ``result_applier.apply()``
except-clause (non-outbox arm) and the new ``_route_apply_publish_failure``
helper in ``handler_wiring.py``. Untouched: a RAW handler/dispatch exception
(never reaching ``apply()``) still keeps the historical staged-rollout
swallow at the default flag state — see
``test_boundary_dlq_omn14507.py::TestBoundaryDlqFlagOff`` (run unmodified
alongside this file as the "other seams are untouched" regression guard).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _BOUNDARY_DLQ_ENV,
    BoundaryApplyPublishError,
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
    """Build the boundary under its required synthetic contract scope
    (OMN-15474 made dispatcher scope a required argument)."""
    return _make_contract_scoped_event_bus_callback(
        topic,
        dispatch_engine,  # type: ignore[arg-type]
        result_applier=result_applier,  # type: ignore[arg-type]
        allowed_dispatcher_ids={"test-dispatcher"},
        **kwargs,  # type: ignore[arg-type]
    )


def _dlq_capable_event_bus() -> MagicMock:
    """Transport-mock-lint compliant (OMN-13026): spec'd to EventBusKafka so
    only real attributes can be set."""
    bus = MagicMock(spec=EventBusKafka)
    bus._publish_raw_to_dlq = AsyncMock()
    return bus


def _succeeding_dispatch_engine(result: object) -> MagicMock:
    """A dispatch engine that returns a non-None result — the exact
    precondition for ``result_applier.apply()`` to be invoked at all."""
    engine = MagicMock()
    engine.dispatch_scoped = AsyncMock(return_value=result)
    return engine


def _failing_result_applier(exc: Exception) -> MagicMock:
    applier = MagicMock()
    applier.apply = AsyncMock(side_effect=exc)
    return applier


def _envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object].model_construct(
        event_type="onex.cmd.test.v1",
        payload={},
        correlation_id=uuid4(),
    )


class TestApplyPublishFailureDefaultFlagState:
    """``ONEX_BOUNDARY_DLQ_ENABLED`` UNSET (production default) — the exact
    configuration the adversarial-verify counter-probe used."""

    @pytest.mark.asyncio
    async def test_apply_publish_failure_routes_to_dlq_at_default_flag_state(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A forced result-applier publish failure on a non-outbox contract
        lands on the DLQ even though ``ONEX_BOUNDARY_DLQ_ENABLED`` is unset."""
        monkeypatch.delenv(_BOUNDARY_DLQ_ENV, raising=False)

        boom = RuntimeError("downstream publish failed (forced)")
        dispatch_engine = _succeeding_dispatch_engine(MagicMock(name="dispatch_result"))
        result_applier = _failing_result_applier(boom)
        event_bus = _dlq_capable_event_bus()
        event_bus._publish_raw_to_dlq = AsyncMock(return_value=True)

        callback = _make_event_bus_callback(
            "onex.evt.platform.context-roi.v1",
            dispatch_engine,
            result_applier=result_applier,
            event_bus=event_bus,
            # Non-outbox contract: propagate_publish_failures defaults False.
        )

        envelope = _envelope()
        await callback(envelope)  # must not raise -- DLQ persisted, safe to ACK

        dispatch_engine.dispatch_scoped.assert_awaited_once()
        result_applier.apply.assert_awaited_once()
        event_bus._publish_raw_to_dlq.assert_awaited_once()
        call_kwargs = event_bus._publish_raw_to_dlq.call_args.kwargs
        assert call_kwargs["original_topic"] == "onex.evt.platform.context-roi.v1"
        assert call_kwargs["error"] is boom
        assert call_kwargs["correlation_id"] == envelope.correlation_id

    @pytest.mark.asyncio
    async def test_apply_publish_failure_no_dlq_route_is_never_silently_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no DLQ-capable bus is wired, the callback must raise (NACK)
        instead of returning normally (ACK) -- the exact silent-drop the
        adversarial-verify counter-probe proved live on this HEAD."""
        monkeypatch.delenv(_BOUNDARY_DLQ_ENV, raising=False)

        boom = RuntimeError("downstream publish failed (forced)")
        dispatch_engine = _succeeding_dispatch_engine(MagicMock(name="dispatch_result"))
        result_applier = _failing_result_applier(boom)

        callback = _make_event_bus_callback(
            "onex.evt.platform.context-roi.v1",
            dispatch_engine,
            result_applier=result_applier,
            # No event_bus supplied -- no DLQ route is possible.
        )

        with pytest.raises(BoundaryApplyPublishError):
            await callback(_envelope())

        dispatch_engine.dispatch_scoped.assert_awaited_once()
        result_applier.apply.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_apply_publish_failure_dlq_write_not_persisted_is_never_acked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_publish_raw_to_dlq`` returning False (documented non-persistence
        contract, OMN-14936) must also raise, not ACK -- a False return is
        not durable and must be treated identically to an unavailable DLQ."""
        monkeypatch.delenv(_BOUNDARY_DLQ_ENV, raising=False)

        boom = RuntimeError("downstream publish failed (forced)")
        dispatch_engine = _succeeding_dispatch_engine(MagicMock(name="dispatch_result"))
        result_applier = _failing_result_applier(boom)
        event_bus = _dlq_capable_event_bus()
        event_bus._publish_raw_to_dlq = AsyncMock(return_value=False)

        callback = _make_event_bus_callback(
            "onex.evt.platform.context-roi.v1",
            dispatch_engine,
            result_applier=result_applier,
            event_bus=event_bus,
        )

        with pytest.raises(BoundaryApplyPublishError):
            await callback(_envelope())

        event_bus._publish_raw_to_dlq.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_apply_publish_failure_dlq_publish_itself_raises_is_never_acked(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DLQ publish that itself raises must not leak the DLQ-internal
        error and must not silently ACK either."""
        monkeypatch.delenv(_BOUNDARY_DLQ_ENV, raising=False)

        boom = RuntimeError("downstream publish failed (forced)")
        dispatch_engine = _succeeding_dispatch_engine(MagicMock(name="dispatch_result"))
        result_applier = _failing_result_applier(boom)
        event_bus = _dlq_capable_event_bus()
        event_bus._publish_raw_to_dlq = AsyncMock(
            side_effect=RuntimeError("dlq topic unreachable")
        )

        callback = _make_event_bus_callback(
            "onex.evt.platform.context-roi.v1",
            dispatch_engine,
            result_applier=result_applier,
            event_bus=event_bus,
        )

        with pytest.raises(BoundaryApplyPublishError):
            await callback(_envelope())

        event_bus._publish_raw_to_dlq.assert_awaited_once()


class TestApplyPublishFailureOutboxPathUnchanged:
    """Contrast case: the outbox (``propagate_publish_failures=True``,
    state_io contract) path already propagates -- this fix must not touch
    that behavior."""

    @pytest.mark.asyncio
    async def test_outbox_path_still_propagates_original_cause(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(_BOUNDARY_DLQ_ENV, raising=False)

        boom = RuntimeError("outbox publish failed (forced)")
        dispatch_engine = _succeeding_dispatch_engine(MagicMock(name="dispatch_result"))
        result_applier = _failing_result_applier(boom)

        callback = _make_event_bus_callback(
            "onex.evt.platform.context-roi.v1",
            dispatch_engine,
            result_applier=result_applier,
            propagate_publish_failures=True,
        )

        with pytest.raises(RuntimeError, match="outbox publish failed"):
            await callback(_envelope())
