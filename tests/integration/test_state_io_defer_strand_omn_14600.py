# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-then-GREEN suite for OMN-14600 — the in_flight-lock defer strand.

**Follow-up to OMN-14493/#2296, not a reopen.** #2296 fixed the original defect
(0 rows EVER reaching terminal). This suite pins the residual it left behind:
`handler_wiring.py`'s in_flight-lock ``defer`` branch (a DIFFERENT input
arriving while a prior leg's outbox batch is committed but un-finalized)
reported ``(1, None)`` — a SUCCESS — with the handler never run. Under a
rapid retry-storm (or simply a winner leg that crashes/stalls and never
finalizes) this drops the deferred leg's own input FOREVER: no retry, no
error, no DLQ, no redelivery, and the boot-time-only recovery sweep never
gets a second chance to clear the row within a live process.

**Fable-gate correction (superseding the first fix attempt).** The original
fix reported the defer as a conflict and relied on
``retry_on_optimistic_conflict`` exhausting and raising
``OptimisticConflictError`` to trigger a Kafka redelivery. That does NOT
happen on this runtime: the state_io stateful callback is registered as a
dispatcher on ``MessageDispatchEngine``, whose per-dispatcher invocation loop
catches every exception a dispatcher raises and converts it to a returned
``HANDLER_ERROR`` status instead of re-raising — the exception never reaches
a layer that could trigger redelivery, and the Kafka offset commits
regardless. The corrected fix INLINE-RECOVERS the stalled winner's own batch
(publish + finalize, from inside the deferring leg) instead of depending on
redelivery — see ``handler_wiring.py``'s in_flight-lock branch-b comment for
the full trace. The tests below assert the CORRECTED (inline-recover)
behavior; the file name/module docstring is kept for ticket continuity.

Reuses the fixture harness from ``test_state_io_outbox_seam_red.py`` (the
in-memory ``_FakeStateStoreAdapter`` with real SQL semantics, the recording
bus, the fan-out handler) rather than re-authoring it — define-and-match-
seams: match the existing harness, don't duplicate it.

Each test drives code that EXISTS today. ``asyncio.sleep`` inside
``retry_on_optimistic_conflict`` is patched to a no-op so any retry path is
deterministic and fast instead of depending on ~1-1.5s of real backoff.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.enums.enum_message_category import EnumMessageCategory
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.utils.util_retry_optimistic import OptimisticConflictError
from tests.integration.test_state_io_outbox_seam_red import (
    CID,
    LEG3_ENVELOPE_ID,
    TENANT,
    TOPIC_INBOUND,
    TOPIC_INFERENCE,
    TOPIC_QUALITY,
    TOPIC_ROUTING,
    _DurableRows,
    _FakeStateStoreAdapter,
    _fanout_batch,
    _FanOutHandler,
    _input_envelope,
    _RecordingBus,
    _stateful_callback,
)

_RETRY_MODULE = "omnibase_infra.utils.util_retry_optimistic"
_SWEEP_INTERVAL = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring."
    "_STATE_IO_RECOVERY_SWEEP_INTERVAL_SECONDS"
)

PROBE_CID = UUID("55555555-5555-5555-5555-555555555555")
WARMUP_CID = UUID("66666666-6666-6666-6666-666666666666")


async def _fast_sleep(_seconds: float) -> None:
    """No-op stand-in for asyncio.sleep so retry backoff costs nothing."""
    return


def _probe_envelope() -> ModelEventEnvelope[object]:
    """An unrelated leg on a DIFFERENT correlation — normal traffic continuing
    on the same live process while CID's row sits stranded."""
    return ModelEventEnvelope[object](
        correlation_id=PROBE_CID,
        payload={"correlation_id": str(PROBE_CID), "tenant_id": TENANT},
    )


def _warmup_envelope() -> ModelEventEnvelope[object]:
    """A throwaway dispatch on ``shared_cb``, on yet another correlation, used
    to consume that closure's first-ever (always-eligible) sweep BEFORE CID's
    row goes stranded — see the vacuity note in
    ``test_periodic_sweep_self_heals_a_stranded_row_without_reboot``."""
    return ModelEventEnvelope[object](
        correlation_id=WARMUP_CID,
        payload={"correlation_id": str(WARMUP_CID), "tenant_id": TENANT},
    )


@pytest.mark.integration
def test_defer_against_a_stalled_winner_inline_recovers_and_lets_leg3_proceed() -> None:
    """Corrected Fix A: the defer must INLINE-RECOVER the stalled winner, not
    depend on redelivery (which is dead code on this runtime — see module
    docstring).

    leg-2 commits its 3-event batch (in_flight=True, pending_emissions set)
    then "crashes" — no bus is wired for its own dispatch, so the row is
    committed but nothing ever publishes/finalizes it, and nothing ever will
    on its own (this simulates the winner leg dying, or a retry-storm cascade
    in which the winner itself never gets to finalize). leg-3 is a DIFFERENT
    input on the SAME correlation, arriving while leg-2's batch is still
    un-finalized.

    Before either fix: leg-3's dispatch returns cleanly (``(1, None)``) with
    the handler never run — a REPORTED SUCCESS that commits the Kafka offset
    as if the leg had been handled, silently losing it forever with no error,
    no DLQ, no redelivery.

    After the CORRECTED fix: leg-3's own dispatch publishes + finalizes
    leg-2's stranded batch INLINE (deterministic, no redelivery dependency),
    then reports a conflict so ``retry_on_optimistic_conflict`` re-attempts
    leg-3 in-process; the lock is now clear, so leg-3's OWN handler runs and
    its own batch is published + finalized too — no exception raised, no
    input lost, both legs' intents reach the bus.
    """
    store = _DurableRows()
    adapter = _FakeStateStoreAdapter(store)

    # leg-2: commit-then-crash. No bus — the external applier would normally
    # publish; we simply never call it. This IS the un-finalized window.
    leg2_handler = _FanOutHandler(_fanout_batch(note="leg2"))
    leg2_cb = _stateful_callback(leg2_handler, adapter)

    # leg-3: a different input on the same correlation.
    bus = _RecordingBus()
    leg3_handler = _FanOutHandler(_fanout_batch(note="leg3"), next_state="RE_ROUTED")
    leg3_cb = _stateful_callback(leg3_handler, adapter, event_bus=bus)

    raised: BaseException | None = None

    async def _run() -> None:
        nonlocal raised
        await leg2_cb(_input_envelope())  # commit, then "crash" (never applied)
        assert store.rows[str(CID)]["in_flight"] is True, (
            "precondition: leg-2's commit landed"
        )
        assert store.rows[str(CID)]["pending_emissions"], (
            "precondition: leg-2's batch is persisted, un-finalized"
        )

        try:
            await leg3_cb(_input_envelope(envelope_id=LEG3_ENVELOPE_ID))
        except OptimisticConflictError as exc:
            raised = exc

    with patch(f"{_RETRY_MODULE}.asyncio.sleep", _fast_sleep):
        asyncio.run(_run())

    assert raised is None, (
        "leg-3's dispatch must NOT raise: the defer branch inline-recovers "
        "leg-2's stranded batch (publish + finalize) instead of depending on "
        "an exhausted-retry exception that this runtime never redelivers on "
        f"— got {raised!r}"
    )
    assert leg3_handler.call_count == 1, (
        "leg-3's OWN handler must run (on the in-process retry) once the "
        "lock is cleared by the inline-recovery of leg-2's batch — the leg "
        "is no longer permanently dropped, even though it then folds (below)"
    )
    assert leg3_handler.folded_count == 1, (
        "leg-3's retry loads leg-2's just-finalized row, whose DOMAIN payload "
        "still carries the FSM's own in_flight=true (a separate concept from "
        "the outbox row's in_flight column, cleared by finalize) -- the "
        "handler's own fold-on-redelivery dedup guard folds it, emitting "
        "nothing new. This mirrors HandlerDelegationWorkflow's real "
        "duplicate-rejection semantics, not a defect of the fix."
    )
    row = store.rows[str(CID)]
    assert row["in_flight"] is False, (
        "leg-2's batch must finalize within leg-3's own dispatch "
        "(inline-recovery) — leg-3's own fold is a no-op persistence-wise, "
        "so this reflects leg-2's finalize alone"
    )
    assert not row["pending_emissions"]
    published = bus.envelopes_for(CID)
    assert len(published) == 3, (
        "leg-2's inline-recovered 3-event batch must reach the bus; leg-3 "
        f"folds and emits nothing new; got {len(published)}"
    )
    assert bus.topics_for(CID) == [TOPIC_ROUTING, TOPIC_INFERENCE, TOPIC_QUALITY], (
        "leg-2's batch is the only one published (inline-recovered by leg-3's "
        "own dispatch, which then folds on its own attempt)"
    )


@pytest.mark.integration
def test_periodic_sweep_self_heals_a_stranded_row_without_reboot() -> None:
    """Fix B: a row whose winner never finalizes must self-heal within a LIVE
    process, not only at the next boot/redeploy.

    Reproduces the exact residual the ticket names: recovery ran once (on the
    first-ever dispatch through a given handler's wired callback) and never
    again, so a batch committed-but-never-finalized by an abandoned winner
    stayed stranded for the rest of the process's life.

    OMN-14600 (Fable-gate correction): this test used to also dispatch a
    "leg-3" on CID before the probe. That is now WRONG — leg-3 arriving on
    the SAME correlation as the stranded row would hit the in_flight-lock
    DEFER branch, which (per the corrected Fix A above) INLINE-RECOVERS the
    row itself, healing it before the sweep ever runs. Leaving that dispatch
    in would make this test pass VACUOUSLY (healed by Fix A, not by the
    sweep this test claims to prove) — see
    ``test_defer_against_a_stalled_winner_inline_recovers_and_lets_leg3_proceed``
    for that scenario. This test now isolates the SWEEP as the ONLY healing
    path: no leg ever arrives on CID again after leg-2 stalls; only an
    UNRELATED probe (a different correlation) drives traffic through the
    same wired closure and triggers its periodic sweep tick.

    OMN-14600 (verify correction, second vacuity): a lone ``shared_cb``
    dispatch after leg-2 stalls is ALSO its closure's first-ever call --
    ``_recovery_last_run_monotonic`` starts at 0.0, so `now - 0.0` always
    exceeds the interval trivially. That makes a single-dispatch version of
    this test indistinguishable from the OLD one-shot-boolean gate the
    module docstring says this replaces: a boot-once sweep would pass this
    exact test too, so it proved nothing about PERIODICITY. The warm-up
    dispatch below (on yet a third correlation, before leg-2 ever stalls)
    consumes that first-ever trigger on nothing to recover; the actual probe
    is then ``shared_cb``'s SECOND call. Under the old one-shot gate the
    second call would never re-sweep (already consumed) and CID would stay
    stranded; under the interval-gated fix (patched to 0.0 below) elapsed
    time re-arms it regardless of how many times it already ran — that
    contrast is what makes this a genuine, non-vacuous proof of Fix B.

    The sweep interval is patched to 0 so the test is deterministic and fast
    — it proves the gate is TIME-based (periodic, re-triggerable) rather than
    a one-shot boolean, without depending on wall-clock sleeps.
    """
    store = _DurableRows()
    adapter = _FakeStateStoreAdapter(store)

    # leg-2: commit-then-crash (no bus — see test above). No leg-3: this
    # test isolates the periodic sweep as CID's ONLY healing path.
    leg2_handler = _FanOutHandler(_fanout_batch(note="leg2"))
    leg2_cb = _stateful_callback(leg2_handler, adapter)

    # The later probe uses this closure (and its bus). A SEPARATE closure
    # from leg2_cb's (no bus) — its own sweep timer starts fresh.
    bus = _RecordingBus()
    probe_handler = _FanOutHandler(_fanout_batch(note="probe"))
    shared_cb = _stateful_callback(probe_handler, adapter, event_bus=bus)

    async def _run() -> None:
        # Consume shared_cb's first-ever (always-eligible) sweep on nothing,
        # BEFORE CID's row ever goes stranded — see the vacuity note above.
        await shared_cb(_warmup_envelope())

        await leg2_cb(_input_envelope())
        assert store.rows[str(CID)]["in_flight"] is True
        assert store.rows[str(CID)]["pending_emissions"]

        # shared_cb's SECOND dispatch (a different, unrelated correlation
        # again). This is the genuinely discriminating call: it only
        # re-sweeps if the gate is interval-based, not one-shot.
        await shared_cb(_probe_envelope())

    with (
        patch(f"{_RETRY_MODULE}.asyncio.sleep", _fast_sleep),
        # create=True: pre-fix, this module constant does not exist yet (the
        # fix introduces it). The patch still applies cleanly in that case --
        # it just has no effect, since pre-fix code never reads it (gated by
        # the old one-shot boolean instead) -- so the RED failure below comes
        # from the real behavior, not a missing-attribute vacuity.
        patch(_SWEEP_INTERVAL, 0.0, create=True),
    ):
        asyncio.run(_run())

    row = store.rows[str(CID)]
    recovered = bus.envelopes_for(CID)
    assert len(recovered) == 3 and row["in_flight"] is False, (
        "STRANDED FOREVER (OMN-14600): leg-2's committed-but-unfinalized batch "
        "was never re-published/finalized by a LATER dispatch on the same "
        "live process. The recovery sweep only ever ran once (boot-time), so "
        "a row abandoned mid-process stays stuck until the next redeploy. "
        f"in_flight={row['in_flight']!r} recovered={len(recovered)}"
    )
    assert bus.topics_for(CID) == [TOPIC_ROUTING, TOPIC_INFERENCE, TOPIC_QUALITY], (
        "self-heal must preserve emission order + per-class topic routing"
    )
    assert not row["pending_emissions"], "finalize must clear pending_emissions"


@pytest.mark.integration
def test_boundary_level_engine_dispatch_reports_success_after_inline_recovery() -> None:
    """Fable #13: prove the fix holds THROUGH the real MessageDispatchEngine,
    not just at the layer directly below it.

    Every other test in this module drives the stateful dispatch callback
    DIRECTLY — bypassing ``MessageDispatchEngine.dispatch()``'s own
    per-dispatcher invocation loop entirely. That loop is EXACTLY the layer
    that swallows a dispatcher's raised exception and converts it to a
    returned ``HANDLER_ERROR`` status instead of re-raising (see module
    docstring) — the layer whose behavior motivated the corrected fix in the
    first place. A suite that never drives dispatch through the real engine
    structurally cannot detect a regression back to "rely on a raised
    exception" (that regression would still pass every test in this module,
    since none of them go through the engine's own catch-all).

    This test registers the state_io stateful callback AS a real dispatcher
    on a real ``MessageDispatchEngine`` and drives ``engine.dispatch()``
    through it for the SAME stalled-winner / deferring-leg scenario, proving:

    (a) the engine reports ``SUCCESS`` — not ``HANDLER_ERROR``. If the defer
        branch ever regresses to depending on ``retry_on_optimistic_conflict``
        exhausting and raising ``OptimisticConflictError`` for redelivery,
        THIS is where it would be caught: the exception is absorbed by the
        engine's catch-all and folded into ``HANDLER_ERROR``, which this
        assertion catches.
    (b) the real terminal disposition (row finalized, winner batch published)
        holds when driven through the actual production dispatch boundary,
        not a bypass.
    """
    store = _DurableRows()
    adapter = _FakeStateStoreAdapter(store)

    # leg-2: commit-then-crash (no bus — see the tests above).
    leg2_handler = _FanOutHandler(_fanout_batch(note="leg2"))
    leg2_cb = _stateful_callback(leg2_handler, adapter)

    # leg-3: registered AS A DISPATCHER on a real engine.
    bus = _RecordingBus()
    leg3_handler = _FanOutHandler(_fanout_batch(note="leg3"), next_state="RE_ROUTED")
    leg3_cb = _stateful_callback(leg3_handler, adapter, event_bus=bus)

    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id="state-io-seam-dispatcher",
        dispatcher=leg3_cb,
        category=EnumMessageCategory.COMMAND,
    )
    engine.register_route(
        ModelDispatchRoute(
            route_id="state-io-seam-route",
            topic_pattern=TOPIC_INBOUND,
            message_category=EnumMessageCategory.COMMAND,
            dispatcher_id="state-io-seam-dispatcher",
        )
    )
    engine.freeze()

    async def _run() -> EnumDispatchStatus:
        await leg2_cb(_input_envelope())  # winner commits, then "crashes"
        assert store.rows[str(CID)]["in_flight"] is True, (
            "precondition: leg-2's commit landed, un-finalized"
        )
        result = await engine.dispatch(
            TOPIC_INBOUND, _input_envelope(envelope_id=LEG3_ENVELOPE_ID)
        )
        return result.status

    with patch(f"{_RETRY_MODULE}.asyncio.sleep", _fast_sleep):
        status = asyncio.run(_run())

    assert status == EnumDispatchStatus.SUCCESS, (
        "REGRESSION (OMN-14600 boundary check): MessageDispatchEngine.dispatch() "
        f"reported {status!r} instead of SUCCESS. A dispatcher exception here "
        "is caught by the engine's per-dispatcher invocation loop and folded "
        "into HANDLER_ERROR rather than re-raised — this is the exact "
        "boundary a redelivery-dependent fix would silently fail at while "
        "every test bypassing the engine kept passing."
    )
    row = store.rows[str(CID)]
    assert row["in_flight"] is False, (
        "the real terminal disposition (leg-2's batch finalized) must hold "
        "when driven through the actual engine.dispatch() boundary"
    )
    assert not row["pending_emissions"]
    assert bus.topics_for(CID) == [TOPIC_ROUTING, TOPIC_INFERENCE, TOPIC_QUALITY], (
        "leg-2's inline-recovered batch must reach the bus via the real dispatch path"
    )
