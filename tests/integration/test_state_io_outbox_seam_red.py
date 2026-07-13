# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RED-first seam suite for P3b — the state_io in-row outbox (OMN-14403 §11).

**This file is authored by a NON-IMPLEMENTER and lands BEFORE P3b.** Every test
below is expected to FAIL against today's code. That ordering is the point: it
is the structural fix for self-certifying evidence (a suite written after the
implementation only ever proves the implementation agrees with itself).

**P3b LANDED (OMN-14493, implementer note).** Tests 1-4 now PASS (their xfail
markers were removed — a strict-xfail that starts passing is a CI failure).
Test 5 remains an HONEST ``xfail(strict=True)``: it drives the applier directly
and asserts the causation-scoped dedupe id, which is authored by the SEPARATE
P3a lane (OMN-14403 §8.1) — P3b must MATCH that seam, not double-author it; it
greens on the P3a rebase (see the test's reason string + the PR body). Two
harness changes, ASSERTIONS UNTOUCHED: (1) ``_stateful_callback`` gained an
optional ``event_bus`` threaded into the two recovery/resume SETUPS (the wrapper,
not the applier, owns the row and the publish-from-row); (2) ``_bus_callback``
passes ``propagate_publish_failures=True`` (the flag the state_io wiring sets for
an outbox contract). Two non-xfail guards were added at the end:
``test_has_bus_wrapper_publishes_once_and_finalizes_no_double_publish`` (no
double-publish + finalize-within-leg) and
``test_non_outbox_boundary_still_swallows_unchanged`` (the boundary no-swallow is
scoped to the outbox path; non-outbox contracts are unchanged). The positive
control is UNCHANGED — its ``in_flight stays TRUE`` models the no-bus commit path.

Spec: ``docs/plans/2026-07-12-multi-event-publish-seam-spec.md`` rev 3.2 §4/§8/§11.

The proof standard (non-negotiable)
-----------------------------------
Each RED is red against the **EXISTS-but-WRONG** state, never against mere
absence. A test that fails only because a symbol is missing is VACUOUS — it
proves "the feature isn't built yet", which we already know. So every test here
drives code that **exists today, executes, and produces the wrong result**:

Test 1 — commit-then-publish. ``_load_handle_persist`` (handler_wiring.py)
commits FSM state and returns; the publish happens OUTSIDE the retry unit, back
in ``_make_event_bus_callback``. A crash in that window loses the batch forever,
and boot recovery then blind-FAILs the row (destroying the intent) or TTL-skips
it entirely.

Test 2 — no in_flight-lock, no resume branch. A redelivered input RE-RUNS the
handler; the FSM's fold-on-redelivery guard folds it without re-emitting, so the
half-published batch is unrecoverable.

Test 3 — the auto-wired consume boundary (handler_wiring.py :2021,
``except Exception: logger.error(...)``) log-and-DISCARDS every exception, so a
publish failure and a conflict-exhaustion are both ACKed with no DLQ and no
redelivery.

Test 4 — ``cas_update`` (state_store_adapter.py :241-280) is unconditional on
``in_flight``, so a concurrent leg CLOBBERS the winner's un-published intent.

Test 5 — the deterministic id is ``uuid5(correlation_id, "class:idx")``
(service_dispatch_result_applier.py :661-664), so two legs of one correlation
COLLIDE and an id-deduping consumer silently drops the second.

`test_positive_control_*` is the guard against a FALSE red: it proves this
harness can observe a real multi-event publish through the real applier TODAY.
If it ever goes red, the harness is broken — not the feature.

Marking
-------
``xfail(strict=True, raises=AssertionError)``:

* ``strict`` — when P3b lands and a test starts PASSING, CI goes red until the
  implementer deletes the marker. The RED is self-retiring; it cannot be
  silently left behind.
* ``raises=AssertionError`` — closes the vacuity hole. An xfail that "passes"
  because the code blew up with ``AttributeError``/``ImportError`` would be
  indistinguishable from one that failed on the real defect. Constraining the
  expected exception to ``AssertionError`` means a *structural* break ERRORS
  loudly instead of hiding inside a green xfail.

Fidelity notes
--------------
* The asyncpg layer is an in-memory fake with the real SQL semantics (the
  established pattern — see ``test_state_io_concurrent_dispatch.py``). It
  faithfully reproduces the WRONG behaviors too (``recover_stale_rows``
  blind-FAILs past TTL and emits nothing), so the RED comes from the wiring
  under test, not from a crippled double. The positive control proves the fake
  does not suppress publishes.
* The "fresh process" boundary is enforced by **serializing the store to JSON
  bytes** and rehydrating a brand-new store from those bytes. Nothing but bytes
  crosses. If the row schema cannot source a value, recovery genuinely cannot
  obtain it from anywhere — which is exactly the tenant trap test 1 pins.
* Handlers return ``ModelHandlerOutput`` (already coerced correctly today at
  ``_normalize_handler_result`` :947-955), NOT a bare ``Sequence[BaseModel]``.
  That deliberately DECOUPLES this suite from P3a's normalize fix, so every RED
  here is attributable to a single named P3b defect and nothing else.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID, uuid5

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_event_bus_callback,
    _make_stateful_dispatch_callback,
)
from omnibase_infra.runtime.service_dispatch_result_applier import DispatchResultApplier
from omnibase_infra.runtime.state_io.state_store_adapter import (
    CONTEXTVAR_STATE_IO_ROWS,
)

# --------------------------------------------------------------------------
# Fixture identities. These are FIXED, not random: the whole suite turns on
# which id the outbox keys its batch and its dedupe id on.
# --------------------------------------------------------------------------

CID = UUID("11111111-1111-1111-1111-111111111111")
"""The correlation_id — one multi-leg FSM workflow."""

INPUT_ENVELOPE_ID = UUID("22222222-2222-2222-2222-222222222222")
"""Leg-2's input envelope_id. THIS is the batch's causation (spec §5/§8.1)."""

GRANDPARENT_CAUSATION_ID = UUID("33333333-3333-3333-3333-333333333333")
"""The input's OWN causation_id — i.e. the batch's GRANDPARENT.

This is deliberately DIFFERENT from ``INPUT_ENVELOPE_ID`` and that difference is
load-bearing: it is the only thing that makes the E1 resume-predicate bug
observable. See ``test_e1_guard_causation_and_envelope_ids_must_differ``.
"""

LEG3_ENVELOPE_ID = UUID("44444444-4444-4444-4444-444444444444")
"""A DIFFERENT input on the same correlation (the re-route leg)."""

TENANT = "acme-tenant"

TOPIC_ROUTING = (
    "onex.cmd.test-seam.routing-request.v1"  # onex-topic-allow: test fixture
)
TOPIC_INFERENCE = (
    "onex.cmd.test-seam.inference-request.v1"  # onex-topic-allow: test fixture
)
TOPIC_QUALITY = (
    "onex.cmd.test-seam.quality-gate-request.v1"  # onex-topic-allow: test fixture
)
TOPIC_TERMINAL = "onex.evt.test-seam.terminal.v1"  # onex-topic-allow: test fixture
TOPIC_INBOUND = (
    "onex.cmd.test-seam.workflow-request.v1"  # onex-topic-allow: test fixture
)

OUTPUT_TOPIC_MAP = {
    "SeamRoutingIntent": TOPIC_ROUTING,
    "SeamInferenceIntent": TOPIC_INFERENCE,
    "SeamQualityGateIntent": TOPIC_QUALITY,
}

STATE_IO = {
    "database": "omnibase_infra",
    "table": "delegation_workflow_state",
    "key": "correlation_id",
    "codec": {"module": "tests.integration", "name": "_SeamCodec"},
}

_PATCH_IMPORT = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)
_PATCH_ADAPTER = "omnibase_infra.runtime.auto_wiring.handler_wiring.StateStoreAdapter"

XFAIL_KW: dict[str, Any] = {"strict": True, "raises": AssertionError}


# --------------------------------------------------------------------------
# Fan-out event models. tenant_id / causation_id default to None so that
# "the applier never stamped them" is observable as None rather than as a
# construction error.
# --------------------------------------------------------------------------


class _SeamEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correlation_id: UUID | None = None
    tenant_id: str | None = None
    causation_id: UUID | None = None
    note: str = ""


class ModelSeamRoutingIntent(_SeamEvent): ...


class ModelSeamInferenceIntent(_SeamEvent): ...


class ModelSeamQualityGateIntent(_SeamEvent): ...


def _fanout_batch(note: str = "leg2") -> tuple[BaseModel, ...]:
    """The N=3 fan-out a def-B orchestrator leg emits from one input."""
    return (
        ModelSeamRoutingIntent(note=note),
        ModelSeamInferenceIntent(note=note),
        ModelSeamQualityGateIntent(note=note),
    )


# --------------------------------------------------------------------------
# The "database". The ONLY thing that survives a process crash.
# --------------------------------------------------------------------------


class _DurableRows:
    """The delegation_workflow_state table, as JSON-serializable rows.

    Carries the P3b target columns (``pending_emissions``, ``publish_attempts``,
    migration 090 / spec §12) so the suite pins the field-match contract. Today's
    wiring simply never writes them — which is the defect, not a gap in the fake.
    """

    def __init__(self, rows: dict[str, dict[str, Any]] | None = None) -> None:
        self.rows: dict[str, dict[str, Any]] = rows or {}

    def crash_to_bytes(self) -> bytes:
        """Serialize the DB. This is the process boundary — nothing else crosses."""
        return json.dumps(self.rows).encode("utf-8")

    @classmethod
    def reboot_from_bytes(cls, blob: bytes) -> _DurableRows:
        """Rehydrate in a FRESH process. No object identity survives."""
        return cls(json.loads(blob.decode("utf-8")))


class _FakeStateStoreAdapter:
    """In-memory StateStoreAdapter with the real SQL semantics.

    Reproduces the production adapter faithfully — INCLUDING the behavior P3b
    must replace: ``recover_stale_rows`` marks a stale in-flight row FAILED and
    emits nothing (it has no bus access). See state_store_adapter.py :282-340.
    """

    def __init__(self, store: _DurableRows) -> None:
        self.store = store

    async def load(self, cid: str) -> tuple[str, int] | None:
        row = self.store.rows.get(cid)
        if row is None:
            return None
        return cast("str", row["payload_json"]), cast("int", row["version"])

    async def seed(
        self,
        cid: str,
        *,
        tenant_id: str,
        state: str,
        in_flight: bool,
        payload_json: str,
        # --- P3b target seam (spec §12). Defaulted so TODAY's wiring, which
        # --- does not pass them, still works — and the row is left with no
        # --- intent, which is precisely the defect under test.
        pending_emissions: list[dict[str, Any]] | None = None,
        publish_attempts: int = 0,
    ) -> bool:
        if cid in self.store.rows:
            return False
        self.store.rows[cid] = {
            "correlation_id": cid,
            "tenant_id": tenant_id,
            "state": state,
            "in_flight": in_flight,
            "payload_json": payload_json,
            "version": 0,
            "pending_emissions": pending_emissions,
            "publish_attempts": publish_attempts,
            "updated_at": time.time(),
        }
        return True

    async def cas_update(
        self,
        cid: str,
        *,
        tenant_id: str,
        state: str,
        in_flight: bool,
        payload_json: str,
        expected_version: int,
        pending_emissions: list[dict[str, Any]] | None = None,
        publish_attempts: int | None = None,
    ) -> int:
        row = self.store.rows.get(cid)
        if row is None or row["version"] != expected_version:
            return 0
        row["tenant_id"] = tenant_id
        row["state"] = state
        row["in_flight"] = in_flight
        row["payload_json"] = payload_json
        row["version"] = expected_version + 1
        row["updated_at"] = time.time()
        # NOTE: unconditional overwrite — this mirrors the real UPDATE, which has
        # no in_flight guard. It is exactly how a concurrent leg clobbers an
        # un-published batch (test 4 / spec §4.1 D1).
        row["pending_emissions"] = pending_emissions
        if publish_attempts is not None:
            row["publish_attempts"] = publish_attempts
        return 1

    async def recover_stale_rows(self, ttl_seconds: int | None = None) -> int:
        """FAITHFUL reproduction of the real (wrong-for-P3b) sweep.

        Blind-FAILs stale in-flight rows and publishes NOTHING. Also TTL-gated,
        so a row that crashed seconds ago is not even looked at (spec E3).
        """
        ttl = 900 if ttl_seconds is None else ttl_seconds
        now = time.time()
        recovered = 0
        for row in self.store.rows.values():
            if (
                row["state"] not in ("COMPLETED", "FAILED")
                and row["in_flight"]
                and row["updated_at"] < now - ttl
            ):
                row["state"] = "FAILED"
                row["in_flight"] = False
                row["version"] += 1
                recovered += 1
        return recovered

    async def select_recoverable_batches(self) -> list[dict[str, Any]]:
        """The P3b target seam (spec §4.1 D2/E3): return in-flight rows carrying a
        non-empty ``pending_emissions``, REGARDLESS of TTL, so the wrapper (which
        owns the bus) can re-publish them. Today's wiring never calls this."""
        return [
            dict(row)
            for row in self.store.rows.values()
            if row["in_flight"] and row.get("pending_emissions")
        ]


class _SeamCodec:
    """state_io codec: the post-handle bridge (OMN-14208 pair-verify M1)."""

    def flush(self, cid: str) -> str | None:
        current = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        entry = current.get(cid)
        return entry[0] if entry is not None else None


class _RecordingBus:
    """Records every publish. Optionally fails on a chosen event index."""

    def __init__(self, fail_at_index: int | None = None) -> None:
        self.published: list[tuple[str, ModelEventEnvelope[Any]]] = []
        self.fail_at_index = fail_at_index

    async def publish_envelope(
        self,
        envelope: ModelEventEnvelope[Any],
        topic: str,
        key: bytes | None = None,
    ) -> None:
        if self.fail_at_index is not None and len(self.published) == self.fail_at_index:
            raise RuntimeError(
                f"broker rejected publish at index {self.fail_at_index} (injected)"
            )
        self.published.append((topic, envelope))

    def envelopes_for(self, correlation_id: UUID) -> list[ModelEventEnvelope[Any]]:
        return [e for _, e in self.published if e.correlation_id == correlation_id]

    def topics_for(self, correlation_id: UUID) -> list[str]:
        return [t for t, e in self.published if e.correlation_id == correlation_id]


class _FanOutHandler:
    """A def-B fan-out orchestrator leg. Counts its own invocations.

    Mirrors HandlerDelegationWorkflow's synchronous in-flight dedup guard: if the
    loaded row is already in_flight, it FOLDS (emits nothing). That guard is what
    makes a redelivery lose the batch today (spec §4.0 F3) — and it is why test 2
    must assert the handler is NOT re-run at all.
    """

    def __init__(
        self,
        events: tuple[BaseModel, ...],
        *,
        next_state: str = "IN_PROGRESS",
    ) -> None:
        self.events = events
        self.next_state = next_state
        self.call_count = 0
        self.folded_count = 0

    async def handle(self, envelope: object) -> ModelHandlerOutput[None] | None:
        self.call_count += 1
        current = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        cid = next(iter(current))
        payload_json, version = current[cid]
        state = json.loads(payload_json) if payload_json else {}

        if state.get("in_flight"):
            self.folded_count += 1
            return None

        new_state = {
            "tenant_id": TENANT,
            "state": self.next_state,
            "in_flight": True,
        }
        CONTEXTVAR_STATE_IO_ROWS.set({cid: (json.dumps(new_state), version)})

        envelope_id = getattr(envelope, "envelope_id", INPUT_ENVELOPE_ID)
        return ModelHandlerOutput.for_orchestrator(
            input_envelope_id=envelope_id,
            correlation_id=CID,
            handler_id="seam-fanout",
            events=self.events,
        )


# --------------------------------------------------------------------------
# Harness wiring — the REAL callback, the REAL applier.
# --------------------------------------------------------------------------


def _input_envelope(
    envelope_id: UUID = INPUT_ENVELOPE_ID,
) -> ModelEventEnvelope[object]:
    """The leg's input. Its causation_id (the grandparent) rides the payload —
    ModelEventEnvelope has no causation_id field today (verified 2026-07-12)."""
    return ModelEventEnvelope[object](
        envelope_id=envelope_id,
        correlation_id=CID,
        payload={
            "correlation_id": str(CID),
            "tenant_id": TENANT,
            "causation_id": str(GRANDPARENT_CAUSATION_ID),
        },
    )


def _make_applier(bus: _RecordingBus) -> DispatchResultApplier:
    return DispatchResultApplier(
        event_bus=cast("Any", bus),
        output_topic=TOPIC_TERMINAL,
        output_topic_map=dict(OUTPUT_TOPIC_MAP),
        allowed_output_topics=list(OUTPUT_TOPIC_MAP.values()) + [TOPIC_TERMINAL],
    )


def _stateful_callback(
    handler: _FanOutHandler,
    adapter: _FakeStateStoreAdapter,
    *,
    event_bus: _RecordingBus | None = None,
) -> Any:
    """Build the REAL stateful dispatch callback over the fake adapter.

    P3b (OMN-14493): ``event_bus`` threads a bus into the stateful wrapper so it
    can publish-from-row for the recovery (boot-sweep) and resume (redelivery)
    paths — the wrapper, not the applier, owns the row and therefore the
    re-publish (spec §4.1 D2). No-bus builds keep the legacy commit-then-return
    shape (external applier publishes), matching the positive control. Only the
    two recovery/resume test SETUPS pass a bus; every assertion is untouched.
    """
    with (
        patch.dict(
            "os.environ",
            {"OMNIBASE_INFRA_DB_URL": "postgresql://user:pass@host:5432/db"},
        ),
        patch(_PATCH_IMPORT, return_value=_SeamCodec),
        patch(_PATCH_ADAPTER, return_value=adapter),
    ):
        return _make_stateful_dispatch_callback(
            cast("Any", handler),
            None,
            dict(STATE_IO),
            event_bus=event_bus,
            output_topic_map=(
                dict(OUTPUT_TOPIC_MAP) if event_bus is not None else None
            ),
        )


# --- outcome-level readers: assert the OUTCOME (tenant/causation carried), not
# --- the carrier P3b happens to choose. Today both are simply absent.


def _tenant_of(env: ModelEventEnvelope[Any]) -> str | None:
    direct = getattr(env, "tenant_id", None)
    if direct is not None:
        return cast("str", direct)
    return getattr(env.payload, "tenant_id", None)


def _causation_of(env: ModelEventEnvelope[Any]) -> UUID | None:
    direct = getattr(env, "causation_id", None)
    if direct is not None:
        return cast("UUID", direct)
    return getattr(env.payload, "causation_id", None)


def _expected_causation_scoped_id(causation: UUID, class_name: str, idx: int) -> UUID:
    """Spec §8.1: uuid5(correlation_id, "{causation}:{class}:{idx}")."""
    return uuid5(CID, f"{causation}:{class_name}:{idx}")


def _todays_correlation_only_id(class_name: str, idx: int) -> UUID:
    """What applier.py :661-664 computes TODAY — no causation scope."""
    return uuid5(CID, f"{class_name}:{idx}")


# ==========================================================================
# POSITIVE CONTROL — must be GREEN today. Guards against a FALSE red.
# ==========================================================================


@pytest.mark.integration
def test_positive_control_harness_observes_a_real_multi_event_publish() -> None:
    """The harness CAN see a 3-event fan-out published through the real applier.

    This is the anti-false-RED guard. Every other test in this file asserts that
    some batch does NOT reach the bus. If the harness were simply broken (bad
    topic map, dead bus, mis-wired applier), those tests would be red for a
    reason that has nothing to do with P3b — the worst possible outcome, a test
    that is right by accident. This proves the publish path works TODAY.

    It also pins the baseline the RED tests are measured against: on the happy
    path (no crash, no contention) the applier already loops and publishes N.
    """
    store = _DurableRows()
    adapter = _FakeStateStoreAdapter(store)
    bus = _RecordingBus()
    handler = _FanOutHandler(_fanout_batch())
    callback = _stateful_callback(handler, adapter)
    applier = _make_applier(bus)

    async def _run() -> None:
        result = await callback(_input_envelope())
        await applier.apply(result, CID)

    asyncio.run(_run())

    assert len(bus.envelopes_for(CID)) == 3, (
        "HARNESS BROKEN: the real applier did not publish the 3-event fan-out on "
        f"the happy path. Every RED in this file is untrustworthy until this "
        f"passes. Got: {bus.published!r}"
    )
    assert bus.topics_for(CID) == [TOPIC_ROUTING, TOPIC_INFERENCE, TOPIC_QUALITY], (
        "publish order / topic resolution regressed (spec §3)"
    )
    assert store.rows[str(CID)]["in_flight"] is True


@pytest.mark.integration
def test_e1_guard_causation_and_envelope_ids_must_differ() -> None:
    """META-GUARD for the E1 deadlock test. Do not delete.

    The §11.11 resume test can ONLY catch the E1 bug (a resume predicate keyed on
    ``causation_id`` instead of ``envelope_id``) if the two values actually differ
    on the input. If a future maintainer "simplifies" the fixture so the input's
    causation_id equals its envelope_id, BOTH the correct and the buggy predicate
    would match, the resume test would pass either way, and E1's liveness deadlock
    would ship GREEN.

    This test exists so that collapsing them is a loud failure, not a silent one.
    """
    env = _input_envelope()
    causation = cast("dict[str, Any]", env.payload)["causation_id"]
    assert UUID(causation) != env.envelope_id, (
        "The input's causation_id (its GRANDPARENT) must differ from its own "
        "envelope_id, or the E1 resume-predicate bug becomes unobservable."
    )


# ==========================================================================
# TEST 1 — CENTERPIECE: fresh-process, row-only crash recovery.
# ==========================================================================


@pytest.mark.integration
def test_crash_after_commit_republishes_byte_identical_batch_from_row_alone() -> None:
    """A crash between state-commit and publish must lose NOTHING.

    The trap this pins: **tenant is never stamped by the applier today**
    (service_dispatch_result_applier.py :665-670 builds the envelope from
    envelope_id/payload/correlation_id/timestamp and nothing else). In a FRESH
    process the input envelope is GONE, so the ONLY possible source of the tenant
    is the persisted row. If the row schema cannot carry it, recovery does not
    fail loudly — it SILENTLY DIVERGES, re-publishing a tenant-less batch that
    looks fine and lands in the wrong (or no) tenant scope.

    The process boundary is enforced with bytes: the store is serialized to JSON
    and rehydrated. No object identity survives. There is nowhere to cheat from.
    """
    # ---- process 1: dispatch, commit, then DIE before publishing.
    store1 = _DurableRows()
    adapter1 = _FakeStateStoreAdapter(store1)
    bus1 = _RecordingBus()
    handler1 = _FanOutHandler(_fanout_batch())
    callback1 = _stateful_callback(handler1, adapter1)

    asyncio.run(callback1(_input_envelope()))
    # <<< CRASH. applier.apply() is never reached. This IS the F3 window: the
    # state commit already landed; the publish never happened.
    assert bus1.published == [], "precondition: nothing was published pre-crash"
    assert store1.rows[str(CID)]["in_flight"] is True, "precondition: row is in-flight"

    crashed_db = store1.crash_to_bytes()

    # ---- process 2: a brand-new process. Only bytes crossed.
    store2 = _DurableRows.reboot_from_bytes(crashed_db)
    adapter2 = _FakeStateStoreAdapter(store2)
    bus2 = _RecordingBus()
    # If recovery re-runs the handler it is not row-only recovery. Count it.
    handler2 = _FanOutHandler(_fanout_batch())
    # P3b: the wrapper owns the bus for the boot-sweep re-publish (spec §4.1 D2).
    callback2 = _stateful_callback(handler2, adapter2, event_bus=bus2)

    async def _boot() -> None:
        # Boot recovery fires on first dispatch (_ensure_stale_rows_recovered).
        # An unrelated correlation arrives; the crashed batch must be recovered.
        other = ModelEventEnvelope[object](
            correlation_id=UUID("99999999-9999-9999-9999-999999999999"),
            payload={
                "correlation_id": "99999999-9999-9999-9999-999999999999",
                "tenant_id": TENANT,
            },
        )
        await callback2(other)

    asyncio.run(_boot())

    recovered = bus2.envelopes_for(CID)

    assert len(recovered) == 3, (
        "P3b: crash-recovery must re-publish all N intended emissions FROM THE "
        "PERSISTED ROW. Today the batch was never persisted (no pending_emissions "
        "write) and recover_stale_rows blind-FAILs / TTL-skips the row, emitting "
        f"nothing — the events are lost forever. Got {len(recovered)} envelope(s)."
    )

    # The recovered batch must be IDENTICAL to what the live path would have
    # published — same ids (so an id-deduping consumer collapses a duplicate),
    # same order, same tenant.
    expected_ids = [
        _expected_causation_scoped_id(INPUT_ENVELOPE_ID, cls, idx)
        for idx, cls in enumerate(
            [
                "ModelSeamRoutingIntent",
                "ModelSeamInferenceIntent",
                "ModelSeamQualityGateIntent",
            ]
        )
    ]
    assert [e.envelope_id for e in recovered] == expected_ids, (
        "recovered envelope_ids must be the causation-scoped deterministic ids "
        "rebuilt FROM THE ROW (spec §8.1), so a recovery re-publish collapses "
        "against the original at the consume-path dedupe."
    )
    assert bus2.topics_for(CID) == [TOPIC_ROUTING, TOPIC_INFERENCE, TOPIC_QUALITY], (
        "recovery must preserve emission order + per-class topic routing (spec §3)"
    )

    # THE TENANT TRAP.
    tenants = [_tenant_of(e) for e in recovered]
    assert tenants == [TENANT, TENANT, TENANT], (
        "SILENT DIVERGENCE: every recovered envelope must carry the tenant, and "
        "in a fresh process the ONLY possible source is the persisted row. The "
        "applier does not stamp tenant today (:665-670), so recovery re-publishes "
        f"a tenant-less batch that looks healthy. Got: {tenants!r}"
    )
    causations = [_causation_of(e) for e in recovered]
    assert causations == [INPUT_ENVELOPE_ID] * 3, (
        f"every recovered envelope must carry causation_id (spec §5). Got: {causations!r}"
    )

    # Row-only means row-only: the handler must NOT be re-run to rebuild the batch.
    assert handler2.call_count <= 1, (
        "recovery re-ran the fan-out handler to rebuild the batch. The committed "
        "ROW is the publish source, never a re-run (spec §4.1) — a re-run against "
        "advanced state emits a DIFFERENT batch under the SAME ids."
    )

    row = store2.rows[str(CID)]
    assert row["state"] != "FAILED", (
        "recover_stale_rows blind-FAILed the row and DESTROYED its intent "
        "(state_store_adapter.py :311-320). Under P3b a row carrying "
        "pending_emissions must be recovered-and-re-published, never given up on."
    )
    assert row["in_flight"] is False, "recovery must CAS-finalize the row (spec §4.1)"
    assert not row["pending_emissions"], "finalize must clear pending_emissions"


# ==========================================================================
# TEST 2 — §11.11 / spec item 12: resume-on-redelivery. GUARDS E1.
# ==========================================================================


@pytest.mark.integration
def test_redelivery_of_same_input_resumes_publish_from_row_without_rerunning_handler() -> (
    None
):
    """§11.11 — the test without which the E1 deadlock ships GREEN.

    Fable found E1 by reading the spec: the resume predicate was written as
    ``incoming.causation_id == batch.causation_envelope_id``. But a redelivered
    input's ``causation_id`` is its GRANDPARENT — the batch's causation is the
    input's own ``envelope_id``. So a causation_id-keyed predicate NEVER matches
    on redelivery, every crash-redelivery falls through to conflict-retry, nobody
    ever finalizes the dead batch, and the workflow DEADLOCKS on exactly the crash
    the rule was written to survive.

    That bug is invisible to every other test in this file:
      * test 1 calls the boot sweep directly — no redelivery involved.
      * test 4's leg-3 is a DIFFERENT input, which is *supposed* to conflict-retry.

    Only a redelivery of the SAME input distinguishes the two predicates, and only
    when causation_id != envelope_id on that input (see the E1 meta-guard above).
    """
    # ---- process 1: commit the batch, publish event[0], then crash mid-batch.
    store1 = _DurableRows()
    adapter1 = _FakeStateStoreAdapter(store1)
    bus1 = _RecordingBus()
    handler1 = _FanOutHandler(_fanout_batch())
    callback1 = _stateful_callback(handler1, adapter1)
    applier1 = _make_applier(bus1)

    async def _partial() -> None:
        result = await callback1(_input_envelope())
        # Crash after the FIRST publish: the batch is now half-emitted.
        applier1._event_bus = cast("Any", _RecordingBus(fail_at_index=1))
        with pytest.raises(RuntimeError):
            await applier1.apply(result, CID)

    asyncio.run(_partial())
    assert store1.rows[str(CID)]["in_flight"] is True, "precondition: row in-flight"

    crashed_db = store1.crash_to_bytes()

    # ---- process 2: Kafka redelivers THE SAME input (same envelope_id).
    store2 = _DurableRows.reboot_from_bytes(crashed_db)
    adapter2 = _FakeStateStoreAdapter(store2)
    bus2 = _RecordingBus()
    handler2 = _FanOutHandler(_fanout_batch())
    # P3b: the wrapper owns the bus for the redelivery resume-from-row (spec §4.1 E2).
    callback2 = _stateful_callback(handler2, adapter2, event_bus=bus2)
    applier2 = _make_applier(bus2)

    async def _redeliver() -> None:
        result = await callback2(_input_envelope(envelope_id=INPUT_ENVELOPE_ID))
        await applier2.apply(result, CID)

    asyncio.run(_redeliver())

    # (ii) the handler must NOT be re-run — the row is the source, not a re-run.
    assert handler2.folded_count == 0, (
        "the redelivered input RE-RAN the handler and the in-flight dedup guard "
        "FOLDED it (emitting nothing). That is the live silent-loss path: the "
        "half-published batch is now unrecoverable. The in_flight-lock rule must "
        "intercept BEFORE the handler runs and resume from the row instead."
    )

    # (iii) the batch re-publishes from the row with the SAME causation-scoped ids.
    resumed = bus2.envelopes_for(CID)
    assert len(resumed) == 3, (
        "resume must re-publish the batch FROM THE ROW (idempotent replay, same "
        f"ids). Got {len(resumed)} envelope(s) — the batch was dropped."
    )
    expected_ids = [
        _expected_causation_scoped_id(INPUT_ENVELOPE_ID, cls, idx)
        for idx, cls in enumerate(
            [
                "ModelSeamRoutingIntent",
                "ModelSeamInferenceIntent",
                "ModelSeamQualityGateIntent",
            ]
        )
    ]
    assert [e.envelope_id for e in resumed] == expected_ids, (
        "the resumed batch must carry the SAME ids as the original attempt so an "
        "id-deduping consumer materializes each effect exactly once (spec §8.2)."
    )

    # (i)+(iv) resume fired AND finalized. THIS is the E1 assertion: a predicate
    # keyed on causation_id never matches here, so the row would stay in_flight
    # forever — a liveness deadlock — and this assertion is what catches it.
    row = store2.rows[str(CID)]
    assert row["in_flight"] is False, (
        "LIVENESS DEADLOCK: the row is still in_flight after the redelivery. The "
        "resume branch never fired, so the batch was never finalized and every "
        "subsequent leg will conflict-retry against a row nobody will ever clear. "
        "This is E1: the resume predicate must key on incoming.envelope_id == "
        "row.causation_envelope_id, NOT on incoming.causation_id (which is the "
        "input's GRANDPARENT and can never match)."
    )
    assert not row["pending_emissions"], "finalize must clear pending_emissions"


# ==========================================================================
# TEST 3 — publish-exception + conflict-exhaustion must PROPAGATE / DLQ (E4).
# ==========================================================================


def _bus_callback(callback: Any, applier: DispatchResultApplier) -> Any:
    """Drive the REAL auto-wired consume boundary on the state_io / outbox path.

    ``propagate_publish_failures=True`` is what the state_io wiring passes for an
    outbox contract (OMN-14403 §4.3): a publish-from-row failure + a conflict-
    retry exhaustion propagate (redeliver) instead of being log-and-discarded.
    Non-outbox contracts never set it, so their swallow behavior is unchanged
    (see ``test_non_outbox_boundary_still_swallows_unchanged``).
    """

    class _Engine:
        async def dispatch(
            self, topic: str, envelope: ModelEventEnvelope[object]
        ) -> Any:
            return await callback(envelope)

    return _make_event_bus_callback(
        TOPIC_INBOUND,
        cast("Any", _Engine()),
        cast("Any", applier),
        propagate_publish_failures=True,
    )


@pytest.mark.integration
def test_publish_exception_mid_batch_propagates_and_is_never_silently_acked() -> None:
    """A broker rejection at event k<N must be LOUD, not an ack.

    Today this is the quietest possible data loss: events 0..k-1 are on the bus,
    events k..N-1 are not, the exception is logged at ERROR, the offset commits,
    and nothing ever retries. "Events lost, nothing surfaced."
    """
    store = _DurableRows()
    adapter = _FakeStateStoreAdapter(store)
    bus = _RecordingBus(fail_at_index=1)  # event 0 lands, event 1 is rejected
    handler = _FanOutHandler(_fanout_batch())
    callback = _stateful_callback(handler, adapter)
    applier = _make_applier(bus)
    on_message = _bus_callback(callback, applier)

    raised: BaseException | None = None

    async def _run() -> None:
        nonlocal raised
        try:
            await on_message(_input_envelope())
        except BaseException as exc:  # noqa: BLE001 — we are asserting on propagation
            raised = exc

    asyncio.run(_run())

    assert len(bus.published) == 1, "precondition: the batch really did fail mid-way"
    assert raised is not None, (
        "SWALLOWED. The publish failure at event index 1 was caught by the "
        "log-and-discard boundary at handler_wiring.py :2021, so the message is "
        "ACKed with 2 of 3 events never emitted and NO redelivery. The failure "
        "must propagate (no offset commit) or route to a DLQ."
    )


@pytest.mark.integration
def test_conflict_retry_exhaustion_propagates_and_is_never_silently_acked() -> None:
    """CAS contention that never resolves must surface, not vanish."""
    store = _DurableRows()
    # Seed a row so the wiring takes the cas_update path, then make every CAS lose.
    store.rows[str(CID)] = {
        "correlation_id": str(CID),
        "tenant_id": TENANT,
        "state": "IN_PROGRESS",
        "in_flight": False,
        "payload_json": json.dumps(
            {"tenant_id": TENANT, "state": "IN_PROGRESS", "in_flight": False}
        ),
        "version": 0,
        "pending_emissions": None,
        "publish_attempts": 0,
        "updated_at": time.time(),
    }
    adapter = _FakeStateStoreAdapter(store)

    async def _always_lose(*_args: Any, **_kwargs: Any) -> int:
        return 0  # rowcount 0 = a concurrent writer always won

    adapter.cas_update = _always_lose  # type: ignore[method-assign]

    bus = _RecordingBus()
    handler = _FanOutHandler(_fanout_batch())
    callback = _stateful_callback(handler, adapter)
    applier = _make_applier(bus)
    on_message = _bus_callback(callback, applier)

    raised: BaseException | None = None

    async def _run() -> None:
        nonlocal raised
        try:
            await on_message(_input_envelope())
        except BaseException as exc:  # noqa: BLE001 — we are asserting on propagation
            raised = exc

    asyncio.run(_run())

    assert raised is not None, (
        "SWALLOWED. retry_on_optimistic_conflict exhausted and raised — correctly "
        "— but the log-and-discard boundary at handler_wiring.py :2021 caught it. "
        "The leg is ACKed and lost with no DLQ and no redelivery."
    )


# ==========================================================================
# TEST 4 — in_flight is NOT a lock: a concurrent leg clobbers un-published intent.
# ==========================================================================


@pytest.mark.integration
def test_concurrent_leg_cannot_clobber_an_in_flight_batchs_unpublished_intent() -> None:
    """The single rule that makes "the committed row is the publish source" TRUE.

    Fable's finding (N2): ``in_flight`` is metadata, not a lock. Without the lock
    rule, this is a legal interleaving that loses events with no crash, no broker
    failure, and no error anywhere — the rev-1 flaw one level deeper.

    Asserted BEHAVIORALLY (every intended event eventually materializes exactly
    once), not by poking at a column — so the test pins the outcome P3b owes,
    not the shape of its implementation.
    """
    store = _DurableRows()
    adapter = _FakeStateStoreAdapter(store)
    bus = _RecordingBus()

    # leg-2 commits its 3-event batch and is then PAUSED before publishing.
    leg2_handler = _FanOutHandler(_fanout_batch(note="leg2"))
    leg2_cb = _stateful_callback(leg2_handler, adapter)

    # leg-3 is a DIFFERENT input on the SAME correlation (an FSM re-route).
    leg3_handler = _FanOutHandler(_fanout_batch(note="leg3"), next_state="RE_ROUTED")
    leg3_cb = _stateful_callback(leg3_handler, adapter)
    applier = _make_applier(bus)

    async def _run() -> None:
        # leg-2 wins CAS: row committed, in_flight=true, intent persisted.
        leg2_result = await leg2_cb(_input_envelope())

        # ...and is descheduled here, mid-publish. leg-3 arrives NOW.
        await leg3_cb(_input_envelope(envelope_id=LEG3_ENVELOPE_ID))

        # leg-2 finally gets to publish, from the row it committed.
        await applier.apply(leg2_result, CID)

    asyncio.run(_run())

    # (i) leg-3 must not have run the handler against the in-flight row.
    assert leg3_handler.call_count == 0, (
        "leg-3 ran its handler against a row that was in_flight with an "
        "un-published batch. The in_flight-lock rule must block the handler and "
        "force leg-3 to resume-or-conflict-retry instead."
    )

    # (iv) THE OUTCOME: nothing leg-2 intended may be lost.
    leg2_notes = [
        e.payload.note
        for e in bus.envelopes_for(CID)
        if getattr(e.payload, "note", None) == "leg2"
    ]
    assert len(leg2_notes) == 3, (
        "EVENT LOSS: leg-3's unconditional cas_update overwrote leg-2's "
        "pending_emissions before leg-2 could publish them, so leg-2's batch is "
        f"gone — no crash, no error, no trace. Published only {len(leg2_notes)}/3 "
        "of leg-2's events."
    )


# ==========================================================================
# TEST 5 — the dedupe id must be CAUSATION-scoped, not envelope-global.
# ==========================================================================


@pytest.mark.integration
@pytest.mark.xfail(
    reason=(
        "OMN-14493 P3b: HONEST cross-lane seam xfail — blocked on the P3a "
        "applier seam (OMN-14403 §8.1 causation-scoped id / §5 tenant-carry). "
        "This test drives the applier DIRECTLY (applier.apply(result, CID)); "
        "the causation-scoped deterministic id it asserts is authored by the "
        "P3a lane (DispatchResultApplier._deterministic_envelope_id), NOT by "
        "P3b's §4 in-row outbox. P3b must NOT double-author it (define-and-"
        "match-seams: match, do not re-implement). Greens on the P3a rebase — "
        "the causation must flow via ModelDispatchResult.causation_envelope_id "
        "so apply(result, CID) (2-arg) resolves it. See PR body + OMN-14403 "
        "comment for the exact bridge."
    ),
    **XFAIL_KW,
)
def test_same_class_emissions_from_different_legs_get_distinct_dedupe_ids() -> None:
    """Two legs of ONE correlation re-emit the same class. Both must materialize.

    This is the purest EXISTS-but-WRONG in the suite: the id computation runs, and
    returns a COLLIDING value. Nothing is missing; the answer is wrong.

    The consequence is a silent no-op — a quality-gate failure triggers a re-route
    that emits a second RoutingIntent, the consumer's dedupe sees an id it has
    already seen, and the re-route never happens. The workflow just stops.
    """
    bus = _RecordingBus()
    applier = _make_applier(bus)

    async def _run() -> None:
        # leg-1: emits a RoutingIntent (caused by INPUT_ENVELOPE_ID).
        leg1 = ModelHandlerOutput.for_orchestrator(
            input_envelope_id=INPUT_ENVELOPE_ID,
            correlation_id=CID,
            handler_id="seam-fanout",
            events=(ModelSeamRoutingIntent(note="leg1"),),
        )
        # leg-3: the quality gate failed -> RE-ROUTE. Same class, same index,
        # same correlation — but a DIFFERENT causation (LEG3_ENVELOPE_ID).
        leg3 = ModelHandlerOutput.for_orchestrator(
            input_envelope_id=LEG3_ENVELOPE_ID,
            correlation_id=CID,
            handler_id="seam-fanout",
            events=(ModelSeamRoutingIntent(note="leg3"),),
        )
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _normalize_handler_result,
        )

        for output in (leg1, leg3):
            result = _normalize_handler_result(output, _input_envelope(), None)
            await applier.apply(result, CID)

    asyncio.run(_run())

    published = bus.envelopes_for(CID)
    assert len(published) == 2, "precondition: both legs reached the applier"
    id_leg1, id_leg3 = published[0].envelope_id, published[1].envelope_id

    assert id_leg1 != id_leg3, (
        "ID COLLISION: both legs computed "
        f"{id_leg1} — uuid5(correlation_id, 'ModelSeamRoutingIntent:0') — because "
        "the id is scoped ONLY to correlation_id. An id-deduping consumer will "
        "drop leg-3's legitimate re-emission and the FSM silently stalls. The id "
        "must be causation-scoped (spec §8.1)."
    )
    assert id_leg1 == _expected_causation_scoped_id(
        INPUT_ENVELOPE_ID, "ModelSeamRoutingIntent", 0
    ), "leg-1's id must be uuid5(cid, '{causation}:{class}:{idx}') (spec §8.1)"
    assert id_leg3 == _expected_causation_scoped_id(
        LEG3_ENVELOPE_ID, "ModelSeamRoutingIntent", 0
    ), "leg-3's id must be causation-scoped to ITS OWN input envelope_id"

    # And the negative control: today's key is what produces the collision.
    assert id_leg1 != _todays_correlation_only_id("ModelSeamRoutingIntent", 0), (
        "the id is still keyed on (correlation_id, class, idx) — the colliding key"
    )


# ==========================================================================
# P3b GUARDS (OMN-14493) — added by the implementer per the build-lead decision.
# These are NOT xfail: they assert the two hard constraints on the §4 fix.
# ==========================================================================


@pytest.mark.integration
def test_has_bus_wrapper_publishes_once_and_finalizes_no_double_publish() -> None:
    """Decision 1 guard: the has-bus wrapper publishes-from-row EXACTLY once.

    When the state_io wrapper owns a bus (the production path), it publishes the
    committed batch FROM THE ROW and CAS-finalizes WITHIN the leg, then returns
    ``None`` — so the external applier at the consume boundary is a no-op and the
    same batch is NEVER published twice. Also pins Decision 2 (finalize-within-
    leg): the row is left ``in_flight=False`` with the batch cleared, so the next
    leg's CAS-retry cannot deadlock on a never-cleared in-flight row (the actual
    stuck-at-INFERENCE_COMPLETED symptom OMN-14493 fixes).
    """
    store = _DurableRows()
    adapter = _FakeStateStoreAdapter(store)
    bus = _RecordingBus()
    handler = _FanOutHandler(_fanout_batch())
    callback = _stateful_callback(handler, adapter, event_bus=bus)
    applier = _make_applier(bus)

    async def _run() -> None:
        result = await callback(_input_envelope())
        assert result is None, (
            "a has-bus wrapper that published-from-row MUST return None so the "
            "external applier does not re-publish the same batch (double-publish)"
        )
        # Running the external applier on the None result must be a no-op.
        await applier.apply(result, CID)

    asyncio.run(_run())

    published = bus.envelopes_for(CID)
    assert len(published) == 3, (
        f"exactly one 3-event batch must reach the bus (no double-publish); "
        f"got {len(published)}"
    )
    assert bus.topics_for(CID) == [TOPIC_ROUTING, TOPIC_INFERENCE, TOPIC_QUALITY]
    row = store.rows[str(CID)]
    assert row["in_flight"] is False, (
        "finalize-within-leg: the has-bus leg must CAS-finalize (in_flight=False) "
        "so the next leg does not deadlock on a never-cleared in-flight row"
    )
    assert not row["pending_emissions"], "finalize must clear pending_emissions"


@pytest.mark.integration
def test_has_bus_wrapper_publish_failure_propagates_for_redelivery() -> None:
    """The production has-bus publish-from-row path must not ACK broker failure."""
    store = _DurableRows()
    adapter = _FakeStateStoreAdapter(store)
    bus = _RecordingBus(fail_at_index=1)  # event 0 lands, event 1 is rejected
    handler = _FanOutHandler(_fanout_batch())
    callback = _stateful_callback(handler, adapter, event_bus=bus)
    applier = _make_applier(bus)
    on_message = _bus_callback(callback, applier)

    raised: BaseException | None = None

    async def _run() -> None:
        nonlocal raised
        try:
            await on_message(_input_envelope())
        except BaseException as exc:  # noqa: BLE001 - asserting on propagation
            raised = exc

    asyncio.run(_run())

    assert len(bus.published) == 1, "precondition: the has-bus batch failed mid-way"
    assert raised is not None, (
        "SWALLOWED. The has-bus publish-from-row path raised a broker failure, "
        "but the auto-wired boundary ACKed it instead of redelivering."
    )
    assert "broker rejected publish" in str(raised)


@pytest.mark.integration
def test_non_outbox_boundary_still_swallows_unchanged() -> None:
    """Decision 3 regression guard: a NON-outbox contract is UNCHANGED.

    The boundary no-swallow is scoped to the outbox path by
    ``propagate_publish_failures``. With the flag at its default (off), a publish
    failure is STILL log-and-discarded exactly as before P3b — P3b must not
    broaden the un-swallow to every auto-wired contract (the platform-wide hole
    is OMN-14498/OMN-14507's, not this ticket's). This is the mirror image of
    ``test_publish_exception_mid_batch_propagates...`` with the flag OFF.
    """
    store = _DurableRows()
    adapter = _FakeStateStoreAdapter(store)
    bus = _RecordingBus(fail_at_index=1)  # event 0 lands, event 1 is rejected
    handler = _FanOutHandler(_fanout_batch())
    callback = _stateful_callback(handler, adapter)  # no bus → external applier
    applier = _make_applier(bus)

    class _Engine:
        async def dispatch(
            self, topic: str, envelope: ModelEventEnvelope[object]
        ) -> Any:
            return await callback(envelope)

    # DEFAULT flag (propagate_publish_failures NOT passed) = non-outbox behavior.
    on_message = _make_event_bus_callback(
        TOPIC_INBOUND, cast("Any", _Engine()), cast("Any", applier)
    )

    raised: BaseException | None = None

    async def _run() -> None:
        nonlocal raised
        try:
            await on_message(_input_envelope())
        except BaseException as exc:  # noqa: BLE001 — asserting on propagation
            raised = exc

    asyncio.run(_run())

    assert len(bus.published) == 1, "precondition: the batch really did fail mid-way"
    assert raised is None, (
        "REGRESSION: a non-outbox contract's publish failure must STILL be "
        "swallowed at the boundary (flag default off) — P3b must not broaden the "
        "un-swallow to every auto-wired contract. Only the outbox path (flag on) "
        "propagates (see test_publish_exception_mid_batch_propagates...)."
    )
