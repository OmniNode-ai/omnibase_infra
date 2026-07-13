# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14576 — single-publisher recovery claim.

The OMN-14493 in-row outbox re-publishes a crashed leg's ``pending_emissions``
from the row on boot recovery. That sweep is gated per-PROCESS (``_recovery_ran``),
so two processes racing boot recovery — a rolling-redeploy overlap or a multi-pod
deploy — could EACH re-publish the same batch (a cross-process double-publish;
verifier finding on omnibase_infra#2296).

This suite proves the fix (a CAS-claim BEFORE publish) makes exactly ONE process
publish a given batch under a forced concurrent-recovery race — and, as the
load-bearing negative control, that WITHOUT the claim the same race double-
publishes. The two adapters differ ONLY in claim support, so the claim is
isolated as the cause of the fix.

The race is forced deterministically: both fake adapters snapshot the recoverable
set and then ``await asyncio.sleep(0)`` before returning, so both sweeps select
the SAME pre-claim version before either claims — the interleaving the claim must
survive. (An in-memory fake with no internal yield would otherwise run each sweep
to completion sequentially and never exercise the race.)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast
from uuid import UUID

import pytest

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope

# Reuse the RED suite's harness — same real callback, same fake-adapter semantics.
from tests.integration.test_state_io_outbox_seam_red import (
    CID,
    TENANT,
    _DurableRows,
    _FakeStateStoreAdapter,
    _fanout_batch,
    _FanOutHandler,
    _input_envelope,
    _RecordingBus,
    _stateful_callback,
)

_OTHER_A = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_OTHER_B = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
_CLAIM_TTL_SECONDS = 120.0


def _snapshot_recoverable(
    store: _DurableRows, *, honor_claim: bool
) -> list[dict[str, Any]]:
    """The recoverable set, optionally excluding a FRESH claim (mirrors the SQL)."""
    now = time.time()
    out: list[dict[str, Any]] = []
    for row in store.rows.values():
        if not (row.get("in_flight") and row.get("pending_emissions")):
            continue
        if honor_claim:
            claimed_at = row.get("recovery_claimed_at")
            if claimed_at is not None and claimed_at >= now - _CLAIM_TTL_SECONDS:
                continue  # another sweep is actively re-publishing this row
        out.append(dict(row))
    return out


class _ClaimingRacingAdapter(_FakeStateStoreAdapter):
    """Adapter WITH the OMN-14576 claim (the fix under test)."""

    async def select_recoverable_batches(self) -> list[dict[str, Any]]:
        snapshot = _snapshot_recoverable(self.store, honor_claim=True)
        # Yield so a concurrent sweep selects the SAME pre-claim version.
        await asyncio.sleep(0)
        return snapshot

    async def claim_recoverable_row(
        self, correlation_id: str, *, expected_version: int, claim_token: str
    ) -> bool:
        row = self.store.rows.get(correlation_id)
        now = time.time()
        if row is None or row["version"] != expected_version:
            return False
        if not (row.get("in_flight") and row.get("pending_emissions")):
            return False
        claimed_at = row.get("recovery_claimed_at")
        if claimed_at is not None and claimed_at >= now - _CLAIM_TTL_SECONDS:
            return False
        row["recovery_claimed_by"] = claim_token
        row["recovery_claimed_at"] = now
        row["version"] += 1
        return True


class _NonClaimingRacingAdapter(_FakeStateStoreAdapter):
    """Adapter WITHOUT the claim — the negative control.

    Same forced race, but no ``claim_recoverable_row`` (so the wrapper's
    ``hasattr`` gate skips the claim), which is exactly the pre-OMN-14576
    behavior that double-publishes.
    """

    async def select_recoverable_batches(self) -> list[dict[str, Any]]:
        snapshot = _snapshot_recoverable(self.store, honor_claim=False)
        await asyncio.sleep(0)
        return snapshot


def _seed_crashed_row(adapter: _FakeStateStoreAdapter) -> None:
    """Persist a crashed-mid-publish recoverable row via a real commit-with-intent."""
    seed_cb = _stateful_callback(_FanOutHandler(_fanout_batch()), adapter)  # no bus
    asyncio.run(seed_cb(_input_envelope()))
    row = adapter.store.rows[str(CID)]
    assert row["in_flight"] is True, (
        "precondition: row committed in-flight (crashed pre-publish)"
    )
    assert row["pending_emissions"], "precondition: the batch was persisted to the row"


def _run_two_concurrent_recovery_sweeps(
    adapter: _FakeStateStoreAdapter,
) -> _RecordingBus:
    """Two independent processes (distinct claim tokens) race boot recovery."""
    bus = _RecordingBus()
    # Each _stateful_callback build mints its own per-process claim token.
    cb_a = _stateful_callback(_FanOutHandler(_fanout_batch()), adapter, event_bus=bus)
    cb_b = _stateful_callback(_FanOutHandler(_fanout_batch()), adapter, event_bus=bus)

    def _other(cid: UUID) -> ModelEventEnvelope[object]:
        return ModelEventEnvelope[object](
            correlation_id=cid,
            payload={"correlation_id": str(cid), "tenant_id": TENANT},
        )

    async def _race() -> None:
        # Each dispatch triggers its process's one-time boot recovery of CID's
        # crashed batch; the trigger correlations differ from CID so recovery
        # (not the in_flight-lock resume) owns it.
        await asyncio.gather(
            cast("Any", cb_a(_other(_OTHER_A))), cast("Any", cb_b(_other(_OTHER_B)))
        )

    asyncio.run(_race())
    return bus


@pytest.mark.integration
def test_two_concurrent_recovery_sweeps_publish_the_batch_exactly_once() -> None:
    """THE FIX: with the CAS-claim, exactly one sweep publishes the batch."""
    adapter = _ClaimingRacingAdapter(_DurableRows())
    _seed_crashed_row(adapter)

    bus = _run_two_concurrent_recovery_sweeps(adapter)

    recovered = bus.envelopes_for(CID)
    assert len(recovered) == 3, (
        "SINGLE-PUBLISHER VIOLATION: two concurrent recovery sweeps published "
        f"{len(recovered)} envelopes for the crashed batch — expected exactly 3 "
        "(one publisher). The CAS-claim must let only one sweep win."
    )
    row = adapter.store.rows[str(CID)]
    assert row["in_flight"] is False, "the winning sweep must CAS-finalize the row"
    assert not row["pending_emissions"], "finalize must clear pending_emissions"
    assert row.get("recovery_claimed_by"), (
        "the winner must have stamped its claim token"
    )


@pytest.mark.integration
def test_without_the_claim_the_same_race_double_publishes() -> None:
    """NEGATIVE CONTROL (load-bearing): remove ONLY the claim and the identical
    forced race double-publishes — proving the claim is the fix, not the harness
    or incidental ordering."""
    adapter = _NonClaimingRacingAdapter(_DurableRows())
    _seed_crashed_row(adapter)

    bus = _run_two_concurrent_recovery_sweeps(adapter)

    recovered = bus.envelopes_for(CID)
    assert len(recovered) == 6, (
        "the negative control must reproduce the cross-process double-publish "
        f"(6 = 2x the 3-event batch); got {len(recovered)}. If this is not 6, the "
        "race is not actually being forced and the positive test proves nothing."
    )
