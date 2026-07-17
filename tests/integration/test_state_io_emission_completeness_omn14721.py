# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed emission-completeness guard for the state_io in-row outbox (OMN-14721).

Root cause (docs/tracking/2026-07-17-delegation-routing-publish-drop-rootcause.md):
the delegation orchestrator handled ``delegation-request.v1``, mutated its FSM to
``RECEIVED`` (a NON-terminal, must-emit state), but the committing leg's
``result.output_events`` was EMPTY — so the wrapper seeded
``delegation_workflow_state`` with ``in_flight=false`` and ``pending_emissions=∅``.
That row is STRUCTURALLY unrecoverable: ``select_recoverable_batches`` only
re-publishes ``in_flight AND jsonb_array_length(pending_emissions) > 0`` rows and
``recover_stale_rows`` only give-up-FAILs it after the stale TTL — so the routing
intent is silently dropped forever and the workflow stalls. This is the worst-case
silent-drop class the seam was built to eliminate.

These tests drive the REAL ``_make_stateful_dispatch_callback`` wrapper over a fake
adapter with the real SQL semantics (the established state_io test pattern — see
``test_state_io_outbox_seam_red.py`` / ``test_state_io_concurrent_dispatch.py``):

* ``test_fresh_nonterminal_seed_with_empty_batch_fails_closed`` — the RED-proof.
  It reproduces the EXACTLY-WRONG state (a leg that mutates FSM to ``RECEIVED`` yet
  emits nothing) and asserts the wrapper now RAISES rather than seeding the
  unrecoverable dead row. Verified RED against pre-fix ``handler_wiring.py`` (with
  the OMN-14721 guard removed, this test FAILS: the callback returns a SUCCESS
  ``ModelDispatchResult`` and a dead row is silently seeded — see the PR body for
  the captured pre-fix output).
* ``test_fresh_received_seed_captures_routing_intent_into_pending_emissions`` — the
  positive control: a NORMAL ``RECEIVED`` leg that emits a routing intent persists
  it into ``pending_emissions`` (non-empty outbox, ``in_flight=True``). Guards
  against a guard that fires on the happy path.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_stateful_dispatch_callback,
)
from omnibase_infra.runtime.state_io.state_store_adapter import CONTEXTVAR_STATE_IO_ROWS

CID = UUID("14721001-1111-4111-8111-141472100001")
INPUT_ENVELOPE_ID = UUID("14721002-2222-4222-8222-141472100002")
TENANT = "acme-tenant"

TOPIC_ROUTING = (
    "onex.cmd.test-seam.routing-request.v1"  # onex-topic-allow: test fixture
)
OUTPUT_TOPIC_MAP = {"SeamRoutingIntent": TOPIC_ROUTING}

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


class ModelSeamRoutingIntent(BaseModel):
    """Stand-in for ModelRoutingIntent — the emission the RECEIVED leg must carry."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: UUID | None = None
    note: str = ""


class _SeamCodec:
    """state_io codec: the post-handle bridge (OMN-14208 pair-verify M1)."""

    def flush(self, cid: str) -> str | None:
        current = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        entry = current.get(cid)
        return entry[0] if entry is not None else None


class _ReceivedLegHandler:
    """A fresh-request leg that advances FSM to a NON-terminal state.

    Parameterized on the emitted batch so the SAME leg models both the correct
    behavior (emits the routing intent) and the OMN-14721 defect (mutates FSM to
    RECEIVED but emits NOTHING — the committing leg severed from its emission).
    """

    def __init__(self, events: tuple[BaseModel, ...]) -> None:
        self.events = events

    async def handle(self, envelope: object) -> ModelHandlerOutput[None]:
        current = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        cid = next(iter(current))
        _payload_json, version = current[cid]
        # Advance FSM to RECEIVED (non-terminal, in_flight=False) — a state that
        # MUST emit to progress. This is exactly the row the live incident left.
        new_state = {"tenant_id": TENANT, "state": "RECEIVED", "in_flight": False}
        CONTEXTVAR_STATE_IO_ROWS.set({cid: (json.dumps(new_state), version)})
        return ModelHandlerOutput.for_orchestrator(
            input_envelope_id=getattr(envelope, "envelope_id", INPUT_ENVELOPE_ID),
            correlation_id=CID,
            handler_id="omn14721-received-leg",
            events=self.events,
        )


class _FakeStateStoreAdapter:
    """In-memory StateStoreAdapter with the real SQL semantics + the outbox seam."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def load(self, cid: str) -> tuple[str, int] | None:
        row = self.rows.get(cid)
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
        pending_emissions: list[dict[str, Any]] | None = None,
        publish_attempts: int = 0,
    ) -> bool:
        if cid in self.rows:
            return False
        self.rows[cid] = {
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
        row = self.rows.get(cid)
        if row is None or row["version"] != expected_version:
            return 0
        row.update(
            tenant_id=tenant_id,
            state=state,
            in_flight=in_flight,
            payload_json=payload_json,
            version=expected_version + 1,
            pending_emissions=pending_emissions,
        )
        return 1

    async def select_recoverable_batches(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.rows.values()
            if row["in_flight"] and row.get("pending_emissions")
        ]

    async def recover_stale_rows(self, ttl_seconds: int | None = None) -> int:
        return 0


def _input_envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](
        envelope_id=INPUT_ENVELOPE_ID,
        correlation_id=CID,
        payload={"correlation_id": str(CID), "tenant_id": TENANT},
    )


def _stateful_callback(
    handler: _ReceivedLegHandler, adapter: _FakeStateStoreAdapter
) -> Any:
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
            event_bus=None,
            output_topic_map=dict(OUTPUT_TOPIC_MAP),
        )


@pytest.mark.integration
def test_fresh_nonterminal_seed_with_empty_batch_fails_closed() -> None:
    """RED-proof: a fresh RECEIVED leg that captures NO emission must fail closed.

    Pre-fix (guard removed) this returned a SUCCESS ``ModelDispatchResult`` and
    silently seeded ``state=RECEIVED, in_flight=false, pending_emissions=∅`` — the
    unrecoverable dead row that stalls the workflow. The guard converts that silent
    permanent drop into a loud, DLQ-able dispatch failure.
    """
    adapter = _FakeStateStoreAdapter()
    handler = _ReceivedLegHandler(events=())  # committing leg severed from emission
    callback = _stateful_callback(handler, adapter)

    with pytest.raises(ModelOnexError) as exc_info:
        asyncio.run(callback(_input_envelope()))

    message = str(exc_info.value)
    assert "OMN-14721" in message
    assert "unrecoverable" in message.lower()
    # And critically: NO dead row was written — the guard fires BEFORE the seed,
    # so the seam does not persist a structurally-unrecoverable row at all.
    assert adapter.rows == {}, (
        "the fail-closed guard must fire before seeding — no unrecoverable "
        f"RECEIVED/in_flight=false/pending=∅ row may be written, got {adapter.rows!r}"
    )


@pytest.mark.integration
def test_fresh_received_seed_captures_routing_intent_into_pending_emissions() -> None:
    """Positive control: a NORMAL RECEIVED leg persists its routing intent.

    The routing intent the handler emits must land in the seeded row's
    ``pending_emissions`` (non-empty outbox, ``in_flight=True``), so the CAS
    winner can publish it FROM the row — the guarantee the OMN-14721 drop violated.
    """
    adapter = _FakeStateStoreAdapter()
    handler = _ReceivedLegHandler(
        events=(ModelSeamRoutingIntent(correlation_id=CID, note="routing"),)
    )
    callback = _stateful_callback(handler, adapter)

    result = asyncio.run(callback(_input_envelope()))
    assert result is not None  # no-bus commit path hands the result back

    assert len(adapter.rows) == 1, (
        f"expected exactly one seeded row, got {adapter.rows!r}"
    )
    (row,) = adapter.rows.values()
    assert row["state"] == "RECEIVED"
    assert row["in_flight"] is True, "a committed outbox batch marks the row in_flight"
    pending = row["pending_emissions"]
    assert pending is not None and len(pending) == 1, (
        f"the routing intent must be captured into pending_emissions, got {pending!r}"
    )
    assert pending[0]["class_name"] == "ModelSeamRoutingIntent"
    assert pending[0]["correlation_id"] == str(CID)
