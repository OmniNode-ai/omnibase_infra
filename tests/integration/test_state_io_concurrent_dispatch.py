# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for concurrent state_io dispatch (OMN-14208).

Proves the load-before/CAS-persist-after boundary hook is replay-safe under
genuine contention: two dispatches racing on the SAME correlation_id — both
observing "no row yet" before either commits — must resolve to exactly one
intent-emitting result. The loser's OCC retry reloads the winner's persisted
row and re-runs ``handle()`` against it, so the synchronous in-flight dedup
guard folds without re-emitting (this is the exact failure mode the design
rejected Candidate-2 over: an async emit-then-persist path would let both
racers hydrate ``in_flight=False`` and both emit, i.e. double LLM spend).

The asyncpg layer is faked with an in-memory adapter that reproduces the real
SQL semantics (``INSERT ... ON CONFLICT DO NOTHING`` / version-gated
``UPDATE``) plus a rendezvous point so both dispatches are forced through
``load()`` before either persists — this is the actual race window, not just
a sequential "first, then second" happy path.
"""

from __future__ import annotations

import asyncio
import json
from typing import cast
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.reducer.model_intent import ModelIntent
from omnibase_core.models.reducer.payloads import ModelPayloadExtension
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_stateful_dispatch_callback,
)
from omnibase_infra.runtime.state_io.state_store_adapter import (
    CONTEXTVAR_STATE_IO_ROWS,
)

_STATE_IO = {
    "database": "omnibase_infra",
    "table": "delegation_workflow_state",
    "key": "correlation_id",
    "codec": {"module": "tests.integration", "name": "_FakeCodec"},
}


class _FakeSharedAdapter:
    """In-memory stand-in for StateStoreAdapter with real CAS/seed semantics.

    Exposes a rendezvous so the first two concurrent ``load()`` callers are
    forced to both observe the pre-commit state (no row) before either
    proceeds — the actual race the OCC-retry design must resolve, not a
    sequential "one completes, then the other starts" approximation.
    """

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, object]] = {}
        self._arrived = 0
        self._both_arrived = asyncio.Event()

    async def load(self, cid: str) -> tuple[str, int] | None:
        if self._arrived < 2:
            self._arrived += 1
            if self._arrived == 2:
                self._both_arrived.set()
            else:
                await self._both_arrived.wait()
        row = self._rows.get(cid)
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
    ) -> bool:
        if cid in self._rows:
            return False
        self._rows[cid] = {
            "tenant_id": tenant_id,
            "state": state,
            "in_flight": in_flight,
            "payload_json": payload_json,
            "version": 0,
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
    ) -> int:
        row = self._rows.get(cid)
        if row is None or row["version"] != expected_version:
            return 0
        row["tenant_id"] = tenant_id
        row["state"] = state
        row["in_flight"] = in_flight
        row["payload_json"] = payload_json
        row["version"] = expected_version + 1
        return 1

    async def recover_stale_rows(self, ttl_seconds: int | None = None) -> int:
        return 0


class _FakeCodec:
    """Stand-in state_io codec exercising the ``flush()`` bridge (OMN-14208
    pair-verify M1). ``_load_handle_persist`` no longer reads
    ``CONTEXTVAR_STATE_IO_ROWS`` back itself post-handle — it calls
    ``codec.flush(cid)`` on the contract-resolved codec instance. This fake
    reads the SAME ContextVar ``_DedupGuardHandler`` mutates: asyncio's
    contextvars are copied per-Task at creation, so each concurrent
    dispatch's mutation is visible only to that SAME task's ``flush()`` call
    — exactly mirroring the production ``StateIoCodec`` /
    ``DelegationWorkflowStateProxy`` split without needing the real
    omnimarket proxy in this infra-only test.
    """

    def flush(self, cid: str) -> str | None:
        current = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        entry = current.get(cid)
        return entry[0] if entry is not None else None


class _DedupGuardHandler:
    """Mimics HandlerDelegationWorkflow's synchronous in-flight dedup guard
    (handler_delegation_workflow.py ~:919/924/927): a synchronous
    read-modify-write BEFORE emitting, so a racer that reloads an
    already-claimed row folds without re-emitting."""

    async def handle(self, envelope: object) -> ModelHandlerOutput[None] | None:
        current = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        cid = next(iter(current))
        payload_json, version = current[cid]
        state = json.loads(payload_json) if payload_json else {}
        if state.get("in_flight"):
            return None
        new_state = {
            "tenant_id": "acme",
            "state": "IN_PROGRESS",
            "in_flight": True,
        }
        CONTEXTVAR_STATE_IO_ROWS.set({cid: (json.dumps(new_state), version)})
        correlation_id = UUID(cid)
        intent = ModelIntent(
            intent_type="extension",
            target="state-io-test",
            payload=ModelPayloadExtension(
                extension_type="test.inference",
                plugin_name="state-io-dedup-test",
                data={"intent": "inference-intent"},
            ),
        )
        return ModelHandlerOutput.for_orchestrator(
            input_envelope_id=_Envelope.envelope_id,
            correlation_id=correlation_id,
            handler_id="dedup-guard-handler",
            intents=(intent,),
        )


class _Envelope:
    envelope_id = UUID("11111111-1111-4111-8111-111111111111")
    payload = {"correlation_id": "22222222-2222-2222-2222-222222222222"}


@pytest.mark.integration
def test_concurrent_same_correlation_id_dispatch_emits_exactly_one_intent() -> None:
    shared_adapter = _FakeSharedAdapter()

    with (
        patch.dict(
            "os.environ",
            {"OMNIBASE_INFRA_DB_URL": "postgresql://user:pass@host:5432/db"},
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_FakeCodec,
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.StateStoreAdapter",
            return_value=shared_adapter,
        ),
    ):
        callback = _make_stateful_dispatch_callback(
            _DedupGuardHandler(), None, _STATE_IO
        )

        async def _run() -> list[object]:
            return await asyncio.gather(
                callback(_Envelope()),  # type: ignore[arg-type]
                callback(_Envelope()),  # type: ignore[arg-type]
            )

        results = asyncio.run(_run())

    non_empty = [r for r in results if r and getattr(r, "output_intents", ())]
    empty = [r for r in results if r is None]
    assert len(non_empty) == 1, (
        f"expected exactly one intent-emitting result, got {results!r}"
    )
    assert len(empty) == 1, f"expected exactly one no-op fold, got {results!r}"
    (intent,) = non_empty[0].output_intents
    assert intent.payload.data == {"intent": "inference-intent"}

    # Exactly one durable row landed — no duplicate/forked row was created by
    # the loser's retry.
    assert len(shared_adapter._rows) == 1
    (row,) = shared_adapter._rows.values()
    assert row["in_flight"] is True
    assert row["version"] == 0, (
        "the loser's no-op retry must not bump version on a byte-identical "
        "re-fold (matches _make_stateful_dispatch_callback's no_mutation skip)"
    )


@pytest.mark.integration
def test_sequential_dispatch_after_completion_also_folds() -> None:
    """Regression companion: a THIRD dispatch after the row is already
    COMPLETED-adjacent (in_flight=True, no race) must also fold on its first
    attempt (no retry needed) — the persisted flag is authoritative
    regardless of whether contention was involved."""
    shared_adapter = _FakeSharedAdapter()
    # Pre-arrive both rendezvous slots so this single dispatch doesn't block.
    shared_adapter._arrived = 2
    shared_adapter._both_arrived.set()
    cid = "22222222-2222-2222-2222-222222222222"
    shared_adapter._rows[cid] = {
        "tenant_id": "acme",
        "state": "IN_PROGRESS",
        "in_flight": True,
        "payload_json": json.dumps(
            {"tenant_id": "acme", "state": "IN_PROGRESS", "in_flight": True}
        ),
        "version": 0,
    }

    with (
        patch.dict(
            "os.environ",
            {"OMNIBASE_INFRA_DB_URL": "postgresql://user:pass@host:5432/db"},
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_FakeCodec,
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.StateStoreAdapter",
            return_value=shared_adapter,
        ),
    ):
        callback = _make_stateful_dispatch_callback(
            _DedupGuardHandler(), None, _STATE_IO
        )
        result = asyncio.run(callback(_Envelope()))  # type: ignore[arg-type]

    assert result is None
    assert shared_adapter._rows[cid]["version"] == 0
