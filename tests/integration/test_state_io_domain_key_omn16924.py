# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-declared domain key for the state_io dispatch seam (OMN-16924).

``node_session_phase_reducer``'s state of record was a cwd-relative
``.onex_state/session/phase_state.yaml``. The runtime container's cwd is
``/app`` (``root:root 0755``) while the process runs as ``omniinfra``, so every
bus dispatch raised ``PermissionError: [Errno 13] Permission denied:
'.onex_state'`` and DLQ'd — a 100% failure rate on all three subscribed topics,
on every lane.

Operator ruling 2026-08-29, verbatim: *"onex_state should be configurable via
contract overlay right? for our purposes, state should only be kept in the
database."* So the reducer's prior state comes from the DATABASE, supplied by
the runtime, through the seam that already does exactly that: ``state_io``
(OMN-14208). Two things had to generalize before a REDUCER could use it, and
both are proven here:

1. **The row key.** ``state_io`` hardcoded ``correlation_id`` — the only id
   every leg of a multi-leg ORCHESTRATOR carries. A REDUCER folds per DOMAIN
   entity: ``HandlerSessionPhaseReducer.delta`` explicitly *rejects* an event
   whose ``session_id`` disagrees with the folded state's, so ``session_id`` is
   its identity, and the omniclaude hooks mint a fresh ``correlation_id`` per
   event whenever tracing supplies one. Keying on ``correlation_id`` would
   scatter one session's fold across N rows. ``state_io.key`` (already present
   in the delegation contract, previously ignored) now names both the wire
   payload field AND the primary-key column.

2. **The emission-completeness guard.** OMN-14721 fails a fresh non-terminal
   seed that carries no durable emission, because ``select_recoverable_batches``
   could never re-publish it. That premise is the in-row OUTBOX. A contract with
   no ``published_events`` map has no outbox at all, its durable output IS the
   row, and there is no emission to strand — so the guard must not convert a
   correct, complete reducer fold into a hard dispatch failure.

Both tests were RED against pre-fix ``handler_wiring.py``: (1) seeded the row
under ``str(CID)`` instead of the session id, and (2) raised the OMN-14721
``ModelOnexError`` and seeded nothing.
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

CID = UUID("16924001-1111-4111-8111-141692400001")
INPUT_ENVELOPE_ID = UUID("16924002-2222-4222-8222-141692400002")
SESSION_ID = "omn16924-session-under-test"

_PATCH_IMPORT = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)
_PATCH_ADAPTER = "omnibase_infra.runtime.auto_wiring.handler_wiring.StateStoreAdapter"

SESSION_STATE_IO: dict[str, object] = {
    "database": "omnibase_infra",
    "table": "session_phase_state",
    "key": "session_id",
    "codec": {
        "module": "tests.integration.test_state_io_domain_key_omn16924",
        "name": "_SeamCodec",
    },
}
DELEGATION_STATE_IO: dict[str, object] = {
    "database": "omnibase_infra",
    "table": "delegation_workflow_state",
    "key": "correlation_id",
    "codec": {
        "module": "tests.integration.test_state_io_domain_key_omn16924",
        "name": "_SeamCodec",
    },
}
OUTPUT_TOPIC_MAP = {
    "SeamRoutingIntent": "onex.cmd.test-seam.routing-request.v1"  # onex-topic-allow: test fixture
}


class ModelSeamRoutingIntent(BaseModel):
    """Stand-in emission for the outbox-carrying (orchestrator) control case."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: UUID | None = None


class _SeamCodec:
    """state_io codec: the post-handle bridge (OMN-14208 pair-verify M1)."""

    def flush(self, key: str) -> str | None:
        current = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        entry = current.get(key)
        return entry[0] if entry is not None else None


class _FoldingHandler:
    """A leg that folds new state into the bound row and emits ``events``.

    Mirrors what a REDUCER does through its omnimarket-side proxy: read the
    runtime-bound row, compute, write the new payload back into the request-
    scoped binding for the codec to flush. It performs no I/O of its own.
    """

    def __init__(self, state: str, events: tuple[BaseModel, ...] = ()) -> None:
        self.state = state
        self.events = events
        self.observed_keys: list[str] = []
        self.observed_prior_payloads: list[str | None] = []

    async def handle(self, envelope: object) -> ModelHandlerOutput[None]:
        current = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        key = next(iter(current))
        prior_payload_json, version = current[key]
        self.observed_keys.append(key)
        self.observed_prior_payloads.append(prior_payload_json)
        folded = {"tenant_id": "", "state": self.state, "in_flight": False}
        CONTEXTVAR_STATE_IO_ROWS.set({key: (json.dumps(folded), version)})
        return ModelHandlerOutput.for_orchestrator(
            input_envelope_id=getattr(envelope, "envelope_id", INPUT_ENVELOPE_ID),
            correlation_id=CID,
            handler_id="omn16924-folding-leg",
            events=self.events,
        )


class _FakeStateStoreAdapter:
    """In-memory StateStoreAdapter with the real SQL semantics."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    async def load(self, key: str) -> tuple[str, int] | None:
        row = self.rows.get(key)
        if row is None:
            return None
        return cast("str", row["payload_json"]), cast("int", row["version"])

    async def seed(
        self,
        key: str,
        *,
        tenant_id: str,
        state: str,
        in_flight: bool,
        payload_json: str,
        pending_emissions: list[dict[str, Any]] | None = None,
        publish_attempts: int = 0,
    ) -> bool:
        if key in self.rows:
            return False
        self.rows[key] = {
            "correlation_id": key,
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
        key: str,
        *,
        tenant_id: str,
        state: str,
        in_flight: bool,
        payload_json: str,
        expected_version: int,
        pending_emissions: list[dict[str, Any]] | None = None,
        publish_attempts: int | None = None,
    ) -> int:
        row = self.rows.get(key)
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


def _envelope(payload: dict[str, object]) -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](
        envelope_id=INPUT_ENVELOPE_ID,
        correlation_id=CID,
        payload=payload,
    )


def _session_started_envelope() -> ModelEventEnvelope[object]:
    """The real ``onex.evt.omniclaude.session-started.v1`` field set.

    ``correlation_id`` is present and is a DIFFERENT value from ``session_id``,
    which is the whole point: the fold must key on the session.
    """
    return _envelope(
        {
            "session_id": SESSION_ID,
            "correlation_id": str(CID),
            "emitted_at": "2026-08-29T07:07:01Z",
            "hook_source": "startup",
            "working_directory": "/workspace",
        }
    )


def _stateful_callback(
    handler: _FoldingHandler,
    adapter: _FakeStateStoreAdapter,
    state_io: dict[str, object],
    *,
    output_topic_map: dict[str, str] | None = None,
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
            dict(state_io),
            event_bus=None,
            output_topic_map=output_topic_map,
        )


@pytest.mark.integration
def test_declared_domain_key_keys_the_row_on_session_id_not_correlation_id() -> None:
    """RED pre-fix: the row was keyed on ``correlation_id``, not the session."""
    adapter = _FakeStateStoreAdapter()
    handler = _FoldingHandler(state="start")
    callback = _stateful_callback(handler, adapter, SESSION_STATE_IO)

    asyncio.run(callback(_session_started_envelope()))

    assert list(adapter.rows) == [SESSION_ID], (
        "the durable row must be keyed on the contract-declared state_io.key "
        f"({SESSION_ID!r}), not on the transport correlation_id; got "
        f"{list(adapter.rows)!r}"
    )
    assert str(CID) not in adapter.rows
    assert handler.observed_keys == [SESSION_ID]


@pytest.mark.integration
def test_second_event_for_the_same_session_folds_onto_the_first_row() -> None:
    """The fold is continuous across events with DIFFERENT correlation_ids.

    This is the behaviour the file state of record was supposed to provide and
    the reason ``correlation_id`` keying would have been wrong even if the file
    had been writable: two hook events in one session carry two correlation_ids.
    """
    adapter = _FakeStateStoreAdapter()
    handler = _FoldingHandler(state="start")
    started = _stateful_callback(handler, adapter, SESSION_STATE_IO)
    asyncio.run(started(_session_started_envelope()))

    ended_handler = _FoldingHandler(state="ended")
    ended = _stateful_callback(ended_handler, adapter, SESSION_STATE_IO)
    asyncio.run(
        ended(
            _envelope(
                {
                    "session_id": SESSION_ID,
                    # A DIFFERENT correlation_id — the hooks mint one per event.
                    "correlation_id": "16924003-3333-4333-8333-141692400003",
                    "emitted_at": "2026-08-29T07:20:00Z",
                    "reason": "clear",
                }
            )
        )
    )

    assert list(adapter.rows) == [SESSION_ID]
    assert adapter.rows[SESSION_ID]["state"] == "ended"
    assert adapter.rows[SESSION_ID]["version"] == 1, "the second event CAS-updated"
    # The second leg saw the FIRST leg's durable state as its prior state — the
    # prior-state provisioning the reducer could not get from the wire.
    assert ended_handler.observed_prior_payloads == [
        json.dumps({"tenant_id": "", "state": "start", "in_flight": False})
    ]


@pytest.mark.integration
def test_missing_declared_key_fails_closed_with_no_local_fallback() -> None:
    """No silent fallback: a message without the declared key is a hard failure."""
    adapter = _FakeStateStoreAdapter()
    handler = _FoldingHandler(state="start")
    callback = _stateful_callback(handler, adapter, SESSION_STATE_IO)

    with pytest.raises(ModelOnexError) as exc_info:
        asyncio.run(
            callback(
                _envelope(
                    {"correlation_id": str(CID), "emitted_at": "2026-08-29T07:07:01Z"}
                )
            )
        )

    assert "session_id" in str(exc_info.value)
    assert adapter.rows == {}


@pytest.mark.integration
def test_reducer_without_an_outbox_seeds_a_nonterminal_row() -> None:
    """RED pre-fix: the OMN-14721 guard rejected a correct, complete fold.

    A contract with no ``published_events`` map has no in-row outbox — there is
    no emission to strand, so a fresh non-terminal seed with an empty batch is
    the NORMAL and complete outcome of a reducer fold, not a dead row.
    """
    adapter = _FakeStateStoreAdapter()
    handler = _FoldingHandler(state="start")
    callback = _stateful_callback(
        handler, adapter, SESSION_STATE_IO, output_topic_map=None
    )

    asyncio.run(callback(_session_started_envelope()))

    (row,) = adapter.rows.values()
    assert row["state"] == "start"
    assert row["in_flight"] is False
    assert not row["pending_emissions"]


@pytest.mark.integration
def test_correlation_id_keying_and_the_outbox_guard_are_unchanged() -> None:
    """Regression control: the delegation shape keeps its exact prior behaviour.

    Default key (``correlation_id``) plus a ``published_events`` map means the
    OMN-14721 guard is still armed and still fires before any row is seeded.
    """
    adapter = _FakeStateStoreAdapter()
    handler = _FoldingHandler(state="RECEIVED")
    callback = _stateful_callback(
        handler,
        adapter,
        DELEGATION_STATE_IO,
        output_topic_map=dict(OUTPUT_TOPIC_MAP),
    )

    with pytest.raises(ModelOnexError) as exc_info:
        asyncio.run(callback(_envelope({"correlation_id": str(CID)})))

    assert "OMN-14721" in str(exc_info.value)
    assert adapter.rows == {}


@pytest.mark.integration
def test_correlation_id_keyed_leg_with_an_emission_still_commits_with_intent() -> None:
    """Positive control for the unchanged delegation path."""
    adapter = _FakeStateStoreAdapter()
    handler = _FoldingHandler(
        state="RECEIVED", events=(ModelSeamRoutingIntent(correlation_id=CID),)
    )
    callback = _stateful_callback(
        handler,
        adapter,
        DELEGATION_STATE_IO,
        output_topic_map=dict(OUTPUT_TOPIC_MAP),
    )

    asyncio.run(callback(_envelope({"correlation_id": str(CID)})))

    assert list(adapter.rows) == [str(CID)]
    row = adapter.rows[str(CID)]
    assert row["in_flight"] is True
    assert row["pending_emissions"] and len(row["pending_emissions"]) == 1


# ---------------------------------------------------------------------------
# Overlay-configurable binding — the second half of the operator ruling.
# ---------------------------------------------------------------------------

# The literal declaration ``node_session_phase_reducer``'s contract carries.
OVERLAY_STATE_IO: dict[str, object] = {
    "database": "${env.ONEX_SESSION_PHASE_STATE_DATABASE:omnibase_infra}",
    "table": "${env.ONEX_SESSION_PHASE_STATE_TABLE:session_phase_state}",
    "key": "session_id",
    "codec": SESSION_STATE_IO["codec"],
}


def _adapter_kwargs(state_io: dict[str, object], env: dict[str, str]) -> dict[str, Any]:
    """Wire a callback under ``env`` and return the adapter's constructor kwargs.

    The binding is decided at WIRING time, so what the adapter was constructed
    with IS the durable target. Asserting on it is the only way to see the
    overlay take effect without a live Postgres.
    """
    adapter = _FakeStateStoreAdapter()
    with (
        patch.dict(
            "os.environ",
            {"OMNIBASE_INFRA_DB_URL": "postgresql://user:pass@host:5432/db", **env},
        ),
        patch(_PATCH_IMPORT, return_value=_SeamCodec),
        patch(_PATCH_ADAPTER, return_value=adapter) as adapter_cls,
    ):
        _make_stateful_dispatch_callback(
            cast("Any", _FoldingHandler(state="start")),
            None,
            dict(state_io),
            event_bus=None,
            output_topic_map=None,
        )
    _dsn, kwargs = adapter_cls.call_args
    return cast("dict[str, Any]", kwargs)


@pytest.mark.integration
def test_overlay_rebinds_the_durable_table_with_no_code_change() -> None:
    """RED pre-fix: ``table`` was taken verbatim, so the overlay ref never resolved.

    Operator ruling 2026-08-29, verbatim: *"onex_state should be configurable via
    contract overlay right? for our purposes, state should only be kept in the
    database."* This is the "configurable via contract overlay" half. Before the
    fix, ``_make_stateful_dispatch_callback`` did ``str(state_io.get("table"))``
    and handed the adapter the literal ``${env.ONEX_SESSION_PHASE_STATE_TABLE:
    session_phase_state}`` string — an overlay could not move the binding at all,
    and the only way to retarget a node's state was to edit its contract.

    Same declaration, two environments, two durable targets, zero code change.
    """
    default_binding = _adapter_kwargs(OVERLAY_STATE_IO, {})
    assert default_binding["table"] == "session_phase_state"
    assert default_binding["key_column"] == "session_id"

    overridden = _adapter_kwargs(
        OVERLAY_STATE_IO,
        {"ONEX_SESSION_PHASE_STATE_TABLE": "session_phase_state_lane_b"},
    )
    assert overridden["table"] == "session_phase_state_lane_b", (
        "the contract-overlay env ref did not rebind the durable table"
    )
    assert overridden["key_column"] == "session_id"


@pytest.mark.integration
def test_overlay_resolving_to_empty_fails_closed_with_no_local_fallback() -> None:
    """An unresolvable binding is a wiring-time failure, never a quiet default.

    ``expand_contract_env_refs`` expands an unset var with no inline default to
    the empty string precisely so the caller fails closed. The node must refuse
    to wire rather than fall back to a local file — the failure mode this whole
    ticket exists to delete.
    """
    unresolvable = dict(OVERLAY_STATE_IO)
    unresolvable["table"] = "${env.ONEX_SESSION_PHASE_STATE_TABLE_UNSET}"

    with pytest.raises(ModelOnexError) as exc_info:
        _adapter_kwargs(unresolvable, {})

    assert "state_io.table is required" in str(exc_info.value)
