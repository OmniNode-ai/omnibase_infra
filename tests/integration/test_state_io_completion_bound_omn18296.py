# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The contract-declared completion bound, and the terminal it must emit (OMN-18296).

RED-first. Both behavioural tests below drive code that EXISTS and executes, and
both fail against the pre-OMN-18296 wiring for a reason that is not absence:

``test_row_past_the_bound_emits_a_real_terminal_event``
    ``recover_stale_rows`` runs, finds the abandoned row, and flips it to
    ``FAILED`` — and publishes nothing. That is a state change no consumer can
    observe, so downstream the workflow stays non-terminal forever. This test
    asserts the give-up TRAVELS, on the contract's declared failure topic.

``test_the_bound_is_swept_without_any_further_dispatch_traffic``
    The pre-fix sweep is reachable only from inside a dispatch
    (``_ensure_stale_rows_recovered``), so the very lane that lost a workflow to
    a restart — and therefore has no traffic left — never sweeps again. This test
    dispatches ONCE and then asserts the row terminalises on the sweeper's own
    timer, with no second dispatch.

Live origin, for anyone reading this later: the lab lane's runtime-effects pod
was recreated at 2026-09-13T09:57:40Z with delegation correlation
``a2fe0848-4b4b-462e-b633-c5f9559afee5`` (submitted 09:55:04Z) in flight. The
inference command's consumer offset was already committed so it was never
redelivered, no response was ever published, and the FSM row sat ``ROUTED`` /
``in_flight`` for more than half an hour past a 900s TTL while the gateway row
stayed ``published``. Both defects above are visible in that one row.

The fake adapter reproduces the real SQL semantics, INCLUDING the pre-fix
row-only ``recover_stale_rows`` — so the red comes from the wiring under test,
never from a crippled double.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict

from omnibase_core.models.errors import ModelOnexError
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_runtime_restart_policy import (
    EnumRuntimeRestartPolicy,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_stateful_dispatch_callback,
    _read_completion_bound,
)
from omnibase_infra.runtime.state_io.model_completion_bound import (
    ModelCompletionBound,
)
from omnibase_infra.runtime.state_io.state_store_adapter import (
    CONTEXTVAR_STATE_IO_ROWS,
)

CID_LIVE = "a2fe0848-4b4b-462e-b633-c5f9559afee5"
"""The real abandoned correlation this ticket was filed on."""

CID_DISPATCH = "11111111-1111-1111-1111-111111111111"
TENANT = "cacacbb1-0e64-4521-9712-ed02ee799907"
TOPIC_DELEGATION_FAILED = (
    "onex.evt.omnibase-infra.delegation-failed.v1"  # onex-topic-allow: test fixture
)
OUTPUT_TOPIC_MAP = {"DelegationFailed": TOPIC_DELEGATION_FAILED}
STATE_IO = {
    "database": "omnibase_infra",
    "table": "delegation_workflow_state",
    "key": "correlation_id",
    "codec": {"module": "tests.integration", "name": "_BoundCodec"},
}
_PATCH_IMPORT = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)
_PATCH_ADAPTER = "omnibase_infra.runtime.auto_wiring.handler_wiring.StateStoreAdapter"

BOUND = ModelCompletionBound(
    max_wall_seconds=900,
    on_runtime_restart=EnumRuntimeRestartPolicy.TERMINALISE_FAILED,
    failure_class="runtime_restart_during_delegation",
    failure_code="ONEX_MARKET_DELEGATION_RUNTIME_RESTART",
)


class ModelDelegationFailed(BaseModel):
    """Stand-in for the real terminal class, resolved by name like the real one."""

    model_config = ConfigDict(extra="forbid")

    correlation_id: UUID
    tenant_id: str | None = None
    causation_id: UUID | None = None
    failure_reason: str = ""
    terminal_failure_reason: str | None = None


class _FakeStateStoreAdapter:
    """In-memory adapter with the real predicates, including the pre-fix sweep."""

    def __init__(self, rows: dict[str, dict[str, Any]]) -> None:
        self.rows = rows
        self.stale_sweeps = 0

    async def load(self, cid: str) -> tuple[str, int] | None:
        row = self.rows.get(cid)
        return (
            (cast("str", row["payload_json"]), cast("int", row["version"]))
            if row
            else None
        )

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
            updated_at=time.time(),
            pending_emissions=pending_emissions,
        )
        if publish_attempts is not None:
            row["publish_attempts"] = publish_attempts
        return 1

    async def recover_stale_rows(self, ttl_seconds: int | None = None) -> int:
        """The pre-fix sweep, reproduced faithfully: flips the row, tells nobody."""
        self.stale_sweeps += 1
        ttl = 900 if ttl_seconds is None else ttl_seconds
        now = time.time()
        recovered = 0
        for row in self.rows.values():
            if (
                row["state"] not in ("COMPLETED", "FAILED")
                and row["in_flight"]
                and not row.get("pending_emissions")
                and row["updated_at"] < now - ttl
            ):
                row["state"] = "FAILED"
                row["in_flight"] = False
                row["version"] += 1
                recovered += 1
        return recovered

    async def select_recoverable_batches(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.rows.values()
            if row["in_flight"] and row.get("pending_emissions")
        ]

    async def select_abandoned_rows(self, ttl_seconds: int) -> list[dict[str, Any]]:
        now = time.time()
        return [
            dict(row)
            for row in self.rows.values()
            if row["state"] not in ("COMPLETED", "FAILED")
            and row["in_flight"]
            and not row.get("pending_emissions")
            and row["updated_at"] < now - ttl_seconds
        ]


class _BoundCodec:
    """Codec whose ``build_abandoned_terminal`` mirrors the omnimarket one."""

    def flush(self, cid: str) -> str | None:
        current = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        entry = current.get(cid)
        return entry[0] if entry is not None else None

    def build_abandoned_terminal(
        self,
        *,
        correlation_id: str,
        tenant_id: str,
        state: str,
        payload_json: str,
        failure_class: str,
        failure_code: str | None,
        max_wall_seconds: int,
    ) -> tuple[str, str, dict[str, object]] | None:
        camel = "".join(part.capitalize() for part in failure_class.split("_"))
        return (
            __name__,
            "ModelDelegationFailed",
            {
                "correlation_id": correlation_id,
                "tenant_id": tenant_id,
                "failure_reason": (
                    f"abandoned in {state} past the {max_wall_seconds}s bound"
                ),
                "terminal_failure_reason": f"{camel}Error: {failure_code}",
            },
        )


class _RecordingBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, ModelEventEnvelope[Any]]] = []

    async def publish_envelope(
        self,
        *,
        envelope: ModelEventEnvelope[Any],
        topic: str,
        key: bytes | None = None,
    ) -> None:
        self.published.append((topic, envelope))


class _NoopHandler:
    async def handle(self, envelope: object) -> Any:
        cid = CID_DISPATCH
        CONTEXTVAR_STATE_IO_ROWS.set(
            {cid: (json.dumps({"tenant_id": TENANT, "state": "COMPLETED"}), 0)}
        )
        return None


def _abandoned_row(updated_at: float) -> dict[str, dict[str, Any]]:
    return {
        CID_LIVE: {
            "correlation_id": CID_LIVE,
            "tenant_id": TENANT,
            "state": "ROUTED",
            "in_flight": True,
            "payload_json": json.dumps({"state": "ROUTED"}),
            "version": 4,
            "pending_emissions": None,
            "publish_attempts": 0,
            "updated_at": updated_at,
        }
    }


def _callback(
    adapter: _FakeStateStoreAdapter,
    bus: _RecordingBus,
    *,
    completion_bound: ModelCompletionBound | None,
) -> Any:
    with (
        patch.dict(
            "os.environ",
            {"OMNIBASE_INFRA_DB_URL": "postgresql://user:pw@host:5432/db"},
        ),
        patch(_PATCH_IMPORT, return_value=_BoundCodec),
        patch(_PATCH_ADAPTER, return_value=adapter),
    ):
        return _make_stateful_dispatch_callback(
            cast("Any", _NoopHandler()),
            None,
            dict(STATE_IO),
            event_bus=cast("Any", bus),
            output_topic_map=dict(OUTPUT_TOPIC_MAP),
            completion_bound=completion_bound,
        )


def _fast_sweeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Collapse the sweeper's cadence so a test observes it in fractions of a second.

    Only the CADENCE is shortened. The bound itself stays the contract's real
    900s and the fixture rows really are older than it, so what the tests assert
    is the production predicate, not a scaled-down one.
    """
    monkeypatch.setattr(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._BOUND_SWEEP_DIVISOR",
        100_000.0,
    )
    monkeypatch.setattr(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._BOUND_SWEEP_MIN_INTERVAL_SECONDS",
        0.05,
    )


def _dispatch_envelope() -> ModelEventEnvelope[object]:
    return ModelEventEnvelope[object](
        envelope_id=UUID("22222222-2222-2222-2222-222222222222"),
        correlation_id=UUID(CID_DISPATCH),
        payload={"correlation_id": CID_DISPATCH, "tenant_id": TENANT},
    )


# --------------------------------------------------------------------------
# The contract field itself.
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_completion_bound_is_a_typed_contract_field_not_a_loose_number(
    tmp_path: Path,
) -> None:
    """AC2: the bound is read as a typed model, and a bad one fails at wiring."""
    good = tmp_path / "good.yaml"
    good.write_text(
        "name: n\n"
        "completion_bound:\n"
        "  max_wall_seconds: 900\n"
        '  on_runtime_restart: "terminalise_failed"\n'
        '  failure_class: "runtime_restart_during_delegation"\n'
        '  failure_code: "ONEX_MARKET_DELEGATION_RUNTIME_RESTART"\n'
    )
    bound = _read_completion_bound(good)
    assert bound is not None
    assert bound.max_wall_seconds == 900
    assert bound.on_runtime_restart is EnumRuntimeRestartPolicy.TERMINALISE_FAILED
    assert bound.failure_class == "runtime_restart_during_delegation"

    absent = tmp_path / "absent.yaml"
    absent.write_text("name: n\n")
    assert _read_completion_bound(absent) is None

    for body in (
        "completion_bound:\n  max_wall_seconds: 0\n  on_runtime_restart: terminalise_failed\n  failure_class: x\n",
        "completion_bound:\n  max_wall_seconds: 900\n  on_runtime_restart: shrug\n  failure_class: x\n",
        "completion_bound:\n  max_wall_seconds: 900\n  on_runtime_restart: terminalise_failed\n  failure_class: x\n  failure_code: not-canonical\n",
        "completion_bound: 900\n",
    ):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: n\n" + body)
        with pytest.raises(ModelOnexError):
            _read_completion_bound(bad)


# --------------------------------------------------------------------------
# The two behavioural REDs.
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_row_past_the_bound_emits_a_real_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC3: the give-up travels on the bus; it is not a silent DB update.

    RED against pre-OMN-18296 wiring: the row ends up ``FAILED`` either way, so
    asserting the row alone proves nothing. What was missing — and what this
    asserts — is the published envelope that the gateway's terminal consumer,
    the projections and the receipt all read.
    """
    _fast_sweeps(monkeypatch)
    adapter = _FakeStateStoreAdapter(_abandoned_row(time.time() - 1000))
    bus = _RecordingBus()
    callback = _callback(adapter, bus, completion_bound=BOUND)

    async def _drive() -> None:
        await callback(_dispatch_envelope())
        await asyncio.sleep(0.4)

    asyncio.run(_drive())

    terminals = [
        (topic, env)
        for topic, env in bus.published
        if str(env.correlation_id) == CID_LIVE
    ]
    assert terminals, (
        "the abandoned row was given up on with no terminal event — a state "
        "change no consumer can observe, which is exactly how a gateway "
        "workflow stays 'published' forever"
    )
    topic, envelope = terminals[0]
    assert topic == TOPIC_DELEGATION_FAILED
    payload = cast("ModelDelegationFailed", envelope.payload)
    assert payload.terminal_failure_reason == (
        "RuntimeRestartDuringDelegationError: ONEX_MARKET_DELEGATION_RUNTIME_RESTART"
    )
    assert "900s bound" in payload.failure_reason
    row = adapter.rows[CID_LIVE]
    assert row["state"] == "FAILED"
    assert row["in_flight"] is False


@pytest.mark.unit
def test_the_bound_is_swept_without_any_further_dispatch_traffic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC1/AC3: the lane that lost the workflow has no traffic left to trigger a sweep.

    One dispatch happens while the row is still INSIDE its bound, so the sweep
    that runs during that dispatch correctly leaves it alone — which is what
    really happened on the lab, where a delegation completed at 10:01 six
    minutes into the abandoned row's life and swept nothing. The row then ages
    past the bound with NO further dispatch. Pre-fix, nothing ever looks at it
    again.
    """
    _fast_sweeps(monkeypatch)
    rows = _abandoned_row(time.time())  # fresh: inside the bound
    adapter = _FakeStateStoreAdapter(rows)
    bus = _RecordingBus()
    callback = _callback(adapter, bus, completion_bound=BOUND)

    async def _drive() -> None:
        await callback(_dispatch_envelope())
        assert not [
            env for _t, env in bus.published if str(env.correlation_id) == CID_LIVE
        ], "a row inside its bound must not be terminalised"
        # The lane goes quiet. Age the row past the bound; no dispatch follows.
        rows[CID_LIVE]["updated_at"] = time.time() - 1000
        await asyncio.sleep(0.4)

    asyncio.run(_drive())

    assert [env for _t, env in bus.published if str(env.correlation_id) == CID_LIVE], (
        "the row aged past its bound on an idle lane and nothing terminalised "
        "it — the sweep is reachable only from inside a dispatch that will "
        "never come"
    )


@pytest.mark.unit
def test_a_contract_with_no_bound_keeps_the_pre_existing_behaviour() -> None:
    """Additive: a contract that declares no bound publishes no terminal."""
    adapter = _FakeStateStoreAdapter(_abandoned_row(time.time() - 1000))
    bus = _RecordingBus()
    callback = _callback(adapter, bus, completion_bound=None)

    async def _drive() -> None:
        await callback(_dispatch_envelope())
        await asyncio.sleep(0.2)

    asyncio.run(_drive())

    assert not [env for _t, env in bus.published if str(env.correlation_id) == CID_LIVE]
    assert adapter.stale_sweeps >= 1, (
        "positive control: the pre-existing row-only sweep must still run, so "
        "the assertion above is about the terminal event and not about a "
        "harness that never swept at all"
    )
