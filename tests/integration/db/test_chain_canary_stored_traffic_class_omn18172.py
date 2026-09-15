# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18172 AC1: the canary's STORED delegation row reads ``traffic_class=synthetic``.

Binding shape (Jonah Gray, OMN-18172 comment 6509da3c, 2026-09-15): a behaviour
check that drives the chain-canary ingress through the handler to terminal
persistence with a real correlation id and asserts ``traffic_class=synthetic``
on the stored row. A run summary or a live dev-lane readback cannot bind.

What runs for real
------------------
* ``HandlerChainCanary.handle()`` builds the ``/skill`` body, including the
  governed ``ModelDelegationProvenance``, and afterwards performs its own link-2
  readback through ``_readback_projection_via_asyncpg``, the production function.
  That readback refuses a SUPERUSER/BYPASSRLS identity, so it connects as a
  throwaway role holding exactly the ``chain_canary_reader`` column grant that
  ``scripts/run-forward-migrations.sh`` (LOGIN_ONLY_ROLE_GRANT_MAP) issues:
  ``correlation_id, state, traffic_class`` and not ``payload``.
* The captured body becomes core's canonical ``ModelDelegationRequest`` (the
  constructor call omnimarket's runtime dispatch port makes), which is folded
  through the real state_io dispatch seam (``_make_stateful_dispatch_callback``)
  into a real ``StateStoreAdapter``: seed at RECEIVED, then a CAS update to
  COMPLETED. Postgres has the forward migrations applied, so the value read back
  is migration 106's STORED generated column, not a value the test wrote.

What does not run, so the green is not over-read
------------------------------------------------
omnimarket is not a dependency of this repository, so
``node_delegation_orchestrator``'s ``DelegationWorkflowState`` and its
``StateIoCodec`` cannot execute here. ``_DelegationWorkflowFold`` stands in for
that leg and places the request at the top-level ``request`` key, which is where
the codec's TypeAdapter dump puts that dataclass field and the path migration
106 reads. A rename on the omnimarket side is outside what this test can see.

Environment
-----------
``OMNIBASE_INFRA_DB_URL`` pointing at a database migrated with
``scripts/run-migrations.py`` (the environment ci.yml's ``integration-guard``
job builds), as a role that may CREATE ROLE. Skips when Postgres is unset or
unreachable, like the sibling real-DB proofs. A reachable database WITHOUT
migration 106 fails. With ``OMN18172_REQUIRE_PG=1`` (set by that CI step and by
the evidence check, Jonah OMN-18172 comment c02674f2) an unset or unreachable
Postgres FAILS instead of skipping: a runner that reads only the exit status
would otherwise take a skip for a pass that asserted nothing.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID, uuid4

import asyncpg
import pytest

from omnibase_core.models.delegation.wire import (
    ModelDelegationProvenance,
    ModelDelegationRequest,
)
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.generated.enum_omnimarket_topic import EnumOmnimarketTopic
from omnibase_infra.nodes.node_chain_canary_effect.handlers import (
    handler_chain_canary,
)
from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link import (
    EnumChainLink,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link_status import (
    EnumChainLinkStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_result import (
    ModelChainCanaryResult,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _make_stateful_dispatch_callback,
)
from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)
from omnibase_infra.runtime.providers.provider_postgres_pool import (
    ProviderPostgresPool,
)
from omnibase_infra.runtime.state_io.state_store_adapter import (
    CONTEXTVAR_STATE_IO_ROWS,
    StateStoreAdapter,
)
from tests.helpers.util_postgres import PostgresConfig, check_postgres_reachable

_postgres_config = PostgresConfig.from_env()

_REQUIRE_PG_ENV = "OMN18172_REQUIRE_PG"
REQUIRE_PG = os.environ.get(_REQUIRE_PG_ENV) == "1"
_POSTGRES_UNAVAILABLE_REASON = (
    "PostgreSQL not available (set OMNIBASE_INFRA_DB_URL to a database "
    "migrated with scripts/run-migrations.py)"
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.postgres,
]


def _postgres_available() -> bool:
    return _postgres_config.is_configured and check_postgres_reachable(
        _postgres_config,
        timeout=5.0,
    )


def _safe_to_mutate_postgres() -> bool:
    return os.environ.get("GITHUB_ACTIONS") == "true" and _postgres_config.host in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


@pytest.fixture(autouse=True)
def _postgres_required_when_flagged() -> None:
    if not _postgres_available():
        message = (
            f"{_POSTGRES_UNAVAILABLE_REASON}: OMNIBASE_INFRA_DB_URL is unset, "
            "malformed, or unreachable."
        )
        if REQUIRE_PG:
            pytest.fail(
                f"{_REQUIRE_PG_ENV}=1 but {message} This proof fails closed "
                "rather than skipping, so a Postgres-absent run cannot read as a pass."
            )
        pytest.skip(message)
    if not _safe_to_mutate_postgres():
        message = (
            "OMN-18172 mutates delegation_workflow_state and creates a temporary "
            "role; it may run only in GitHub Actions against loopback Postgres."
        )
        if REQUIRE_PG:
            pytest.fail(f"{_REQUIRE_PG_ENV}=1 but {message}")
        pytest.skip(message)


@pytest.fixture
def probe_ids() -> tuple[UUID, UUID, UUID]:
    return uuid4(), uuid4(), uuid4()


def _only_state_row(
    rows: dict[str, tuple[str | None, int]],
) -> tuple[str, str | None, int]:
    assert len(rows) == 1, (
        "state_io seam must expose exactly one correlation row to the fold; "
        f"got {len(rows)} keys: {sorted(rows)}"
    )
    key, (prior_payload_json, version) = next(iter(rows.items()))
    return key, prior_payload_json, version


_TABLE = "delegation_workflow_state"
# scripts/run-forward-migrations.sh LOGIN_ONLY_ROLE_GRANT_MAP, chain_canary_reader.
_READER_GRANT_COLUMNS = "correlation_id, state, traffic_class"
_PROJECTION_DSN_ENV = "OMN18172_CANARY_READER_DSN"
_SUCCESS_TOPIC = EnumOmnimarketTopic.EVT_DELEGATE_SKILL_COMPLETED_V1.value
_FULL_CHAIN = ("received", "routed", "inference_completed", "terminal")

# The delegation orchestrator's state_io declaration (omnimarket contract.yaml).
_STATE_IO: dict[str, object] = {
    "database": "omnibase_infra",
    "table": _TABLE,
    "key": "correlation_id",
    "codec": {
        "module": "omnimarket.nodes.node_delegation_orchestrator.state_codec",
        "name": "StateIoCodec",
    },
}
_PATCH_IMPORT = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)
_PATCH_ADAPTER = "omnibase_infra.runtime.auto_wiring.handler_wiring.StateStoreAdapter"


class _FlushBoundRowCodec:
    """state_io codec: hands the row the leg bound back to the seam."""

    def flush(self, key: str) -> str | None:
        current = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        entry = current.get(key)
        return entry[0] if entry is not None else None


class _DelegationWorkflowFold:
    """Stand-in for node_delegation_orchestrator's legs (see module docstring).

    First leg for a correlation: accept the wire ``ModelDelegationRequest`` and
    store it unchanged under ``request``. Any later leg: advance the persisted
    workflow to its terminal. No I/O of its own; the seam loads and persists.
    """

    async def handle(self, envelope: object) -> ModelHandlerOutput[None]:
        rows = CONTEXTVAR_STATE_IO_ROWS.get() or {}
        key, prior_payload_json, version = _only_state_row(rows)
        if prior_payload_json is None:
            request = ModelDelegationRequest.model_validate(
                getattr(envelope, "payload", None)
            )
            workflow: dict[str, object] = {
                "correlation_id": str(request.correlation_id),
                "tenant_id": request.tenant_id or "",
                "request": request.model_dump(mode="json"),
                "state": "RECEIVED",
            }
        else:
            workflow = json.loads(prior_payload_json)
            workflow["state"] = "COMPLETED"
        workflow["in_flight"] = False
        CONTEXTVAR_STATE_IO_ROWS.set({key: (json.dumps(workflow), version)})
        return ModelHandlerOutput.for_orchestrator(
            input_envelope_id=cast("UUID", getattr(envelope, "envelope_id", uuid4())),
            correlation_id=UUID(key),
            handler_id="omn18172-delegation-workflow-fold",
            events=(),
        )


def _delegation_request_from_skill_body(
    body: dict[str, object],
) -> ModelDelegationRequest:
    """The ``ModelDelegationRequest`` the runtime dispatch port builds from /skill."""
    payload = cast("dict[str, Any]", body["payload"])
    raw_provenance = payload.get("provenance")
    return ModelDelegationRequest(
        prompt=payload["prompt"],
        task_type=payload["task_type"],
        correlation_id=UUID(payload["correlation_id"]),
        max_tokens=payload["max_tokens"],
        emitted_at=datetime.now(UTC),
        provenance=(
            None
            if raw_provenance is None
            else ModelDelegationProvenance.model_validate(raw_provenance)
        ),
    )


class _RuntimeIngress:
    """POST /skill plus the delegation chain behind it, persisting to Postgres."""

    def __init__(self, adapter: StateStoreAdapter) -> None:
        self.bodies: list[dict[str, object]] = []
        with (
            patch(_PATCH_IMPORT, return_value=_FlushBoundRowCodec),
            patch(_PATCH_ADAPTER, return_value=adapter),
        ):
            self._dispatch = _make_stateful_dispatch_callback(
                cast("Any", _DelegationWorkflowFold()),
                None,
                dict(_STATE_IO),
                event_bus=None,
                output_topic_map=None,
            )

    async def __call__(
        self, url: str, body: dict[str, object], timeout_s: float
    ) -> tuple[dict[str, object], str, int]:
        self.bodies.append(body)
        request = _delegation_request_from_skill_body(body)
        legs: tuple[object, ...] = (
            request.model_dump(mode="json"),
            {"correlation_id": str(request.correlation_id)},
        )
        for leg_payload in legs:
            await self._dispatch(
                ModelEventEnvelope[object](
                    correlation_id=request.correlation_id,
                    payload=leg_payload,
                )
            )
        return (
            {
                "ok": True,
                "command_name": "node_delegate_skill_orchestrator",
                "terminal_event": "omnimarket.delegate-skill-completed",
                "output_payloads": [{"status": "completed"}],
            },
            "",
            42,
        )


async def _quarantine_clean(
    bootstrap: str, topic: str, correlation_id: str, max_records: int, timeout_s: float
) -> tuple[bool, int, str]:
    return False, 0, ""


async def _terminal_present(
    bootstrap: str,
    topics: tuple[str, ...],
    correlation_id: str,
    max_records: int,
    timeout_s: float,
) -> tuple[str, int, str]:
    return _SUCCESS_TOPIC, 1, ""


async def _ledger_verified(
    source: str, correlation_id: str, timeout_s: float
) -> tuple[tuple[str, ...], bool, str, str]:
    return _FULL_CHAIN, True, "pass", ""


async def _run_canary(
    monkeypatch: pytest.MonkeyPatch,
    ingress: _RuntimeIngress,
    reader_dsn: str,
    probe_correlation_id: UUID,
    run_correlation_id: UUID,
) -> ModelChainCanaryResult:
    # The handler mints its probe id with uuid4(); pin it to a known real UUID.
    monkeypatch.setattr(handler_chain_canary, "uuid4", lambda: probe_correlation_id)
    handler = HandlerChainCanary(
        ingress=ingress,
        quarantine_scan=_quarantine_clean,
        terminal_readback=_terminal_present,
        projection_dsn_lookup=lambda name: (
            reader_dsn if name == _PROJECTION_DSN_ENV else ""
        ),
        ledger_replay=_ledger_verified,
        ledger_dsn_lookup=lambda name: "postgresql://ledger.invalid/omnibase_infra",
        kill_switch_disabled=False,
    )
    return await handler.handle(
        ModelChainCanaryRequest(
            correlation_id=run_correlation_id,
            probe_url="http://runtime.invalid:8085",
            budget_ms=5_000,
            terminal_bootstrap_servers="broker.invalid:19092",
            projection_dsn_env=_PROJECTION_DSN_ENV,
            ledger_source_env="OMN18172_LEDGER_DSN",
            expected_ledger_hops=_FULL_CHAIN,
        )
    )


def _link_two(result: ModelChainCanaryResult) -> tuple[EnumChainLinkStatus, str]:
    (verdict,) = (
        v for v in result.link_verdicts if v.link is EnumChainLink.ROUTING_PROJECTED
    )
    return verdict.status, verdict.detail


@pytest.fixture
async def admin_connection(
    probe_ids: tuple[UUID, UUID, UUID],
) -> AsyncIterator[asyncpg.Connection]:
    connection = await asyncpg.connect(_postgres_config.build_dsn(), timeout=10.0)
    try:
        generated = await connection.fetchval(
            "SELECT is_generated FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = $1 "
            "AND column_name = 'traffic_class'",
            _TABLE,
        )
        if generated != "ALWAYS":
            pytest.fail(
                f"{_TABLE}.traffic_class is not a generated column "
                f"(is_generated={generated!r}): apply the forward migrations "
                "through 106 with scripts/run-migrations.py first"
            )
        canary_cid, control_cid, _ = probe_ids
        test_keys = [str(canary_cid), str(control_cid)]
        await connection.execute(
            f"DELETE FROM {_TABLE} WHERE correlation_id = ANY($1::text[])",  # noqa: S608 - module constant
            test_keys,
        )
        yield connection
        await connection.execute(
            f"DELETE FROM {_TABLE} WHERE correlation_id = ANY($1::text[])",  # noqa: S608 - module constant
            test_keys,
        )
    finally:
        await connection.close()


@pytest.fixture
async def canary_reader_dsn(
    admin_connection: asyncpg.Connection,
) -> AsyncIterator[str]:
    """A least-privilege LOGIN role with the chain_canary_reader column grant."""
    role = f"omn18172_canary_reader_{secrets.token_hex(4)}"
    password = secrets.token_hex(24)
    await admin_connection.execute(
        f"CREATE ROLE {role} WITH LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB "
        f"NOCREATEROLE NOREPLICATION PASSWORD '{password}'"
    )
    try:
        database = await admin_connection.fetchval("SELECT current_database()")
        await admin_connection.execute(
            f'GRANT CONNECT ON DATABASE "{database}" TO {role}'
        )
        await admin_connection.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        await admin_connection.execute(
            f"GRANT SELECT ({_READER_GRANT_COLUMNS}) ON public.{_TABLE} TO {role}"
        )
        yield PostgresConfig(
            host=_postgres_config.host,
            port=_postgres_config.port,
            database=_postgres_config.database,
            user=role,
            password=password,
        ).build_dsn()
    finally:
        # DROP OWNED also revokes the role's database-level and column grants.
        await admin_connection.execute(f"DROP OWNED BY {role}")
        await admin_connection.execute(f"DROP ROLE IF EXISTS {role}")


@pytest.fixture
async def state_adapter() -> AsyncIterator[StateStoreAdapter]:
    dsn = _postgres_config.build_dsn()
    provider = ProviderPostgresPool(
        ModelPostgresPoolConfig.from_dsn(dsn, min_size=1, max_size=5)
    )
    adapter = StateStoreAdapter(dsn, table=_TABLE, pool_factory=provider.create)
    try:
        yield adapter
    finally:
        await adapter.close()


async def _read_as_canary_reader(reader_dsn: str, cid: UUID) -> asyncpg.Record | None:
    connection = await asyncpg.connect(reader_dsn, timeout=10.0)
    try:
        return await connection.fetchrow(
            f"SELECT state, traffic_class FROM {_TABLE} WHERE correlation_id = $1",  # noqa: S608 - module constant
            str(cid),
        )
    finally:
        await connection.close()


async def test_canary_delegation_terminal_row_stores_traffic_class_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    probe_ids: tuple[UUID, UUID, UUID],
    admin_connection: asyncpg.Connection,
    canary_reader_dsn: str,
    state_adapter: StateStoreAdapter,
) -> None:
    canary_cid, _, run_cid = probe_ids
    ingress = _RuntimeIngress(state_adapter)

    result = await _run_canary(
        monkeypatch,
        ingress,
        canary_reader_dsn,
        canary_cid,
        run_cid,
    )

    # One correlation id, traced: minted by the handler, on the wire, on the row.
    assert result.probe_correlation_id == canary_cid
    (body,) = ingress.bodies
    assert body["correlation_id"] == str(canary_cid)
    wire_payload = cast("dict[str, Any]", body["payload"])
    assert wire_payload["provenance"]["traffic_class"] == "synthetic"

    # The seam really seeded and then CAS-advanced the row to a terminal.
    stored = await admin_connection.fetchrow(
        f"SELECT state, version FROM {_TABLE} WHERE correlation_id = $1",  # noqa: S608 - module constant
        str(canary_cid),
    )
    assert stored is not None
    assert (stored["state"], stored["version"]) == ("COMPLETED", 1)

    # The canary's own link-2 readback, as the least-privilege reader.
    assert result.projection_traffic_class == "synthetic"
    status, detail = _link_two(result)
    assert status is EnumChainLinkStatus.PASS, detail
    assert "traffic_class=synthetic" in detail

    # And the column itself, read with only the column-scoped grant.
    row = await _read_as_canary_reader(canary_reader_dsn, canary_cid)
    assert row is not None
    assert (row["state"], row["traffic_class"]) == ("COMPLETED", "synthetic")


async def test_request_without_provenance_stores_unclassified_never_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    probe_ids: tuple[UUID, UUID, UUID],
    admin_connection: asyncpg.Connection,
    canary_reader_dsn: str,
    state_adapter: StateStoreAdapter,
) -> None:
    """Negative control: the same path with provenance omitted from the body."""
    _, control_cid, run_cid = probe_ids
    build_body = HandlerChainCanary._build_body

    def _build_body_without_provenance(
        request: ModelChainCanaryRequest, probe_correlation_id: str
    ) -> dict[str, object]:
        body = build_body(request, probe_correlation_id)
        payload = dict(cast("dict[str, object]", body["payload"]))
        del payload["provenance"]
        return {**body, "payload": payload}

    monkeypatch.setattr(
        HandlerChainCanary,
        "_build_body",
        staticmethod(_build_body_without_provenance),
    )
    ingress = _RuntimeIngress(state_adapter)

    result = await _run_canary(
        monkeypatch,
        ingress,
        canary_reader_dsn,
        control_cid,
        run_cid,
    )

    (body,) = ingress.bodies
    assert "provenance" not in cast("dict[str, object]", body["payload"])
    assert result.projection_traffic_class == "unclassified"
    status, detail = _link_two(result)
    assert status is EnumChainLinkStatus.PASS, detail
    assert "traffic_class=unclassified" in detail

    row = await _read_as_canary_reader(canary_reader_dsn, control_cid)
    assert row is not None
    assert (row["state"], row["traffic_class"]) == ("COMPLETED", "unclassified")
