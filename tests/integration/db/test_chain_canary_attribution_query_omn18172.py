# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18172 AC4: the attribution query partitions canary from non-canary traffic.

AC1 proved the STORED generated column ``traffic_class`` stores correctly per
row. This test defines the **attribution query** — a ``SELECT … GROUP BY
traffic_class`` — and proves it returns the right partition when a synthetic
(canary) row and an unclassified (non-canary) row coexist.

Binding shape (Jonah Gray, OMN-18172 ruling 2ee54366, 2026-09-15): AC4 needs
no ORGANIC row. Non-canary = canary's own body with provenance omitted. The
attribution query must be defined in the AC4 PR. Bind like AC1.

Environment
-----------
Same as AC1 — ``OMNIBASE_INFRA_DB_URL`` pointing at a loopback-migrated
Postgres, ``OMN18172_REQUIRE_PG=1`` / ``OMN18172_ALLOW_LAB_PG=1`` semantics.
See the docstring in ``test_chain_canary_stored_traffic_class_omn18172`` for
the full guard logic.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import patch
from uuid import UUID, uuid4

import asyncpg
import pytest

from omnibase_infra.nodes.node_chain_canary_effect.handlers.handler_chain_canary import (
    HandlerChainCanary,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_canary_request import (
    ModelChainCanaryRequest,
)
from omnibase_infra.runtime.models.model_postgres_pool_config import (
    ModelPostgresPoolConfig,
)
from omnibase_infra.runtime.providers.provider_postgres_pool import (
    ProviderPostgresPool,
)
from omnibase_infra.runtime.state_io.state_store_adapter import StateStoreAdapter
from tests.helpers.util_postgres import PostgresConfig, check_postgres_reachable
from tests.integration.db.test_chain_canary_stored_traffic_class_omn18172 import (
    _FULL_CHAIN,
    _PROJECTION_DSN_ENV,
    _READER_GRANT_COLUMNS,
    _TABLE,
    _ledger_verified,
    _quarantine_clean,
    _RuntimeIngress,
    _terminal_present,
)

_postgres_config = PostgresConfig.from_env()

_REQUIRE_PG_ENV = "OMN18172_REQUIRE_PG"
REQUIRE_PG = os.environ.get(_REQUIRE_PG_ENV) == "1"
_ALLOW_LAB_PG_ENV = "OMN18172_ALLOW_LAB_PG"
ALLOW_LAB_PG = os.environ.get(_ALLOW_LAB_PG_ENV) == "1"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_POSTGRES_UNAVAILABLE_REASON = (
    "PostgreSQL not available (set OMNIBASE_INFRA_DB_URL to a database "
    "migrated with scripts/run-migrations.py)"
)

# ---------------------------------------------------------------------------
# Attribution query — the deliverable AC4 defines.
#
# Groups delegation_workflow_state rows by traffic_class and returns
# (traffic_class, row_count) tuples.  With a correlation-id scope the query
# counts only the rows under test; without it an operator can run the same
# query against the full table to see the canary-vs-real traffic split.
# ---------------------------------------------------------------------------
ATTRIBUTION_QUERY = """\
SELECT traffic_class,
       COUNT(*) AS row_count
  FROM delegation_workflow_state
 WHERE correlation_id = ANY($1::text[])
 GROUP BY traffic_class
 ORDER BY traffic_class
"""

pytestmark = [
    pytest.mark.integration,
    pytest.mark.postgres,
]


def _postgres_available() -> bool:
    return _postgres_config.is_configured and check_postgres_reachable(
        _postgres_config,
        timeout=5.0,
    )


def _postgres_host_is_loopback() -> bool:
    return _postgres_config.host in _LOOPBACK_HOSTS


def _safe_to_mutate_postgres() -> bool:
    in_actions = os.environ.get("GITHUB_ACTIONS") == "true"
    return (in_actions or ALLOW_LAB_PG) and _postgres_host_is_loopback()


@pytest.fixture(autouse=True)
def _postgres_required_when_flagged() -> None:
    if ALLOW_LAB_PG and not _postgres_host_is_loopback():
        pytest.fail(
            f"{_ALLOW_LAB_PG_ENV}=1 permits a non-Actions run only against loopback "
            f"Postgres; OMNIBASE_INFRA_DB_URL host {_postgres_config.host!r} is not "
            "loopback. Refusing to mutate delegation_workflow_state or create a role."
        )
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
            "role; it may run only against loopback Postgres, in GitHub Actions or "
            f"with the explicit lab opt-in {_ALLOW_LAB_PG_ENV}=1."
        )
        pytest.skip(message)


def _quote_ident(value: str) -> str:
    escaped = value.replace('"', '""')
    return f'"{escaped}"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


@pytest.fixture
def created_correlation_ids() -> list[UUID]:
    return []


@pytest.fixture
async def admin_connection(
    created_correlation_ids: list[UUID],
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
        yield connection
        if created_correlation_ids:
            test_keys = [str(cid) for cid in created_correlation_ids]
            await connection.execute(
                f"DELETE FROM {_TABLE} WHERE correlation_id = ANY($1::text[])",  # noqa: S608 - module constant
                test_keys,
            )
    finally:
        await connection.close()


@pytest.fixture
async def canary_reader_dsn(
    admin_connection: asyncpg.Connection,
) -> AsyncIterator[tuple[str, str]]:
    """A least-privilege LOGIN role with the chain_canary_reader column grant."""
    role = f"omn18172_ac4_reader_{secrets.token_hex(4)}"
    role_sql = _quote_ident(role)
    password = secrets.token_hex(24)
    await admin_connection.execute(
        f"CREATE ROLE {role_sql} WITH LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB "
        f"NOCREATEROLE NOREPLICATION PASSWORD {_quote_literal(password)}"
    )
    try:
        database = cast(
            "str",
            await admin_connection.fetchval("SELECT current_database()"),
        )
        await admin_connection.execute(
            f"GRANT CONNECT ON DATABASE {_quote_ident(database)} TO {role_sql}"
        )
        await admin_connection.execute(f"GRANT USAGE ON SCHEMA public TO {role_sql}")
        await admin_connection.execute(
            f"GRANT SELECT ({_READER_GRANT_COLUMNS}) ON public.{_TABLE} TO {role_sql}"
        )
        dsn = PostgresConfig(
            host=_postgres_config.host,
            port=_postgres_config.port,
            database=_postgres_config.database,
            user=role,
            password=password,
        ).build_dsn()
        yield dsn, password
    finally:
        cleanup_errors: list[str] = []
        try:
            await admin_connection.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE usename = $1 AND pid <> pg_backend_pid()",
                role,
            )
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"terminate backends: {exc}")
        try:
            await admin_connection.execute(f"DROP OWNED BY {role_sql}")
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"drop owned: {exc}")
        try:
            await admin_connection.execute(f"DROP ROLE IF EXISTS {role_sql}")
        except Exception as exc:  # noqa: BLE001
            cleanup_errors.append(f"drop role: {exc}")
        if cleanup_errors and __import__("sys").exception() is None:
            pytest.fail("temporary reader cleanup failed: " + "; ".join(cleanup_errors))


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


async def _run_canary_delegation(
    ingress: _RuntimeIngress,
    reader_dsn: str,
    run_correlation_id: UUID,
    created_correlation_ids: list[UUID],
) -> UUID:
    """Run a canary through the handler and return the probe correlation id."""
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
    request = ModelChainCanaryRequest(
        correlation_id=run_correlation_id,
        probe_url="http://runtime.invalid:8085",
        budget_ms=5_000,
        terminal_bootstrap_servers="broker.invalid:19092",
        projection_dsn_env=_PROJECTION_DSN_ENV,
        ledger_source_env="OMN18172_LEDGER_DSN",
        expected_ledger_hops=_FULL_CHAIN,
    )
    result = await handler.handle(request)
    created_correlation_ids.extend((result.probe_correlation_id, run_correlation_id))
    return result.probe_correlation_id


async def _run_noncanary_delegation(
    ingress: _RuntimeIngress,
    reader_dsn: str,
    run_correlation_id: UUID,
    created_correlation_ids: list[UUID],
) -> UUID:
    """Run a non-canary (provenance stripped) and return its probe correlation id."""
    build_body = HandlerChainCanary._build_body

    def _build_body_without_provenance(
        request: ModelChainCanaryRequest, probe_correlation_id: str
    ) -> dict[str, object]:
        body = build_body(request, probe_correlation_id)
        payload = dict(cast("dict[str, object]", body["payload"]))
        del payload["provenance"]
        return {**body, "payload": payload}

    with patch.object(
        HandlerChainCanary,
        "_build_body",
        staticmethod(_build_body_without_provenance),
    ):
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
        request = ModelChainCanaryRequest(
            correlation_id=run_correlation_id,
            probe_url="http://runtime.invalid:8085",
            budget_ms=5_000,
            terminal_bootstrap_servers="broker.invalid:19092",
            projection_dsn_env=_PROJECTION_DSN_ENV,
            ledger_source_env="OMN18172_LEDGER_DSN",
            expected_ledger_hops=_FULL_CHAIN,
        )
        result = await handler.handle(request)
        created_correlation_ids.extend(
            (result.probe_correlation_id, run_correlation_id)
        )
        return result.probe_correlation_id


async def test_attribution_query_partitions_synthetic_from_unclassified(
    created_correlation_ids: list[UUID],
    admin_connection: asyncpg.Connection,
    canary_reader_dsn: tuple[str, str],
    state_adapter: StateStoreAdapter,
) -> None:
    """The attribution query returns one row per traffic_class with correct counts."""
    reader_dsn, _ = canary_reader_dsn

    canary_ingress = _RuntimeIngress(state_adapter)
    canary_cid = await _run_canary_delegation(
        canary_ingress, reader_dsn, uuid4(), created_correlation_ids
    )

    noncanary_ingress = _RuntimeIngress(state_adapter)
    noncanary_cid = await _run_noncanary_delegation(
        noncanary_ingress, reader_dsn, uuid4(), created_correlation_ids
    )

    scoped_ids = [str(canary_cid), str(noncanary_cid)]
    rows = await admin_connection.fetch(ATTRIBUTION_QUERY, scoped_ids)
    result = {r["traffic_class"]: r["row_count"] for r in rows}

    assert result == {"synthetic": 1, "unclassified": 1}, (
        f"attribution query must partition exactly one synthetic and one "
        f"unclassified row; got {result}"
    )


async def test_attribution_query_counts_multiple_rows_per_class(
    created_correlation_ids: list[UUID],
    admin_connection: asyncpg.Connection,
    canary_reader_dsn: tuple[str, str],
    state_adapter: StateStoreAdapter,
) -> None:
    """Two canaries and one non-canary: counts are 2 and 1."""
    reader_dsn, _ = canary_reader_dsn

    cids: list[str] = []
    for _ in range(2):
        ingress = _RuntimeIngress(state_adapter)
        cid = await _run_canary_delegation(
            ingress, reader_dsn, uuid4(), created_correlation_ids
        )
        cids.append(str(cid))

    ingress = _RuntimeIngress(state_adapter)
    cid = await _run_noncanary_delegation(
        ingress, reader_dsn, uuid4(), created_correlation_ids
    )
    cids.append(str(cid))

    rows = await admin_connection.fetch(ATTRIBUTION_QUERY, cids)
    result = {r["traffic_class"]: r["row_count"] for r in rows}

    assert result == {"synthetic": 2, "unclassified": 1}, (
        f"attribution query must count 2 synthetic and 1 unclassified; got {result}"
    )


async def test_attribution_query_works_through_reader_role(
    created_correlation_ids: list[UUID],
    admin_connection: asyncpg.Connection,
    canary_reader_dsn: tuple[str, str],
    state_adapter: StateStoreAdapter,
) -> None:
    """The least-privilege canary reader can run the attribution query."""
    reader_dsn, _ = canary_reader_dsn

    ingress = _RuntimeIngress(state_adapter)
    canary_cid = await _run_canary_delegation(
        ingress, reader_dsn, uuid4(), created_correlation_ids
    )
    ingress2 = _RuntimeIngress(state_adapter)
    noncanary_cid = await _run_noncanary_delegation(
        ingress2, reader_dsn, uuid4(), created_correlation_ids
    )

    scoped_ids = [str(canary_cid), str(noncanary_cid)]

    reader_conn = await asyncpg.connect(reader_dsn, timeout=10.0)
    try:
        rows = await reader_conn.fetch(ATTRIBUTION_QUERY, scoped_ids)
        result = {r["traffic_class"]: r["row_count"] for r in rows}
        assert result == {"synthetic": 1, "unclassified": 1}
    finally:
        await reader_conn.close()
