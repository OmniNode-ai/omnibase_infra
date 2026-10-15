# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kernel derived writes on the tenant path, using owned real PostgreSQL.

Failures: borrowing the unscoped transaction, losing the GUC before the
derived update, leaking another tenant's row, committing a callback failure,
or refusing a derived write whose caller requests no public row readback.
The callback derives a text identity; monetary computation belongs to the
market caller tests. This is not a proof of a full migration replay.
"""

from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_projection_db_adapter,
)
from tests.helpers.application_db_topology import (
    projection_database_target,
    projection_database_urls,
)
from tests.helpers.projection_tenant_authority import verified_tenant_dispatch
from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.postgres]
_FORWARD = Path(__file__).resolve().parents[3] / "docker/migrations/forward"
_DATABASE = "omnidash_analytics"
_ROLE = "tenant_projection_writer"
_PASSWORD = uuid4().hex


@pytest.fixture
def delegation_pg(ephemeral_postgres: EphemeralPostgres) -> EphemeralPostgres:
    pg = ephemeral_postgres
    conn = pg.connect()
    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE {_DATABASE}")
    finally:
        conn.close()
    for migration in (
        _FORWARD / "103_create_tenant_projection_writer_role.sql",
        _FORWARD / "nodes/node_projection_delegation/0007_delegation_events.sql",
        _FORWARD
        / "nodes/node_projection_delegation/0059_delegation_events_modelled_run_savings.sql",
        _FORWARD
        / "nodes/node_projection_delegation_inference_response/0004_grant_tenant_projection_writer.sql",
    ):
        result = pg.psql(
            "-v", "ON_ERROR_STOP=1", "-f", str(migration), dbname=_DATABASE
        )
        assert result.returncode == 0, result.stderr
    with pg.connect(dbname=_DATABASE) as conn, conn.cursor() as cur:
        # This isolated tenant fixture adds the live UUID/policy boundary;
        # base shape, savings columns, writer role and grants come from SQL.
        cur.execute(
            "ALTER TABLE delegation_events ADD COLUMN tenant_id uuid NOT NULL;"
            "ALTER TABLE delegation_events ENABLE ROW LEVEL SECURITY;"
            "ALTER TABLE delegation_events FORCE ROW LEVEL SECURITY;"
            "CREATE POLICY tenant_isolation ON delegation_events FOR ALL "
            "USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid) "
            "WITH CHECK (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)"
        )
        cur.execute(f"ALTER ROLE {_ROLE} LOGIN PASSWORD %s", (_PASSWORD,))
        cur.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname=%s", (_ROLE,)
        )
        assert cur.fetchone() == (False, False)
        cur.execute(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
            "WHERE oid='public.delegation_events'::regclass"
        )
        assert cur.fetchone() == (True, True)
    return pg


@pytest.mark.parametrize("public_returning", [True, False])
def test_kernel_derivation_stays_in_the_verified_tenant_transaction(
    delegation_pg: EphemeralPostgres,
    public_returning: bool,
) -> None:
    pg = delegation_pg
    tenant, other = uuid4(), uuid4()
    target = projection_database_target("delegation_events", schema="public")
    dsn = (
        f"host={pg.socket_dir} port={pg.port} dbname={_DATABASE} "
        f"user={_ROLE} password={_PASSWORD}"
    )

    def adapter_for(tenant_id):
        authority, event = verified_tenant_dispatch(tenant_id)
        return _build_projection_db_adapter(
            projection_database_urls(target, dsn), target, authority, event
        )

    adapter, other_adapter = adapter_for(tenant), adapter_for(other)
    run, other_run = str(uuid4()), str(uuid4())

    def observed():
        with pg.connect(dbname=_DATABASE) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT task_type, model_name, tenant_id::text FROM delegation_events "
                "WHERE correlation_id=%s",
                (run,),
            )
            return cur.fetchone()

    def derive(stored):
        writer = next(iter(adapter._binding_connections.connections.values()))
        with writer.cursor() as cur:
            cur.execute("SELECT current_user, current_setting('app.tenant_id')")
            assert cur.fetchone() == (_ROLE, str(tenant))
        assert observed() == ("initial", "", str(tenant)), "UPSERT committed too early"
        return {"model_name": "derived:" + stored["task_type"]}

    try:
        adapter.upsert(
            "delegation_events",
            "correlation_id",
            {
                "correlation_id": run,
                "task_type": "initial",
            },
        )
        other_adapter.upsert(
            "delegation_events",
            "correlation_id",
            {
                "correlation_id": other_run,
                "task_type": "other",
            },
        )
        with pg.connect(dbname=_DATABASE) as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM delegation_events")
            assert cur.fetchone() == (2,)
        assert adapter.query("delegation_events", {"correlation_id": other_run}) == []
        columns = (
            ("correlation_id", "task_type", "model_name") if public_returning else ()
        )
        result = adapter.upsert_returning(
            "delegation_events",
            "correlation_id",
            {
                "correlation_id": run,
                "task_type": "document",
            },
            derive=derive,
            input_columns=("task_type",),
            derived_columns=("model_name",),
            returning=columns,
        )
        expected = [
            {
                "correlation_id": run,
                "task_type": "document",
                "model_name": "derived:document",
            }
        ]
        assert result == (expected if public_returning else [])
        assert observed() == ("document", "derived:document", str(tenant))

        def broken(_):
            raise RuntimeError("owned tenant callback failure")

        with pytest.raises(RuntimeError, match="owned tenant callback failure"):
            adapter.upsert_returning(
                "delegation_events",
                "correlation_id",
                {
                    "correlation_id": run,
                    "task_type": "must-roll-back",
                },
                derive=broken,
                input_columns=("task_type",),
                derived_columns=("model_name",),
                returning=columns,
            )
        assert observed() == ("document", "derived:document", str(tenant))
        assert all(
            c.autocommit for c in adapter._binding_connections.connections.values()
        )
    finally:
        adapter.close()
        other_adapter.close()
