# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Apply 095 to rows written after 091 on an isolated PostgreSQL cluster."""

from pathlib import Path

import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

pytestmark = [pytest.mark.integration, pytest.mark.ephemeral_pg]

SAVINGS = (
    Path(__file__).resolve().parents[3]
    / "docker/migrations/forward/nodes/node_projection_savings"
)
PREVIOUS = SAVINGS / "091_savings_estimates_house_tenant_uuid_backfill.sql"
MIGRATION = SAVINGS / "095_savings_estimates_house_tenant_uuid_rebackfill.sql"
HOUSE = "820272f9-4aaf-5add-a2df-0af942852ab2"


@pytest.mark.parametrize(
    "as_owner", [False, True], ids=["superuser", "forced-rls-owner"]
)
@pytest.mark.parametrize("mirror", ["absent", "unreconciled", "matching"])
def test_095_moves_post_091_slug_rows_without_changing_other_tenants_or_rls(
    ephemeral_postgres: EphemeralPostgres, as_owner: bool, mirror: str
) -> None:
    pg = ephemeral_postgres
    conn = pg.connect()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE ROLE omn19438_owner NOSUPERUSER NOBYPASSRLS")
            cur.execute(
                "CREATE TABLE savings_estimates (id integer PRIMARY KEY, "
                "tenant_id text NOT NULL DEFAULT 'omninode', amount integer NOT NULL)"
            )
            cur.execute("ALTER TABLE savings_estimates OWNER TO omn19438_owner")
            cur.execute("ALTER TABLE savings_estimates ENABLE ROW LEVEL SECURITY")
            cur.execute("ALTER TABLE savings_estimates FORCE ROW LEVEL SECURITY")
            cur.execute(
                "CREATE POLICY tenant_isolation ON savings_estimates "
                "USING (tenant_id = current_setting('app.current_tenant_id', true)) "
                "WITH CHECK (tenant_id = current_setting('app.current_tenant_id', true))"
            )
        previous = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(PREVIOUS))
        assert previous.returncode == 0, previous.stderr
        with conn.cursor() as cur:
            # Reproduce the incident: 091 already ran, then the old writer
            # explicitly supplied the slug despite the repaired column default.
            cur.execute(
                "INSERT INTO savings_estimates "
                "SELECT id, 'omninode', id * 2 FROM generate_series(1, 770) id"
            )
            cur.execute(
                "INSERT INTO savings_estimates VALUES "
                "(771, 'other-tenant', 1542), (772, %s, 1544)",
                (HOUSE,),
            )
            cur.execute(
                "SELECT count(*) FROM savings_estimates WHERE tenant_id = 'omninode'"
            )
            assert cur.fetchone() == (770,)
            if mirror == "unreconciled":
                cur.execute("CREATE TABLE tenant_registry_mirror (tenant_slug text)")
            elif mirror == "matching":
                cur.execute(
                    "CREATE TABLE tenant_registry_mirror (tenant_slug text, tenant_uuid uuid)"
                )
                cur.execute(
                    "INSERT INTO tenant_registry_mirror VALUES ('omninode', %s)",
                    (HOUSE,),
                )
                cur.execute("GRANT SELECT ON tenant_registry_mirror TO omn19438_owner")
        for _ in range(2):
            applied = pg.psql(
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                "SET ROLE omn19438_owner" if as_owner else "RESET ROLE",
                "-f",
                str(MIGRATION),
            )
            assert applied.returncode == 0, applied.stderr
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, tenant_id, amount FROM savings_estimates ORDER BY id"
                )
                assert cur.fetchall() == [
                    (i, HOUSE if i != 771 else "other-tenant", i * 2)
                    for i in range(1, 773)
                ]
                cur.execute(
                    "SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                    "WHERE oid = 'savings_estimates'::regclass"
                )
                assert cur.fetchone() == (True, True)
                cur.execute(
                    "SELECT polname FROM pg_policy "
                    "WHERE polrelid = 'savings_estimates'::regclass"
                )
                assert cur.fetchall() == [("tenant_isolation",)]
        with conn.cursor() as cur:
            cur.execute("INSERT INTO savings_estimates (id, amount) VALUES (773, 1546)")
            cur.execute("SELECT tenant_id FROM savings_estimates WHERE id = 773")
            assert cur.fetchone() == (HOUSE,)
            cur.execute("SET ROLE omn19438_owner")
            cur.execute("SELECT count(*) FROM savings_estimates")
            assert cur.fetchone() == (0,), (
                "forced RLS must still hide rows without a tenant"
            )
        print(
            f"095 readback: mirror={mirror} owner={as_owner}; 770 slug rows moved, "
            "0 remain; other tenant and forced RLS preserved; second apply unchanged"
        )
    finally:
        conn.close()


def test_095_refuses_registry_drift_without_changing_slug_rows(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    pg = ephemeral_postgres
    conn = pg.connect()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE savings_estimates (tenant_id text DEFAULT 'omninode')"
            )
            cur.execute("INSERT INTO savings_estimates VALUES ('omninode')")
            cur.execute(
                "CREATE TABLE tenant_registry_mirror (tenant_slug text, tenant_uuid uuid)"
            )
            cur.execute(
                "INSERT INTO tenant_registry_mirror VALUES "
                "('omninode', '11111111-1111-4111-8111-111111111111')"
            )
        applied = pg.psql("-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION))
        assert applied.returncode != 0
        assert "tenant registry drift" in applied.stderr
        with conn.cursor() as cur:
            cur.execute("SELECT tenant_id FROM savings_estimates")
            assert cur.fetchall() == [("omninode",)]
    finally:
        conn.close()
