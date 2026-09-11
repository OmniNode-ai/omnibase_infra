# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The two delegation-savings views read as their invoker (OMN-18159).

``node_projection_delegation``'s 0040 gave the FOUR delegation aggregate views
invoker rights, so a read evaluates row-level security against the CALLER rather
than the view owner. A live read-only readback of onex-dev on 2026-09-11 then
found two MORE views selecting from ``delegation_events`` with
``security_invoker`` unset -- ``projection_delegation_savings`` and
``projection_delegation_savings_series``. They were missed because they belong
to a different node with a different migration lineage, not because anything
about them is different. Operator ruling 2026-09-11T14:40:49Z corrected the
exposure set from four to six, and ``node_projection_savings/088`` is that fix.

WHY THE PROOF IS HERE RATHER THAN IN THE OWNING REPO

This node's migrations are schema-qualified to ``public``. They cannot be
isolated into a disposable schema the way the delegation node's can -- applying
them against any live database REPLACES that database's public views. The only
safe before/after is a throwaway cluster where ``public`` itself is disposable,
which is what ``EphemeralPostgres`` provides.

WHY THE PAIR, NOT THE POST-STATE

Asserting only that the option is true after 088 would pass just as happily
against a database where it had always been true -- which is not hypothetical:
the ``.201`` dev lane already carries ``security_invoker = true`` on both views
while onex-dev does not, so a post-state-only assertion run against the lane
would have reported success for a migration that had not run. The discriminating
assertion is that the SAME corpus, stopped one migration earlier, reports the
option unset.
"""

from __future__ import annotations

from pathlib import Path

import psycopg2
import pytest

from tests.integration.migrations.conftest import EphemeralPostgres

REPO_ROOT = Path(__file__).parent.parent.parent.parent
NODES = REPO_ROOT / "docker" / "migrations" / "forward" / "nodes"
DELEGATION = NODES / "node_projection_delegation"
SAVINGS = NODES / "node_projection_savings"

MIGRATION_FILE = SAVINGS / "088_savings_views_invoker_scoped.sql"
PREVIOUS_MIGRATION = "087_savings_views_read_persisted_provenance.sql"

SAVINGS_VIEWS = (
    "projection_delegation_savings",
    "projection_delegation_savings_series",
)

#: Superseded by omnibase_infra's own 0032 and fenced (OMN-15349); the corpus
#: below performs the post-0032 conversion itself.
_FENCED = {"0031_delegation_events_tenant_id_to_uuid.sql"}


def _schema_safe(sql: str) -> str:
    """A plain CREATE INDEX is schema-equivalent on a single-connection cluster.

    ``CONCURRENTLY`` cannot run inside the implicit transaction psycopg2 opens.
    A property of the driver, not of the migration.
    """
    return sql.replace("CREATE INDEX CONCURRENTLY", "CREATE INDEX")


def _require_pg15(conn: psycopg2.extensions.connection) -> None:
    """Skip, with the reason, on a server older than the corpus needs.

    The delegation corpus contains ``GRANT SET ON PARAMETER``, which Postgres
    only understands from 15. On an older server every statement after it fails
    with ``unrecognized privilege type "SET"`` -- a fact about the cluster the
    runner happened to put on PATH, not about this migration. Skipping names
    that; failing would report a defect in the wrong place.
    """
    if conn.server_version < 150000:
        pytest.skip(
            "ephemeral cluster is PostgreSQL "
            f"{conn.server_version // 10000}; the delegation corpus needs 15+ "
            "for GRANT SET ON PARAMETER"
        )


def _apply(conn: psycopg2.extensions.connection, *, stop_after: str | None) -> None:
    """Apply the delegation corpus, then the savings corpus up to ``stop_after``."""
    _require_pg15(conn)
    with conn.cursor() as cur:
        # Two roles these corpora GRANT to but do not create; a real lane
        # provisions them from other migrations. Created with the same guarded
        # form the owning migrations use, so the fixture matches the lane rather
        # than a migration being weakened to match the fixture.
        cur.execute(
            "DO $$ BEGIN "
            "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'app_dashboard') "
            "THEN CREATE ROLE app_dashboard; END IF; "
            "IF NOT EXISTS (SELECT 1 FROM pg_roles "
            "WHERE rolname = 'tenant_projection_writer') THEN "
            "CREATE ROLE tenant_projection_writer WITH NOLOGIN NOSUPERUSER "
            "NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION; END IF; "
            "IF NOT EXISTS (SELECT 1 FROM pg_roles "
            "WHERE rolname = 'omninode_runtime') THEN "
            "CREATE ROLE omninode_runtime WITH NOLOGIN NOSUPERUSER "
            "NOBYPASSRLS NOCREATEDB NOCREATEROLE NOREPLICATION; END IF; "
            "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_omnidash') "
            "THEN CREATE ROLE role_omnidash; END IF; "
            "END$$;"
        )
        for path in sorted(DELEGATION.glob("*.sql")):
            if path.name in _FENCED:
                continue
            cur.execute(_schema_safe(path.read_text(encoding="utf-8")))
        for path in sorted(SAVINGS.glob("*.sql")):
            cur.execute(_schema_safe(path.read_text(encoding="utf-8")))
            if stop_after is not None and path.name == stop_after:
                return


def _security_invoker(
    conn: psycopg2.extensions.connection,
) -> dict[str, str | None]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT c.relname, "
            "(SELECT option_value FROM pg_options_to_table(c.reloptions) "
            " WHERE option_name = 'security_invoker') "
            "FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = 'public' AND c.relkind = 'v' "
            "AND c.relname = ANY(%s)",
            (list(SAVINGS_VIEWS),),
        )
        return {row[0]: row[1] for row in cur.fetchall()}


@pytest.mark.integration
def test_088_turns_invoker_rights_on_and_087_leaves_them_off(
    ephemeral_postgres: EphemeralPostgres,
) -> None:
    before = ephemeral_postgres.connect(dbname="postgres")
    before.autocommit = True
    try:
        with before.cursor() as cur:
            cur.execute("CREATE DATABASE before_088")
    finally:
        before.close()

    conn = ephemeral_postgres.connect(dbname="before_088")
    conn.autocommit = True
    try:
        _apply(conn, stop_after=PREVIOUS_MIGRATION)
        observed = _security_invoker(conn)
    finally:
        conn.close()

    assert set(observed) == set(SAVINGS_VIEWS), (
        "positive control: both views must exist after 087, or the pair below "
        "compares nothing"
    )
    assert observed == dict.fromkeys(SAVINGS_VIEWS, None), (
        "087 must leave security_invoker unset, or 088 proves nothing"
    )

    after_conn = ephemeral_postgres.connect(dbname="postgres")
    after_conn.autocommit = True
    try:
        with after_conn.cursor() as cur:
            cur.execute("CREATE DATABASE after_088")
    finally:
        after_conn.close()

    conn = ephemeral_postgres.connect(dbname="after_088")
    conn.autocommit = True
    try:
        _apply(conn, stop_after=None)
        assert _security_invoker(conn) == dict.fromkeys(SAVINGS_VIEWS, "true")
        # Idempotent: a corpus re-run must not fail on it.
        with conn.cursor() as cur:
            cur.execute(_schema_safe(MIGRATION_FILE.read_text(encoding="utf-8")))
        assert _security_invoker(conn) == dict.fromkeys(SAVINGS_VIEWS, "true")
    finally:
        conn.close()


@pytest.mark.integration
def test_088_alters_exactly_the_two_savings_views_and_nothing_else() -> None:
    """Text contract: no DROP, no re-CREATE, no widening beyond these two views.

    Nothing about either view's shape changes, so nothing needs dropping -- and
    a DROP is precisely what re-owns a view to whoever ran the migration, which
    is the defect 0040 recorded and this migration must not reintroduce.
    """
    executable = [
        line.strip()
        for line in MIGRATION_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    assert executable == [
        "ALTER VIEW public.projection_delegation_savings "
        "SET (security_invoker = true);",
        "ALTER VIEW public.projection_delegation_savings_series "
        "SET (security_invoker = true);",
    ]
