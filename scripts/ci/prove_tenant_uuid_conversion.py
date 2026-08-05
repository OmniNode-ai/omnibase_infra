# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Docker fixture proof for OMN-15356: capability_scores tenant_id TEXT->UUID.

Proves, against a rebuilt, disposable PostgreSQL 16 fixture -- never a live,
shared, or deployed database -- the predicate this ticket exists to satisfy:
every tenant value maps totally to a canonical UUID, and no sentinel/default
survives an unmapped value.

Three phases, each printing a ``tenant_uuid_conversion_phase=... status=...``
line so a passing run is grep-able the same way the sibling ACL proof harness
(``docker/application-acl-proof``) is:

1. fail_closed   -- an unmapped legacy value present anywhere in the table
                    aborts the WHOLE conversion transaction. Column stays
                    TEXT, zero rows change, no invented UUID appears.
2. total_mapping -- once every row is a known value, the conversion succeeds
                    and every row resolves to the correct canonical UUID.
3. continuity    -- the pre-existing UNIQUE(model_key, task_type) constraint
                    and the tenant_id index both survive the type change and
                    still function afterward.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import psycopg2
import psycopg2.extras

# Decode native `uuid` columns as `uuid.UUID` objects rather than plain str,
# so the assertions below prove the column is genuinely typed uuid (an
# equality check against a str would pass just as well for a text column
# holding a uuid-shaped string, which is not what this migration claims).
psycopg2.extras.register_uuid()

DSN = os.environ["ADMIN_DSN"]
MIGRATION_SQL = Path("/app/0003_capability_scores_tenant_id_to_uuid.sql").read_text(
    encoding="utf-8"
)
HOUSE_TENANT_UUID = uuid.UUID("820272f9-4aaf-5add-a2df-0af942852ab2")
UNMAPPED_VALUE = "acme-legacy-unmapped"


def _connect() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    return conn


def _reset_table(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM public.capability_scores")
    conn.commit()


def _column_type(conn: psycopg2.extensions.connection) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'capability_scores' "
            "AND column_name = 'tenant_id'"
        )
        row = cur.fetchone()
        assert row is not None, "tenant_id column not found"
        return str(row[0])


def phase_fail_closed(conn: psycopg2.extensions.connection) -> None:
    _reset_table(conn)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.capability_scores (model_key, task_type, tenant_id) "
            "VALUES (%s, %s, %s), (%s, %s, %s)",
            (
                "legit-model",
                "legit-task",
                "omninode",
                "rogue-model",
                "rogue-task",
                UNMAPPED_VALUE,
            ),
        )
    conn.commit()

    aborted = False
    error_text = ""
    with conn.cursor() as cur:
        try:
            cur.execute(MIGRATION_SQL)
            conn.commit()
        except psycopg2.Error as exc:
            conn.rollback()
            aborted = True
            error_text = str(exc)

    assert aborted, "conversion must raise when an unmapped tenant value is present"
    assert UNMAPPED_VALUE in error_text, (
        f"error must name the unmapped value so the operator can find it, got: {error_text}"
    )

    column_type = _column_type(conn)
    assert column_type == "text", (
        f"tenant_id must remain TEXT after an aborted (rolled back) conversion, "
        f"found {column_type!r}"
    )

    with conn.cursor() as cur:
        cur.execute("SELECT tenant_id FROM public.capability_scores ORDER BY model_key")
        remaining = sorted(r[0] for r in cur.fetchall())
    assert remaining == sorted(["omninode", UNMAPPED_VALUE]), (
        f"aborted transaction must leave every row exactly as seeded (no partial "
        f"write, no invented sentinel), found {remaining}"
    )
    print("tenant_uuid_conversion_phase=fail_closed status=PASS")


def phase_total_mapping(conn: psycopg2.extensions.connection) -> None:
    _reset_table(conn)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.capability_scores (model_key, task_type, tenant_id) "
            "VALUES (%s, %s, %s), (%s, %s, %s)",
            ("model-a", "task-a", "omninode", "model-b", "task-b", "omninode"),
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(MIGRATION_SQL)
    conn.commit()

    column_type = _column_type(conn)
    assert column_type == "uuid", (
        f"tenant_id must be uuid after conversion, found {column_type!r}"
    )

    with conn.cursor() as cur:
        cur.execute("SELECT tenant_id FROM public.capability_scores ORDER BY model_key")
        rows = [r[0] for r in cur.fetchall()]
    assert len(rows) == 2, f"expected 2 rows, found {len(rows)}"
    for value in rows:
        assert isinstance(value, uuid.UUID), f"row value {value!r} is not a UUID object"
        assert value == HOUSE_TENANT_UUID, (
            f"expected every seeded 'omninode' row to resolve to the pinned canonical "
            f"UUID {HOUSE_TENANT_UUID}, found {value}"
        )

    # Re-applying is idempotent: the guard DO block detects the column is
    # already uuid and skips, rather than erroring or double-converting.
    with conn.cursor() as cur:
        cur.execute(MIGRATION_SQL)
    conn.commit()
    column_type_after_reapply = _column_type(conn)
    assert column_type_after_reapply == "uuid"
    print("tenant_uuid_conversion_phase=total_mapping status=PASS")


def phase_continuity(conn: psycopg2.extensions.connection) -> None:
    # Table is already converted from phase_total_mapping; verify the
    # pre-existing UNIQUE constraint and the tenant_id index both survived.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT conname FROM pg_constraint "
            "WHERE conrelid = 'public.capability_scores'::regclass AND contype = 'u'"
        )
        unique_constraints = {r[0] for r in cur.fetchall()}
    assert "capability_scores_model_key_task_type_key" in unique_constraints, (
        f"UNIQUE(model_key, task_type) constraint missing after conversion, "
        f"found: {unique_constraints}"
    )

    with conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname = 'public' AND tablename = 'capability_scores'"
        )
        indexes = {r[0] for r in cur.fetchall()}
    assert "idx_capability_scores_tenant_id" in indexes, (
        f"tenant_id index missing after conversion, found: {indexes}"
    )

    # Behavioral proof, not just catalog presence: the UNIQUE constraint must
    # still reject a real duplicate insert.
    duplicate_rejected = False
    with conn.cursor() as cur:
        try:
            cur.execute(
                "INSERT INTO public.capability_scores (model_key, task_type, tenant_id) "
                "VALUES (%s, %s, %s)",
                ("model-a", "task-a", str(HOUSE_TENANT_UUID)),
            )
            conn.commit()
        except psycopg2.errors.UniqueViolation:
            conn.rollback()
            duplicate_rejected = True
    assert duplicate_rejected, (
        "UNIQUE(model_key, task_type) must still reject duplicates post-conversion"
    )

    # RLS policy must compare with an explicit ::uuid cast now that the
    # column itself is uuid -- a malformed/unset GUC must still fail closed
    # (zero visible rows), never silently pass through as a text compare.
    #
    # The ADMIN_DSN connects as the postgres SUPERUSER, which bypasses RLS
    # unconditionally regardless of FORCE (documented in the migration's own
    # header) -- so this must run as a real non-superuser, non-bypassrls role
    # for the policy to apply at all. `SET ROLE` switches `current_user`
    # (what the policy evaluates against) for the rest of the transaction
    # without requiring app_dashboard to be a LOGIN role; only a superuser or
    # a role already a member may SET ROLE to it, and postgres, as the role
    # that created it, may.
    # Deliberately do NOT set app.tenant_id in this transaction:
    # current_setting('app.tenant_id', true) then returns SQL NULL (not an
    # empty string -- which would itself fail ``::uuid`` cast with a syntax
    # error, a different failure this check does not exercise), NULL::uuid
    # is a valid no-op cast to NULL, and `tenant_id = NULL` is NULL under
    # three-valued logic, so the policy predicate excludes every row.
    with conn.cursor() as cur:
        cur.execute("SET LOCAL ROLE app_dashboard")
        cur.execute("SELECT count(*) FROM public.capability_scores")
        (visible_with_unset_guc,) = cur.fetchone()
    conn.rollback()
    assert visible_with_unset_guc == 0, (
        f"unset tenant GUC must see zero rows under the tenant_isolation "
        f"policy for a non-superuser reader, saw {visible_with_unset_guc}"
    )

    print("tenant_uuid_conversion_phase=continuity status=PASS")


def main() -> int:
    conn = _connect()
    try:
        phase_fail_closed(conn)
        phase_total_mapping(conn)
        phase_continuity(conn)
    finally:
        _reset_table(conn)
        conn.close()
    print("tenant_uuid_conversion_status=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
