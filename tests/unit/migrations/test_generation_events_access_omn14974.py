# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-14974: vendored generation_events access migration stays complete."""

from pathlib import Path

_MIGRATION = (
    Path(__file__).resolve().parents[3]
    / "docker"
    / "migrations"
    / "forward"
    / "nodes"
    / "node_projection_delegation"
    / "0027_generation_events_tenant_rls.sql"
)


def _normalized_sql() -> str:
    return " ".join(_MIGRATION.read_text().split())


def test_vendored_generation_events_migration_is_tenant_scoped() -> None:
    sql = _normalized_sql()

    assert (
        "ALTER TABLE generation_events ADD COLUMN IF NOT EXISTS tenant_id text "
        "NOT NULL DEFAULT 'omninode'" in sql
    )
    assert "ALTER TABLE generation_events ENABLE ROW LEVEL SECURITY" in sql
    assert "ALTER TABLE generation_events FORCE ROW LEVEL SECURITY" not in sql
    assert "CREATE POLICY tenant_isolation ON generation_events" in sql
    assert "GRANT SELECT, INSERT, UPDATE ON generation_events TO role_omnidash" in sql
    assert "GRANT SELECT ON generation_events TO app_dashboard" in sql
    assert "GRANT DELETE ON generation_events" not in sql


def test_absent_role_omnidash_warns_and_does_not_abort_the_deploy() -> None:
    """OMN-15351: role_omnidash is environment-provisioned, so its absence WARNs.

    Execution proof (both role states, real psql, real ephemeral Postgres) lives
    in tests/integration/db/test_generation_events_role_tolerance_omn15351.py.
    This is the cheap static ratchet against a silent re-tightening.
    """
    sql = _normalized_sql()

    assert "RAISE WARNING 'role_omnidash role missing" in sql
    assert "RAISE EXCEPTION 'role_omnidash role missing" not in sql
    # Not a silent skip: the warning names both grants it skips.
    assert "SKIPPING 2 grants" in sql
    assert "(1) GRANT USAGE ON SCHEMA public TO role_omnidash" in sql
    assert (
        "(2) GRANT SELECT, INSERT, UPDATE ON generation_events TO role_omnidash" in sql
    )
    # The grants themselves are guarded by the same existence check.
    assert (
        "IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'role_omnidash') THEN "
        "EXECUTE 'GRANT USAGE ON SCHEMA public TO role_omnidash'; "
        "EXECUTE 'GRANT SELECT, INSERT, UPDATE ON generation_events TO role_omnidash';"
        in sql
    )


def test_app_dashboard_guard_stays_fail_closed() -> None:
    """OMN-15351 relaxed the role_omnidash guard ONLY.

    Forward migration 094 (OMN-14899) creates app_dashboard in-repo, so its
    absence is a migration-ordering bug, not an environment difference.
    """
    sql = _normalized_sql()

    assert "RAISE EXCEPTION 'app_dashboard role missing" in sql
    assert "RAISE WARNING 'app_dashboard role missing" not in sql
