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


def test_vendored_generation_events_migration_is_tenant_scoped() -> None:
    sql = " ".join(_MIGRATION.read_text().split())

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
