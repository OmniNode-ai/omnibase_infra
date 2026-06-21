# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for node-owned migration auto-discovery (OMN-12559).

These tests are Docker/DB-independent. They prove:

  1. Marketplace node migrations are vendored durably into the namespaced
     forward location, so a clean clone reproduces node-owned schemas and
     views with NO manual copy.
  2. ``run-forward-migrations.sh`` contains the namespaced discovery pass
     that applies node migrations under ``node:<node>:<file>`` ids — a
     separate identity space from the flat infra sequence, so a node
     migration numbered 076 never collides with infra's flat 076 and no
     renumber is ever required.
  3. The vendored tree stays in sync with the omnimarket source when the
     source tree is resolvable (drift guard via sync-node-migrations.sh), with
     no node allowlist.

Ticket: OMN-12559
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
NODES_DIR = FORWARD_DIR / "nodes"
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync-node-migrations.sh"
SYNCED_NODES_ALLOWLIST = NODES_DIR / ".synced-nodes"

DELEGATION_VIEW = (
    NODES_DIR
    / "node_projection_delegation"
    / "0010_create_delegation_dashboard_projection_views.sql"
)
SAVINGS_VIEW = (
    NODES_DIR
    / "node_projection_savings"
    / "076_create_delegation_savings_projection_view.sql"
)
SAVINGS_CREATE = (
    NODES_DIR / "node_projection_savings" / "074_create_savings_estimates.sql"
)
SAVINGS_UPDATED_AT = (
    NODES_DIR / "node_projection_savings" / "075_add_savings_estimates_updated_at.sql"
)
REGISTRATION_CREATE = (
    NODES_DIR / "node_projection_registration" / "0000_create_node_service_registry.sql"
)
REGISTRATION_HEARTBEAT = (
    NODES_DIR / "node_projection_registration" / "0001_add_heartbeat_columns.sql"
)
DEPLOYMENT_EVIDENCE_CREATE = (
    NODES_DIR
    / "node_deployment_evidence_reducer"
    / "0001_create_deployment_evidence_projection_tables.sql"
)
EVIDENCE_DASHBOARD_CREATE = (
    NODES_DIR
    / "node_evidence_dashboard_reducer"
    / "0001_create_evidence_dashboard_projection_tables.sql"
)

pytestmark = pytest.mark.unit


class TestVendoredViewMigrations:
    """Marketplace node migrations are durably vendored (no manual copy)."""

    def test_delegation_dashboard_views_vendored(self) -> None:
        assert DELEGATION_VIEW.is_file(), (
            "delegation dashboard views must be vendored under "
            "docker/migrations/forward/nodes/node_projection_delegation/"
        )
        sql = DELEGATION_VIEW.read_text(encoding="utf-8")
        # Authoritative dashboard read views from PR #1005 / OMN-12489.
        assert "CREATE OR REPLACE VIEW projection_delegation_summary" in sql
        assert "CREATE OR REPLACE VIEW projection_delegation_model_routing" in sql

    def test_savings_projection_view_vendored(self) -> None:
        assert SAVINGS_VIEW.is_file(), (
            "savings projection view must be vendored under "
            "docker/migrations/forward/nodes/node_projection_savings/"
        )
        assert SAVINGS_CREATE.is_file(), (
            "savings_estimates base table must be vendored with the savings view "
            "because node-owned migrations apply to omnidash_analytics"
        )
        assert SAVINGS_UPDATED_AT.is_file(), (
            "savings_estimates additive columns must be vendored with the savings view"
        )
        sql = SAVINGS_VIEW.read_text(encoding="utf-8")
        assert "CREATE OR REPLACE VIEW projection_delegation_savings" in sql

    def test_delegation_base_tables_vendored_so_views_apply(self) -> None:
        """Views depend on delegation_events/generation_events base tables.

        Those node-owned base-table migrations must be vendored alongside the
        views so a clean clone can apply the whole chain.
        """
        node_dir = NODES_DIR / "node_projection_delegation"
        files = {p.name for p in node_dir.glob("*.sql")}
        assert "0007_delegation_events.sql" in files
        assert "0008_generation_events.sql" in files
        assert "0010_create_delegation_dashboard_projection_views.sql" in files
        assert "0012_generation_output_columns.sql" in files
        assert "0013_generation_proof_fields.sql" in files

    def test_premium_counterfactual_migration_vendored(self) -> None:
        """OMN-13355: the pinned premium-counterfactual column migration is vendored.

        The auditable saving (counterfactual - actual) is persisted as a JSONB
        column on delegation_events. The node-owned migration must be vendored so
        a clean clone materializes the column under the forward-migration runner.
        """
        migration = (
            NODES_DIR / "node_projection_delegation" / "0017_premium_counterfactual.sql"
        )
        assert migration.is_file(), (
            "0017_premium_counterfactual.sql must be vendored under "
            "docker/migrations/forward/nodes/node_projection_delegation/ "
            "(run scripts/sync-node-migrations.sh)"
        )
        sql = migration.read_text(encoding="utf-8")
        assert (
            "ALTER TABLE delegation_events" in sql
            and "ADD COLUMN IF NOT EXISTS premium_counterfactual JSONB" in sql
        )

    def test_cost_measurements_migration_vendored(self) -> None:
        """OMN-13401/OMN-13234: the per-tier actual-cost measurement migration is vendored.

        0018 records HOW cost_usd was measured (cost_tier_type, cost_tier_name,
        cost_measurement_source, budget_headroom_consumed_usd) so honest savings
        (counterfactual - actual) is auditable from the projection. The node-owned
        migration must be vendored so a clean clone materializes the columns under
        the forward-migration runner.
        """
        migration = (
            NODES_DIR
            / "node_projection_delegation"
            / "0018_delegation_cost_measurements.sql"
        )
        assert migration.is_file(), (
            "0018_delegation_cost_measurements.sql must be vendored under "
            "docker/migrations/forward/nodes/node_projection_delegation/ "
            "(run scripts/sync-node-migrations.sh)"
        )
        sql = migration.read_text(encoding="utf-8")
        assert "ALTER TABLE delegation_events" in sql
        for column in (
            "cost_tier_type",
            "cost_tier_name",
            "cost_measurement_source",
            "budget_headroom_consumed_usd",
        ):
            assert f"ADD COLUMN IF NOT EXISTS {column}" in sql

    def test_budget_state_migration_vendored(self) -> None:
        """OMN-13401/OMN-13235: the per-tenant ceiling budget-state table is vendored.

        0019 creates the durable, event-sourced delegation_budget_state surface
        (cap + consumption + headroom) so the dashboard/API can show how much of a
        tenant's monthly ceiling budget is consumed. The node-owned migration must
        be vendored so a clean clone materializes the table under the
        forward-migration runner.
        """
        migration = (
            NODES_DIR
            / "node_projection_delegation"
            / "0019_delegation_budget_state.sql"
        )
        assert migration.is_file(), (
            "0019_delegation_budget_state.sql must be vendored under "
            "docker/migrations/forward/nodes/node_projection_delegation/ "
            "(run scripts/sync-node-migrations.sh)"
        )
        sql = migration.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS delegation_budget_state" in sql
        assert (
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_delegation_budget_state_identity"
            in sql
        )

    def test_delegation_base_migration_repairs_warm_table_shape(self) -> None:
        """Base migration must be safe when delegation_events already exists."""
        migration = (
            NODES_DIR / "node_projection_delegation" / "0007_delegation_events.sql"
        )
        sql = migration.read_text(encoding="utf-8")
        alter_pos = sql.index("ALTER TABLE delegation_events")
        first_index_pos = min(
            sql.index(index_name)
            for index_name in (
                "idx_delegation_events_timestamp",
                "idx_delegation_events_created_at",
            )
        )

        assert alter_pos < first_index_pos
        expected_columns = (
            "created_at",
            "quality_gates_checked_jsonb",
            "quality_gates_failed_jsonb",
            "latency_ms",
            "tokens_input",
            "tokens_output",
        )
        assert all(
            f"ADD COLUMN IF NOT EXISTS {column}" in sql for column in expected_columns
        )

    def test_registration_projection_migrations_vendored(self) -> None:
        """The node_service_registry owner migrations are vendored durably."""
        assert REGISTRATION_CREATE.is_file(), (
            "node_service_registry create migration must be vendored under "
            "docker/migrations/forward/nodes/node_projection_registration/"
        )
        assert REGISTRATION_HEARTBEAT.is_file(), (
            "node_service_registry heartbeat-column migration must be vendored "
            "beside the create migration"
        )

        create_sql = REGISTRATION_CREATE.read_text(encoding="utf-8")
        heartbeat_sql = REGISTRATION_HEARTBEAT.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS node_service_registry" in create_sql
        assert "uptime_seconds BIGINT DEFAULT 0" in create_sql
        assert "ADD COLUMN IF NOT EXISTS last_heartbeat_at" in heartbeat_sql
        assert "ADD COLUMN IF NOT EXISTS uptime_seconds" in heartbeat_sql

    def test_deployment_evidence_projection_migrations_vendored(self) -> None:
        """Deployment evidence tables must materialize from node contracts."""
        assert DEPLOYMENT_EVIDENCE_CREATE.is_file(), (
            "deployment evidence reducer migrations must be vendored under "
            "docker/migrations/forward/nodes/node_deployment_evidence_reducer/"
        )
        sql = DEPLOYMENT_EVIDENCE_CREATE.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS deployment_evidence_projection" in sql
        assert "CREATE TABLE IF NOT EXISTS deployment_readiness_projection" in sql

    def test_evidence_dashboard_projection_migrations_vendored(self) -> None:
        """Evidence dashboard projection tables must be available to omnidash."""
        assert EVIDENCE_DASHBOARD_CREATE.is_file(), (
            "evidence dashboard reducer migrations must be vendored under "
            "docker/migrations/forward/nodes/node_evidence_dashboard_reducer/"
        )
        sql = EVIDENCE_DASHBOARD_CREATE.read_text(encoding="utf-8")
        assert "CREATE TABLE IF NOT EXISTS evidence_dashboard_projection" in sql
        assert "CREATE TABLE IF NOT EXISTS evidence_correlation_trace_projection" in sql
        assert (
            "CREATE TABLE IF NOT EXISTS evidence_readiness_aggregate_projection" in sql
        )


class TestNamespacedDiscoveryWiring:
    """run-forward-migrations.sh applies node migrations under namespaced ids."""

    def test_runner_scans_node_migrations_dir(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        assert "NODE_MIGRATIONS_DIR" in script
        # Iterates per-node subdirectories.
        assert "${NODE_MIGRATIONS_DIR}" in script

    def test_runner_uses_namespaced_migration_id(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        # The namespaced id form node:<node>:<filename> is what prevents the
        # numeric collision with the flat docker/<filename> sequence.
        assert 'migration_id="node:${node_name}:${filename}"' in script
        assert "source_set" in script

    def test_runner_applies_node_migrations_to_projection_database(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        assert 'NODE_PGDB="${NODE_POSTGRES_DB:-${PGDB}}"' in script
        assert '-d "$NODE_PGDB"' in script

    def test_runner_flat_and_node_ids_are_distinct_namespaces(self) -> None:
        """The flat sequence and node sequence use different id prefixes."""
        script = RUNNER.read_text(encoding="utf-8")
        # Flat infra files are tracked as docker/<filename>.
        assert 'migration_id="docker/${filename}"' in script
        # Node files are tracked as node:<node>:<filename>.
        assert 'migration_id="node:${node_name}:${filename}"' in script

    def test_runner_rejects_malformed_create_database_directive(
        self, tmp_path: Path
    ) -> None:
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()
        (migrations_dir / "001_bad_directive.sql").write_text(
            "--   onex-create-database: bad/name\nSELECT 1;\n",
            encoding="utf-8",
        )
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        psql = bin_dir / "psql"
        psql.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        psql.chmod(0o755)

        result = subprocess.run(
            ["sh", str(RUNNER)],
            check=False,
            env={
                **os.environ,
                "MIGRATIONS_DIR": str(migrations_dir),
                "NODE_MIGRATIONS_DIR": str(tmp_path / "no-node-migrations"),
                "POSTGRES_PASSWORD": "postgres",
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
            },
            text=True,
            capture_output=True,
        )

        assert result.returncode == 1
        assert "invalid database identifier in migration directive: bad/name" in (
            result.stderr
        )


class TestNoNodeMigrationAllowlist:
    """All marketplace node migrations are eligible for vendoring."""

    def test_synced_nodes_allowlist_removed(self) -> None:
        assert not SYNCED_NODES_ALLOWLIST.exists()

    def test_sync_script_has_no_node_allowlist(self) -> None:
        script = SYNC_SCRIPT.read_text(encoding="utf-8")
        assert "SYNCED_NODES_FILE" not in script
        assert "load_synced_nodes" not in script
        assert "file_is_allowed" not in script
        assert ".synced-nodes" not in script


def _resolve_omnimarket_src() -> Path | None:
    """Resolve the omnimarket repo root if available for the drift check."""
    explicit = os.environ.get("OMNIMARKET_SRC")
    if explicit and (Path(explicit) / "src/omnimarket/nodes").is_dir():
        return Path(explicit)
    omni_home = os.environ.get("OMNI_HOME")
    if omni_home and (Path(omni_home) / "omnimarket/src/omnimarket/nodes").is_dir():
        return Path(omni_home) / "omnimarket"
    return None


class TestVendoredTreeMatchesSource:
    """Vendored node migrations stay in sync with omnimarket (drift guard)."""

    def test_sync_check_reports_in_sync(self) -> None:
        omk = _resolve_omnimarket_src()
        if omk is None:
            pytest.skip(
                "omnimarket source tree not resolvable "
                "(set OMNIMARKET_SRC or OMNI_HOME)"
            )
        result = subprocess.run(
            ["bash", str(SYNC_SCRIPT), "--check"],
            cwd=str(REPO_ROOT),
            env={**os.environ, "OMNIMARKET_SRC": str(omk)},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, (
            "vendored node migrations are out of sync with omnimarket:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
