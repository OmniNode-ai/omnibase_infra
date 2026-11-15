# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the migration sentinel (OMN-3737).

Validates:
1. Migration SQL file exists and is idempotent (ON CONFLICT / IF NOT EXISTS).
2. Healthcheck script exists and is executable.
3. Docker-compose references the migration-gate service.
4. Migration SQL does not create new tables (only modifies db_metadata).
"""

from __future__ import annotations

import re
import stat
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent.parent

MIGRATION_FILE = (
    REPO_ROOT / "docker" / "migrations" / "forward" / "037_migration_sentinel.sql"
)
ROLLBACK_FILE = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_037_migration_sentinel.sql"
)
HEALTHCHECK_SCRIPT = REPO_ROOT / "scripts" / "check_migrations_complete.sh"
COMPOSE_FILE = REPO_ROOT / "docker" / "docker-compose.infra.yml"


@pytest.mark.unit
class TestMigrationSentinelSQL:
    """Validate the migration SQL file."""

    def test_migration_file_exists(self) -> None:
        assert MIGRATION_FILE.exists(), f"Migration file not found: {MIGRATION_FILE}"

    def test_rollback_file_exists(self) -> None:
        assert ROLLBACK_FILE.exists(), f"Rollback file not found: {ROLLBACK_FILE}"

    def test_migration_is_idempotent(self) -> None:
        """Migration must use IF NOT EXISTS for column additions."""
        sql = MIGRATION_FILE.read_text()
        assert "IF NOT EXISTS" in sql, (
            "Migration must use ADD COLUMN IF NOT EXISTS for idempotency"
        )

    def test_migration_sets_sentinel_true(self) -> None:
        """Migration must set migrations_complete = TRUE."""
        sql = MIGRATION_FILE.read_text().upper()
        assert "MIGRATIONS_COMPLETE" in sql
        assert "TRUE" in sql

    def test_migration_does_not_create_tables(self) -> None:
        """Sentinel migration must not create new tables."""
        sql = MIGRATION_FILE.read_text().upper()
        assert "CREATE TABLE" not in sql, (
            "Migration 037 must not create new tables -- "
            "it only adds a column to the existing db_metadata table"
        )

    def test_migration_updates_schema_version(self) -> None:
        """Migration must update schema_version to '037'."""
        sql = MIGRATION_FILE.read_text()
        assert "'037'" in sql, "Migration must set schema_version = '037'"


@pytest.mark.unit
class TestHealthcheckScript:
    """Validate the healthcheck script."""

    def test_healthcheck_script_exists(self) -> None:
        assert HEALTHCHECK_SCRIPT.exists(), (
            f"Healthcheck script not found: {HEALTHCHECK_SCRIPT}"
        )

    def test_healthcheck_script_is_executable(self) -> None:
        mode = HEALTHCHECK_SCRIPT.stat().st_mode
        assert mode & stat.S_IXUSR, "Healthcheck script must be executable (chmod +x)"

    def test_healthcheck_queries_migrations_complete(self) -> None:
        """Script must query the migrations_complete column."""
        content = HEALTHCHECK_SCRIPT.read_text()
        assert "migrations_complete" in content, (
            "Healthcheck must query db_metadata.migrations_complete"
        )

    def test_healthcheck_requires_projection_tables(self) -> None:
        """Runtime gate must prove projection tables in omnidash_analytics."""
        content = HEALTHCHECK_SCRIPT.read_text()
        assert "NODE_POSTGRES_DB" in content
        assert "REQUIRED_PROJECTION_TABLES" in content
        assert "delegation_events" in content
        assert "node_service_registry" in content
        assert "to_regclass" in content

    def test_healthcheck_uses_psql(self) -> None:
        """Script must use psql to query PostgreSQL."""
        content = HEALTHCHECK_SCRIPT.read_text()
        assert "psql" in content, "Healthcheck must use psql"


@pytest.mark.unit
class TestDockerComposeIntegration:
    """Validate docker-compose waits on the forward migration (OMN-24483)."""

    def test_compose_has_no_migration_gate_service(self) -> None:
        content = COMPOSE_FILE.read_text()
        assert not re.search(r"^  migration-gate:", content, re.MULTILINE), (
            "docker-compose.infra.yml must not define a migration-gate service"
        )

    def test_runtime_services_depend_on_forward_migration(self) -> None:
        """Key runtime services wait on the forward-migration one-shot."""
        content = COMPOSE_FILE.read_text()
        services_needing_migration = [
            "omninode-runtime",
            "agent-actions-consumer",
            "skill-lifecycle-consumer",
        ]
        for service in services_needing_migration:
            pattern = (
                rf"\n  {service}:.*?depends_on:.*?"
                r"forward-migration:\s+condition: service_completed_successfully"
            )
            match = re.search(pattern, content, re.DOTALL)
            assert match is not None, (
                f"{service} must depend on forward-migration completing"
            )

    def test_runtime_env_opts_in_to_the_boot_wait(self) -> None:
        """The runtime itself waits for the sentinel flag and the tables."""
        content = COMPOSE_FILE.read_text()
        assert 'ONEX_BOOT_WAIT_FOR_MIGRATIONS: "true"' in content

    def test_forward_migration_targets_projection_database(self) -> None:
        content = COMPOSE_FILE.read_text()
        assert "NODE_POSTGRES_DB: omnidash_analytics" in content
