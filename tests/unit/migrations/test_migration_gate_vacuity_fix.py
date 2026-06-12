# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for OMN-13062 — migration-gate vacuity fix (retro A-10).

Three bugs fixed:

(1) RUNNER SENTINEL DISCIPLINE
    run-forward-migrations.sh must clear migrations_complete to FALSE at the
    start of every run, and set it TRUE only as the FINAL act after all
    migrations succeed.  Any mid-run failure must leave the gate UNHEALTHY.

(2) SYNC-NODE-MIGRATIONS VACUOUS GATE
    sync-node-migrations.sh --check must exit nonzero (2) when the
    omnimarket source tree is unresolvable.  Silently passing hides drift.
    SYNC_NODE_MIGRATIONS_SKIP_UNRESOLVABLE=1 provides the single opt-out.

(3) WAIT-FOR-POSTGRES GUARD
    run-forward-migrations.sh must wait for Postgres to be ready before
    proceeding, guarding against the first-boot initdb race.

(4) SKIP-MANIFEST
    A committed docker/migrations/skip-manifest.yaml is the sole escape for
    intentionally-skipped migrations.  The runner reads this at startup.

Recurrences: OMN-12885, OMN-12934
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync-node-migrations.sh"
SKIP_MANIFEST = REPO_ROOT / "docker" / "migrations" / "skip-manifest.yaml"
MIGRATION_085 = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "forward"
    / "085_add_runner_completed_at_to_db_metadata.sql"
)
ROLLBACK_085 = (
    REPO_ROOT
    / "docker"
    / "migrations"
    / "rollback"
    / "rollback_085_add_runner_completed_at_to_db_metadata.sql"
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# (1) Runner sentinel discipline
# ---------------------------------------------------------------------------
class TestRunnerSentinelDiscipline:
    """run-forward-migrations.sh clears sentinel at start, sets at end only."""

    def test_runner_clears_sentinel_at_start(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        # The runner must set migrations_complete = FALSE before any migration.
        assert "migrations_complete = FALSE" in script, (
            "run-forward-migrations.sh must clear migrations_complete=FALSE "
            "at the start of every run (OMN-13062)"
        )

    def test_runner_sets_sentinel_only_at_end(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        # The sentinel-set TRUE block must come AFTER the node-migration loop.
        node_discovery_pos = script.index("Node-owned migration auto-discovery")
        sentinel_set_pos = script.index("Setting sentinel TRUE")
        assert sentinel_set_pos > node_discovery_pos, (
            "The sentinel TRUE set must be the runner's final act, "
            "after all infra and node migrations complete (OMN-13062)"
        )

    def test_runner_sets_runner_completed_at(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        assert "runner_completed_at = NOW()" in script, (
            "run-forward-migrations.sh must stamp runner_completed_at "
            "as its final act (OMN-13062)"
        )

    def test_runner_uses_on_error_stop_for_sentinel_set(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        # Find the sentinel-set block at the end.
        sentinel_pos = script.rindex("Setting sentinel TRUE")
        block_after = script[sentinel_pos:]
        assert "ON_ERROR_STOP=1" in block_after, (
            "The final sentinel-set psql call must use -v ON_ERROR_STOP=1 "
            "so a DB error at that point still fails the runner (OMN-13062)"
        )

    def test_runner_migration_failure_leaves_gate_unhealthy(self) -> None:
        """A failing migration must exit nonzero (via set -e + ON_ERROR_STOP)."""
        script = RUNNER.read_text(encoding="utf-8")
        # The runner uses set -e; psql uses -v ON_ERROR_STOP=1 for each apply.
        assert "set -e" in script, (
            "run-forward-migrations.sh must use 'set -e' so any psql failure "
            "aborts the runner before the sentinel is set (OMN-13062)"
        )
        assert "-v ON_ERROR_STOP=1 -f" in script, (
            "Each migration apply must use -v ON_ERROR_STOP=1 so a SQL error "
            "causes psql to exit nonzero and the runner to abort (OMN-13062)"
        )


# ---------------------------------------------------------------------------
# (2) sync-node-migrations --check vacuous gate
# ---------------------------------------------------------------------------
class TestSyncNodeMigrationsCheckVacuity:
    """sync-node-migrations.sh --check must exit nonzero when source is absent."""

    def test_check_mode_exits_nonzero_when_source_unresolvable(self) -> None:
        """--check with no source must exit 2, not 0."""
        result = subprocess.run(
            ["bash", str(SYNC_SCRIPT), "--check"],
            cwd=str(REPO_ROOT),
            env={
                k: v
                for k, v in os.environ.items()
                # Unset OMNI_HOME and OMNIMARKET_SRC to make source unresolvable.
                if k not in ("OMNI_HOME", "OMNIMARKET_SRC")
            },
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 2, (
            f"sync-node-migrations.sh --check must exit 2 when omnimarket "
            f"source is unresolvable (OMN-13062 vacuous-gate sibling fix). "
            f"Got returncode={result.returncode}.\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )

    def test_check_mode_skip_env_allows_exit_zero(self) -> None:
        """SYNC_NODE_MIGRATIONS_SKIP_UNRESOLVABLE=1 provides the single opt-out."""
        result = subprocess.run(
            ["bash", str(SYNC_SCRIPT), "--check"],
            cwd=str(REPO_ROOT),
            env={
                k: v
                for k, v in os.environ.items()
                if k not in ("OMNI_HOME", "OMNIMARKET_SRC")
            }
            | {"SYNC_NODE_MIGRATIONS_SKIP_UNRESOLVABLE": "1"},
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, (
            f"sync-node-migrations.sh --check with "
            f"SYNC_NODE_MIGRATIONS_SKIP_UNRESOLVABLE=1 must exit 0. "
            f"Got returncode={result.returncode}.\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )

    def test_sync_script_documents_exit_code_2_for_unresolvable(self) -> None:
        script = SYNC_SCRIPT.read_text(encoding="utf-8")
        assert "exit 2" in script, (
            "sync-node-migrations.sh must document and implement exit code 2 "
            "for unresolvable omnimarket source (OMN-13062)"
        )

    def test_sync_script_has_skip_unresolvable_env_escape(self) -> None:
        script = SYNC_SCRIPT.read_text(encoding="utf-8")
        assert "SYNC_NODE_MIGRATIONS_SKIP_UNRESOLVABLE" in script, (
            "sync-node-migrations.sh must honour "
            "SYNC_NODE_MIGRATIONS_SKIP_UNRESOLVABLE=1 as the single escape "
            "for --check when source is absent (OMN-13062)"
        )


# ---------------------------------------------------------------------------
# (3) wait-for-postgres guard
# ---------------------------------------------------------------------------
class TestWaitForPostgresGuard:
    """run-forward-migrations.sh must wait for Postgres before proceeding."""

    def test_runner_has_postgres_wait_loop(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        assert "Waiting for Postgres" in script, (
            "run-forward-migrations.sh must wait for Postgres before "
            "applying migrations (OMN-13062 first-boot initdb race guard)"
        )

    def test_runner_wait_loop_has_retry_limit(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        assert "PG_WAIT_RETRIES" in script, (
            "The Postgres wait loop must honour PG_WAIT_RETRIES so the "
            "runner eventually exits nonzero on a stuck Postgres (OMN-13062)"
        )

    def test_runner_postgres_wait_exits_nonzero_on_timeout(
        self, tmp_path: Path
    ) -> None:
        """A fake psql that always fails triggers the retry limit and exits 1."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        fake_psql = bin_dir / "psql"
        fake_psql.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        fake_psql.chmod(0o755)

        # Use an empty migrations dir so the runner stops after the wait loop.
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        result = subprocess.run(
            ["sh", str(RUNNER)],
            check=False,
            env={
                # Minimal env — no real Postgres vars so fake psql handles all calls.
                "POSTGRES_PASSWORD": "test",
                "POSTGRES_HOST": "127.0.0.1",
                "POSTGRES_PORT": "65534",  # unused port
                "MIGRATIONS_DIR": str(migrations_dir),
                "NODE_MIGRATIONS_DIR": str(tmp_path / "no-nodes"),
                "PG_WAIT_RETRIES": "2",  # fast timeout for tests
                "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
                "HOME": os.environ.get("HOME", str(tmp_path)),
            },
            text=True,
            capture_output=True,
            timeout=30,
        )

        assert result.returncode != 0, (
            "run-forward-migrations.sh must exit nonzero when Postgres is not "
            "reachable within PG_WAIT_RETRIES retries (OMN-13062)"
        )
        assert "not ready" in result.stdout or "Aborting" in result.stdout, (
            f"Expected 'not ready' or 'Aborting' in runner output.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


# ---------------------------------------------------------------------------
# (4) Skip-manifest
# ---------------------------------------------------------------------------
class TestSkipManifest:
    """The committed skip-manifest.yaml is the sole escape for skipped migrations."""

    def test_skip_manifest_exists(self) -> None:
        assert SKIP_MANIFEST.exists(), (
            f"docker/migrations/skip-manifest.yaml must exist (OMN-13062): {SKIP_MANIFEST}"
        )

    def test_skip_manifest_has_skipped_migrations_key(self) -> None:
        content = SKIP_MANIFEST.read_text(encoding="utf-8")
        assert "skipped_migrations:" in content, (
            "skip-manifest.yaml must contain a 'skipped_migrations:' key (OMN-13062)"
        )

    def test_runner_reads_skip_manifest(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        assert "skip-manifest.yaml" in script, (
            "run-forward-migrations.sh must read skip-manifest.yaml (OMN-13062)"
        )

    def test_runner_has_is_skipped_by_manifest_function(self) -> None:
        script = RUNNER.read_text(encoding="utf-8")
        assert "is_skipped_by_manifest" in script, (
            "run-forward-migrations.sh must implement is_skipped_by_manifest "
            "to check each migration against the skip-manifest (OMN-13062)"
        )

    def test_runner_honours_skip_manifest_in_migration_loop(
        self, tmp_path: Path
    ) -> None:
        """A migration listed in skip-manifest.yaml must not be executed."""
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()

        # psql that exits 0 on SELECT 1 (wait guard), and records calls.
        call_log = tmp_path / "psql_calls.log"
        fake_psql = bin_dir / "psql"
        fake_psql.write_text(
            textwrap.dedent(f"""\
                #!/bin/sh
                printf '%s\\n' "$*" >> "{call_log}"
                # If the -c arg contains SELECT 1, succeed (wait guard).
                if echo "$*" | grep -q 'SELECT 1'; then
                  exit 0
                fi
                # CREATE TABLE for schema_migrations — succeed.
                if echo "$*" | grep -q 'CREATE TABLE'; then
                  exit 0
                fi
                # INSERT INTO schema_migrations — succeed.
                if echo "$*" | grep -q 'INSERT INTO'; then
                  exit 0
                fi
                # DO block for sentinel clear — succeed.
                if echo "$*" | grep -q 'BEGIN'; then
                  exit 0
                fi
                # Actual -f migration file — succeed.
                exit 0
            """),
            encoding="utf-8",
        )
        fake_psql.chmod(0o755)

        migrations_dir = tmp_path / "migrations" / "forward"
        migrations_dir.mkdir(parents=True)

        # One migration that is listed in the skip-manifest.
        skip_sql = migrations_dir / "001_should_be_skipped.sql"
        skip_sql.write_text("SELECT 'THIS SHOULD NOT RUN';\n", encoding="utf-8")

        # skip-manifest one level above forward/.
        skip_manifest_path = tmp_path / "migrations" / "skip-manifest.yaml"
        skip_manifest_path.write_text(
            textwrap.dedent("""\
                skipped_migrations:
                  - id: "docker/001_should_be_skipped.sql"
                    reason: "test skip"
                    ticket: "OMN-13062"
            """),
            encoding="utf-8",
        )

        result = subprocess.run(
            ["sh", str(RUNNER)],
            check=False,
            env={
                # Minimal env — deliberately exclude real POSTGRES_HOST so
                # the runner uses its default "postgres" (fake psql handles it).
                "POSTGRES_PASSWORD": "test",
                "POSTGRES_HOST": "localhost",
                "POSTGRES_PORT": "5432",
                "MIGRATIONS_DIR": str(migrations_dir),
                "NODE_MIGRATIONS_DIR": str(tmp_path / "no-nodes"),
                "PG_WAIT_RETRIES": "1",
                # Put fake psql first so it shadows any real psql.
                "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}",
                "HOME": os.environ.get("HOME", str(tmp_path)),
            },
            text=True,
            capture_output=True,
            timeout=30,
        )

        assert "skip-manifest" in result.stdout, (
            f"Runner must log 'skip-manifest' for the skipped migration.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # The SQL file must not have been passed to psql with -f.
        # The fake psql logs every call; check that the migration file never
        # appears in a -f argument (it was skipped, not executed).
        calls = call_log.read_text(encoding="utf-8") if call_log.exists() else ""
        file_executed = f"-f {skip_sql}" in calls or f"-f {skip_sql.name}" in calls
        assert not file_executed, (
            f"Skip-manifest migration must not be passed to psql with -f.\n"
            f"psql calls logged:\n{calls}"
        )


# ---------------------------------------------------------------------------
# (5) Migration 085 — runner_completed_at schema
# ---------------------------------------------------------------------------
class TestMigration085RunnerCompletedAt:
    """Migration 085 adds runner_completed_at column to db_metadata."""

    def test_migration_085_exists(self) -> None:
        assert MIGRATION_085.exists(), f"Migration 085 must exist: {MIGRATION_085}"

    def test_rollback_085_exists(self) -> None:
        assert ROLLBACK_085.exists(), (
            f"Rollback for migration 085 must exist: {ROLLBACK_085}"
        )

    def test_migration_085_is_idempotent(self) -> None:
        sql = MIGRATION_085.read_text(encoding="utf-8")
        assert "IF NOT EXISTS" in sql, (
            "Migration 085 must use ADD COLUMN IF NOT EXISTS for idempotency"
        )

    def test_migration_085_adds_runner_completed_at(self) -> None:
        sql = MIGRATION_085.read_text(encoding="utf-8")
        assert "runner_completed_at" in sql
        assert "TIMESTAMPTZ" in sql.upper()

    def test_migration_085_does_not_create_tables(self) -> None:
        sql = MIGRATION_085.read_text(encoding="utf-8").upper()
        assert "CREATE TABLE" not in sql, (
            "Migration 085 must only ALTER db_metadata, not create new tables"
        )
