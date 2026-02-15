# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the schema fingerprint CI twin script (OMN-2149).

Tests cover fingerprint computation from migration files, artifact
read/write, verify success/failure, and stamp behavior.  All tests
use ``tmp_path`` -- no real migration directory is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_schema_fingerprint import (
    cmd_stamp,
    cmd_verify,
    compute_migration_fingerprint,
    main,
    read_artifact,
    write_artifact,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_migration(migrations_dir: Path, name: str, content: str) -> Path:
    """Create a fake migration SQL file."""
    path = migrations_dir / name
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# TestComputeMigrationFingerprint
# ---------------------------------------------------------------------------


class TestComputeMigrationFingerprint:
    """Tests for compute_migration_fingerprint()."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty migrations directory produces a valid fingerprint with zero files."""
        fp, count = compute_migration_fingerprint(tmp_path)
        assert len(fp) == 64
        assert count == 0

    def test_single_file(self, tmp_path: Path) -> None:
        """Single migration file produces a deterministic fingerprint."""
        _write_migration(tmp_path, "001_init.sql", "CREATE TABLE foo (id INT);")
        fp, count = compute_migration_fingerprint(tmp_path)
        assert len(fp) == 64
        assert count == 1

    def test_deterministic_ordering(self, tmp_path: Path) -> None:
        """Fingerprint is deterministic regardless of file creation order."""
        _write_migration(tmp_path, "002_second.sql", "ALTER TABLE foo ADD col TEXT;")
        _write_migration(tmp_path, "001_first.sql", "CREATE TABLE foo (id INT);")

        fp1, _ = compute_migration_fingerprint(tmp_path)

        # Recreate in different order
        tmp2 = tmp_path / "alt"
        tmp2.mkdir()
        _write_migration(tmp2, "001_first.sql", "CREATE TABLE foo (id INT);")
        _write_migration(tmp2, "002_second.sql", "ALTER TABLE foo ADD col TEXT;")

        fp2, _ = compute_migration_fingerprint(tmp2)
        assert fp1 == fp2

    def test_content_change_changes_fingerprint(self, tmp_path: Path) -> None:
        """Changing file content produces a different fingerprint."""
        _write_migration(tmp_path, "001_init.sql", "CREATE TABLE foo (id INT);")
        fp1, _ = compute_migration_fingerprint(tmp_path)

        _write_migration(tmp_path, "001_init.sql", "CREATE TABLE bar (id INT);")
        fp2, _ = compute_migration_fingerprint(tmp_path)

        assert fp1 != fp2

    def test_added_file_changes_fingerprint(self, tmp_path: Path) -> None:
        """Adding a new migration file changes the fingerprint."""
        _write_migration(tmp_path, "001_init.sql", "CREATE TABLE foo (id INT);")
        fp1, count1 = compute_migration_fingerprint(tmp_path)

        _write_migration(tmp_path, "002_more.sql", "ALTER TABLE foo ADD col TEXT;")
        fp2, count2 = compute_migration_fingerprint(tmp_path)

        assert fp1 != fp2
        assert count1 == 1
        assert count2 == 2

    def test_non_sql_files_ignored(self, tmp_path: Path) -> None:
        """Non-.sql files are ignored."""
        _write_migration(tmp_path, "001_init.sql", "CREATE TABLE foo (id INT);")
        (tmp_path / "README.md").write_text("docs", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")

        _fp, count = compute_migration_fingerprint(tmp_path)
        assert count == 1


# ---------------------------------------------------------------------------
# TestReadArtifact
# ---------------------------------------------------------------------------


class TestReadArtifact:
    """Tests for read_artifact()."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """Missing artifact file returns None."""
        assert read_artifact(tmp_path / "nonexistent") is None

    def test_pending_returns_none(self, tmp_path: Path) -> None:
        """Artifact with PENDING value returns None."""
        artifact = tmp_path / "fp.sha256"
        artifact.write_text("sha256:PENDING\n", encoding="utf-8")
        assert read_artifact(artifact) is None

    def test_valid_artifact(self, tmp_path: Path) -> None:
        """Valid artifact returns the hash."""
        artifact = tmp_path / "fp.sha256"
        artifact.write_text(
            "# comment\nsha256:abc123\ngenerated_at: now\n",
            encoding="utf-8",
        )
        assert read_artifact(artifact) == "abc123"

    def test_no_sha256_line_returns_none(self, tmp_path: Path) -> None:
        """Artifact without sha256: line returns None."""
        artifact = tmp_path / "fp.sha256"
        artifact.write_text("# just a comment\n", encoding="utf-8")
        assert read_artifact(artifact) is None


# ---------------------------------------------------------------------------
# TestWriteArtifact
# ---------------------------------------------------------------------------


class TestWriteArtifact:
    """Tests for write_artifact()."""

    def test_writes_valid_artifact(self, tmp_path: Path) -> None:
        """write_artifact produces a readable artifact."""
        artifact = tmp_path / "fp.sha256"
        write_artifact(artifact, "abc123", 5)

        assert artifact.exists()
        content = artifact.read_text(encoding="utf-8")
        assert "sha256:abc123" in content
        assert "migration_file_count: 5" in content

    def test_round_trip(self, tmp_path: Path) -> None:
        """Written artifact can be read back correctly."""
        artifact = tmp_path / "fp.sha256"
        write_artifact(artifact, "deadbeef" * 8, 10)
        assert read_artifact(artifact) == "deadbeef" * 8


# ---------------------------------------------------------------------------
# TestCmdVerify
# ---------------------------------------------------------------------------


class TestCmdVerify:
    """Tests for cmd_verify()."""

    def test_verify_passes_when_matching(self, tmp_path: Path) -> None:
        """Verify returns 0 when artifact matches migrations."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        _write_migration(migrations, "001_init.sql", "CREATE TABLE foo (id INT);")

        artifact = tmp_path / "fp.sha256"
        fp, count = compute_migration_fingerprint(migrations)
        write_artifact(artifact, fp, count)

        assert cmd_verify(migrations, artifact) == 0

    def test_verify_fails_when_stale(self, tmp_path: Path) -> None:
        """Verify returns 2 when artifact does not match."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        _write_migration(migrations, "001_init.sql", "CREATE TABLE foo (id INT);")

        artifact = tmp_path / "fp.sha256"
        write_artifact(artifact, "wrong_hash", 1)

        assert cmd_verify(migrations, artifact) == 2

    def test_verify_fails_when_artifact_missing(self, tmp_path: Path) -> None:
        """Verify returns 2 when artifact file does not exist."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        _write_migration(migrations, "001_init.sql", "CREATE TABLE foo (id INT);")

        artifact = tmp_path / "nonexistent.sha256"
        assert cmd_verify(migrations, artifact) == 2

    def test_verify_fails_when_artifact_pending(self, tmp_path: Path) -> None:
        """Verify returns 2 when artifact has PENDING value."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        _write_migration(migrations, "001_init.sql", "CREATE TABLE foo (id INT);")

        artifact = tmp_path / "fp.sha256"
        artifact.write_text("sha256:PENDING\n", encoding="utf-8")

        assert cmd_verify(migrations, artifact) == 2


# ---------------------------------------------------------------------------
# TestCmdStamp
# ---------------------------------------------------------------------------


class TestCmdStamp:
    """Tests for cmd_stamp()."""

    def test_stamp_creates_artifact(self, tmp_path: Path) -> None:
        """Stamp creates a valid artifact file."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        _write_migration(migrations, "001_init.sql", "CREATE TABLE foo (id INT);")

        artifact = tmp_path / "fp.sha256"
        assert cmd_stamp(migrations, artifact) == 0
        assert artifact.exists()

        # Verify the artifact is correct
        assert cmd_verify(migrations, artifact) == 0

    def test_stamp_dry_run_does_not_write(self, tmp_path: Path) -> None:
        """Stamp with dry_run=True does not create the file."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        _write_migration(migrations, "001_init.sql", "CREATE TABLE foo (id INT);")

        artifact = tmp_path / "fp.sha256"
        assert cmd_stamp(migrations, artifact, dry_run=True) == 0
        assert not artifact.exists()


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for main() CLI entry point."""

    def test_no_args_returns_1(self) -> None:
        """No subcommand returns exit code 1."""
        assert main([]) == 1

    def test_verify_with_matching_artifact(self, tmp_path: Path) -> None:
        """verify subcommand returns 0 when artifact matches."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        _write_migration(migrations, "001_init.sql", "CREATE TABLE foo (id INT);")

        artifact = tmp_path / "fp.sha256"
        fp, count = compute_migration_fingerprint(migrations)
        write_artifact(artifact, fp, count)

        result = main(
            [
                "verify",
                "--migrations-dir",
                str(migrations),
                "--artifact",
                str(artifact),
            ]
        )
        assert result == 0

    def test_verify_with_stale_artifact(self, tmp_path: Path) -> None:
        """verify subcommand returns 2 when artifact is stale."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        _write_migration(migrations, "001_init.sql", "CREATE TABLE foo (id INT);")

        artifact = tmp_path / "fp.sha256"
        write_artifact(artifact, "stale_hash", 1)

        result = main(
            [
                "verify",
                "--migrations-dir",
                str(migrations),
                "--artifact",
                str(artifact),
            ]
        )
        assert result == 2

    def test_stamp_creates_artifact(self, tmp_path: Path) -> None:
        """stamp subcommand creates artifact and returns 0."""
        migrations = tmp_path / "migrations"
        migrations.mkdir()
        _write_migration(migrations, "001_init.sql", "CREATE TABLE foo (id INT);")

        artifact = tmp_path / "fp.sha256"
        result = main(
            [
                "stamp",
                "--migrations-dir",
                str(migrations),
                "--artifact",
                str(artifact),
            ]
        )
        assert result == 0
        assert artifact.exists()
