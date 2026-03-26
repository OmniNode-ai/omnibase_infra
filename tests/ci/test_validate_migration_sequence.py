# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for migration sequence number duplicate detection.

Covers:
  - extract_sequence_number() — filename parsing variants
  - validate_migration_sequence() — same-set, cross-set, non-sql, non-migration
  - generate_report() — pass and failure message content
  - Main exit codes

Ticket: OMN-3570
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from scripts.validation.validate_migration_sequence import (
    DuplicateConflict,
    SequenceValidationResult,
    extract_sequence_number,
    generate_report,
    validate_migration_sequence,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

DOCKER_MIGRATIONS = "docker/migrations/forward"
SRC_MIGRATIONS = "src/omnibase_infra/migrations/forward"


def _make_migration_dirs(repo_path: Path) -> tuple[Path, Path]:
    """Create both migration directories and return them."""
    docker_dir = repo_path / DOCKER_MIGRATIONS
    src_dir = repo_path / SRC_MIGRATIONS
    docker_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    return docker_dir, src_dir


def _write_sql(directory: Path, name: str, content: str = "-- migration\n") -> Path:
    """Write a .sql file in directory and return the path."""
    f = directory / name
    f.write_text(content, encoding="utf-8")
    return f


def _mock_staged(repo_path: Path, paths: list[str]) -> Any:
    """Return a context manager that patches _get_staged_paths to return paths."""
    return patch(
        "scripts.validation.validate_migration_sequence._get_staged_paths",
        return_value=paths,
    )


# ── extract_sequence_number ───────────────────────────────────────────────────


class TestExtractSequenceNumber:
    """Unit tests for extract_sequence_number()."""

    def test_three_digit_seq(self) -> None:
        assert extract_sequence_number("006_foo.sql") == 6

    def test_two_digit_seq(self) -> None:
        assert extract_sequence_number("036_bar.sql") == 36

    def test_one_digit_seq(self) -> None:
        assert extract_sequence_number("1_create_table.sql") == 1

    def test_zero_padded_seq(self) -> None:
        assert extract_sequence_number("001_init.sql") == 1

    def test_non_sql_returns_none(self) -> None:
        assert extract_sequence_number("006_foo.sh") is None

    def test_non_sql_txt_returns_none(self) -> None:
        assert extract_sequence_number("006_foo.txt") is None

    def test_no_leading_digits_returns_none(self) -> None:
        assert extract_sequence_number("foo_migration.sql") is None

    def test_bare_number_only_stem(self) -> None:
        assert extract_sequence_number("007.sql") == 7

    def test_uppercase_extension_is_sql(self) -> None:
        # .SQL should be treated as sql (case-insensitive suffix check)
        assert extract_sequence_number("010_foo.SQL") == 10

    def test_path_with_dir_component(self) -> None:
        # Only the filename part matters
        assert extract_sequence_number("docker/migrations/forward/036_bar.sql") == 36


# ── validate_migration_sequence ──────────────────────────────────────────────


class TestValidateMigrationSequence:
    """Integration tests for validate_migration_sequence()."""

    # ── no staged migrations → hook does not fire ─────────────────────────

    def test_no_staged_files_exits_cleanly(self, tmp_path: Path) -> None:
        """Hook exits 0 when no migration files are staged."""
        _make_migration_dirs(tmp_path)
        with _mock_staged(tmp_path, []):
            result = validate_migration_sequence(tmp_path)
        assert result.is_valid
        assert not result.has_staged_migrations

    def test_staged_readme_only_does_not_trigger(self, tmp_path: Path) -> None:
        """Non-migration staged file alone does not trigger the hook (R4 AC)."""
        _make_migration_dirs(tmp_path)
        with _mock_staged(tmp_path, ["README.md"]):
            result = validate_migration_sequence(tmp_path)
        assert result.is_valid
        assert not result.has_staged_migrations

    def test_staged_non_sql_in_migration_dir_does_not_trigger(
        self, tmp_path: Path
    ) -> None:
        """A .sh file staged in the migration dir is not treated as a migration."""
        docker_dir, _ = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "000_create.sh", "#!/bin/bash\n")
        with _mock_staged(tmp_path, [f"{DOCKER_MIGRATIONS}/000_create.sh"]):
            result = validate_migration_sequence(tmp_path)
        assert result.is_valid
        assert not result.has_staged_migrations

    # ── unique sequences ──────────────────────────────────────────────────

    def test_all_unique_within_docker_set(self, tmp_path: Path) -> None:
        """Unique sequences in docker set → exit 0."""
        docker_dir, _ = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "001_a.sql")
        _write_sql(docker_dir, "002_b.sql")
        with _mock_staged(tmp_path, [f"{DOCKER_MIGRATIONS}/002_b.sql"]):
            result = validate_migration_sequence(tmp_path)
        assert result.is_valid
        assert result.has_staged_migrations

    def test_all_unique_cross_sets(self, tmp_path: Path) -> None:
        """Unique sequences across both sets → exit 0 (R2 AC)."""
        docker_dir, src_dir = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "001_a.sql")
        _write_sql(src_dir, "002_b.sql")
        with _mock_staged(tmp_path, [f"{SRC_MIGRATIONS}/002_b.sql"]):
            result = validate_migration_sequence(tmp_path)
        assert result.is_valid
        assert len(result.conflicts) == 0

    # ── duplicate detection ───────────────────────────────────────────────

    def test_same_set_duplicate_detected(self, tmp_path: Path) -> None:
        """seq 036 in docker set twice → exit 1, message names both files."""
        docker_dir, _ = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "036_original.sql")
        # Simulate staging a new file with duplicate seq
        staged_path = f"{DOCKER_MIGRATIONS}/036_duplicate.sql"
        with _mock_staged(tmp_path, [staged_path]):
            # Also write the staged file to disk so the fs scan picks it up
            _write_sql(docker_dir, "036_duplicate.sql")
            result = validate_migration_sequence(tmp_path)
        assert not result.is_valid
        assert len(result.conflicts) == 1
        assert result.conflicts[0].sequence == 36
        filenames = {
            Path(result.conflicts[0].file_a).name,
            Path(result.conflicts[0].file_b).name,
        }
        assert "036_original.sql" in filenames
        assert "036_duplicate.sql" in filenames

    def test_cross_set_duplicate_detected(self, tmp_path: Path) -> None:
        """seq 036 in docker set + seq 036 in src set → exit 1 (R2 AC)."""
        docker_dir, src_dir = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "036_docker.sql")
        staged_path = f"{SRC_MIGRATIONS}/036_cross_set.sql"
        with _mock_staged(tmp_path, [staged_path]):
            _write_sql(src_dir, "036_cross_set.sql")
            result = validate_migration_sequence(tmp_path)
        assert not result.is_valid
        assert len(result.conflicts) == 1
        conflict = result.conflicts[0]
        assert conflict.sequence == 36
        names = {Path(conflict.file_a).name, Path(conflict.file_b).name}
        assert "036_docker.sql" in names
        assert "036_cross_set.sql" in names

    def test_multiple_duplicate_pairs(self, tmp_path: Path) -> None:
        """Multiple independent duplicate pairs are all reported."""
        docker_dir, src_dir = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "010_a.sql")
        _write_sql(docker_dir, "010_b.sql")
        _write_sql(src_dir, "010_c.sql")
        staged = [f"{SRC_MIGRATIONS}/010_c.sql"]
        with _mock_staged(tmp_path, staged):
            result = validate_migration_sequence(tmp_path)
        assert not result.is_valid
        # Two pairs share seq 10: (010_a, 010_b) and one of those with 010_c
        conflict_seqs = [c.sequence for c in result.conflicts]
        assert all(s == 10 for s in conflict_seqs)

    # ── staged file not yet on disk (staged-only file) ────────────────────

    def test_staged_only_file_included_in_scan(self, tmp_path: Path) -> None:
        """A staged .sql file not yet on disk is included in the duplicate scan."""
        docker_dir, _ = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "005_existing.sql")
        # Staged path that does not exist on disk yet
        staged_path = f"{DOCKER_MIGRATIONS}/005_staged_only.sql"
        with _mock_staged(tmp_path, [staged_path]):
            result = validate_migration_sequence(tmp_path)
        assert not result.is_valid
        conflict = result.conflicts[0]
        assert conflict.sequence == 5


# ── generate_report ───────────────────────────────────────────────────────────


class TestGenerateReport:
    def test_no_staged_migrations_report(self) -> None:
        result = SequenceValidationResult(has_staged_migrations=False)
        report = generate_report(result)
        assert "skipped" in report

    def test_pass_report(self) -> None:
        result = SequenceValidationResult(has_staged_migrations=True, files_scanned=10)
        report = generate_report(result)
        assert "PASS" in report
        assert "10" in report

    def test_failure_report_contains_conflict_info(self) -> None:
        result = SequenceValidationResult(
            has_staged_migrations=True,
            files_scanned=4,
            conflicts=[
                DuplicateConflict(
                    sequence=36,
                    file_a="docker/migrations/forward/036_original.sql",
                    file_b="docker/migrations/forward/036_duplicate.sql",
                )
            ],
        )
        report = generate_report(result)
        assert "DUPLICATE" in report
        assert "036" in report
        assert "036_original.sql" in report
        assert "036_duplicate.sql" in report

    def test_failure_report_mentions_both_dirs(self) -> None:
        result = SequenceValidationResult(
            has_staged_migrations=True,
            files_scanned=2,
            conflicts=[
                DuplicateConflict(
                    sequence=36,
                    file_a="docker/migrations/forward/036_docker.sql",
                    file_b="src/omnibase_infra/migrations/forward/036_src.sql",
                )
            ],
        )
        report = generate_report(result)
        assert "036_docker.sql" in report
        assert "036_src.sql" in report
        assert "namespace" in report


# ── RuntimeError on git failure ───────────────────────────────────────────────


class TestGitErrorHandling:
    def test_git_not_found_raises_runtime_error(self, tmp_path: Path) -> None:
        """If git is not available, validate raises RuntimeError (exit 2)."""
        with patch(
            "scripts.validation.validate_migration_sequence._get_staged_paths",
            side_effect=RuntimeError("git executable not found"),
        ):
            with pytest.raises(RuntimeError, match="git executable not found"):
                validate_migration_sequence(tmp_path)

    def test_git_nonzero_exit_raises_runtime_error(self, tmp_path: Path) -> None:
        """Non-zero git exit code raises RuntimeError (exit 2)."""
        with patch(
            "scripts.validation.validate_migration_sequence._get_staged_paths",
            side_effect=RuntimeError("git diff --cached failed"),
        ):
            with pytest.raises(RuntimeError, match="git diff --cached failed"):
                validate_migration_sequence(tmp_path)
