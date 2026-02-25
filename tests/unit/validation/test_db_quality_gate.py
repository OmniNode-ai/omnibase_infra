"""Tests for the DB quality gate validator (OMN-1785).

Validates that the quality gate correctly detects and reports:
- Domain-specific DB adapter classes
- Direct SQL in domain code
- Direct DB connection calls
- Proper exemption of infra and test directories
- Escape-hatch comment markers
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest


@pytest.fixture
def tmp_src(tmp_path: Path) -> Path:
    """Create a temporary src directory with the expected structure."""
    src = tmp_path / "src"
    src.mkdir()
    return src


def _write_py(directory: Path, filename: str, content: str) -> Path:
    """Helper to write a Python file with dedented content."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(dedent(content), encoding="utf-8")
    return path


class TestAdapterClassDetection:
    """Tests for forbidden adapter class pattern detection."""

    def test_detects_adapter_postgres_class(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "adapters.py",
            """\
            class UserAdapterPostgres:
                pass
            """,
        )
        violations = scan_file(f)
        assert len(violations) == 1
        assert violations[0].category == "adapter_class"
        assert "UserAdapterPostgres" in violations[0].matched_text

    def test_detects_postgres_adapter_class(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "db.py",
            """\
            class PostgresAdapterUser:
                pass
            """,
        )
        violations = scan_file(f)
        assert len(violations) == 1
        assert violations[0].category == "adapter_class"

    def test_escape_hatch_suppresses_adapter_check(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "adapters.py",
            """\
            class UserAdapterPostgres:  # db-adapter-ok
                pass
            """,
        )
        violations = scan_file(f)
        assert len(violations) == 0


class TestDirectSqlDetection:
    """Tests for direct SQL pattern detection."""

    def test_detects_select_from(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "queries.py",
            """\
            query = "SELECT id, name FROM users WHERE active = true"
            """,
        )
        violations = scan_file(f)
        assert len(violations) == 1
        assert violations[0].category == "direct_sql"

    def test_detects_insert_into(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "writer.py",
            """\
            query = "INSERT INTO users (name) VALUES ($1)"
            """,
        )
        violations = scan_file(f)
        assert len(violations) == 1
        assert violations[0].category == "direct_sql"

    def test_detects_update_set(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "updater.py",
            """\
            query = "UPDATE users SET name = $1 WHERE id = $2"
            """,
        )
        violations = scan_file(f)
        assert len(violations) == 1
        assert violations[0].category == "direct_sql"

    def test_detects_delete_from(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "cleaner.py",
            """\
            query = "DELETE FROM users WHERE expired = true"
            """,
        )
        violations = scan_file(f)
        assert len(violations) == 1
        assert violations[0].category == "direct_sql"

    def test_escape_hatch_suppresses_sql_check(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "queries.py",
            """\
            query = "SELECT id FROM users"  # sql-ok
            """,
        )
        violations = scan_file(f)
        assert len(violations) == 0

    def test_comment_lines_are_not_flagged(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "notes.py",
            """\
            # SELECT id FROM users
            # This is just a comment about the query
            """,
        )
        violations = scan_file(f)
        assert len(violations) == 0


class TestDirectConnectDetection:
    """Tests for direct DB connection pattern detection."""

    def test_detects_psycopg_connect(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "db.py",
            """\
            import psycopg
            conn = psycopg.connect("dbname=test")
            """,
        )
        violations = scan_file(f)
        assert any(v.category == "direct_connect" for v in violations)

    def test_detects_asyncpg_connect(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import scan_file

        f = _write_py(
            tmp_src / "myapp",
            "db.py",
            """\
            import asyncpg
            conn = await asyncpg.connect("postgresql://localhost/test")
            """,
        )
        violations = scan_file(f)
        assert any(v.category == "direct_connect" for v in violations)


class TestExemptions:
    """Tests for directory-level exemptions."""

    def test_omnibase_infra_files_exempt(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import validate_db_quality_gate

        _write_py(
            tmp_src / "omnibase_infra" / "handlers",
            "handler_db.py",
            """\
            class UserAdapterPostgres:
                pass
            query = "SELECT id FROM users"
            """,
        )
        result = validate_db_quality_gate([tmp_src])
        assert result.is_valid
        assert result.files_skipped >= 1

    def test_test_files_exempt(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import validate_db_quality_gate

        _write_py(
            tmp_src / "tests" / "unit",
            "test_db.py",
            """\
            class TestAdapterPostgres:
                pass
            query = "SELECT id FROM users"
            """,
        )
        result = validate_db_quality_gate([tmp_src])
        assert result.is_valid

    def test_domain_code_not_exempt(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import validate_db_quality_gate

        _write_py(
            tmp_src / "omniclaude",
            "persistence.py",
            """\
            class ChatAdapterPostgres:
                pass
            """,
        )
        result = validate_db_quality_gate([tmp_src])
        assert not result.is_valid
        assert len(result.adapter_violations) == 1


class TestFullScan:
    """Integration tests for the full scan pipeline."""

    def test_clean_codebase_passes(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import validate_db_quality_gate

        _write_py(
            tmp_src / "myapp",
            "service.py",
            """\
            from omnibase_infra.runtime.db import PostgresRepositoryRuntime

            class UserService:
                def __init__(self, repo: PostgresRepositoryRuntime) -> None:
                    self.repo = repo
            """,
        )
        result = validate_db_quality_gate([tmp_src])
        assert result.is_valid
        assert result.files_checked >= 1

    def test_nonexistent_path_skipped(self, tmp_path: Path) -> None:
        from scripts.validation.validate_db_quality_gate import validate_db_quality_gate

        result = validate_db_quality_gate([tmp_path / "nonexistent"], verbose=False)
        assert result.is_valid
        assert result.files_checked == 0

    def test_report_generation(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import (
            generate_report,
            validate_db_quality_gate,
        )

        _write_py(
            tmp_src / "myapp",
            "bad.py",
            """\
            class FooAdapterPostgres:
                pass
            """,
        )
        result = validate_db_quality_gate([tmp_src])
        report = generate_report(result)
        assert "FAIL" in report
        assert "OMN-1785" in report
        assert "adapter_class" in report.lower() or "Adapter class" in report

    def test_report_pass(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import (
            generate_report,
            validate_db_quality_gate,
        )

        _write_py(
            tmp_src / "myapp",
            "clean.py",
            """\
            x = 1
            """,
        )
        result = validate_db_quality_gate([tmp_src])
        report = generate_report(result)
        assert "PASS" in report


class TestMigrationDirectoryExempt:
    """Tests that migration files are exempt."""

    def test_migration_files_exempt(self, tmp_src: Path) -> None:
        from scripts.validation.validate_db_quality_gate import validate_db_quality_gate

        _write_py(
            tmp_src / "migrations" / "versions",
            "001_create_users.py",
            """\
            def upgrade():
                op.execute("SELECT 1 FROM users")
                op.execute("INSERT INTO users (name) VALUES ('admin')")
            """,
        )
        result = validate_db_quality_gate([tmp_src])
        assert result.is_valid
