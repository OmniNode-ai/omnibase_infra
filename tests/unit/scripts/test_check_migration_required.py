"""Tests for scripts/check_migration_required.py writer-without-migration gate."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "check_migration_required.py"


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_migration_required", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.unit
class TestWriterDetection:
    def test_detects_writer_postgres_files(self):
        checker = load_checker()
        assert checker.is_writer_file(
            "src/omnibase_infra/nodes/agent_actions/writer_postgres.py"
        )
        assert not checker.is_writer_file("src/omnibase_infra/nodes/other/model.py")

    def test_detects_handler_postgres_files(self):
        checker = load_checker()
        assert checker.is_writer_file(
            "src/omnibase_infra/nodes/foo/handler_registration_storage_postgres.py"
        )

    def test_no_migration_comment_bypasses_check(self):
        checker = load_checker()
        assert checker.has_bypass_comment(
            "# no-migration: table already exists in docker set\n"
        )
        assert not checker.has_bypass_comment("# unrelated comment\n")

    def test_migration_file_present_passes(self):
        checker = load_checker()
        changed_files = [
            "src/omnibase_infra/nodes/foo/writer_postgres.py",
            "docker/migrations/forward/036_add_foo_table.sql",
        ]
        violations = checker.find_violations(changed_files, bypass_comment=None)
        assert violations == []

    def test_writer_without_migration_fails(self):
        checker = load_checker()
        changed_files = [
            "src/omnibase_infra/nodes/foo/writer_postgres.py",
        ]
        violations = checker.find_violations(changed_files, bypass_comment=None)
        assert len(violations) == 1
        assert "writer_postgres.py" in violations[0]
