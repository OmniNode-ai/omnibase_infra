"""Tests for scripts/check_migration_required.py writer-without-migration gate."""

import importlib.util
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT_PATH = Path(__file__).parents[3] / "scripts" / "check_migration_required.py"


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_migration_required", SCRIPT_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_git_mock(merge_base_sha: str, base_files: dict[str, str]):
    """Return a side_effect for subprocess.check_output that dispatches by git subcommand."""

    def side_effect(cmd, **kwargs):
        if cmd[:2] == ["git", "merge-base"]:
            return merge_base_sha + "\n"
        if cmd[:2] == ["git", "show"]:
            ref_path = cmd[2]
            path = ref_path.split(":", 1)[1]
            if path in base_files:
                return base_files[path]
            raise subprocess.CalledProcessError(1, cmd)
        if cmd[:2] == ["git", "diff"]:
            return ""
        raise subprocess.CalledProcessError(1, cmd)

    return side_effect


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
        violations = checker.find_violations(changed_files)
        assert violations == []

    def test_writer_without_migration_fails(self):
        checker = load_checker()
        changed_files = [
            "src/omnibase_infra/nodes/foo/writer_postgres.py",
        ]
        violations = checker.find_violations(changed_files)
        assert len(violations) == 1
        assert "writer_postgres.py" in violations[0]


@pytest.mark.unit
class TestCosmeticDetection:
    def test_docstring_only_change_is_cosmetic(self, tmp_path):
        checker = load_checker()
        base_src = 'def write():\n    """Old docstring."""\n    return 1\n'
        current_src = 'def write():\n    """New docstring."""\n    return 1\n'
        writer = tmp_path / "writer_postgres.py"
        writer.write_text(current_src)
        with patch.object(
            subprocess,
            "check_output",
            side_effect=make_git_mock("abc123", {str(writer): base_src}),
        ):
            result = checker.is_cosmetic_only(str(writer), "abc123")
        assert result is True

    def test_comment_only_change_is_cosmetic(self, tmp_path):
        checker = load_checker()
        base_src = "# old comment\nx = 1\n"
        current_src = "# new comment\nx = 1\n"
        writer = tmp_path / "writer_postgres.py"
        writer.write_text(current_src)
        with patch.object(
            subprocess,
            "check_output",
            side_effect=make_git_mock("abc123", {str(writer): base_src}),
        ):
            result = checker.is_cosmetic_only(str(writer), "abc123")
        assert result is True

    def test_code_change_not_cosmetic(self, tmp_path):
        checker = load_checker()
        base_src = "x = 1\n"
        current_src = "x = 2\n"
        writer = tmp_path / "writer_postgres.py"
        writer.write_text(current_src)
        with patch.object(
            subprocess,
            "check_output",
            side_effect=make_git_mock("abc123", {str(writer): base_src}),
        ):
            result = checker.is_cosmetic_only(str(writer), "abc123")
        assert result is False

    def test_new_file_not_cosmetic(self, tmp_path):
        checker = load_checker()
        current_src = "x = 1\n"
        writer = tmp_path / "writer_postgres.py"
        writer.write_text(current_src)
        # base_files is empty → git show raises CalledProcessError → new file
        with patch.object(
            subprocess,
            "check_output",
            side_effect=make_git_mock("abc123", {}),
        ):
            result = checker.is_cosmetic_only(str(writer), "abc123")
        assert result is False

    def test_syntax_error_not_cosmetic(self, tmp_path):
        checker = load_checker()
        base_src = "def foo(\n"  # invalid syntax
        current_src = "def foo(\n"
        writer = tmp_path / "writer_postgres.py"
        writer.write_text(current_src)
        with patch.object(
            subprocess,
            "check_output",
            side_effect=make_git_mock("abc123", {str(writer): base_src}),
        ):
            result = checker.is_cosmetic_only(str(writer), "abc123")
        assert result is False

    def test_empty_merge_base_not_cosmetic(self):
        checker = load_checker()
        result = checker.is_cosmetic_only("some/path.py", "")
        assert result is False

    def test_docstring_removed_empty_body(self, tmp_path):
        checker = load_checker()
        # A function with only a docstring — stripping it leaves empty body
        # ast.walk + body mutation shouldn't crash; just returns False or True
        base_src = 'def foo():\n    """Only a docstring."""\n'
        current_src = 'def foo():\n    """Different docstring."""\n'
        writer = tmp_path / "writer_postgres.py"
        writer.write_text(current_src)
        with patch.object(
            subprocess,
            "check_output",
            side_effect=make_git_mock("abc123", {str(writer): base_src}),
        ):
            # Should not crash — may return True or False depending on AST
            checker.is_cosmetic_only(str(writer), "abc123")

    def test_merge_base_used_not_head(self, tmp_path):
        checker = load_checker()
        base_src = "x = 1\n"
        current_src = "x = 1\n"
        writer = tmp_path / "writer_postgres.py"
        writer.write_text(current_src)
        calls = []

        def tracking_mock(cmd, **kwargs):
            calls.append(cmd)
            return make_git_mock("deadbeef", {str(writer): base_src})(cmd, **kwargs)

        with patch.object(subprocess, "check_output", side_effect=tracking_mock):
            checker.is_cosmetic_only(str(writer), "deadbeef")

        # Verify git show was called with the merge-base SHA, not HEAD
        show_calls = [c for c in calls if c[:2] == ["git", "show"]]
        assert len(show_calls) == 1
        assert show_calls[0][2].startswith("deadbeef:")
