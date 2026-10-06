# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Migration-sequence COMPUTE node and runtime tests (OMN-20568).

Ports every case from test_validate_migration_sequence.py (OMN-3570), covering
filename parsing, staged-file gating, shared sequence conflicts, namespaced
subtree exclusions, report wording, and Git errors. Also pins CLI exit codes.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationReport,
)
from omnibase_infra.nodes.node_migration_sequence_check_compute import (
    runtime_migration_sequence_check as runtime,
)
from omnibase_infra.nodes.node_migration_sequence_check_compute.handler import (
    NodeMigrationSequenceCheckCompute,
    migration_scan_paths,
)
from omnibase_infra.nodes.node_migration_sequence_check_compute.handler import (
    _sequence_number as extract_sequence_number,
)
from omnibase_infra.nodes.node_migration_sequence_check_compute.models import (
    ModelMigrationSequenceCheckInput,
)

pytestmark = pytest.mark.unit

DOCKER_MIGRATIONS = "docker/migrations/forward"
SRC_MIGRATIONS = "src/omnibase_infra/migrations/forward"


def _make_migration_dirs(repo_path: Path) -> tuple[Path, Path]:
    docker_dir = repo_path / DOCKER_MIGRATIONS
    src_dir = repo_path / SRC_MIGRATIONS
    docker_dir.mkdir(parents=True)
    src_dir.mkdir(parents=True)
    return docker_dir, src_dir


def _write_sql(directory: Path, name: str, content: str = "-- migration\n") -> Path:
    path = directory / name
    path.write_text(content, encoding="utf-8")
    return path


def _collect(
    repo_path: Path, staged: tuple[str, ...]
) -> ModelMigrationSequenceCheckInput:
    """Exercise the filesystem collector with explicit staged paths."""
    with patch.object(runtime, "_staged_paths", return_value=staged):
        return runtime.collect_input(repo_path)


def _check(request: ModelMigrationSequenceCheckInput) -> ModelValidationReport:
    return NodeMigrationSequenceCheckCompute().handle(request)


class TestExtractSequenceNumber:
    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("006_foo.sql", 6),
            ("036_bar.sql", 36),
            ("1_create_table.sql", 1),
            ("001_init.sql", 1),
            ("006_foo.sh", None),
            ("006_foo.txt", None),
            ("foo_migration.sql", None),
            ("007.sql", 7),
            ("010_foo.SQL", 10),
            (f"{DOCKER_MIGRATIONS}/036_bar.sql", 36),
        ],
        ids=[
            "three-digit",
            "two-digit-value",
            "one-digit",
            "zero-padded",
            "shell",
            "text",
            "no-leading-digits",
            "bare-number",
            "uppercase-sql",
            "directory-component",
        ],
    )
    def test_filename_parser(self, filename: str, expected: int | None) -> None:
        assert extract_sequence_number(filename) == expected


class TestMigrationSequenceCheck:
    def test_no_staged_files_exits_cleanly(self, tmp_path: Path) -> None:
        _make_migration_dirs(tmp_path)
        request = _collect(tmp_path, ())
        assert _check(request).overall_status == "PASS"
        assert migration_scan_paths(request) == ()

    def test_staged_readme_only_does_not_trigger(self, tmp_path: Path) -> None:
        _make_migration_dirs(tmp_path)
        request = _collect(tmp_path, ("README.md",))
        assert _check(request).overall_status == "PASS"
        assert migration_scan_paths(request) == ()

    def test_staged_non_sql_in_migration_dir_does_not_trigger(
        self, tmp_path: Path
    ) -> None:
        docker_dir, _ = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "000_create.sh", "#!/bin/bash\n")
        request = _collect(tmp_path, (f"{DOCKER_MIGRATIONS}/000_create.sh",))
        assert _check(request).overall_status == "PASS"
        assert migration_scan_paths(request) == ()

    def test_all_unique_within_docker_set(self, tmp_path: Path) -> None:
        docker_dir, _ = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "001_a.sql")
        _write_sql(docker_dir, "002_b.sql")
        request = _collect(tmp_path, (f"{DOCKER_MIGRATIONS}/002_b.sql",))
        report = _check(request)
        assert report.overall_status == "PASS"
        assert report.findings == ()
        assert len(migration_scan_paths(request)) == 2

    def test_all_unique_cross_sets(self, tmp_path: Path) -> None:
        docker_dir, src_dir = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "001_a.sql")
        _write_sql(src_dir, "002_b.sql")
        request = _collect(tmp_path, (f"{SRC_MIGRATIONS}/002_b.sql",))
        report = _check(request)
        assert report.overall_status == "PASS"
        assert report.findings == ()
        assert len(migration_scan_paths(request)) == 2

    def test_same_set_duplicate_detected(self, tmp_path: Path) -> None:
        docker_dir, _ = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "036_original.sql")
        _write_sql(docker_dir, "036_duplicate.sql")
        request = _collect(tmp_path, (f"{DOCKER_MIGRATIONS}/036_duplicate.sql",))
        report = _check(request)
        assert report.overall_status == "FAIL"
        assert len(report.findings) == 1
        finding = report.findings[0]
        assert finding.validator_id == "migration-sequence"
        assert finding.severity == "FAIL"
        assert finding.rule_id == "duplicate-sequence"
        assert finding.evidence == {
            "sequence": 36,
            "file_a": f"{DOCKER_MIGRATIONS}/036_duplicate.sql",
            "file_b": f"{DOCKER_MIGRATIONS}/036_original.sql",
        }
        assert finding.location == f"{DOCKER_MIGRATIONS}/036_original.sql"
        assert finding.remediation is not None
        assert "renumber" in finding.remediation

    def test_cross_set_duplicate_detected(self, tmp_path: Path) -> None:
        docker_dir, src_dir = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "036_docker.sql")
        _write_sql(src_dir, "036_cross_set.sql")
        request = _collect(tmp_path, (f"{SRC_MIGRATIONS}/036_cross_set.sql",))
        report = _check(request)
        assert report.overall_status == "FAIL"
        assert len(report.findings) == 1
        assert report.findings[0].evidence == {
            "sequence": 36,
            "file_a": f"{DOCKER_MIGRATIONS}/036_docker.sql",
            "file_b": f"{SRC_MIGRATIONS}/036_cross_set.sql",
        }

    def test_multiple_duplicate_pairs(self, tmp_path: Path) -> None:
        docker_dir, src_dir = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "010_a.sql")
        _write_sql(docker_dir, "010_b.sql")
        _write_sql(src_dir, "010_c.sql")
        request = _collect(tmp_path, (f"{SRC_MIGRATIONS}/010_c.sql",))
        report = _check(request)
        assert report.overall_status == "FAIL"
        assert [finding.evidence for finding in report.findings] == [
            {
                "sequence": 10,
                "file_a": f"{DOCKER_MIGRATIONS}/010_a.sql",
                "file_b": f"{DOCKER_MIGRATIONS}/010_b.sql",
            },
            {
                "sequence": 10,
                "file_a": f"{DOCKER_MIGRATIONS}/010_a.sql",
                "file_b": f"{SRC_MIGRATIONS}/010_c.sql",
            },
        ]

    def test_staged_only_file_included_in_scan(self, tmp_path: Path) -> None:
        docker_dir, _ = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "005_existing.sql")
        staged_path = f"{DOCKER_MIGRATIONS}/005_staged_only.sql"
        request = _collect(tmp_path, (staged_path,))
        assert not (tmp_path / staged_path).exists()
        assert staged_path in migration_scan_paths(request)
        report = _check(request)
        assert report.overall_status == "FAIL"
        assert len(report.findings) == 1
        assert report.findings[0].evidence == {
            "sequence": 5,
            "file_a": f"{DOCKER_MIGRATIONS}/005_existing.sql",
            "file_b": staged_path,
        }

    def test_node_subtree_file_does_not_collide_with_flat_sequence(
        self, tmp_path: Path
    ) -> None:
        docker_dir, _ = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "076_add_savings_estimate_provenance.sql")
        node_dir = docker_dir / "nodes" / "node_projection_savings"
        node_dir.mkdir(parents=True)
        node_file = "076_create_delegation_savings_projection_view.sql"
        _write_sql(node_dir, node_file)
        staged_path = f"{DOCKER_MIGRATIONS}/nodes/node_projection_savings/{node_file}"
        request = _collect(tmp_path, (staged_path,))
        assert _check(request).overall_status == "PASS"
        assert migration_scan_paths(request) == ()

    def test_node_subtree_excluded_even_when_flat_migration_staged(
        self, tmp_path: Path
    ) -> None:
        docker_dir, _ = _make_migration_dirs(tmp_path)
        _write_sql(docker_dir, "090_new_flat.sql")
        node_dir = docker_dir / "nodes" / "node_projection_savings"
        node_dir.mkdir(parents=True)
        _write_sql(node_dir, "090_node_view.sql")
        staged_path = f"{DOCKER_MIGRATIONS}/090_new_flat.sql"
        request = _collect(tmp_path, (staged_path,))
        assert _check(request).overall_status == "PASS"
        assert migration_scan_paths(request) == (staged_path,)
        # The pure handler also excludes node paths supplied explicitly.
        explicit = request.model_copy(
            update={
                "migration_paths": (
                    *request.migration_paths,
                    f"{DOCKER_MIGRATIONS}/nodes/node_projection_savings/090_node_view.sql",
                ),
            }
        )
        assert _check(explicit).overall_status == "PASS"
        assert migration_scan_paths(explicit) == (staged_path,)


class TestRuntimeWording:
    def test_no_staged_migrations_report(self) -> None:
        request = ModelMigrationSequenceCheckInput()
        assert "skipped" in runtime._render(_check(request), request)

    def test_pass_report(self) -> None:
        paths = tuple(
            f"{DOCKER_MIGRATIONS}/{sequence:03d}_a.sql" for sequence in range(10)
        )
        request = ModelMigrationSequenceCheckInput(
            staged_paths=paths[:1], migration_paths=paths
        )
        report = runtime._render(_check(request), request)
        assert "PASS" in report
        assert "10 file(s) scanned" in report

    def test_failure_report_contains_conflict_info(self) -> None:
        paths = (
            f"{DOCKER_MIGRATIONS}/036_original.sql",
            f"{DOCKER_MIGRATIONS}/036_duplicate.sql",
        )
        request = ModelMigrationSequenceCheckInput(
            staged_paths=paths[:1], migration_paths=paths
        )
        report = runtime._render(_check(request), request)
        assert "DUPLICATE" in report
        assert "036" in report
        assert "036_original.sql" in report
        assert "036_duplicate.sql" in report

    def test_failure_report_mentions_both_dirs(self) -> None:
        paths = (f"{DOCKER_MIGRATIONS}/036_docker.sql", f"{SRC_MIGRATIONS}/036_src.sql")
        request = ModelMigrationSequenceCheckInput(
            staged_paths=paths[:1], migration_paths=paths
        )
        report = runtime._render(_check(request), request)
        assert "036_docker.sql" in report
        assert "036_src.sql" in report
        assert "namespace" in report


class TestGitErrorHandling:
    @pytest.mark.parametrize(
        ("failure", "message"),
        [
            (FileNotFoundError(), "git executable not found"),
            (subprocess.TimeoutExpired("git", 10), "git diff --cached timed out"),
        ],
        ids=["git-missing", "git-timeout"],
    )
    def test_git_execution_error_raises_runtime_error(
        self, tmp_path: Path, failure: Exception, message: str
    ) -> None:
        with patch.object(subprocess, "run", autospec=True, side_effect=failure):
            with pytest.raises(RuntimeError, match=message):
                runtime.collect_input(tmp_path)

    def test_git_nonzero_exit_raises_runtime_error(self, tmp_path: Path) -> None:
        completed = subprocess.CompletedProcess(
            args=["git", "diff", "--cached", "--name-only"],
            returncode=128,
            stdout="",
            stderr="not a git repository\n",
        )
        with patch.object(subprocess, "run", autospec=True, return_value=completed):
            with pytest.raises(
                RuntimeError,
                match=r"git diff --cached failed \(exit 128\): not a git repository",
            ):
                runtime.collect_input(tmp_path)


class TestMain:
    @pytest.mark.parametrize(
        ("paths", "verbose", "expected_exit", "message"),
        [
            ((), False, 0, ""),
            ((), True, 0, "skipped"),
            ((f"{DOCKER_MIGRATIONS}/001_a.sql",), False, 0, "PASS"),
            (
                (f"{DOCKER_MIGRATIONS}/001_a.sql", f"{SRC_MIGRATIONS}/001_b.sql"),
                False,
                1,
                "DUPLICATE",
            ),
        ],
        ids=["quiet-skip", "verbose-skip", "pass", "fail"],
    )
    def test_exit_codes_and_output(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        paths: tuple[str, ...],
        verbose: bool,
        expected_exit: int,
        message: str,
    ) -> None:
        request = ModelMigrationSequenceCheckInput(
            staged_paths=paths, migration_paths=paths
        )
        argv = [str(tmp_path), "--verbose"] if verbose else [str(tmp_path)]
        with patch.object(runtime, "collect_input", return_value=request) as collect:
            assert runtime.main(argv) == expected_exit
        collect.assert_called_once_with(tmp_path.resolve())
        captured = capsys.readouterr()
        assert captured.err == ""
        if message:
            assert message in captured.out
        else:
            assert captured.out == ""

    @pytest.mark.parametrize(
        ("failure", "expected_error"),
        [
            (
                RuntimeError("git executable not found"),
                "Error: git executable not found\n",
            ),
            (OSError("scan failed"), "Unexpected error: scan failed\n"),
        ],
        ids=["git-error", "filesystem-error"],
    )
    def test_runtime_failure_exits_two(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        failure: Exception,
        expected_error: str,
    ) -> None:
        with patch.object(runtime, "collect_input", side_effect=failure):
            assert runtime.main([str(tmp_path)]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == expected_error
