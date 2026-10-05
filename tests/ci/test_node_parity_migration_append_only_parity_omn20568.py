# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Real-repository parity for the append-only script and COMPUTE node."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationReport,
    ModelValidationRequestRef,
)
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.nodes.node_migration_append_only_check_compute import (
    NodeMigrationAppendOnlyCheckCompute,
)
from omnibase_infra.nodes.node_migration_append_only_check_compute import (
    runtime_migration_append_only_check as node_runtime,
)
from omnibase_infra.nodes.node_migration_append_only_check_compute.models import (
    ModelMigrationAppendOnlyCheckInput,
)
from scripts.validation import check_migration_append_only as old
from tests.ci.test_migration_append_only_guard_omn16705 import (
    _APPLIED,
    _APPLIED_BODY,
    _SUCCESSOR,
    _SUCCESSOR_BODY,
    _edit_applied_migration,
    _manifest_row,
    _write,
)

pytestmark = pytest.mark.unit

FORWARD = "docker/migrations/forward"
MANIFEST = f"{FORWARD}/_ledger/application-migrations.tsv"
SUPERSESSIONS = f"{FORWARD}/_ledger/migration-supersessions.tsv"
SECOND = "nodes/node_example/0003_second.sql"
SCENARIOS = (
    "empty",
    "modify",
    "delete",
    "rename",
    "add",
    "flat",
    "valid_supersession",
    "missing_successor",
    "bad_ticket",
    "different_node",
    "descending_ordinal",
    "equal_ordinal",
    "two_violations",
    "undeclared_successor",
    "invalid_path",
    "malformed_tsv",
    "empty_tsv_field",
    "blank_tsv_lines",
    "multiple_rows_one_valid",
    "multiple_rows_invalid",
    "rename_with_supersession",
    "delete_with_supersession",
    "stale_supersession",
    "amend_new_migration",
    "empty_base_manifest",
    "absent_base_manifest",
)


@pytest.fixture(autouse=True)
def _no_git_location_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.email=guard@test.invalid",
            "-c",
            "user.name=guard test",
            *args,
        ],
        cwd=repo,
        env=scrub_git_location_env(),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _build_repo(root: Path, scenario: str = "empty") -> Path:
    root.mkdir()
    _git(root, "init", "-q", "-b", "dev")
    _write(root, f"{FORWARD}/{_APPLIED}", _APPLIED_BODY)
    _write(root, f"{FORWARD}/{SECOND}", "SELECT 3;\n")
    _write(root, f"{FORWARD}/001_flat.sql", "SELECT 1;\n")
    manifest = (
        _manifest_row(_APPLIED, _APPLIED_BODY)
        + "\n"
        + _manifest_row(SECOND, "SELECT 3;\n")
        + "\n"
    )
    if scenario == "empty_base_manifest":
        manifest = "\n"
    if scenario != "absent_base_manifest":
        _write(root, MANIFEST, manifest)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    _git(root, "branch", "base-marker")
    _git(root, "update-ref", "refs/remotes/origin/dev", "HEAD")
    return root


def _supersede(
    root: Path,
    successor: str = _SUCCESSOR,
    ticket: str = "OMN-16705",
    *,
    add: bool = True,
    declare: bool = True,
) -> None:
    _edit_applied_migration(root, "-- restored to applied bytes\n" + _APPLIED_BODY)
    if add:
        _write(root, f"{FORWARD}/{successor}", _SUCCESSOR_BODY)
    if declare:
        _write(
            root,
            MANIFEST,
            (root / MANIFEST).read_text()
            + _manifest_row(successor, _SUCCESSOR_BODY)
            + "\n",
        )
    _write(
        root,
        SUPERSESSIONS,
        "\t".join([_APPLIED, successor, ticket, "restore applied bytes"]) + "\n",
    )


def _apply(root: Path, scenario: str) -> None:
    if scenario in {"empty", "empty_base_manifest", "absent_base_manifest"}:
        return
    if scenario in {"modify", "two_violations"}:
        _edit_applied_migration(
            root, _APPLIED_BODY + "ALTER TABLE example ADD COLUMN x INT;\n"
        )
        if scenario == "two_violations":
            _write(root, f"{FORWARD}/{SECOND}", "SELECT 4;\n")
        return
    if scenario == "delete":
        (root / FORWARD / _APPLIED).unlink()
        _write(root, MANIFEST, "")
        return
    if scenario == "rename":
        (root / FORWARD / _APPLIED).rename(root / FORWARD / _SUCCESSOR)
        return
    if scenario in {"add", "amend_new_migration"}:
        _write(root, f"{FORWARD}/{_SUCCESSOR}", _SUCCESSOR_BODY)
        _write(
            root,
            MANIFEST,
            (root / MANIFEST).read_text()
            + _manifest_row(_SUCCESSOR, _SUCCESSOR_BODY)
            + "\n",
        )
        if scenario == "amend_new_migration":
            _git(root, "add", "-A")
            _git(root, "commit", "-qm", "add successor on branch")
            _write(
                root,
                f"{FORWARD}/{_SUCCESSOR}",
                _SUCCESSOR_BODY + "-- amended on same branch\n",
            )
        return
    if scenario == "flat":
        _write(root, f"{FORWARD}/001_flat.sql", "SELECT 2;\n")
        return
    successor = {
        "different_node": "nodes/node_other/0002_other.sql",
        "descending_ordinal": "nodes/node_example/0000_previous.sql",
        "equal_ordinal": "nodes/node_example/0001_other.sql",
        "invalid_path": "nodes/node_example/not_numbered.sql",
    }.get(scenario, _SUCCESSOR)
    _supersede(
        root,
        successor,
        "bad-ticket" if scenario == "bad_ticket" else "OMN-16705",
        add=scenario != "missing_successor",
        declare=scenario != "undeclared_successor",
    )
    if scenario == "malformed_tsv":
        _write(root, SUPERSESSIONS, "bad\trow\n")
    elif scenario == "empty_tsv_field":
        _write(root, SUPERSESSIONS, f"{_APPLIED}\t{_SUCCESSOR}\tOMN-16705\t\n")
    elif scenario == "blank_tsv_lines":
        _write(
            root, SUPERSESSIONS, "\n  \n" + (root / SUPERSESSIONS).read_text() + "\n"
        )
    elif scenario.startswith("multiple_rows"):
        bad_row = f"{_APPLIED}\t{_SUCCESSOR}\tbad-ticket\treason\n"
        original = (root / SUPERSESSIONS).read_text()
        _write(
            root,
            SUPERSESSIONS,
            bad_row
            + (
                original
                if scenario == "multiple_rows_one_valid"
                else bad_row.replace("bad-ticket", "another-bad-ticket")
            ),
        )
    elif scenario == "rename_with_supersession":
        (root / FORWARD / _APPLIED).unlink()
        _write(root, f"{FORWARD}/{_SUCCESSOR}", _APPLIED_BODY)
    elif scenario == "delete_with_supersession":
        (root / FORWARD / _APPLIED).unlink()
    elif scenario == "stale_supersession":
        _git(root, "add", "-A")
        _git(root, "commit", "-qm", "land supersession")
        _git(root, "branch", "landed")
        _edit_applied_migration(root, "-- second unauthorised edit\n")


@dataclass(frozen=True)
class Verdict:
    exit_code: int
    violations: frozenset[str]
    output: str
    error: str


def _run_main(
    main: Callable[[list[str] | None], int],
    args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> Verdict:
    capsys.readouterr()
    code = main(args)
    captured = capsys.readouterr()
    return Verdict(
        code,
        frozenset(
            line.removeprefix("  - ")
            for line in captured.err.splitlines()
            if line.startswith("  - ")
        ),
        captured.out,
        captured.err,
    )


def _assert_parity(expected: Verdict, actual: Verdict) -> None:
    assert actual.exit_code == expected.exit_code
    assert actual.violations == expected.violations
    assert actual.output == expected.output
    assert actual.error == expected.error


@pytest.mark.parametrize("staged", [True, False], ids=["staged", "committed"])
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_node_parity_migration_append_only_parity_matrix(
    scenario: str, staged: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _build_repo(tmp_path / "repo", scenario)
    _apply(root, scenario)
    _git(root, "add", "-A")
    if not staged:
        _git(root, "commit", "-qm", "change", "--allow-empty")
    base = "landed" if scenario == "stale_supersession" else "base-marker"
    args = ["--repo-root", str(root), "--base", base] + (["--staged"] if staged else [])
    expected = _run_main(old.main, args, capsys)
    actual = _run_main(node_runtime.main, args, capsys)
    _assert_parity(expected, actual)
    if expected.exit_code == 2:
        assert expected.error.startswith("FAIL: ")
        return
    request = node_runtime.collect_request(root, base=base, staged=staged)
    report = NodeMigrationAppendOnlyCheckCompute().handle(request)
    assert frozenset(f.message for f in report.findings) == expected.violations
    assert frozenset(f.location for f in report.findings) == frozenset(
        v.split(" (", 1)[0] for v in expected.violations
    )
    assert len(report.findings) == len(expected.violations)
    assert report.provenance.validators_run == ("migration-append-only",)
    assert all(
        f.severity == "FAIL"
        and f.validator_id == "migration-append-only"
        and f.rule_id == "applied-migration-mutated"
        and f.remediation
        for f in report.findings
    )
    if scenario in {"modify", "delete", "rename"}:
        assert expected.exit_code == 1
    if scenario == "two_violations":
        assert len(expected.violations) == 2


@pytest.mark.parametrize("staged", [True, False])
@pytest.mark.parametrize("missing_base", [True, False])
def test_node_parity_migration_append_only_parity_base_errors(
    staged: bool, missing_base: bool, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _build_repo(tmp_path / "repo")
    _apply(root, "modify")
    _git(root, "add", "-A")
    args = (
        ["--repo-root", str(root)]
        + (["--base", "missing-ref"] if missing_base else [])
        + (["--staged"] if staged else [])
    )
    expected = _run_main(old.main, args, capsys)
    _assert_parity(expected, _run_main(node_runtime.main, args, capsys))
    assert expected.exit_code == (1 if staged and not missing_base else 2)


@pytest.mark.parametrize("staged", [True, False])
@pytest.mark.parametrize("change", ["manifest", "supersessions", "successor"])
def test_node_parity_migration_append_only_parity_unstaged_data(
    staged: bool, change: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Index mode ignores unstaged edits; committed mode reads the working tree."""
    root = _build_repo(tmp_path / "repo")
    _apply(root, "valid_supersession")
    if change == "manifest":
        _git(
            root,
            "add",
            f"{FORWARD}/{_APPLIED}",
            f"{FORWARD}/{_SUCCESSOR}",
            SUPERSESSIONS,
        )
    elif change == "supersessions":
        _git(root, "add", "-A")
        _write(root, SUPERSESSIONS, "")
    else:
        _git(root, "add", "-A")
        (root / FORWARD / _SUCCESSOR).unlink()
    if not staged:
        _git(root, "commit", "-qm", "staged changes")
    args = ["--repo-root", str(root), "--base", "base-marker"] + (
        ["--staged"] if staged else []
    )
    expected = _run_main(old.main, args, capsys)
    _assert_parity(expected, _run_main(node_runtime.main, args, capsys))
    assert expected.exit_code == (
        1
        if (staged and change == "manifest") or (not staged and change != "manifest")
        else 0
    )


class _BrokenNode:
    def handle(
        self, request: ModelMigrationAppendOnlyCheckInput
    ) -> ModelValidationReport:
        return ModelValidationReport.from_findings(
            findings=(),
            request=ModelValidationRequestRef(profile="default"),
            validators_run=("migration-append-only",),
        )


def test_node_parity_migration_append_only_parity_comparison_detects_broken_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _build_repo(tmp_path / "repo")
    _apply(root, "modify")
    _git(root, "add", "-A")
    args = ["--repo-root", str(root), "--base", "base-marker", "--staged"]
    expected = _run_main(old.main, args, capsys)
    monkeypatch.setattr(
        node_runtime, "NodeMigrationAppendOnlyCheckCompute", _BrokenNode
    )
    actual = _run_main(node_runtime.main, args, capsys)
    assert expected.exit_code == 1 and expected.violations
    assert actual.exit_code == 0 and not actual.violations
    with pytest.raises(AssertionError):
        _assert_parity(expected, actual)


def test_node_parity_migration_append_only_parity_handler_is_pure_and_input_frozen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _build_repo(tmp_path / "repo")
    _apply(root, "valid_supersession")
    _git(root, "add", "-A")
    request = node_runtime.collect_request(root, base="base-marker", staged=True)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("handler attempted I/O")

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "run", forbidden)
        patch.setattr(Path, "read_text", forbidden)
        patch.setattr(Path, "is_file", forbidden)
        assert (
            NodeMigrationAppendOnlyCheckCompute().handle(request).overall_status
            == "PASS"
        )
    with pytest.raises(ValidationError, match="frozen"):
        request.__setattr__("base_ref", "changed")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelMigrationAppendOnlyCheckInput.model_validate(
            {**request.model_dump(), "unknown": "field"}
        )
