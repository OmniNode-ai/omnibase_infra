# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Real-git migration-sequence parity against the removed validator script (OMN-20568).

The script ``scripts/validation/validate_migration_sequence.py`` and the node were run
over this matrix of real git repositories while both existed (commit 1a4e78742); the
script's exit code, stdout, stderr and conflict pairs were recorded in
``tests/fixtures/validator_parity/migration_sequence/golden.json`` and are replayed here.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from omnibase_core.models.validation.model_validation_report import (
    ModelValidationReport,
    ModelValidationRequestRef,
)
from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.nodes.node_migration_sequence_check_compute import (
    NodeMigrationSequenceCheckCompute,
)
from omnibase_infra.nodes.node_migration_sequence_check_compute import (
    runtime_migration_sequence_check as node_runtime,
)
from omnibase_infra.nodes.node_migration_sequence_check_compute.models import (
    ModelMigrationSequenceCheckInput,
)
from tests.ci.recorded_verdicts import RecordedVerdicts, normalise_git_error

RECORDED = RecordedVerdicts(
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "validator_parity"
    / "migration_sequence"
    / "golden.json"
)


@dataclass(frozen=True)
class _DuplicateConflict:
    sequence: int
    file_a: str
    file_b: str

    def __str__(self) -> str:
        return f"  seq {self.sequence:03d}: {self.file_a!r}  <-->  {self.file_b!r}"


@dataclass(frozen=True)
class _RecordedResult:
    conflicts: tuple[_DuplicateConflict, ...]


class _RemovedScript:
    """The removed script's verdicts, replayed from the recording."""

    DuplicateConflict = _DuplicateConflict

    def main(self) -> int:
        record = RECORDED.take("seq.main", skip=("seq.validate",))
        sys.stdout.write(record["out"])
        sys.stderr.write(record["err"])
        exit_code: int = record["exit"]
        return exit_code

    def validate_migration_sequence(self, repo: Path) -> _RecordedResult:
        record = RECORDED.take("seq.validate")
        return _RecordedResult(
            tuple(_DuplicateConflict(*c) for c in record["conflicts"])
        )


old_script = _RemovedScript()


@pytest.fixture(autouse=True)
def _bind_recording(request: pytest.FixtureRequest) -> None:
    RECORDED.bind(request.node.nodeid)


DOCKER = "docker/migrations/forward"
SRC = "src/omnibase_infra/migrations/forward"
BASE = f"{DOCKER}/001_base.sql"


@dataclass(frozen=True)
class Scenario:
    initial: tuple[str, ...] = (BASE,)
    added: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    removed_after_staging: tuple[str, ...] = ()
    unstaged: tuple[str, ...] = ()
    expected_pairs: tuple[tuple[int, str, str], ...] = ()


SCENARIOS = {
    "nothing-staged": Scenario(),
    "non-migration": Scenario(added=("README.md",)),
    "new-unique": Scenario(added=(f"{DOCKER}/002_new.sql",)),
    "docker-duplicate": Scenario(
        added=(f"{DOCKER}/001_duplicate.sql",),
        expected_pairs=((1, BASE, f"{DOCKER}/001_duplicate.sql"),),
    ),
    "cross-directory-duplicate": Scenario(
        added=(f"{SRC}/001_cross.sql",),
        expected_pairs=((1, BASE, f"{SRC}/001_cross.sql"),),
    ),
    "excluded-subtree": Scenario(added=(f"{DOCKER}/nodes/node_x/001_node.sql",)),
    "excluded-subtree-with-scan": Scenario(
        added=(f"{DOCKER}/nodes/node_x/001_node.sql", f"{DOCKER}/002_new.sql"),
    ),
    "non-sql": Scenario(added=(f"{DOCKER}/001_notes.txt",)),
    "no-leading-digits": Scenario(added=(f"{DOCKER}/schema.sql",)),
    "three-way-duplicate": Scenario(
        added=(f"{SRC}/001_cross.sql", f"{DOCKER}/001_z.sql"),
        expected_pairs=(
            (1, BASE, f"{DOCKER}/001_z.sql"),
            (1, BASE, f"{SRC}/001_cross.sql"),
        ),
    ),
    "staged-deletion": Scenario(deleted=(BASE,)),
    "staged-deletion-still-conflicts": Scenario(
        initial=(BASE, f"{DOCKER}/001_z.sql"),
        deleted=(BASE,),
        expected_pairs=((1, BASE, f"{DOCKER}/001_z.sql"),),
    ),
    "staged-file-removed-from-disk": Scenario(
        added=(f"{SRC}/001_cross.sql",),
        removed_after_staging=(f"{SRC}/001_cross.sql",),
        expected_pairs=((1, BASE, f"{SRC}/001_cross.sql"),),
    ),
    "existing-duplicate-does-not-trigger": Scenario(
        initial=(BASE, f"{SRC}/001_cross.sql"),
        added=("README.md",),
    ),
    "unstaged-duplicate-counts": Scenario(
        added=(f"{DOCKER}/002_new.sql",),
        unstaged=(f"{SRC}/001_cross.sql",),
        expected_pairs=((1, BASE, f"{SRC}/001_cross.sql"),),
    ),
    "nested-staged-file-counts": Scenario(
        added=(f"{SRC}/nested/001_nested.sql",),
        expected_pairs=((1, BASE, f"{SRC}/nested/001_nested.sql"),),
    ),
    "nested-unstaged-file-is-not-globbed": Scenario(
        added=(f"{DOCKER}/002_new.sql",),
        unstaged=(f"{SRC}/nested/001_nested.sql",),
    ),
    "uppercase-staged-extension": Scenario(
        added=(f"{SRC}/001_cross.SQL",),
        expected_pairs=((1, BASE, f"{SRC}/001_cross.SQL"),),
    ),
    "uppercase-existing-extension": Scenario(
        initial=(BASE, f"{SRC}/001_cross.SQL"),
        added=(f"{DOCKER}/002_new.sql",),
    ),
    "different-padding-same-number": Scenario(
        added=(f"{SRC}/1.sql",),
        expected_pairs=((1, BASE, f"{SRC}/1.sql"),),
    ),
    "absent-migration-directories": Scenario(
        initial=("README.md",), added=("CHANGELOG.md",)
    ),
    "non-numeric-sql-contributes-to-scan-count": Scenario(
        initial=(BASE, f"{DOCKER}/schema.sql"), added=(f"{DOCKER}/002_new.sql",)
    ),
}


@pytest.fixture(autouse=True)
def _no_git_location_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [key for key in os.environ if key.startswith("GIT_")]:
        monkeypatch.delenv(key)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.email=t@example.test", "-c", "user.name=t", *args],
        cwd=repo,
        env=scrub_git_location_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _write_files(repo: Path, paths: tuple[str, ...]) -> None:
    for path in paths:
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("select 1;\n", encoding="utf-8")


def _build_repo(repo: Path, scenario: Scenario) -> None:
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _write_files(repo, scenario.initial)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _write_files(repo, scenario.added)
    for path in scenario.deleted:
        (repo / path).unlink()
    _git(repo, "add", "-A")
    for path in scenario.removed_after_staging:
        (repo / path).unlink()
    _write_files(repo, scenario.unstaged)


@dataclass(frozen=True)
class Verdict:
    exit_code: int
    pairs: tuple[tuple[int, str, str], ...]
    output: str
    error: str


def _old(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    verbose: bool = False,
) -> Verdict:
    capsys.readouterr()
    argv = ["validate_migration_sequence.py", str(repo)]
    if verbose:
        argv.append("--verbose")
    monkeypatch.setattr(sys, "argv", argv)
    exit_code = old_script.main()
    captured = capsys.readouterr()
    pairs = old_script.validate_migration_sequence(repo).conflicts
    return Verdict(
        exit_code,
        tuple((pair.sequence, pair.file_a, pair.file_b) for pair in pairs),
        captured.out,
        captured.err,
    )


def _node(
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    *,
    verbose: bool = False,
) -> Verdict:
    monkeypatch.chdir(repo)
    capsys.readouterr()
    exit_code = node_runtime.main(["--verbose"] if verbose else [])
    captured = capsys.readouterr()
    report = NodeMigrationSequenceCheckCompute().handle(
        node_runtime.collect_input(repo)
    )
    pairs: list[tuple[int, str, str]] = []
    for finding in report.findings:
        assert finding.validator_id == "migration-sequence"
        assert finding.severity == "FAIL"
        assert finding.rule_id == "duplicate-sequence"
        sequence = finding.evidence["sequence"]
        file_a = finding.evidence["file_a"]
        file_b = finding.evidence["file_b"]
        assert isinstance(sequence, int)
        assert isinstance(file_a, str)
        assert isinstance(file_b, str)
        assert finding.location == file_b
        assert finding.message == str(
            old_script.DuplicateConflict(sequence, file_a, file_b)
        )
        assert finding.remediation
        pairs.append((sequence, file_a, file_b))
    assert report.provenance.validators_run == ("migration-sequence",)
    assert report.profile == "default"
    assert (report.overall_status == "FAIL") == bool(pairs)
    return Verdict(exit_code, tuple(pairs), captured.out, captured.err)


def _assert_parity(old: Verdict, node: Verdict) -> None:
    assert node.exit_code == old.exit_code
    assert node.pairs == old.pairs
    assert node.output == old.output
    assert node.error == old.error


@pytest.mark.unit
@pytest.mark.parametrize("scenario_id", SCENARIOS)
@pytest.mark.parametrize("verbose", [False, True], ids=["quiet", "verbose"])
def test_node_parity_migration_sequence_parity_matrix(
    scenario_id: str,
    verbose: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scenario = SCENARIOS[scenario_id]
    repo = tmp_path / "repo"
    _build_repo(repo, scenario)
    old = _old(repo, monkeypatch, capsys, verbose=verbose)
    assert old.pairs == scenario.expected_pairs
    assert old.exit_code == (1 if scenario.expected_pairs else 0)
    _assert_parity(old, _node(repo, monkeypatch, capsys, verbose=verbose))


class _BrokenNode:
    def handle(
        self, request: ModelMigrationSequenceCheckInput
    ) -> ModelValidationReport:
        return ModelValidationReport.from_findings(
            findings=(),
            request=ModelValidationRequestRef(profile="default"),
            validators_run=("migration-sequence",) if request.staged_paths else (),
        )


@pytest.mark.unit
def test_node_parity_migration_sequence_parity_comparison_detects_broken_node(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    _build_repo(repo, SCENARIOS["docker-duplicate"])
    old = _old(repo, monkeypatch, capsys)
    monkeypatch.setattr(NodeMigrationSequenceCheckCompute, "handle", _BrokenNode.handle)
    broken = _node(repo, monkeypatch, capsys)
    assert old.exit_code == 1
    assert broken.exit_code == 0
    with pytest.raises(AssertionError):
        _assert_parity(old, broken)
    with pytest.raises(AssertionError):
        _assert_parity(old, Verdict(old.exit_code, (), old.output, old.error))


@pytest.mark.unit
def test_node_parity_migration_sequence_parity_git_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_migration_sequence.py", str(tmp_path)])
    old_exit = old_script.main()
    old_output = capsys.readouterr()
    node_exit = node_runtime.main([str(tmp_path)])
    node_output = capsys.readouterr()
    assert node_exit == old_exit == 2
    assert node_output.out == old_output.out
    assert normalise_git_error(node_output.err) == normalise_git_error(old_output.err)


@pytest.mark.unit
@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("missing git"), subprocess.TimeoutExpired("git", 10)],
    ids=["git-missing", "git-timeout"],
)
def test_node_parity_migration_sequence_parity_git_execution_errors(
    failure: FileNotFoundError | subprocess.TimeoutExpired,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = tmp_path / "repo"
    _build_repo(repo, SCENARIOS["new-unique"])
    monkeypatch.setattr(sys, "argv", ["validate_migration_sequence.py", str(repo)])
    with patch.object(subprocess, "run", autospec=True, side_effect=failure):
        old_exit = old_script.main()
        old_output = capsys.readouterr()
        node_exit = node_runtime.main([str(repo)])
        node_output = capsys.readouterr()
    assert node_exit == old_exit == 2
    assert node_output == old_output


@pytest.mark.unit
def test_node_parity_migration_sequence_parity_input_is_frozen_and_forbids_extra() -> (
    None
):
    request = ModelMigrationSequenceCheckInput()
    for field_name in ModelMigrationSequenceCheckInput.model_fields:
        with pytest.raises(ValidationError, match="frozen"):
            setattr(request, field_name, (BASE,))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelMigrationSequenceCheckInput.model_validate({"unknown": "field"})
