# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Verdict parity of the migration-freeze node vs the three implementations it replaced (OMN-20568).

``scripts/validation/validate_migration_freeze.py`` (pre-commit, through
``validate.py migration_freeze``), ``scripts/check_migration_freeze.sh`` (CI) and the
python ``--check-committed`` mode enforced the freeze before this node. While they
existed, the same matrix of freeze files and diffs below was run through all three and
through the node in real git repositories; the three agreed on every exit code and the
node agreed with them (commit badcc132e). The recorded verdicts are
``tests/fixtures/validator_parity/migration_freeze/golden.json``. This file keeps that
matrix as the node's regression test.

AC3: every refusal any of the three produced on the matrix is still produced by the node.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute import (
    NodeMigrationFreezeCheckCompute,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute import (
    runtime_migration_freeze_check as node_runtime,
)
from omnibase_infra.nodes.node_migration_freeze_check_compute.models import (
    ModelMigrationFreezeCheckInput,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN = json.loads(
    (
        REPO_ROOT
        / "tests"
        / "fixtures"
        / "validator_parity"
        / "migration_freeze"
        / "golden.json"
    ).read_text(encoding="utf-8")
)

MIG = "docker/migrations/forward"


def _ago(days: int) -> str:
    return (datetime.now(tz=UTC).date() - timedelta(days=days)).isoformat()


# Freeze files. None means no .migration_freeze at all.
FREEZES: dict[str, str | None] = {
    "absent": None,
    "no_date": "# Migration Freeze\nticket=OMN-2073\n",
    "fresh": f"freeze_date={_ago(5)}\n",
    "warn": f"freeze_date={_ago(35)}\n",
    "expired": f"freeze_date={_ago(65)}\n",
    "garbage_date": "freeze_date=not-a-date\n",
    "commented_date": f"# freeze_date={_ago(65)}\n",
}


@dataclass(frozen=True)
class Change:
    """One working-tree change applied on top of the base commit."""

    kind: str  # add | modify | delete | rename
    path: str
    source: str = ""  # rename source


DIFFS: dict[str, tuple[Change, ...]] = {
    "empty": (),
    "add_migration": (Change("add", f"{MIG}/101_new.sql"),),
    "add_outside": (Change("add", "src/new_module.py"),),
    "add_both": (
        Change("add", f"{MIG}/102_new.sql"),
        Change("add", "src/new_module.py"),
    ),
    "add_rollback_dir": (Change("add", "docker/migrations/rollback/101_down.sql"),),
    "add_non_sql": (Change("add", f"{MIG}/NOTES.txt"),),
    "modify_migration": (Change("modify", f"{MIG}/001_base.sql"),),
    "delete_migration": (Change("delete", f"{MIG}/001_base.sql"),),
    "rename_into": (Change("rename", f"{MIG}/103_moved.sql", "docs/old_notes.sql"),),
    "rename_within": (
        Change("rename", f"{MIG}/104_renamed.sql", f"{MIG}/001_base.sql"),
    ),
    "add_two_migrations": (
        Change("add", f"{MIG}/105_a.sql"),
        Change("add", f"{MIG}/106_b.sql"),
    ),
}


@pytest.fixture(autouse=True)
def _no_git_location_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in [k for k in os.environ if k.startswith("GIT_")]:
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


def _build_repo(root: Path, freeze: str | None, changes: tuple[Change, ...]) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "-q", "-b", "main")
    (root / MIG).mkdir(parents=True)
    (root / MIG / "001_base.sql").write_text("select 1;\n")
    (root / "docs").mkdir()
    (root / "docs" / "old_notes.sql").write_text("select 2;\n" * 20)
    (root / "src").mkdir()
    (root / "src" / "mod.py").write_text("x = 1\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    for change in changes:
        target = root / change.path
        target.parent.mkdir(parents=True, exist_ok=True)
        if change.kind == "add":
            target.write_text("select 3;\n")
        elif change.kind == "modify":
            target.write_text("select 1; -- edited\n")
        elif change.kind == "delete":
            target.unlink()
        elif change.kind == "rename":
            _git(root, "mv", change.source, change.path)
    _git(root, "add", "-A")
    if freeze is not None:
        (root / ".migration_freeze").write_text(freeze)


def _commit(root: Path) -> None:
    _git(root, "commit", "-q", "-m", "change", "--allow-empty")


@dataclass(frozen=True)
class Verdict:
    exit_code: int
    refused: tuple[str, ...]


def _node(root: Path, monkeypatch: pytest.MonkeyPatch, *args: str) -> Verdict:
    monkeypatch.chdir(root)
    report_exit = node_runtime.main(list(args))
    # Re-derive the refused files through the handler to compare finding sets.
    staged_or_base = node_runtime._added_paths(args[1] if args else None)
    freeze_text = (root / ".migration_freeze").read_text()
    report = NodeMigrationFreezeCheckCompute().handle(
        ModelMigrationFreezeCheckInput(
            freeze_active=True,
            freeze_text=freeze_text,
            today=datetime.now(tz=UTC).date(),
            added_paths=staged_or_base,
        )
    )
    return Verdict(
        report_exit,
        tuple(
            sorted(
                f.location or ""
                for f in report.findings
                if f.rule_id == "new-migration-file"
            )
        ),
    )


SCENARIOS = [(f, d) for f in FREEZES for d in DIFFS]
IDS = [f"{f}-{d}" for f, d in SCENARIOS]


def _expected(freeze_id: str, diff_id: str, mode: str) -> Verdict:
    recorded = GOLDEN["scenarios"][f"{freeze_id}-{diff_id}"][mode]
    return Verdict(recorded["exit"], tuple(sorted(recorded["refused"])))


@pytest.mark.unit
@pytest.mark.parametrize(("freeze_id", "diff_id"), SCENARIOS, ids=IDS)
def test_node_parity_migration_freeze_parity_precommit_mode(
    freeze_id: str,
    diff_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Staged-file mode matches what the python validator and bash script recorded."""
    repo = tmp_path / "repo"
    _build_repo(repo, FREEZES[freeze_id], DIFFS[diff_id])
    expected = _expected(freeze_id, diff_id, "staged")
    if FREEZES[freeze_id] is None:
        monkeypatch.chdir(repo)
        assert node_runtime.main([]) == expected.exit_code == 0
        capsys.readouterr()
        return
    node = _node(repo, monkeypatch)
    capsys.readouterr()
    assert node == expected


@pytest.mark.unit
@pytest.mark.parametrize(("freeze_id", "diff_id"), SCENARIOS, ids=IDS)
def test_node_parity_migration_freeze_parity_ci_mode(
    freeze_id: str,
    diff_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Committed-diff mode matches what --check-committed and --ci recorded."""
    repo = tmp_path / "repo"
    _build_repo(repo, FREEZES[freeze_id], DIFFS[diff_id])
    _commit(repo)
    expected = _expected(freeze_id, diff_id, "ci")
    if FREEZES[freeze_id] is None:
        monkeypatch.chdir(repo)
        assert node_runtime.main(["--base", "origin/main"]) == expected.exit_code == 0
        capsys.readouterr()
        return
    node = _node(repo, monkeypatch, "--base", "origin/main")
    capsys.readouterr()
    assert node == expected


class _FixedClock(datetime):
    @classmethod
    def now(cls, tz: object = None) -> _FixedClock:
        return cls(2026, 10, 5, 12, 0, tzinfo=UTC)


@pytest.mark.unit
@pytest.mark.parametrize("age_days", [0, 29, 30, 31, 59, 60, 61, 400])
def test_node_parity_migration_freeze_parity_age_thresholds_match_recorded(
    age_days: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact 30 and 60 day thresholds under a fixed clock."""
    freeze_date = _FixedClock.now().date() - timedelta(days=age_days)
    repo = tmp_path / "repo"
    _build_repo(repo, f"freeze_date={freeze_date.isoformat()}\n", ())
    monkeypatch.setattr(node_runtime, "datetime", _FixedClock)
    monkeypatch.chdir(repo)
    assert node_runtime.main([]) == GOLDEN["age_thresholds"][str(age_days)]


@pytest.mark.unit
def test_node_parity_migration_freeze_parity_indented_date_follows_python_not_bash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Documented drift: the bash script ignored an indented freeze_date line (exit 0)
    and the python validator enforced it (exit 1). The node takes the stricter python
    behaviour, so an expired indented date fails."""
    repo = tmp_path / "repo"
    _build_repo(repo, f"  freeze_date={_ago(65)}\n", ())
    assert _node(repo, monkeypatch).exit_code == 1


@pytest.mark.unit
def test_node_parity_migration_freeze_parity_comparison_detects_a_broken_node(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The matrix has teeth: a node that stops refusing new migrations disagrees."""
    repo = tmp_path / "repo"
    _build_repo(repo, FREEZES["fresh"], DIFFS["add_migration"])
    monkeypatch.setattr(
        NodeMigrationFreezeCheckCompute,
        "_violation_findings",
        staticmethod(lambda request: []),
    )
    assert _node(repo, monkeypatch) != _expected("fresh", "add_migration", "staged")
