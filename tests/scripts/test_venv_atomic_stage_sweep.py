# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The sweep of earlier installs' leftover `.atomic-stage.*` folders (OMN-24508).

A stopped or failed run of the installer leaves its stage dir in the venv root.
The sweep runs under the venv's exclusive reconcile lock on every path (the script's
exit, the command line and the typed operation), refuses a relative path and a folder
that is not a venv root, keeps a stage that holds a journal (the previous packages),
and reports a removal that failed instead of counting it as swept (OMN-24568).
"""

from __future__ import annotations

import fcntl
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIB = (
    _REPO_ROOT
    / "src/omnibase_infra/handlers/venv_atomic_install/atomic_venv_install.py"
)
_RUNTIME = (
    _REPO_ROOT
    / "src/omnibase_infra/nodes/node_venv_atomic_install/runtime_venv_atomic_install.py"
)
_INSTALLER = _REPO_ROOT / "scripts/install-node-skill-package.sh"

_SPEC = importlib.util.spec_from_file_location("atomic_venv_install", _LIB)
assert _SPEC is not None and _SPEC.loader is not None
plan = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plan)


def _leftover(venv: Path, suffix: str, *, journal: bool, site: bool) -> Path:
    """One dead run's stage dir: a journal and a site tree when requested."""
    stage = venv / f"{plan.STAGE_DIR_PREFIX}{suffix}"
    if site:
        (stage / "site" / "pkg").mkdir(parents=True)
        (stage / "site" / "pkg" / "m0.py").write_text("X = 0\n", encoding="utf-8")
    if journal:
        (stage / "journal.json").write_text("[]", encoding="utf-8")
    if not site and not journal:
        stage.mkdir(parents=True)
    return stage


def _venv_root(root: Path) -> Path:
    """A dispatch venv root: live children and the leftovers of nine dead runs."""
    venv = root / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    (venv / "lib").mkdir(parents=True)
    (venv / ".atomic-stage-note").write_text("a note, not a stage", encoding="utf-8")
    (venv / "atomic-stage.x").mkdir(parents=True)
    for i in range(9):
        _leftover(venv, f"dead{i:02d}", journal=i < 3, site=i < 6)
    return venv


def _stages(venv: Path) -> list[str]:
    return sorted(
        p.name for p in venv.iterdir() if p.name.startswith(plan.STAGE_DIR_PREFIX)
    )


def test_finalize_sweeps_every_unjournaled_leftover_and_keeps_the_journaled_ones(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    venv = _venv_root(tmp_path)
    stage = _leftover(venv, "current", journal=True, site=True)

    status = plan.run_command("finalize", stage_dir=stage)

    assert status == plan.EXIT_OK
    assert not stage.exists(), "the finished stage was not removed"
    assert _stages(venv) == [
        ".atomic-stage.dead00",
        ".atomic-stage.dead01",
        ".atomic-stage.dead02",
    ], "only the three stages holding a journal (the previous packages) may remain"
    assert "swept 6 stale stage dir(s), kept 3, failed 0" in capsys.readouterr().out
    assert (venv / "bin").is_dir() and (venv / "lib").is_dir(), "live dirs disturbed"
    assert (venv / ".atomic-stage-note").is_file(), "the note file was swept"
    assert (venv / "atomic-stage.x").is_dir(), "a dir without the dot prefix was swept"


def test_a_symlink_named_like_a_stage_is_not_followed(tmp_path: Path) -> None:
    venv = _venv_root(tmp_path)
    outside = tmp_path / "outside"
    (outside / "data").mkdir(parents=True)
    (outside / "data" / "keep").write_text("x", encoding="utf-8")
    (venv / ".atomic-stage.link").symlink_to(outside)
    stage = _leftover(venv, "current", journal=False, site=True)

    status = plan.run_command("finalize", stage_dir=stage)

    assert status == plan.EXIT_OK
    assert (venv / ".atomic-stage.link").is_symlink(), "the sweep touched the symlink"
    assert (outside / "data" / "keep").is_file(), "the symlink's target was removed"


def test_failed_install_keeps_stage(tmp_path: Path) -> None:
    """No finalize on the failure path: the rollback touches no sibling stage."""
    venv = _venv_root(tmp_path)
    stage = _leftover(venv, "failed", journal=False, site=True)

    status = plan.run_command("rollback", stage_dir=stage)

    assert status == plan.EXIT_OK
    siblings = sorted(
        p.name for p in venv.iterdir() if p.name.startswith(plan.STAGE_DIR_PREFIX)
    )
    assert len(siblings) == 10, "the failure path swept stages it must keep"
    assert all((venv / name).is_dir() for name in siblings)


def test_the_runtime_cli_sweeps_through_the_engine(tmp_path: Path) -> None:
    venv = _venv_root(tmp_path)
    stage = _leftover(venv, "current", journal=True, site=True)

    result = subprocess.run(
        [sys.executable, str(_RUNTIME), "finalize", "--stage-dir", str(stage)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert len(_stages(venv)) == 3, (
        "the CLI finalize swept a journaled stage or left others"
    )
    assert "kept 3" in result.stdout


def test_installer_script_template_matches_engine_prefix() -> None:
    script = _INSTALLER.read_text(encoding="utf-8")
    literal = ".atomic-stage.XXXXXX"
    assert literal in script, "the installer's mktemp template drifted from the engine"
    assert literal == plan.STAGE_DIR_PREFIX + "XXXXXX"


def test_sweep_takes_the_lock_and_removes_nothing_while_another_install_holds_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venv = _venv_root(tmp_path)
    stage = _leftover(venv, "current", journal=False, site=True)
    before = _stages(venv)
    holder = os.open(venv / ".onex-venv-reconcile.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        for command in ("sweep", "finalize"):
            status = plan.run_command(command, stage_dir=stage, lock_timeout_s=1.0)
            assert status == plan.EXIT_OK
            assert "stale-stage sweep skipped" in capsys.readouterr().err
            assert _stages(venv) == sorted(
                set(before) - ({stage.name} if command == "finalize" else set())
            ), f"{command} removed a stage while the lock was held"
    finally:
        os.close(holder)
    assert (
        plan.run_command("sweep", stage_dir=venv / ".atomic-stage.none") == plan.EXIT_OK
    )
    assert len(_stages(venv)) == 3, "with the lock free the sweep did not run"


def test_a_caller_that_already_holds_the_lock_is_not_made_to_take_it_twice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv = _venv_root(tmp_path)
    lock = venv.resolve() / ".onex-venv-reconcile.lock"
    holder = os.open(lock, os.O_CREAT | os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        result = plan.sweep_stale_stages(venv, lock_held=str(lock), lock_timeout_s=1.0)
    finally:
        os.close(holder)
    assert len(result.removed) == 6 and len(result.kept) == 3


def test_a_marker_naming_another_venv_does_not_suppress_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    venv = _venv_root(tmp_path)
    holder = os.open(venv / ".onex-venv-reconcile.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        with pytest.raises(plan.VenvLockBusyError):
            plan.sweep_stale_stages(
                venv, lock_held=str(tmp_path / "other.lock"), lock_timeout_s=1.0
            )
    finally:
        os.close(holder)
    assert len(_stages(venv)) == 9


def test_a_failed_removal_is_reported_as_failed_and_not_counted_as_swept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    venv = _venv_root(tmp_path)
    stuck = venv / ".atomic-stage.dead05"
    real_rmtree = shutil.rmtree

    def refusing_rmtree(path: Path) -> None:
        if Path(path) == stuck:
            raise PermissionError(13, "Permission denied", str(path))
        real_rmtree(path)

    monkeypatch.setattr(shutil, "rmtree", refusing_rmtree)
    result = plan.sweep_stale_stages(venv)

    assert result.failed == (stuck,)
    assert stuck not in result.removed and stuck.is_dir()
    assert len(result.removed) == 5 and len(result.kept) == 3
    captured = capsys.readouterr()
    assert "cannot remove stage" in captured.err and "dead05" in captured.err
    assert "swept stale stage" in captured.out
    assert f"swept stale stage {stuck}" not in captured.out


def test_a_stage_holding_a_journal_is_kept_and_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    venv = _venv_root(tmp_path)
    result = plan.sweep_stale_stages(venv)
    assert sorted(p.name for p in result.kept) == [
        ".atomic-stage.dead00",
        ".atomic-stage.dead01",
        ".atomic-stage.dead02",
    ]
    out = capsys.readouterr().out
    assert "kept stage" in out and "holds a journal" in out


def test_a_relative_stage_dir_is_refused_and_the_working_directory_is_not_swept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoy = tmp_path / "cwd"
    (decoy / ".atomic-stage.decoy").mkdir(parents=True)
    (decoy / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    monkeypatch.chdir(decoy)
    for command in ("finalize", "sweep"):
        status = plan.run_command(command, stage_dir=Path(".atomic-stage.x"))
        assert status == plan.EXIT_USAGE, command
    with pytest.raises(ValueError, match="relative"):
        plan.sweep_stale_stages(Path())
    assert (decoy / ".atomic-stage.decoy").is_dir()


def test_a_folder_that_is_not_a_venv_root_is_refused(tmp_path: Path) -> None:
    not_a_venv = tmp_path / "plain"
    (not_a_venv / ".atomic-stage.keepme").mkdir(parents=True)
    with pytest.raises(ValueError, match="not a venv root"):
        plan.sweep_stale_stages(not_a_venv)
    assert (not_a_venv / ".atomic-stage.keepme").is_dir()


def _commit_fixture(tmp_path: Path) -> Path:
    stage = tmp_path / "venv" / ".atomic-stage.c1"
    stage.mkdir(parents=True)
    (stage / "plan.json").write_text("[]", encoding="utf-8")
    (stage / "journal.json").write_text("[]", encoding="utf-8")
    return stage


def test_a_commit_that_restored_everything_drops_its_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _commit_fixture(tmp_path)

    def broken_footprint(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(plan, "_live_footprint", broken_footprint)
    assert plan.commit_stage(sys.executable, stage) == plan.EXIT_COMMIT_FAILED
    assert not (stage / "journal.json").exists(), "a finished restore kept its journal"


def test_a_commit_whose_restore_failed_keeps_its_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = _commit_fixture(tmp_path)

    def broken_footprint(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("probe failed")

    def broken_undo(journal: object) -> None:
        raise RuntimeError("restore failed")

    monkeypatch.setattr(plan, "_live_footprint", broken_footprint)
    monkeypatch.setattr(plan, "_undo", broken_undo)
    assert plan.commit_stage(sys.executable, stage) == plan.EXIT_COMMIT_FAILED
    assert (stage / "journal.json").is_file(), "the only copy's journal was dropped"
