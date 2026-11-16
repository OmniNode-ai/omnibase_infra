# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The sweep of earlier installs' leftover `.atomic-stage.*` folders (OMN-24508).

A stopped or failed run of the installer leaves its stage dir in the venv root;
a successful finish sweeps them all, because the whole sequence runs under the
venv's exclusive reconcile lock. The failure path keeps its own stage.
"""

from __future__ import annotations

import importlib.util
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
    (venv / "lib").mkdir(parents=True)
    (venv / ".atomic-stage-note").write_text("a note, not a stage", encoding="utf-8")
    (venv / "atomic-stage.x").mkdir(parents=True)
    for i in range(9):
        _leftover(venv, f"dead{i:02d}", journal=i < 3, site=i < 6)
    return venv


def test_finalize_sweeps_every_earlier_leftover_and_keeps_the_live_children(
    tmp_path: Path,
) -> None:
    venv = _venv_root(tmp_path)
    stage = _leftover(venv, "current", journal=True, site=True)

    status = plan.run_command("finalize", stage_dir=stage)

    assert status == plan.EXIT_OK
    assert not stage.exists(), "the finished stage was not removed"
    leftovers = [p for p in venv.iterdir() if p.name.startswith(plan.STAGE_DIR_PREFIX)]
    assert not leftovers, f"the sweep left stages: {leftovers}"
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
    assert not any(p.name.startswith(plan.STAGE_DIR_PREFIX) for p in venv.iterdir()), (
        "the CLI finalize left the leftovers"
    )


def test_installer_script_template_matches_engine_prefix() -> None:
    script = _INSTALLER.read_text(encoding="utf-8")
    literal = ".atomic-stage.XXXXXX"
    assert literal in script, "the installer's mktemp template drifted from the engine"
    assert literal == plan.STAGE_DIR_PREFIX + "XXXXXX"


def test_finalize_warns_and_keeps_a_stage_it_cannot_remove(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stage another user's files stick in stays, named on stderr (OMN-24899).

    Root's bytecode inside the owner's venv made ``rmtree`` of that stage a
    silent no-op under ``ignore_errors``; the sweep used to count it as swept
    anyway and the folder outlived every "swept" line.
    """
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    stuck = _leftover(venv, "stuck", journal=False, site=False)
    gone = _leftover(venv, "dead00", journal=False, site=False)
    stage = _leftover(venv, "current", journal=True, site=True)
    real_rmtree = shutil.rmtree

    def stuck_rmtree(path: Path, ignore_errors: bool = False) -> None:
        if path.name.startswith(plan.STAGE_DIR_PREFIX) and path.name.endswith("stuck"):
            return
        real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(plan.shutil, "rmtree", stuck_rmtree)

    status = plan.run_command("finalize", stage_dir=stage)

    assert status == plan.EXIT_OK
    assert stuck.is_dir(), "the unremovable stage was reported as swept anyway"
    assert not gone.exists(), "the removable stage was not swept"
    out, err = capsys.readouterr()
    assert "swept 1 stale stage dir(s)" in out, out
    assert f"  swept stale stage {stuck}" not in out, out
    assert "WARNING" in err, err
    assert str(stuck) in err, err
    assert "1 stale stage dir(s) remain" in err, err
