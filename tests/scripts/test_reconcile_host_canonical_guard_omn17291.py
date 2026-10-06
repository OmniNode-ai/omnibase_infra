# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The reconciler reads the installed canonical-clone guard back (OMN-17291).

THE INCIDENT

The tracked reference-transaction guard learned on 2026-09-26 that
``checkout -B dev`` on the branch HEAD already sits on is not a branch switch
(OMN-18608). The installed copy under ``$OMNI_HOME/scripts/git-hooks`` stayed at
the 2026-09-24 text for nine days. Every tick's own ``checkout --force -B dev``
was refused by a guard that had already been fixed, a clone that fell behind
``origin/dev`` could not advance, and the tick failed from 2026-10-03T22:33Z.
Nothing compared the installed guard with the tracked one, so nothing said so.

``reconcile-host.sh`` now carries a ``canonical-guard`` surface that compares
the three installed guard scripts with the tracked source beside it, by content.
These tests hold the three outcomes: no installed guard is silent, an identical
one passes, and a drifted one fails the run in both modes and names the repair.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.scripts.test_reconcile_host_omn17307 import (
    EXIT_FAILED,
    EXIT_OK,
    Workspace,
    _build_green_fixture,
    _run,
    build_workspace,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACKED_GUARDS = _REPO_ROOT / "scripts" / "git-hooks"
_GUARD_SCRIPTS = (
    "canonical_clone_guard.sh",
    "canonical_clone_paths.sh",
    "canonical_clone_ref_guard.sh",
)


@pytest.fixture
def ws(tmp_path: Path) -> Workspace:
    return build_workspace(tmp_path)


def _install_guards(ws: Workspace) -> Path:
    """Tracked source beside the script, and a byte-identical live copy."""
    source = ws.scripts / "git-hooks"
    source.mkdir()
    live = ws.root / "scripts" / "git-hooks"
    live.mkdir(parents=True)
    for name in _GUARD_SCRIPTS:
        shutil.copy2(_TRACKED_GUARDS / name, source / name)
        shutil.copy2(_TRACKED_GUARDS / name, live / name)
    return live


def test_host_with_no_installed_guard_records_nothing(ws: Workspace) -> None:
    """Positive control: absence of an installed guard is not a failure."""
    _build_green_fixture(ws)

    proc = _run(ws)

    assert proc.returncode == EXIT_OK, proc.stderr
    assert "canonical-guard" not in proc.stderr


def test_identical_installed_guard_passes(ws: Workspace) -> None:
    """Positive control for the drift test: the surface can read green."""
    _build_green_fixture(ws)
    _install_guards(ws)

    proc = _run(ws)

    assert proc.returncode == EXIT_OK, proc.stderr
    assert "canonical-guard: ALREADY_AT_TARGET" in proc.stderr


@pytest.mark.parametrize("script", _GUARD_SCRIPTS)
def test_drifted_installed_guard_fails_in_both_modes(
    ws: Workspace, script: str
) -> None:
    _build_green_fixture(ws)
    live = _install_guards(ws)
    # The shape of the incident: the installed copy lacks a line the tracked
    # source has.
    (live / script).write_text(
        (live / script).read_text(encoding="utf-8") + "\n# stale copy\n",
        encoding="utf-8",
    )

    for mode_args in ((), ("--check",)):
        proc = _run(ws, *mode_args)
        assert proc.returncode == EXIT_FAILED, (mode_args, proc.stderr)
        assert "canonical-guard: DRIFT" in proc.stderr, (mode_args, proc.stderr)
        assert script in proc.stderr, (mode_args, proc.stderr)


def test_missing_installed_script_is_drift_when_a_live_directory_exists(
    ws: Workspace,
) -> None:
    _build_green_fixture(ws)
    live = _install_guards(ws)
    (live / "canonical_clone_ref_guard.sh").unlink()

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED, proc.stderr
    assert "canonical-guard: DRIFT" in proc.stderr
    assert "canonical_clone_ref_guard.sh" in proc.stderr


def test_drift_names_the_sanctioned_installer_in_the_receipt(ws: Workspace) -> None:
    _build_green_fixture(ws)
    live = _install_guards(ws)
    (live / "canonical_clone_paths.sh").write_text("# stale\n", encoding="utf-8")

    proc = _run(ws)

    assert proc.returncode == EXIT_FAILED, proc.stderr
    receipt = ws.receipt.read_text(encoding="utf-8")
    assert '"surface": "canonical-guard"' in receipt
    assert "install-canonical-clone-git-hooks.sh" in receipt


def test_drift_does_not_hold_the_dispatch_floor(ws: Workspace) -> None:
    """A stale guard says nothing about which build a dispatch would run."""
    _build_green_fixture(ws)
    live = _install_guards(ws)
    (live / "canonical_clone_paths.sh").write_text("# stale\n", encoding="utf-8")

    _run(ws)

    receipt = ws.receipt.read_text(encoding="utf-8")
    row = next(line for line in receipt.splitlines() if '"canonical-guard"' in line)
    assert '"dispatch_premise": false' in row
