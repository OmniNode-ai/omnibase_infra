# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the venv reconcile lock and the post-repair readback (OMN-18663).

Two defects, both observed on 2026-09-18 against the shared plugin CLI venv
(omni_home/CLAUDE.md rule 11):

1. ``check-omnimarket-venv-drift.sh --repair`` exited on the install script's
   status without re-running the comparison it opened with, so a repair that
   changed nothing reported success and the drift guard kept refusing the same
   interpreter.
2. Two lanes used the venv at once. One ``--dry-run`` classified a VCS-ref bump
   0.4.118 -> None as REMOVE and refused (exit 3); an identical run seconds
   later planned cleanly. The first had read a venv mid-install -- uv had
   removed the old distribution and not yet written the new one -- so the
   refusal described a state that never existed as a steady state.

Fully hermetic: a local bare git repo stands in for the canonical remote, the
"venv" is a shim that answers the metadata probes and execs the real
interpreter for everything else, and the provider co-install is replaced via
``ONEX_DRIFT_INSTALL_SCRIPT``. No network, no uv, no real install.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.unit


def _scrubbed_git_env() -> dict[str, str]:
    """A git environment that cannot reach out of ``tmp_path`` (OMN-14891).

    git exports GIT_DIR / GIT_WORK_TREE / GIT_INDEX_FILE into every hook
    environment, and those OVERRIDE both ``cwd=`` and ``git -C``: a fixture that
    shells out to git while running under a pre-commit or pre-push hook would
    mutate the REAL invoking worktree. ``scrub_git_location_env`` removes them.

    It also removes every ``GIT_CONFIG*`` key, including the conftest fixture's
    protective ones (OMN-16584), so the neutral overrides are put back after the
    scrub -- otherwise a developer's ``[tag] gpgsign = true`` (or any global
    hook config) leaks into these fixtures.
    """
    env = scrub_git_location_env(os.environ)
    # Named literally as well as scrubbed: the OMN-14891 guard verifies a
    # module-local scrubber by reading the keys it drops, and a delegated call
    # is invisible to that check.
    for key in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        env.pop(key, None)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["GIT_EDITOR"] = "true"
    return env


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DRIFT_SCRIPT = _REPO_ROOT / "scripts" / "check-omnimarket-venv-drift.sh"
_INSTALL_SCRIPT = _REPO_ROOT / "scripts" / "install-node-skill-package.sh"
_LOCK_LIB = _REPO_ROOT / "scripts" / "lib" / "venv_reconcile_lock.sh"

_EXIT_DRIFT = 1
_EXIT_BUSY = 4


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
        env=_scrubbed_git_env(),
    ).stdout.strip()


def _make_omni_home(root: Path, *, version: str = "0.4.119") -> tuple[Path, str]:
    """An $OMNI_HOME whose omnimarket clone tracks a local bare 'origin'."""
    work = root / "upstream"
    work.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet", "-b", "dev"],
        cwd=work,
        check=True,
        capture_output=True,
        env=_scrubbed_git_env(),
    )
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Test")
    (work / "pyproject.toml").write_text(
        f'[project]\nname = "omnimarket"\nversion = "{version}"\n'
        'dependencies = ["omnibase-compat>=0.5.7,<0.6.0"]\n',
        encoding="utf-8",
    )
    _git(work, "add", "pyproject.toml")
    _git(work, "commit", "--quiet", "-m", "init")

    bare = root / "bare.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(work), str(bare)],
        check=True,
        capture_output=True,
        env=_scrubbed_git_env(),
    )

    omni_home = root / "omni_home"
    omni_home.mkdir()
    clone = omni_home / "omnimarket"
    subprocess.run(
        ["git", "clone", "--quiet", str(bare), str(clone)],
        check=True,
        capture_output=True,
        env=_scrubbed_git_env(),
    )
    _git(clone, "config", "user.email", "test@example.com")
    _git(clone, "config", "user.name", "Test")
    return omni_home, _git(clone, "rev-parse", "HEAD")


def _make_venv_shim(
    root: Path, *, installed_sha: str | None, facts: dict[str, dict[str, str | None]]
) -> Path:
    """A venv whose python answers both metadata probes and runs real scripts.

    ``python -`` (heredoc) is the drift script's installed-commit probe;
    ``python -c`` is the readback's. Anything else -- heavy_lock.py,
    venv_readback.py -- execs the real interpreter, because those are real
    programs that have to actually run.
    """
    venv = root / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    shim = venv / "bin" / "python"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${1:-}" == "-" ]]; then\n'
        f"  printf '%s\\n' {json.dumps(installed_sha or '')}\n"
        "  exit 0\n"
        "fi\n"
        'if [[ "${1:-}" == "-c" ]]; then\n'
        f"  printf '%s' {json.dumps(json.dumps(facts))}\n"
        "  exit 0\n"
        "fi\n"
        f'exec {sys.executable} "$@"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def _noop_installer(root: Path) -> Path:
    """An install script that reports success and changes nothing."""
    script = root / "noop-install.sh"
    script.write_text(
        "#!/usr/bin/env bash\necho '== fake co-install: exiting 0 =='\nexit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _run_drift(
    omni_home: Path, shim: Path, *args: str, **env_extra: str
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["OMNI_HOME"] = str(omni_home)
    env.pop("ONEX_VENV_RECONCILE_LOCK", None)
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(_DRIFT_SCRIPT), *args, str(shim)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


# --------------------------------------------------------------------------- #
# The readback: --repair may not report success it cannot prove
# --------------------------------------------------------------------------- #
def test_repair_that_changes_nothing_is_reported_as_drifted(tmp_path: Path) -> None:
    """THE DEFECT. Before OMN-18663 this exited 0 on the installer's status."""
    omni_home, head = _make_omni_home(tmp_path)
    stale = "c" * 40
    shim = _make_venv_shim(
        tmp_path,
        installed_sha=stale,
        facts={"omnimarket": {"version": "0.4.118", "commit": stale}},
    )
    result = _run_drift(
        omni_home,
        shim,
        "--repair",
        ONEX_DRIFT_INSTALL_SCRIPT=str(_noop_installer(tmp_path)),
    )
    assert result.returncode == _EXIT_DRIFT, result.stdout + result.stderr
    assert "DRIFTED" in result.stderr
    # Both values, so the operator needs nothing else to act.
    assert stale in result.stderr + result.stdout
    assert head in result.stderr + result.stdout


def test_repair_whose_result_the_venv_carries_passes(tmp_path: Path) -> None:
    omni_home, head = _make_omni_home(tmp_path)
    shim = _make_venv_shim(
        tmp_path,
        installed_sha="c" * 40,
        facts={"omnimarket": {"version": "0.4.119", "commit": head}},
    )
    result = _run_drift(
        omni_home,
        shim,
        "--repair",
        ONEX_DRIFT_INSTALL_SCRIPT=str(_noop_installer(tmp_path)),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PROVEN" in result.stdout


def test_no_drift_still_short_circuits_without_a_repair(tmp_path: Path) -> None:
    omni_home, head = _make_omni_home(tmp_path)
    shim = _make_venv_shim(
        tmp_path,
        installed_sha=head,
        facts={"omnimarket": {"version": "0.4.119", "commit": head}},
    )
    result = _run_drift(omni_home, shim)
    assert result.returncode == 0
    assert "OK: installed omnimarket matches" in result.stdout


# --------------------------------------------------------------------------- #
# The lock: a reader that cannot take it reports BUSY, never a verdict
# --------------------------------------------------------------------------- #
def _hold_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("w")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def test_dry_run_under_contention_reports_busy_not_a_plan(tmp_path: Path) -> None:
    """The 2026-09-18 misread: a reader must not classify a half-applied venv."""
    omni_home, _head = _make_omni_home(tmp_path)
    shim = _make_venv_shim(
        tmp_path,
        installed_sha="c" * 40,
        facts={"omnimarket": {"version": "0.4.118", "commit": "c" * 40}},
    )
    lock_path = shim.parent.parent / ".onex-venv-reconcile.lock"
    handle = _hold_lock(lock_path)
    try:
        result = _run_drift(omni_home, shim, "--dry-run", ONEX_VENV_LOCK_TIMEOUT="0")
    finally:
        handle.close()
    assert result.returncode == _EXIT_BUSY, result.stdout + result.stderr
    assert "BUSY" in result.stderr
    # It got nowhere near a verdict, so it printed none.
    combined = result.stdout + result.stderr
    assert "REMOVE" not in combined
    assert "DRIFT:" not in combined


def test_repair_under_contention_reports_busy_and_mutates_nothing(
    tmp_path: Path,
) -> None:
    omni_home, _ = _make_omni_home(tmp_path)
    shim = _make_venv_shim(
        tmp_path,
        installed_sha="c" * 40,
        facts={"omnimarket": {"version": "0.4.118", "commit": "c" * 40}},
    )
    marker = tmp_path / "installer-ran"
    installer = tmp_path / "recording-install.sh"
    installer.write_text(
        f"#!/usr/bin/env bash\ntouch {marker}\nexit 0\n", encoding="utf-8"
    )
    installer.chmod(0o755)

    handle = _hold_lock(shim.parent.parent / ".onex-venv-reconcile.lock")
    try:
        result = _run_drift(
            omni_home,
            shim,
            "--repair",
            ONEX_VENV_LOCK_TIMEOUT="0",
            ONEX_DRIFT_INSTALL_SCRIPT=str(installer),
        )
    finally:
        handle.close()
    assert result.returncode == _EXIT_BUSY
    assert not marker.exists(), "a BUSY run must not have touched the venv"


def test_the_lock_is_released_so_a_later_run_proceeds(tmp_path: Path) -> None:
    omni_home, head = _make_omni_home(tmp_path)
    shim = _make_venv_shim(
        tmp_path,
        installed_sha=head,
        facts={"omnimarket": {"version": "0.4.119", "commit": head}},
    )
    for _ in range(2):
        result = _run_drift(omni_home, shim)
        assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------- #
# Structural: both entry points are inside the lock, and neither can opt out
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("script", [_DRIFT_SCRIPT, _INSTALL_SCRIPT])
def test_entry_point_runs_its_critical_section_under_the_lock(script: Path) -> None:
    source = script.read_text(encoding="utf-8")
    assert "lib/venv_reconcile_lock.sh" in source
    assert "venv_reconcile_run_locked" in source
    assert "venv_reconcile_say_busy" in source


@pytest.mark.parametrize("script", [_DRIFT_SCRIPT, _INSTALL_SCRIPT])
def test_entry_point_reads_the_venv_back(script: Path) -> None:
    assert "venv_readback.py" in script.read_text(encoding="utf-8")


@pytest.mark.parametrize("script", [_DRIFT_SCRIPT, _INSTALL_SCRIPT, _LOCK_LIB])
def test_no_bypass_flag_is_advertised(script: Path) -> None:
    """Rule 10: a gate with an off switch is the first thing a failing lane reaches for."""
    source = script.read_text(encoding="utf-8")
    for shape in ("--skip-readback", "--no-readback", "--no-lock", "--force"):
        assert shape not in source, f"{shape} would turn the gate off"


def test_the_lock_uses_fcntl_not_flock1() -> None:
    """macOS ships no flock(1); the idiom exits 127 without running anything."""
    source = _LOCK_LIB.read_text(encoding="utf-8")
    assert "heavy_lock.py" in source
    assert "fcntl" in source
