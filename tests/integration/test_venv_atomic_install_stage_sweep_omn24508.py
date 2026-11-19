# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The installer's real command line, run as a subprocess the way ``install-node-skill-package.sh`` runs it
(OMN-24508).

The host reconcile and the script run the engine through the node runtime's command line (``nodes/node_venv_atomic_install/
runtime_venv_atomic_install.py``) as a subprocess. A failed or stopped install leaves its ``.atomic-stage.*``
side folder behind, and a rollback removes no sibling; a later finalize sweeps those leftovers and keeps the
venv's own ``bin`` and ``lib``. No network and no venv are needed: an interpreter path that does not exist fails
the stage at its plan, as a failed install does.
"""

from __future__ import annotations

import fcntl
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = (
    _REPO_ROOT
    / "src/omnibase_infra/nodes/node_venv_atomic_install/runtime_venv_atomic_install.py"
)
_REQUIREMENT = (
    "omnimarket @ git+https://github.com/OmniNode-ai/omnimarket.git"
    "@0123456789abcdef0123456789abcdef01234567"
)


def _run(
    cache: Path, *argv: str, python: str | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [python or sys.executable, str(_TOOL), *argv],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={**os.environ, "UV_CACHE_DIR": str(cache)},
    )


def test_cli_finalize_sweeps_the_stages_that_failed_runs_left_behind(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "lib").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")

    for name in ("dead1", "dead2"):
        stage_dir = venv / f".atomic-stage.{name}"
        staged = _run(
            cache,
            "stage",
            "--python",
            str(tmp_path / "no-such-python"),
            "--stage-dir",
            str(stage_dir),
            "--no-deps",
            "--",
            _REQUIREMENT,
        )
        assert staged.returncode != 0, staged.stdout + staged.stderr
        assert stage_dir.is_dir(), "a failed install removed its own stage folder"

    rolled_back = _run(
        cache, "rollback", "--stage-dir", str(venv / ".atomic-stage.dead2")
    )
    assert rolled_back.returncode == 0, rolled_back.stdout + rolled_back.stderr
    for name in ("dead1", "dead2"):
        assert (venv / f".atomic-stage.{name}").is_dir(), (
            "a rollback removed a sibling stage"
        )

    current = venv / ".atomic-stage.current"
    current.mkdir()
    finalized = _run(cache, "finalize", "--stage-dir", str(current))
    assert finalized.returncode == 0, finalized.stdout + finalized.stderr
    assert not [
        child for child in venv.iterdir() if child.name.startswith(".atomic-stage.")
    ]
    assert (venv / "bin").is_dir(), "finalize removed the venv's bin"
    assert (venv / "lib").is_dir(), "finalize removed the venv's lib"
    assert "swept 2 stale stage dir(s), kept 0, failed 0" in finalized.stdout


_SCRIPT = _REPO_ROOT / "scripts/install-node-skill-package.sh"


def _git(*argv: str, cwd: Path) -> str:
    done = subprocess.run(
        ["git", *argv],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(os.environ),
    )
    return done.stdout.strip()


def _installer_world(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """A venv root, an omnimarket clone, a uv that always fails, and the env the script reads."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text(
        f"home = {Path(sys.executable).parent}\n", encoding="utf-8"
    )
    (venv / "bin" / "python").symlink_to(sys.executable)
    home = tmp_path / "home"
    clone = home / "omnimarket"
    clone.mkdir(parents=True)
    _git("init", "-q", cwd=clone)
    (clone / "pyproject.toml").write_text(
        '[project]\nname = "omnimarket"\nversion = "0"\n'
        'dependencies = ["omnibase-compat>=1.0", "omninode-memory>=1.0"]\n',
        encoding="utf-8",
    )
    _git("add", ".", cwd=clone)
    _git(
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "commit",
        "-qm",
        "pins",
        cwd=clone,
    )
    ref = _git("rev-parse", "HEAD", cwd=clone)
    shim = tmp_path / "shim"
    shim.mkdir()
    uv = shim / "uv"
    uv.write_text("#!/bin/sh\necho 'uv: stub refuses every install' >&2\nexit 1\n")
    uv.chmod(uv.stat().st_mode | stat.S_IXUSR)
    env = {
        **os.environ,
        "OMNI_HOME": str(home),
        "OMNIMARKET_REF": ref,
        "ONEX_ATOMIC_INSTALL": "1",
        "ONEX_VENV_LOCK_TIMEOUT": "2s",
        "PATH": f"{shim}:{os.environ['PATH']}",
    }
    env.pop("ONEX_VENV_RECONCILE_LOCK", None)
    return venv, home, env


def _run_script(venv: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_SCRIPT), "--execute", str(venv / "bin" / "python")],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
        env=env,
    )


def test_failure_of_the_real_script_leaves_no_stage_and_keeps_only_a_journaled_one(
    tmp_path: Path,
) -> None:
    """The script's own exit sweeps after a FAILED install, under the lock it already holds."""
    venv, _home, env = _installer_world(tmp_path)
    (venv / ".atomic-stage.dead1").mkdir()
    (venv / ".atomic-stage.dead1" / "plan.json").write_text("[]", encoding="utf-8")
    (venv / ".atomic-stage.undo").mkdir()
    (venv / ".atomic-stage.undo" / "journal.json").write_text("[]", encoding="utf-8")

    done = _run_script(venv, env)

    assert done.returncode != 0, done.stdout + done.stderr
    assert "stub refuses every install" in done.stderr, done.stdout + done.stderr
    left = sorted(p.name for p in venv.iterdir() if p.name.startswith(".atomic-stage."))
    assert left == [".atomic-stage.undo"], done.stdout + done.stderr
    assert "kept stage" in done.stdout and "holds a journal" in done.stdout


def test_the_real_script_waits_for_the_lock_and_touches_no_stage_while_it_is_held(
    tmp_path: Path,
) -> None:
    """A concurrent install holds the lock: the script reports BUSY and removes nothing."""
    venv, _home, env = _installer_world(tmp_path)
    concurrent = venv / ".atomic-stage.concurrent"
    concurrent.mkdir()
    (concurrent / "plan.json").write_text("[]", encoding="utf-8")
    holder = os.open(venv / ".onex-venv-reconcile.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(holder, fcntl.LOCK_EX)
    try:
        done = _run_script(venv, env)
    finally:
        os.close(holder)
    assert done.returncode == 4, done.stdout + done.stderr
    assert "BUSY" in done.stderr
    assert (concurrent / "plan.json").is_file(), "a held install's stage was removed"
