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

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

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
    assert "swept 2 stale stage dir(s)" in finalized.stdout


def test_cli_finalize_reports_a_stale_stage_it_cannot_remove(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    venv = tmp_path / "venv"
    current = venv / ".atomic-stage.current"
    removable = venv / ".atomic-stage.removable"
    protected = venv / ".atomic-stage.protected"
    for stage in (current, removable, protected):
        stage.mkdir(parents=True)
    payload = protected / "bytecode.pyc"
    payload.write_bytes(b"retained")

    if os.geteuid() == 0:
        # Root can unlink from a read-only directory. Inject the failed
        # deletion in the subprocess while leaving other rmtree calls real.
        (tmp_path / "sitecustomize.py").write_text(
            dedent("""\
                import shutil
                from pathlib import Path

                real_rmtree = shutil.rmtree

                def rmtree(path, *, ignore_errors=False):
                    if Path(path).name == ".atomic-stage.protected":
                        if ignore_errors:
                            return
                        raise PermissionError(path)
                    return real_rmtree(path, ignore_errors=ignore_errors)

                shutil.rmtree = rmtree
                """),
            encoding="utf-8",
        )
        monkeypatch.setenv("PYTHONPATH", str(tmp_path), prepend=os.pathsep)

    protected.chmod(0o555)
    try:
        finalized = _run(tmp_path / "cache", "finalize", "--stage-dir", str(current))
    finally:
        if protected.is_dir():
            protected.chmod(0o755)

    assert finalized.returncode == 0, finalized.stdout + finalized.stderr
    assert not current.exists()
    assert not removable.exists()
    assert payload.read_bytes() == b"retained"
    assert f"swept stale stage {removable}" in finalized.stdout
    assert f"swept stale stage {protected}" not in finalized.stdout
    assert "swept 1 stale stage dir(s)" in finalized.stdout
    assert f"stale stage {protected} could not be removed" in finalized.stderr
    assert "1 stale stage dir(s) remain" in finalized.stderr
