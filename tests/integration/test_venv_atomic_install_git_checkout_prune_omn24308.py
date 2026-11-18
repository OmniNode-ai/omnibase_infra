# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The atomic install's real command line prunes the uv git checkout it recorded (OMN-24308).

The host reconcile and ``install-node-skill-package.sh`` run the engine through the node
runtime's command line (``nodes/node_venv_atomic_install/runtime_venv_atomic_install.py``) as a
subprocess, under the interpreter of the venv they repair. This test joins the pieces the same
way the script does: the real command line, the engine it loads by path, and the real ``uv cache
dir`` that honours ``UV_CACHE_DIR``. A stage run records the revision it was asked to install; a
finalize run removes that revision's checkout from a seeded cache and nothing else; a rollback
removes nothing. No network and no venv are needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOOL = (
    _REPO_ROOT
    / "src/omnibase_infra/nodes/node_venv_atomic_install/runtime_venv_atomic_install.py"
)
_SHA = "0123456789abcdef0123456789abcdef01234567"
_OTHER = "fedcba9876543210fedcba9876543210fedcba98"
_REQUIREMENT = f"omnimarket @ git+https://github.com/OmniNode-ai/omnimarket.git@{_SHA}"


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


def _seed(cache: Path, layout: str, repo: str, sha: str) -> Path:
    folder = cache / layout / "checkouts" / repo / sha[:7]
    folder.mkdir(parents=True)
    (folder / "file").write_text("x", encoding="utf-8")
    (folder.parent / f"{sha[:7]}.ok").write_text("", encoding="utf-8")
    return folder


def test_cli_finalize_prunes_the_recorded_revision_and_rollback_does_not(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    recorded = _seed(cache, "git-v1", "repoA", _SHA)
    bystander = _seed(cache, "git-v1", "repoA", _OTHER)
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "git-revisions.txt").write_text(f"{_SHA}\n", encoding="utf-8")

    rolled_back = _run(cache, "rollback", "--stage-dir", str(stage))
    assert rolled_back.returncode == 0, rolled_back.stdout + rolled_back.stderr
    assert recorded.is_dir(), "rollback pruned a checkout"

    finalized = _run(cache, "finalize", "--stage-dir", str(stage))
    assert finalized.returncode == 0, finalized.stdout + finalized.stderr
    assert not recorded.exists(), "finalize left the recorded checkout"
    assert not recorded.with_name(f"{recorded.name}.ok").exists()
    assert bystander.is_dir(), "finalize pruned a checkout it never recorded"
    assert not stage.exists(), "finalize left the stage folder"
    assert "pruned uv git checkout" in finalized.stdout


def test_cli_stage_records_the_revision_before_it_resolves_anything(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "cache"
    seeded = _seed(cache, "git-v0", "repoA", _SHA)
    stage = tmp_path / "stage"

    # An interpreter path that does not exist: the plan step fails, as a failed install does.
    staged = _run(
        cache,
        "stage",
        "--python",
        str(tmp_path / "no-such-python"),
        "--stage-dir",
        str(stage),
        "--no-deps",
        "--",
        _REQUIREMENT,
    )

    assert staged.returncode != 0, "the stage was expected to fail at its plan"
    assert (stage / "git-revisions.txt").read_text(encoding="utf-8").splitlines() == [
        _SHA
    ]
    assert seeded.is_dir(), "a failed stage pruned a checkout"
