# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration coverage for the omnimarket venv drift repair script."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check-omnimarket-venv-drift.sh"


def _init_bare_remote(root: Path) -> Path:
    work = root / "work"
    work.mkdir()
    subprocess.run(["git", "init", "--quiet", "-b", "dev"], cwd=work, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=work, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=work, check=True)
    (work / "f.txt").write_text("x", encoding="utf-8")
    subprocess.run(["git", "add", "f.txt"], cwd=work, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=work, check=True)

    bare = root / "bare.git"
    subprocess.run(
        ["git", "clone", "--quiet", "--bare", str(work), str(bare)], check=True
    )
    return bare


def _clone_dev(bare_remote: Path, destination: Path) -> str:
    subprocess.run(
        ["git", "clone", "--quiet", str(bare_remote), str(destination)], check=True
    )
    subprocess.run(["git", "checkout", "--quiet", "dev"], cwd=destination, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=destination,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=destination, check=True)
    return _head(destination)


def _head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _fake_python(root: Path, installed_sha: str) -> Path:
    shim = root / "fake_python.sh"
    shim.write_text(
        "#!/usr/bin/env bash\ncat >/dev/null\necho " + f'"{installed_sha}"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def _script_harness(root: Path) -> tuple[Path, Path]:
    harness = root / "harness"
    harness.mkdir()

    script = harness / "check-omnimarket-venv-drift.sh"
    script.write_text(_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(0o755)

    marker = harness / "install_calls.log"
    installer = harness / "install-node-skill-package.sh"
    installer.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "ref=${{OMNIMARKET_REF:-}} args=$*" >> "{marker}"\n',
        encoding="utf-8",
    )
    installer.chmod(0o755)
    return script, marker


def test_repair_fast_forwards_canonical_clone_before_install(tmp_path: Path) -> None:
    bare = _init_bare_remote(tmp_path)
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir()
    omnimarket_root = omni_home / "omnimarket"
    behind_sha = _clone_dev(bare, omnimarket_root)

    advance_clone = tmp_path / "advance_clone"
    _clone_dev(bare, advance_clone)
    (advance_clone / "f2.txt").write_text("y", encoding="utf-8")
    subprocess.run(["git", "add", "f2.txt"], cwd=advance_clone, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "advance"], cwd=advance_clone, check=True
    )
    subprocess.run(["git", "push", "--quiet"], cwd=advance_clone, check=True)
    ahead_sha = _head(advance_clone)

    script, marker = _script_harness(tmp_path)
    result = subprocess.run(
        ["bash", str(script), "--repair", str(_fake_python(tmp_path, behind_sha))],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={"PATH": "/usr/bin:/bin", "OMNI_HOME": str(omni_home)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert _head(omnimarket_root) == ahead_sha
    assert f"ref={ahead_sha}" in marker.read_text(encoding="utf-8")
