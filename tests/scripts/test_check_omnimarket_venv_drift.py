# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/check-omnimarket-venv-drift.sh (OMN-14060).

Fully hermetic and offline: the "canonical" remote is a local bare git repo
(not github.com), reached via `git fetch origin dev` over the filesystem, and
the "installed SHA" probe is a fake python shim that echoes a canned commit id
instead of a real venv. This exercises the script's actual detection logic end
-to-end without a network call or a real omnimarket install. The --repair
mutation path (which shells out to install-node-skill-package.sh against a
real venv over the network) is proven separately by manual verification in the
OMN-14060 PR body, not re-exercised here.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check-omnimarket-venv-drift.sh"

_CANON_SHA_LEN = 40


def _init_bare_remote(root: Path) -> Path:
    """Create a local bare git repo with one commit on `dev`; return its path."""
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


def _make_local_omnimarket_clone(root: Path, bare_remote: Path) -> Path:
    """Clone the bare remote into $OMNI_HOME/omnimarket (dev checked out)."""
    omnimarket_root = root / "omnimarket"
    subprocess.run(
        ["git", "clone", "--quiet", str(bare_remote), str(omnimarket_root)],
        check=True,
    )
    subprocess.run(
        ["git", "checkout", "--quiet", "dev"], cwd=omnimarket_root, check=True
    )
    return omnimarket_root


def _canon_head(omnimarket_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=omnimarket_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _make_fake_python_shim(root: Path, installed_sha: str) -> Path:
    """A fake 'python' that reads (and discards) the heredoc script piped to
    it via stdin, then prints a canned installed-commit SHA (or empty string
    for "not installed") -- exactly what the real probe expects on stdout.
    """
    shim = root / "fake_python.sh"
    shim.write_text(
        "#!/usr/bin/env bash\ncat >/dev/null\necho " + f'"{installed_sha}"\n',
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim


def test_errors_when_omni_home_unset() -> None:
    result = subprocess.run(
        ["bash", str(_SCRIPT), "/usr/bin/python3"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode != 0
    assert "OMNI_HOME is not set" in (result.stdout + result.stderr)


def test_errors_when_no_canonical_clone(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(_SCRIPT), "/usr/bin/python3"],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={"PATH": "/usr/bin:/bin", "OMNI_HOME": str(tmp_path)},
    )
    assert result.returncode != 0
    assert "no canonical omnimarket clone" in (result.stdout + result.stderr)


def test_reports_ok_when_installed_matches_canonical(tmp_path: Path) -> None:
    bare = _init_bare_remote(tmp_path)
    omnimarket_root = _make_local_omnimarket_clone(tmp_path, bare)
    canon_sha = _canon_head(omnimarket_root)
    fake_python = _make_fake_python_shim(tmp_path, canon_sha)

    result = subprocess.run(
        ["bash", str(_SCRIPT), str(fake_python)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={"PATH": "/usr/bin:/bin", "OMNI_HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK: installed omnimarket matches canonical" in result.stdout
    assert canon_sha in result.stdout


def test_reports_drift_when_installed_diverges(tmp_path: Path) -> None:
    bare = _init_bare_remote(tmp_path)
    omnimarket_root = _make_local_omnimarket_clone(tmp_path, bare)
    canon_sha = _canon_head(omnimarket_root)
    stale_sha = "f" * _CANON_SHA_LEN
    fake_python = _make_fake_python_shim(tmp_path, stale_sha)

    result = subprocess.run(
        ["bash", str(_SCRIPT), str(fake_python)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={"PATH": "/usr/bin:/bin", "OMNI_HOME": str(tmp_path)},
    )
    assert result.returncode == 1
    assert f"DRIFT: installed {stale_sha} != canonical {canon_sha}" in result.stdout
    assert "Re-run with --repair to fix" in result.stdout
    # Must not attempt a repair without the flag.
    assert "repairing" not in result.stdout.lower()


def test_reports_drift_when_not_installed(tmp_path: Path) -> None:
    bare = _init_bare_remote(tmp_path)
    _make_local_omnimarket_clone(tmp_path, bare)
    fake_python = _make_fake_python_shim(tmp_path, "")  # "" == not installed

    result = subprocess.run(
        ["bash", str(_SCRIPT), str(fake_python)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={"PATH": "/usr/bin:/bin", "OMNI_HOME": str(tmp_path)},
    )
    assert result.returncode == 1
    assert "omnimarket is not installed" in result.stdout


def test_script_exists_and_executable() -> None:
    assert _SCRIPT.is_file(), f"missing script: {_SCRIPT}"
    assert _SCRIPT.stat().st_mode & 0o111, "script must be executable"


def test_script_has_spdx_header() -> None:
    head = "\n".join(_SCRIPT.read_text(encoding="utf-8").splitlines()[:4])
    assert "SPDX-License-Identifier: MIT" in head
    assert "SPDX-FileCopyrightText" in head


def test_no_hardcoded_absolute_machine_paths() -> None:
    text = _SCRIPT.read_text(encoding="utf-8")
    for token in ("/Users/", "/Volumes/"):
        assert token not in text, f"hardcoded machine path {token!r} present"
