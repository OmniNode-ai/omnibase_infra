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


# ---------------------------------------------------------------------------
# --repair reversed-drift regression (OMN-16366)
#
# `--repair` installs the resolved `origin/dev` SHA into the target venv but,
# before this fix, never advanced the canonical clone's own checked-out HEAD
# to match -- so the very next run of the in-process guard
# (`omnimarket_drift_guard.canonical_local_omnimarket_commit`, which reads
# only the clone's local `git rev-parse HEAD`, never `origin/dev`) would
# immediately re-fail, just with drift reversed. These tests exercise the
# --repair fast-forward step hermetically by stubbing out
# install-node-skill-package.sh (a no-op that records its args) alongside a
# copy of the real script, so no network `uv pip install` runs.
# ---------------------------------------------------------------------------


def _make_repair_harness(tmp_path: Path, *, stub_exit: int = 0) -> tuple[Path, Path]:
    """Copy the real script into an isolated dir alongside a stub
    install-node-skill-package.sh (SCRIPT_DIR-relative, so the copy picks up
    the stub instead of the real installer). Returns (script_copy, marker).
    """
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()

    script_copy = harness_dir / "check-omnimarket-venv-drift.sh"
    script_copy.write_text(_SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    script_copy.chmod(0o755)

    marker = harness_dir / "install_calls.log"
    stub_installer = harness_dir / "install-node-skill-package.sh"
    stub_installer.write_text(
        "#!/usr/bin/env bash\n"
        f'echo "ref=${{OMNIMARKET_REF:-}} args=$*" >> "{marker}"\n'
        f"exit {stub_exit}\n",
        encoding="utf-8",
    )
    stub_installer.chmod(0o755)

    return script_copy, marker


def test_repair_fast_forwards_canonical_clone_to_installed_commit(
    tmp_path: Path,
) -> None:
    """A clean ff case: local canonical clone starts strictly behind
    origin/dev. --repair must leave the canonical clone's checked-out HEAD
    equal to the SHA it installed, so the in-process guard does not
    immediately re-fail on the next dispatch.
    """
    bare = _init_bare_remote(tmp_path)
    omnimarket_root = _make_local_omnimarket_clone(tmp_path, bare)
    behind_sha = _canon_head(omnimarket_root)

    # Advance origin/dev past what the local clone has checked out, via a
    # second, throwaway clone of the bare remote (which has `origin`
    # configured, unlike the `work` repo the bare remote was mirrored from).
    advance_clone = tmp_path / "advance_clone"
    subprocess.run(
        ["git", "clone", "--quiet", str(bare), str(advance_clone)], check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=advance_clone,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=advance_clone, check=True
    )
    (advance_clone / "f2.txt").write_text("y", encoding="utf-8")
    subprocess.run(["git", "add", "f2.txt"], cwd=advance_clone, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "advance"], cwd=advance_clone, check=True
    )
    subprocess.run(["git", "push", "--quiet"], cwd=advance_clone, check=True)
    ahead_sha = _canon_head(advance_clone)
    assert ahead_sha != behind_sha

    fake_python = _make_fake_python_shim(tmp_path, behind_sha)  # stale/drifted
    script_copy, marker = _make_repair_harness(tmp_path)

    result = subprocess.run(
        ["bash", str(script_copy), "--repair", str(fake_python)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={"PATH": "/usr/bin:/bin", "OMNI_HOME": str(tmp_path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.exists(), "stub installer was never invoked"
    assert f"ref={ahead_sha}" in marker.read_text(encoding="utf-8")

    # The regression: the canonical clone itself must now be at the same
    # commit that was installed, not left behind at `behind_sha`.
    assert _canon_head(omnimarket_root) == ahead_sha


def test_repair_refuses_when_canonical_clone_diverged(tmp_path: Path) -> None:
    """A diverged case (the OMN-14638 shape): the local canonical clone has
    its own commit `origin/dev` does not have. A clean fast-forward is
    impossible, so --repair must refuse loudly and NOT install a commit the
    canonical clone can never reach (which would strand the guard failing
    with drift reversed until a human intervenes) -- and must not touch the
    target venv at all.
    """
    bare = _init_bare_remote(tmp_path)
    omnimarket_root = _make_local_omnimarket_clone(tmp_path, bare)

    # Diverge the local clone: a commit origin/dev does not have (a sibling,
    # not an ancestor, of what origin/dev advances to below -- a mere
    # "local is ahead by one commit" case is trivially ff-able and would not
    # exercise the refusal path at all).
    (omnimarket_root / "local_only.txt").write_text("z", encoding="utf-8")
    subprocess.run(["git", "add", "local_only.txt"], cwd=omnimarket_root, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "local-only"],
        cwd=omnimarket_root,
        check=True,
    )
    diverged_sha = _canon_head(omnimarket_root)

    # Independently advance origin/dev past the SAME base commit, via a
    # separate throwaway clone -- so origin/dev's new tip is a sibling of
    # `diverged_sha`, not its ancestor, and a fast-forward is impossible.
    advance_clone = tmp_path / "advance_clone"
    subprocess.run(
        ["git", "clone", "--quiet", str(bare), str(advance_clone)], check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=advance_clone,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=advance_clone, check=True
    )
    (advance_clone / "remote_only.txt").write_text("w", encoding="utf-8")
    subprocess.run(["git", "add", "remote_only.txt"], cwd=advance_clone, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "remote-only"],
        cwd=advance_clone,
        check=True,
    )
    subprocess.run(["git", "push", "--quiet"], cwd=advance_clone, check=True)

    fake_python = _make_fake_python_shim(tmp_path, "f" * _CANON_SHA_LEN)
    script_copy, marker = _make_repair_harness(tmp_path)

    result = subprocess.run(
        ["bash", str(script_copy), "--repair", str(fake_python)],
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
        env={"PATH": "/usr/bin:/bin", "OMNI_HOME": str(tmp_path)},
    )
    assert result.returncode != 0
    assert not marker.exists(), "stub installer must not run on a diverged clone"
    assert "cannot fast-forward" in (result.stdout + result.stderr)
    # The clone must be left exactly as it was -- refusal, not a mutation.
    assert _canon_head(omnimarket_root) == diverged_sha
