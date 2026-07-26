# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""ensure_runner_clones.sh must provision EVERY sibling the workspace build's
sibling-pin preflight requires -- not just the 5 it hardcoded before this fix
(OMN-15137).

Live failure: OMN-14900 stability deploy-hop re-fire iteration N+3 (run
https://github.com/OmniNode-ai/omnibase_infra/actions/runs/30175447119,
2026-07-25) got past ensure_runner_clones.sh ("all 5 private clones present
and operable"), then failed 3 steps later in stage_workspace.sh's sibling-pin
preflight:

    ERROR: cannot resolve clone pin for omnibase-spi: missing pyproject.toml
    for omnibase-spi: /data/omninode/runner_omni_home/omnibase_spi/pyproject.toml

Root cause: ``RUNNER_CLONE_REPOS`` in ensure_runner_clones.sh never included
``omnibase_spi``, even though stage_workspace.sh's preflight
(``PREFLIGHT_REPO_ARGS`` -> check_sibling_lock_pins.py's
``DEFAULT_PACKAGE_REPO_DIRS``) has required a live ``OMNI_HOME/omnibase_spi``
clone since OMN-12977. The two lists were independently hardcoded and drifted
apart silently.

The fix sources a new shared manifest (``sibling_clone_manifest.sh``) into
both scripts so they can never diverge again. These tests exercise the
REAL ``ensure_runner_clones.sh`` script end-to-end against real (local
``file://``) git remotes -- proving the artifact that runs, not just that the
source text mentions "omnibase_spi".
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "ensure_runner_clones.sh"
MANIFEST = REPO_ROOT / "scripts" / "runtime_build" / "sibling_clone_manifest.sh"

# The full sibling set the workspace build's pin-preflight resolves clones
# for (check_sibling_lock_pins.py's DEFAULT_PACKAGE_REPO_DIRS). Mirrored here
# (not imported) so a regression in the manifest itself still fails this test
# independently -- see test_sibling_clone_manifest_parity.py for the
# cross-language authoritative-source check.
EXPECTED_REPOS = (
    "omnibase_infra",
    "omnibase_core",
    "omnibase_spi",
    "omnibase_compat",
    "onex_change_control",
    "omnimarket",
)


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_bare_upstream(upstream_root: Path, repo: str) -> None:
    """Create a real bare git repo with one commit, clonable via file://."""
    bare = upstream_root / f"{repo}.git"
    _git("init", "-q", "--bare", str(bare), cwd=upstream_root)

    work = upstream_root / f".work-{repo}"
    work.mkdir()
    _git("init", "-q", cwd=work)
    _git(
        "-c",
        "user.email=test@example.com",
        "-c",
        "user.name=test",
        "commit",
        "--allow-empty",
        "-q",
        "-m",
        "init",
        cwd=work,
    )
    _git("remote", "add", "origin", str(bare), cwd=work)
    _git("push", "-q", "origin", "HEAD:refs/heads/main", cwd=work)
    _git("symbolic-ref", "HEAD", "refs/heads/main", cwd=bare)


def _run_ensure_clones(
    omni_home: Path,
    base_url: str,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        "OMNI_HOME": str(omni_home),
        "RUNNER_CLONE_BASE_URL": base_url,
        "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        "HOME": str(omni_home.parent),
    }
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    """A directory of real bare git repos for all 6 canonical siblings."""
    root = tmp_path / "upstream"
    root.mkdir()
    for repo in EXPECTED_REPOS:
        _make_bare_upstream(root, repo)
    return root


def test_manifest_declares_all_six_canonical_siblings() -> None:
    """sibling_clone_manifest.sh enumerates the full set, including spi."""
    text = MANIFEST.read_text(encoding="utf-8")
    for repo in EXPECTED_REPOS:
        assert f'"{repo}"' in text, f"{repo} missing from sibling_clone_manifest.sh"


def test_ensure_runner_clones_provisions_all_six_including_spi(
    tmp_path: Path, upstream: Path
) -> None:
    """RED against the exact OMN-15137 gap: omnibase_spi must be cloned."""
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir()

    result = _run_ensure_clones(omni_home, f"file://{upstream}")

    assert result.returncode == 0, result.stderr
    assert "all 6 private clones present and operable" in result.stderr

    for repo in EXPECTED_REPOS:
        clone = omni_home / repo
        assert (clone / ".git").exists(), f"{repo} was not cloned under OMNI_HOME"

    # The specific regression: omnibase_spi must exist and be a real repo.
    spi = omni_home / "omnibase_spi"
    rev_parse = _git("rev-parse", "--git-dir", cwd=spi)
    assert rev_parse.returncode == 0


def test_ensure_runner_clones_is_idempotent(tmp_path: Path, upstream: Path) -> None:
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir()

    first = _run_ensure_clones(omni_home, f"file://{upstream}")
    assert first.returncode == 0, first.stderr

    heads_before = {
        repo: _git("rev-parse", "HEAD", cwd=omni_home / repo).stdout.strip()
        for repo in EXPECTED_REPOS
    }

    second = _run_ensure_clones(omni_home, f"file://{upstream}")
    assert second.returncode == 0, second.stderr
    assert "clone missing" not in second.stderr

    for repo in EXPECTED_REPOS:
        head_after = _git("rev-parse", "HEAD", cwd=omni_home / repo).stdout.strip()
        assert head_after == heads_before[repo]


def test_missing_upstream_repo_fails_closed_naming_the_repo(
    tmp_path: Path, upstream: Path
) -> None:
    """If ANY sibling's upstream is unreachable, fail loud and name it --
    the exact shape of the OMN-15137 defect (a sibling silently never
    provisioned) must be impossible to reproduce for any repo in the set."""
    # Remove the omnibase_spi upstream to simulate "not provisioned anywhere"
    import shutil

    shutil.rmtree(upstream / "omnibase_spi.git")

    omni_home = tmp_path / "omni_home"
    omni_home.mkdir()

    result = _run_ensure_clones(omni_home, f"file://{upstream}")

    assert result.returncode == 64
    assert "omnibase_spi" in result.stderr
    assert "git clone failed" in result.stderr
    # Must not silently report success.
    assert "all 6 private clones present" not in result.stderr


def test_unwritable_clone_git_dir_fails_closed(tmp_path: Path, upstream: Path) -> None:
    """A clone whose .git is not writable by the current euid must fail --
    this is the dubious-ownership / FETCH_HEAD-EACCES failure mode the
    private-OMNI_HOME provisioning (OMN-14900) exists to catch."""
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir()

    first = _run_ensure_clones(omni_home, f"file://{upstream}")
    assert first.returncode == 0, first.stderr

    spi_git_dir = omni_home / "omnibase_spi" / ".git"
    spi_git_dir.chmod(0o500)
    try:
        result = _run_ensure_clones(omni_home, f"file://{upstream}")
        assert result.returncode == 64
        assert "not writable" in result.stderr
        assert "omnibase_spi" in result.stderr
    finally:
        spi_git_dir.chmod(0o700)


def test_missing_omni_home_fails_closed() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 64
    assert "OMNI_HOME must be set" in result.stderr
