# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RT-1 (OMN-14438): end-to-end wiring of the clean-ref deploy source into
stage_workspace.sh.

Proves, against a REAL behind clone (exists-but-WRONG, not absent):
  * with DEPLOY_REF set, the ambient BEHIND clone is checked out to the intended
    ref before staging, so the vendored-SHA manifest carries the NEW ref SHA (if
    the checkout were a no-op the manifest would carry the stale behind SHA);
  * without DEPLOY_REF the build is loudly stamped unpinned and NOT asserted;
  * the exact assertion command stage_workspace.sh runs goes RED on a poisoned
    (real stale) vendored SHA;
  * an unresolvable DEPLOY_REF fails the build closed (exit 4).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "stage_workspace.sh"
DEPLOY_SOURCE_REF = REPO_ROOT / "scripts" / "runtime_build" / "deploy_source_ref.py"

SIBLING_REPOS = (
    "omnibase_core",
    "omnibase_compat",
    "onex_change_control",
    "omnimarket",
)
_DIST_NAME = {
    "omnibase_core": "omnibase-core",
    "omnibase_compat": "omnibase-compat",
    "onex_change_control": "onex-change-control",
    "omnimarket": "omnimarket",
    "omnibase_infra": "omnibase-infra",
    "omnibase_spi": "omnibase-spi",
}
_PIN_VERSION = "9.9.9"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def _init_repo(path: Path, dist: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q", "-b", "dev")
    _git(path, "config", "user.email", "t@t.t")
    _git(path, "config", "user.name", "t")
    (path / "pyproject.toml").write_text(
        f"[project]\nname = '{dist}'\nversion = '{_PIN_VERSION}'\n", encoding="utf-8"
    )
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "init")


def _advance_dev_keep_version(path: Path) -> str:
    """Add a second commit on dev that keeps the pinned version, and return the new
    dev HEAD SHA."""
    (path / "marker.txt").write_text("advanced\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "advance dev (same version)")
    return _git(path, "rev-parse", "HEAD")


def _write_consumer_lock(omni_home: Path) -> None:
    # uv.lock is a TRACKED file in reality (committed on dev), so it must be
    # committed here too -- RT-1's `git clean -fdx` would wipe an untracked lock.
    blocks = [
        f'[[package]]\nname = "{dist}"\nversion = "{_PIN_VERSION}"\n'
        for dist in _DIST_NAME.values()
    ]
    market = omni_home / "omnimarket"
    (market / "uv.lock").write_text("\n".join(blocks), encoding="utf-8")
    _git(market, "add", "uv.lock")
    _git(market, "commit", "-q", "-m", "add uv.lock")


def _make_omni_home(tmp_path: Path) -> Path:
    omni_home = tmp_path / "omni_home"
    for repo in SIBLING_REPOS:
        _init_repo(omni_home / repo, _DIST_NAME[repo])
    for repo in ("omnibase_infra", "omnibase_spi"):
        _init_repo(omni_home / repo, _DIST_NAME[repo])
    _write_consumer_lock(omni_home)
    return omni_home


def _run_stage(
    omni_home: Path,
    build_ctx: Path,
    *,
    deploy_ref: str | None = None,
    hotpatch: bool = False,
) -> subprocess.CompletedProcess[str]:
    (build_ctx / "workspace").mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMNI_HOME": str(omni_home),
        "CONSUMER_LOCK": str(omni_home / "omnimarket" / "uv.lock"),
    }
    if deploy_ref is not None:
        env["DEPLOY_REF"] = deploy_ref
    if hotpatch:
        env["DEPLOY_HOTPATCH"] = "1"
    return subprocess.run(
        ["bash", str(STAGE_SCRIPT)],
        cwd=build_ctx,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _behind_core(omni_home: Path) -> tuple[str, str]:
    """Advance omnibase_core's dev by a commit, then leave the clone BEHIND
    (detached at the old commit). Returns (old_sha, new_dev_sha)."""
    core = omni_home / "omnibase_core"
    old_sha = _git(core, "rev-parse", "HEAD")
    new_sha = _advance_dev_keep_version(core)
    _git(core, "checkout", "-q", "--detach", old_sha)  # behind dev
    assert _git(core, "rev-parse", "HEAD") == old_sha
    return old_sha, new_sha


@pytest.mark.unit
def test_deploy_ref_checks_out_behind_clone_and_asserts_green(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    old_sha, new_sha = _behind_core(omni_home)
    assert old_sha != new_sha

    build_ctx = tmp_path / "ctx"
    result = _run_stage(omni_home, build_ctx, deploy_ref="dev")
    assert result.returncode == 0, result.stderr

    # The clone was checked out to dev HEAD before staging.
    assert _git(omni_home / "omnibase_core", "rev-parse", "HEAD") == new_sha

    # The vendored-SHA manifest carries the NEW ref SHA -- proof the checkout
    # actually moved the tree during real staging (a no-op would leave old_sha).
    vcs = json.loads(
        (build_ctx / "workspace" / "sibling-vcs-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert vcs["siblings"]["omnibase_core"]["vcs_ref"] == new_sha
    assert vcs["siblings"]["omnibase_core"]["vcs_ref"] != old_sha

    # The expected-refs manifest exists and the in-script assertion passed.
    expected_refs = build_ctx / "workspace" / "deploy-source-refs.json"
    assert expected_refs.exists()
    exp = json.loads(expected_refs.read_text(encoding="utf-8"))
    assert exp["ref_pinned"] is True
    assert exp["repos"]["omnibase_core"]["expected_sha"] == new_sha
    assert "manifest assertion passed" in result.stderr


@pytest.mark.unit
def test_without_deploy_ref_is_unpinned_and_unasserted(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    old_sha, _new_sha = _behind_core(omni_home)

    build_ctx = tmp_path / "ctx"
    result = _run_stage(omni_home, build_ctx, deploy_ref=None)
    assert result.returncode == 0, result.stderr

    # Unpinned: no clean-checkout ran, so the ambient BEHIND SHA was vendored.
    vcs = json.loads(
        (build_ctx / "workspace" / "sibling-vcs-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert vcs["siblings"]["omnibase_core"]["vcs_ref"] == old_sha
    # No expected-refs manifest, and the unpinned warning fired.
    assert not (build_ctx / "workspace" / "deploy-source-refs.json").exists()
    assert "DEPLOY_REF unset" in result.stderr
    assert "NOT asserted" in result.stderr


@pytest.mark.unit
def test_stage_assert_command_goes_red_on_poisoned_provenance(tmp_path: Path) -> None:
    """The exact command stage_workspace.sh runs (deploy_source_ref.py assert)
    fails closed when the vendored SHA is the real stale (behind) commit."""
    omni_home = _make_omni_home(tmp_path)
    old_sha, _new_sha = _behind_core(omni_home)

    build_ctx = tmp_path / "ctx"
    assert _run_stage(omni_home, build_ctx, deploy_ref="dev").returncode == 0

    workspace = build_ctx / "workspace"
    provenance = workspace / "sibling-vcs-provenance.json"
    expected_refs = workspace / "deploy-source-refs.json"

    # Poison the vendored provenance with the REAL old (behind) SHA -- the clone
    # exists and old_sha is a valid commit in it, it is simply the WRONG one.
    vcs = json.loads(provenance.read_text(encoding="utf-8"))
    vcs["siblings"]["omnibase_core"]["vcs_ref"] = old_sha
    provenance.write_text(json.dumps(vcs), encoding="utf-8")

    red = subprocess.run(
        [
            "python3",
            str(DEPLOY_SOURCE_REF),
            "assert",
            "--vcs-provenance",
            str(provenance),
            "--expected-refs",
            str(expected_refs),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert red.returncode == 4, red.stdout + red.stderr
    assert old_sha[:12] in red.stderr


@pytest.mark.unit
def test_unresolvable_deploy_ref_fails_build_closed(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    build_ctx = tmp_path / "ctx"
    result = _run_stage(omni_home, build_ctx, deploy_ref="no-such-ref-xyz")
    assert result.returncode == 4, result.stderr
    assert "clean-ref checkout failed" in result.stderr
    # A failed checkout must NOT leave a provenance manifest claiming success.
    assert not (build_ctx / "workspace" / "sibling-vcs-provenance.json").exists()
