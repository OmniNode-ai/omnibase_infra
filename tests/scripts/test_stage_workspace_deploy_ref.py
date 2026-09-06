# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RT-1 (OMN-14438): end-to-end wiring of the clean-ref deploy source into
stage_workspace.sh.

Proves, against a REAL behind clone (exists-but-WRONG, not absent):
  * with DEPLOY_REF set, the ambient BEHIND clone is checked out to the intended
    ref before staging, so the vendored-SHA manifest carries the NEW ref SHA (if
    the checkout were a no-op the manifest would carry the stale behind SHA);
  * without DEPLOY_REF the build is REFUSED outright (OMN-17291), and the
    ambient-tree build is reachable only behind the named opt-in;
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
    allow_unpinned: bool = False,
    repo_refs: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    (build_ctx / "workspace").mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMNI_HOME": str(omni_home),
        "CONSUMER_LOCK": str(omni_home / "omnimarket" / "uv.lock"),
    }
    env.pop("DEPLOY_REF", None)
    env.pop("DEPLOY_HOTPATCH", None)
    env.pop("ALLOW_UNPINNED_DEPLOY_SOURCE", None)
    if deploy_ref is not None:
        env["DEPLOY_REF"] = deploy_ref
    if hotpatch:
        env["DEPLOY_HOTPATCH"] = "1"
    if allow_unpinned:
        env["ALLOW_UNPINNED_DEPLOY_SOURCE"] = "1"
    command = ["bash", str(STAGE_SCRIPT)]
    for repo_ref in repo_refs or []:
        command.extend(["--repo-ref", repo_ref])
    return subprocess.run(
        command,
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
def test_without_deploy_ref_the_build_is_refused(tmp_path: Path) -> None:
    """OMN-17291 superseded the warn-and-build-anyway contract this test used to
    pin. An unasserted source ref must not be able to produce a build at all."""
    omni_home = _make_omni_home(tmp_path)
    _behind_core(omni_home)

    build_ctx = tmp_path / "ctx"
    result = _run_stage(omni_home, build_ctx, deploy_ref=None)
    assert result.returncode == 5, result.stderr
    assert "DEPLOY_REF unset" in result.stderr

    # Refused before staging: no provenance, no expected-refs, nothing vendored.
    assert not (build_ctx / "workspace" / "sibling-vcs-provenance.json").exists()
    assert not (build_ctx / "workspace" / "deploy-source-refs.json").exists()


@pytest.mark.unit
def test_unpinned_ambient_build_behind_explicit_opt_in(tmp_path: Path) -> None:
    """The ambient-tree build survives as a NAMED opt-in: still unasserted (the
    behind SHA is vendored), but never the silent default."""
    omni_home = _make_omni_home(tmp_path)
    old_sha, _new_sha = _behind_core(omni_home)

    build_ctx = tmp_path / "ctx"
    result = _run_stage(omni_home, build_ctx, deploy_ref=None, allow_unpinned=True)
    assert result.returncode == 0, result.stderr

    # Unpinned: no clean-checkout ran, so the ambient BEHIND SHA was vendored.
    vcs = json.loads(
        (build_ctx / "workspace" / "sibling-vcs-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert vcs["siblings"]["omnibase_core"]["vcs_ref"] == old_sha
    # No expected-refs manifest, and the opt-in is named in the log.
    assert not (build_ctx / "workspace" / "deploy-source-refs.json").exists()
    assert "ALLOW_UNPINNED_DEPLOY_SOURCE=1" in result.stderr
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


PINNED_SIBLINGS = ("omnibase_core", "omnibase_compat", "omnimarket")


def _current_repo_refs(omni_home: Path) -> list[str]:
    return [
        f"{repo}={_git(omni_home / repo, 'rev-parse', 'HEAD')}"
        for repo in PINNED_SIBLINGS
    ]


def _make_pinned_clones(tmp_path: Path) -> Path:
    """Only the checkout targets are needed for pre-staging refusal tests."""
    omni_home = tmp_path / "omni_home"
    for repo in PINNED_SIBLINGS:
        _init_repo(omni_home / repo, _DIST_NAME[repo])
    return omni_home


@pytest.mark.unit
def test_per_repo_pins_stage_distinct_immutable_commits(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    old_core, target_core = _behind_core(omni_home)
    market = omni_home / "omnimarket"
    _git(market, "checkout", "-q", "-b", "unmerged-change")
    target_market = _advance_dev_keep_version(market)
    _git(market, "checkout", "-q", "dev")
    target_compat = _git(omni_home / "omnibase_compat", "rev-parse", "HEAD")
    targets = {
        "omnibase_core": target_core,
        "omnibase_compat": target_compat,
        "omnimarket": target_market,
    }
    result = _run_stage(
        omni_home,
        tmp_path / "ctx",
        repo_refs=[f"{repo}={sha}" for repo, sha in targets.items()],
    )
    assert result.returncode == 0, result.stderr
    assert target_core != old_core
    expected = json.loads(
        (tmp_path / "ctx/workspace/deploy-source-refs.json").read_text()
    )
    vcs = json.loads(
        (tmp_path / "ctx/workspace/sibling-vcs-provenance.json").read_text()
    )
    assert expected["ref_pinned"] is True
    for repo, sha in targets.items():
        assert _git(omni_home / repo, "rev-parse", "HEAD") == sha
        assert expected["repos"][repo]["expected_sha"] == sha
        assert expected["repos"][repo]["hotpatch"] is False
        assert vcs["siblings"][repo]["vcs_ref"] == sha


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid",
    ["missing", "duplicate", "unknown", "short_sha", "global", "hotpatch", "unpinned"],
)
def test_per_repo_pin_errors_preserve_every_clone(tmp_path: Path, invalid: str) -> None:
    omni_home = tmp_path / "omni_home"
    _init_repo(omni_home / "omnibase_core", "omnibase-core")
    before_core, target_core = _behind_core(omni_home)
    refs = [
        f"omnibase_core={target_core}",
        f"omnibase_compat={'b' * 40}",
        f"omnimarket={'c' * 40}",
    ]
    if invalid == "missing":
        refs.pop()
    elif invalid == "duplicate":
        refs.append(refs[0])
    elif invalid == "unknown":
        refs.append(f"other={target_core}")
    elif invalid == "short_sha":
        refs[-1] = "omnimarket=1234567"
    result = _run_stage(
        omni_home,
        tmp_path / "ctx",
        repo_refs=refs,
        deploy_ref="dev" if invalid == "global" else None,
        hotpatch=invalid == "hotpatch",
        allow_unpinned=invalid == "unpinned",
    )
    assert result.returncode != 0
    assert _git(omni_home / "omnibase_core", "rev-parse", "HEAD") == before_core
    assert not (tmp_path / "ctx/workspace/deploy-source-refs.json").exists()


@pytest.mark.unit
def test_per_repo_late_missing_commit_preserves_earlier_clone(tmp_path: Path) -> None:
    omni_home = _make_pinned_clones(tmp_path)
    before_core, target_core = _behind_core(omni_home)
    refs = _current_repo_refs(omni_home)
    refs[0] = f"omnibase_core={target_core}"
    refs[-1] = f"omnimarket={'f' * 40}"
    result = _run_stage(omni_home, tmp_path / "ctx", repo_refs=refs)
    assert result.returncode == 4, result.stderr
    assert _git(omni_home / "omnibase_core", "rev-parse", "HEAD") == before_core
    assert not (tmp_path / "ctx/workspace/deploy-source-refs.json").exists()


@pytest.mark.unit
@pytest.mark.parametrize("ignored", [False, True])
def test_per_repo_dirty_target_is_refused_without_deleting_work(
    tmp_path: Path, ignored: bool
) -> None:
    omni_home = _make_pinned_clones(tmp_path)
    before_core, target_core = _behind_core(omni_home)
    market = omni_home / "omnimarket"
    if ignored:
        (market / ".gitignore").write_text("operator-work.txt\n")
        _git(market, "add", ".gitignore")
        _git(market, "commit", "-q", "-m", "ignore operator work")
    sentinel = market / "operator-work.txt"
    sentinel.write_text("must survive\n")
    refs = _current_repo_refs(omni_home)
    refs[0] = f"omnibase_core={target_core}"
    result = _run_stage(omni_home, tmp_path / "ctx", repo_refs=refs)
    assert result.returncode == 4, result.stderr
    assert "dirty" in result.stderr.lower()
    assert sentinel.read_text() == "must survive\n"
    assert _git(omni_home / "omnibase_core", "rev-parse", "HEAD") == before_core
