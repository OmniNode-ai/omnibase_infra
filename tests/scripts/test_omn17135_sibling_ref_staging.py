# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17135 defect 1: a sibling is staged at a ref of its OWN, never at the
omnibase_infra commit SHA the rebuild request carries.

The CI path (``runtime-rebuild-trigger.yml`` -> ``scripts/trigger_rebuild_on_merge.py``)
constrains ``git_ref`` to a lowercase hex commit SHA **of omnibase_infra**. The
deploy agent exports that verbatim as ``DEPLOY_REF``, and RT-1 (OMN-14438) then
clean-checks-out EVERY sibling at it. An omnibase_infra SHA cannot exist in
omnibase_core / omnibase_compat / omnimarket, so RT-1 exits 3 and the staging
script exits 4 -- ``ERROR: omnibase_core: cannot resolve ref '<infra sha>'``.
Job ``a5b200d5`` died this way 13 seconds after acceptance on 2026-09-09, which
is why rule 24(a)'s automatic lab pass has never once worked through CI.

The contract these tests pin, design (i): the INFRA ref is the requested SHA;
each SIBLING resolves a ref of its own -- its declared tracking head at staging
time -- and the ref actually used is RECORDED per repo in the expected-refs
manifest that the end-of-staging assertion is resolved against.

Fail-closed is unchanged (OMN-14438/OMN-17291): a sibling whose fallback ref is
itself unresolvable still aborts the build. The fallback is a named, recorded
substitution, never a silent "build whatever the ambient tree holds".
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "stage_workspace.sh"

# The repos RT-1 clean-checks-out (stage_workspace.sh SIBLING_REPOS) plus the
# ones the lock-pin preflight resolves out of SIBLING_CLONE_MANIFEST.
_STAGED_SIBLINGS = ("omnibase_core", "omnibase_compat", "omnimarket")
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


def _advance_dev(path: Path) -> str:
    (path / "marker.txt").write_text("advanced\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "advance dev (same version)")
    return _git(path, "rev-parse", "HEAD")


def _write_consumer_lock(omni_home: Path) -> None:
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
    for repo, dist in _DIST_NAME.items():
        _init_repo(omni_home / repo, dist)
    _write_consumer_lock(omni_home)
    return omni_home


def _refs_out(build_ctx: Path) -> Path:
    return build_ctx.parent / "refs-state" / f"{build_ctx.name}.json"


def _run_stage(
    omni_home: Path,
    build_ctx: Path,
    *,
    deploy_ref: str,
    sibling_fallback_ref: str | None = None,
) -> subprocess.CompletedProcess[str]:
    (build_ctx / "workspace").mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "OMNI_HOME": str(omni_home),
        "CONSUMER_LOCK": str(omni_home / "omnimarket" / "uv.lock"),
        "DEPLOY_SOURCE_REFS_OUT": str(_refs_out(build_ctx)),
        "DEPLOY_REF": deploy_ref,
    }
    env.pop("DEPLOY_HOTPATCH", None)
    env.pop("ALLOW_UNPINNED_DEPLOY_SOURCE", None)
    env.pop("DEPLOY_SIBLING_FALLBACK_REF", None)
    if sibling_fallback_ref is not None:
        env["DEPLOY_SIBLING_FALLBACK_REF"] = sibling_fallback_ref
    return subprocess.run(
        ["bash", str(STAGE_SCRIPT)],
        cwd=build_ctx,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _infra_only_sha(omni_home: Path) -> str:
    """A real 40-hex commit SHA that exists in omnibase_infra and in NO sibling.

    This is the exists-but-WRONG shape the live defect has: not a malformed
    string, but a perfectly valid commit object of the wrong repository. A
    sibling asked to resolve it fails for the only reason that matters.
    """
    infra = omni_home / "omnibase_infra"
    sha = _advance_dev(infra)
    for repo in _STAGED_SIBLINGS:
        probe = subprocess.run(
            [
                "git",
                "-C",
                str(omni_home / repo),
                "rev-parse",
                "--verify",
                f"{sha}^{{commit}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert probe.returncode != 0, f"{repo} unexpectedly resolves the infra SHA"
    return sha


# ---------------------------------------------------------------------------
# (a) the live defect: a SHA-form git_ref stages each sibling at its OWN head
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_infra_sha_deploy_ref_stages_each_sibling_at_its_own_head(
    tmp_path: Path,
) -> None:
    omni_home = _make_omni_home(tmp_path)
    infra_sha = _infra_only_sha(omni_home)

    # Every sibling's dev advances and its clone is left BEHIND (detached at the
    # old commit) -- exists-but-WRONG, so a no-op checkout is visible as a stale
    # vendored SHA rather than passing by luck.
    expected: dict[str, str] = {}
    for repo in _STAGED_SIBLINGS:
        clone = omni_home / repo
        old = _git(clone, "rev-parse", "HEAD")
        new = _advance_dev(clone)
        _git(clone, "checkout", "-q", "--detach", old)
        assert old != new
        expected[repo] = new

    build_ctx = tmp_path / "ctx"
    result = _run_stage(
        omni_home, build_ctx, deploy_ref=infra_sha, sibling_fallback_ref="dev"
    )
    assert result.returncode == 0, result.stderr

    vcs = json.loads(
        (build_ctx / "workspace" / "sibling-vcs-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    for repo, sha in expected.items():
        assert vcs["siblings"][repo]["vcs_ref"] == sha
        assert vcs["siblings"][repo]["vcs_ref"] != infra_sha

    # (d) the per-repo ref actually used is RECORDED, alongside the infra ref it
    # was substituted for -- the substitution is evidence, not a silent rescue.
    exp = json.loads(_refs_out(build_ctx).read_text(encoding="utf-8"))
    for repo, sha in expected.items():
        row = exp["repos"][repo]
        assert row["ref"] == "dev"
        assert row["expected_sha"] == sha
        assert row["fallback_from"] == infra_sha


# ---------------------------------------------------------------------------
# (b) fail-closed is unchanged: an unresolvable sibling ref still aborts
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_unresolvable_sibling_fallback_still_fails_closed(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    infra_sha = _infra_only_sha(omni_home)

    build_ctx = tmp_path / "ctx"
    result = _run_stage(
        omni_home,
        build_ctx,
        deploy_ref=infra_sha,
        sibling_fallback_ref="origin/does-not-exist",
    )
    assert result.returncode == 4, result.stdout + result.stderr
    assert "omnibase_core" in result.stderr
    # Both refs are named, so the operator is not left guessing which one failed.
    assert infra_sha in result.stderr
    assert "origin/does-not-exist" in result.stderr
    assert not (build_ctx / "workspace" / "sibling-vcs-provenance.json").exists()


# ---------------------------------------------------------------------------
# (c) a symbolic ref that resolves everywhere behaves exactly as before
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_symbolic_deploy_ref_behaviour_unchanged(tmp_path: Path) -> None:
    omni_home = _make_omni_home(tmp_path)
    core = omni_home / "omnibase_core"
    old = _git(core, "rev-parse", "HEAD")
    new = _advance_dev(core)
    _git(core, "checkout", "-q", "--detach", old)

    build_ctx = tmp_path / "ctx"
    result = _run_stage(
        omni_home, build_ctx, deploy_ref="dev", sibling_fallback_ref="origin/dev"
    )
    assert result.returncode == 0, result.stderr

    vcs = json.loads(
        (build_ctx / "workspace" / "sibling-vcs-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert vcs["siblings"]["omnibase_core"]["vcs_ref"] == new

    exp = json.loads(_refs_out(build_ctx).read_text(encoding="utf-8"))
    for repo in _STAGED_SIBLINGS:
        row = exp["repos"][repo]
        # The primary ref resolved, so the fallback never engaged and nothing
        # records a substitution that did not happen.
        assert row["ref"] == "dev"
        assert row["fallback_from"] == ""
    assert "manifest assertion passed" in result.stderr
