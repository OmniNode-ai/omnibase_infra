# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Behavioral tests for cut_release_train_tag.sh ref resolution (OMN-14956)
and its GITHUB_OUTPUT tag publication (OMN-14957).

Live failure being eliminated (run 29977699589, exit 128): usage advertises
``--ref <branch|tag|sha>``, but the execute loop ran ``git rev-parse
"${REF}^{commit}"`` in EACH of the 5 sibling clones. A raw omnibase_infra
commit SHA resolves only in the anchor clone, so the loop died at
omnibase_core with ``fatal: ambiguous argument`` under ``set -euo pipefail``
-- making the OMN-14900 TAG_SPEC instruction "pass the merge SHA explicitly"
structurally impossible.

The fix resolves --ref in the ANCHOR clone only (named error if even that
fails); siblings use their own resolution of --ref when it resolves there
(branch/tag names) and otherwise fall back, loudly, to their own origin/dev.

These tests drive the REAL script against local file:// git fixtures for all
5 repos, with a recording ``gh`` shim on PATH -- no network, no GitHub.

Per reference_git_env_vars_override_c_and_cwd: strip GIT_DIR/GIT_INDEX_FILE/
GIT_WORK_TREE from the subprocess env so an inherited pre-push hook export
cannot redirect these git operations onto the real worktree.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "cut_release_train_tag.sh"

TAG_REPOS = (
    "omnibase_infra",
    "omnibase_core",
    "omnibase_compat",
    "onex_change_control",
    "omnimarket",
)
ANCHOR = "omnibase_infra"

_HERMETIC_ENV = {
    k: v
    for k, v in os.environ.items()
    if k not in {"GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GITHUB_OUTPUT"}
}

GH_SHIM = """#!/usr/bin/env bash
printf '%s\\n' "$*" >> "${GH_SHIM_LOG:?}"
exit 0
"""


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=_HERMETIC_ENV,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _build_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """5 repos, each: seed commit -> bare origin -> working clone under
    OMNI_HOME. Unique content per repo so every repo has a DIFFERENT dev SHA
    (the raw anchor SHA can never resolve in a sibling)."""
    omni_home = tmp_path / "omni_home"
    omni_home.mkdir()
    dev_shas: dict[str, str] = {}
    for repo in TAG_REPOS:
        seed = tmp_path / "seeds" / repo
        seed.mkdir(parents=True)
        _git(["init", "-b", "dev"], cwd=seed)
        (seed / "f.txt").write_text(f"content-{repo}\n")
        _git(["add", "f.txt"], cwd=seed)
        _git(
            ["-c", "user.email=t@t.com", "-c", "user.name=t", "commit", "-m", repo],
            cwd=seed,
        )
        origin = tmp_path / "origins" / f"{repo}.git"
        origin.parent.mkdir(exist_ok=True)
        _git(["clone", "--bare", str(seed), str(origin)], cwd=tmp_path)
        _git(["clone", str(origin), str(omni_home / repo)], cwd=tmp_path)
        dev_shas[repo] = _git(["rev-parse", "origin/dev"], cwd=omni_home / repo)
    return omni_home, dev_shas


def _run_script(
    tmp_path: Path, omni_home: Path, args: list[str]
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    gh = bin_dir / "gh"
    gh.write_text(GH_SHIM)
    gh.chmod(0o755)
    gh_log = tmp_path / "gh.log"
    gh_log.touch()
    github_output = tmp_path / "github_output"
    github_output.touch()
    env = dict(_HERMETIC_ENV)
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "OMNI_HOME": str(omni_home),
            "GH_SHIM_LOG": str(gh_log),
            "GITHUB_OUTPUT": str(github_output),
        }
    )
    result = subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, gh_log, github_output


@pytest.mark.unit
def test_raw_anchor_sha_tags_anchor_at_sha_and_siblings_at_their_dev(
    tmp_path: Path,
) -> None:
    """The OMN-14956 headline: --ref <raw anchor SHA> must succeed (pre-fix:
    exit 128 at the first sibling), tagging the anchor AT that SHA and each
    sibling at its own origin/dev, with a loud fallback NOTE."""
    omni_home, dev_shas = _build_fixture(tmp_path)
    anchor_sha = dev_shas[ANCHOR]
    result, gh_log, github_output = _run_script(
        tmp_path, omni_home, ["--lane", "stability", "--ref", anchor_sha, "--execute"]
    )
    assert result.returncode == 0, f"exit {result.returncode}; stderr:\n{result.stderr}"
    # The cut tag name is published for the chained deploy job (OMN-14957).
    out_lines = github_output.read_text().splitlines()
    tags = [line.split("=", 1)[1] for line in out_lines if line.startswith("tag=")]
    assert len(tags) == 1, f"expected one tag= output line, got {out_lines}"
    tag = tags[0]
    assert tag.startswith("lab/stability/")
    assert tag.endswith(anchor_sha[:12])
    # Anchor tagged at exactly the requested SHA.
    assert (
        _git(["rev-parse", f"{tag}^{{commit}}"], cwd=omni_home / ANCHOR) == anchor_sha
    )
    # Every sibling tagged at ITS OWN origin/dev (fallback), loudly.
    for repo in TAG_REPOS:
        if repo == ANCHOR:
            continue
        assert (
            _git(["rev-parse", f"{tag}^{{commit}}"], cwd=omni_home / repo)
            == dev_shas[repo]
        ), f"{repo} must be tagged at its own origin/dev"
    assert "falling back to this sibling's own origin/dev" in result.stderr
    # The GitHub ref was created on the anchor repo at the anchor SHA.
    gh_calls = gh_log.read_text()
    assert "repos/OmniNode-ai/omnibase_infra/git/refs" in gh_calls
    assert anchor_sha in gh_calls
    # The stale trigger claim is gone: GITHUB_TOKEN refs fire nothing.
    assert "fires .github/workflows/release-train-lab.yml" not in result.stderr


@pytest.mark.unit
def test_branch_ref_resolves_per_clone_without_fallback(tmp_path: Path) -> None:
    """origin/dev still resolves in every clone independently -- no fallback
    NOTE, each repo tagged at its own origin/dev (pre-fix behavior kept)."""
    omni_home, dev_shas = _build_fixture(tmp_path)
    result, _, github_output = _run_script(
        tmp_path, omni_home, ["--lane", "stability", "--ref", "origin/dev", "--execute"]
    )
    assert result.returncode == 0, result.stderr
    assert "falling back" not in result.stderr
    tag = next(
        line.split("=", 1)[1]
        for line in github_output.read_text().splitlines()
        if line.startswith("tag=")
    )
    for repo in TAG_REPOS:
        assert (
            _git(["rev-parse", f"{tag}^{{commit}}"], cwd=omni_home / repo)
            == dev_shas[repo]
        )


@pytest.mark.unit
def test_unresolvable_ref_is_named_anchor_error(tmp_path: Path) -> None:
    """A ref that resolves nowhere must die with the NAMED anchor-resolution
    error (exit 1), never a bare git 'ambiguous argument' crash mid-loop."""
    omni_home, _ = _build_fixture(tmp_path)
    result, gh_log, _ = _run_script(
        tmp_path,
        omni_home,
        ["--lane", "stability", "--ref", "no-such-ref-anywhere", "--execute"],
    )
    assert result.returncode == 1, (
        f"expected exit 1, got {result.returncode}; stderr:\n{result.stderr}"
    )
    assert "does not resolve to a commit in the anchor clone" in result.stderr
    assert "ambiguous argument" not in result.stderr
    # Nothing was pushed to GitHub.
    assert "git/refs" not in gh_log.read_text()
