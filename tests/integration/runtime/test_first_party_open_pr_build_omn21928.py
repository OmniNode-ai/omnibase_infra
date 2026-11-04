# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A declared open pull request's branch audits through the real staging and install path.

The dev lane's workspace staging pins the siblings at their dev-merge tags and records
the infra clone, which the deploy agent has put at the pull request head, as a build
context with no tag. Real wheels are installed from those trees and the first-party
audit runs on them in each build context. Only disposable origins and build contexts
are used: no Docker daemon, broker, credentials, package index, or deployed lane.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_SHARED_PATH = Path(__file__).with_name("test_first_party_tag_sources_omn17427.py")
_spec = importlib.util.spec_from_file_location(
    "first_party_tag_sources_fixtures_omn21928", _SHARED_PATH
)
assert _spec and _spec.loader
shared = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = shared
_spec.loader.exec_module(shared)

CONTEXTS = {
    "dev": "dev-lane",
    "open_pr": "dev-lane-open-pr",
}


def _stage_at(home: Path, tmp_path: Path, deploy_ref: str) -> tuple[int, Path, str]:
    env = {
        **shared.scrub_git_location_env(),
        "OMNI_HOME": str(home),
        "DEPLOY_SOURCE_CLONE_ROOT": str(home),
        "DEPLOY_REF": deploy_ref,
        "DEPLOY_SIBLING_FALLBACK_REF": "origin/dev",
        "DEPLOY_HOTPATCH": "0",
        "ALLOW_UNPINNED_DEPLOY_SOURCE": "0",
        "ALLOW_SIBLING_PIN_DRIFT": "0",
        "CONSUMER_LOCK": str(home / "omnimarket/uv.lock"),
        "DEPLOY_SOURCE_REFS_OUT": str(tmp_path / "refs.json"),
        "DEPLOY_SOURCE_WORKTREE_ROOT": str(tmp_path / "trees"),
        "PATH": f"{Path(sys.executable).parent}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    done = shared._run(
        "bash",
        str(shared.BUILD / "stage_workspace.sh"),
        cwd=home / "omnibase_infra",
        env=env,
    )
    return done.returncode, home / "omnibase_infra/workspace", done.stdout + done.stderr


def _install(home: Path, workspace: Path, site: Path, tmp_path: Path) -> None:
    uv = shutil.which("uv")
    assert uv is not None
    site.mkdir()
    for repo in shared.INSTALLED:
        source = (
            home / repo
            if repo == "omnibase_infra"
            else workspace / "sibling-repos" / repo
        )
        installed = shared._run(
            uv,
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(site),
            "--no-deps",
            "--no-build-isolation",
            "--no-cache",
            "--no-sources",
            f"{shared.distribution(repo)} @ {source.as_uri()}",
            cwd=tmp_path,
        )
        assert installed.returncode == 0, installed.stdout + installed.stderr


def _audit(
    site: Path,
    staging: Path,
    manifest: Path,
    context: str,
    declared_pr: str,
) -> subprocess.CompletedProcess[str]:
    manifest.write_text("{}", encoding="utf-8")
    code = (
        "import runpy, sys; from pathlib import Path; "
        f"m = runpy.run_path({str(shared.CHECKER)!r}); "
        f"sys.exit(m['run_audit'](Path({str(staging)!r}), "
        f"record_into=Path({str(manifest)!r}), search_paths=[{str(site)!r}], "
        f"require={[shared.distribution(name) for name in shared.INSTALLED]!r}, "
        f"context={context!r}, declared_pr={declared_pr!r}))"
    )
    return shared._run(sys.executable, "-c", code, cwd=site)


def test_the_dev_lane_builds_the_declared_open_pr_head_and_records_it(
    tmp_path: Path,
) -> None:
    home, tags, commits = shared._sources(tmp_path)
    infra = home / "omnibase_infra"
    shared._git(infra, "config", "user.name", "test")
    shared._git(infra, "config", "user.email", "test@example.com")
    shared._git(infra, "checkout", "-q", "-b", "omn-18087-prepr-slot-audit")
    (infra / "src/omnibase_infra/__init__.py").write_text(
        'SOURCE = "open-pr"\n', encoding="utf-8"
    )
    shared._git(infra, "add", ".")
    shared._git(infra, "commit", "-qm", "open pull request head")
    head = shared._git(infra, "rev-parse", "HEAD")
    assert head != commits["omnibase_infra"]
    declared = f"omnibase_infra#5564@{head}"

    code, workspace, output = _stage_at(home, tmp_path, head)
    assert code == 0, output
    record = json.loads((workspace / "sibling-vcs-provenance.json").read_text())
    target = record["build_context"]["omnibase_infra"]
    assert target["vcs_ref"] == head
    assert target["dev_merge_tag"] == ""
    for repo in shared.SIBLINGS:
        assert record["siblings"][repo]["dev_merge_tag"] == tags[repo]

    site = tmp_path / "installed"
    _install(home, workspace, site, tmp_path)
    staging = workspace / "sibling-vcs-provenance.json"

    # The dev lane's default context refuses the branch head.
    refused = _audit(site, staging, tmp_path / "default.json", CONTEXTS["dev"], "")
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert "FAIL omnibase-infra: branch" in refused.stdout

    # The open-PR context refuses it too until a declaration names this head.
    undeclared = _audit(
        site, staging, tmp_path / "undeclared.json", CONTEXTS["open_pr"], ""
    )
    assert undeclared.returncode == 1, undeclared.stdout + undeclared.stderr
    assert "needs a declared open pull request" in undeclared.stdout
    other = _audit(
        site,
        staging,
        tmp_path / "other.json",
        CONTEXTS["open_pr"],
        f"omnibase_infra#5564@{commits['omnibase_infra']}",
    )
    assert other.returncode == 1, other.stdout + other.stderr

    manifest = tmp_path / "build-provenance.json"
    accepted = _audit(site, staging, manifest, CONTEXTS["open_pr"], declared)
    assert accepted.returncode == 0, accepted.stdout + accepted.stderr
    assert f"[declared open pull request {declared}]" in accepted.stdout
    written = json.loads(manifest.read_text())
    assert written["first_party_build_context"] == CONTEXTS["open_pr"]
    assert written["first_party_declared_pr"] == declared
    rows = {row["repo"]: row for row in written["first_party_sources"]}
    # The target is a branch build naming the head; the siblings stay dev-merge tags.
    assert rows["omnibase_infra"]["tag"] == ""
    assert rows["omnibase_infra"]["commit"] == head
    assert rows["omnibase_infra"]["kind"] in ("sha", "branch")
    assert rows["omnibase_infra"]["declared_pr"] == declared
    for repo in ("omnibase_core", "omnibase_spi"):
        assert rows[repo]["kind"] == "dev-merge-tag"
        assert rows[repo]["tag"] == tags[repo]
        assert rows[repo]["declared_pr"] == ""
