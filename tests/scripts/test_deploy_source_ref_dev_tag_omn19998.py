# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19998: a lab build stages omnimarket at its newest dev tag, not a release.

Operator RULING 2026-10-16T09:35:49Z: in the lab, omnimarket deploys from a tag
cut on each dev merge (``dev-<UTC YYYYMMDDHHMMSS>-<sha12>``), and that is part
of the process rather than something somebody remembers. RT-1's checkout
(``deploy_source_ref.py``) builds the newest such tag on origin in place of a
remote-tracking branch for every repository named with ``--follow-dev-tag``;
the deploy agent and the stability refresh name omnimarket.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "runtime_build" / "deploy_source_ref.py"

_spec = importlib.util.spec_from_file_location(
    "deploy_source_ref_dev_tag_under_test", MODULE_PATH
)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)

pytestmark = pytest.mark.unit


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**scrub_git_location_env(), "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def _tag_name(stamp: str, sha: str) -> str:
    return f"dev-{stamp}-{sha[:12]}"


def _origin_and_clone(tmp_path: Path, commits: int = 3) -> tuple[Path, list[str]]:
    """An origin with ``commits`` commits on ``dev`` and a clone of it."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "dev")
    _git(origin, "config", "user.email", "t@t.t")
    _git(origin, "config", "user.name", "t")
    shas = []
    for index in range(commits):
        (origin / "f.txt").write_text(f"{index}\n", encoding="utf-8")
        _git(origin, "add", "-A")
        _git(origin, "commit", "-q", "-m", f"c{index}")
        shas.append(_git(origin, "rev-parse", "HEAD"))
    clone = tmp_path / "omnimarket"
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(clone)],
        check=True,
        capture_output=True,
        env={**scrub_git_location_env(), "GIT_TERMINAL_PROMPT": "0"},
    )
    return origin, shas


class TestParseDevTags:
    def test_newest_last_and_only_well_formed(self) -> None:
        a, b = "a" * 40, "b" * 40
        text = "\n".join(
            [
                f"{b}\trefs/tags/{_tag_name('20261016093000', b)}",
                f"{a}\trefs/tags/{_tag_name('20261016080000', a)}",
                f"{a}\trefs/tags/dev-hand-made",
                # the name's sha12 is not the commit it points at
                f"{a}\trefs/tags/{_tag_name('20261016100000', b)}",
                f"{a}\trefs/tags/v0.4.314",
            ]
        )
        tags = mod.parse_dev_tags(text)
        assert [t.sha for t in tags] == [a, b]
        assert tags[-1].cut_at == datetime(2026, 10, 16, 9, 30, tzinfo=UTC)

    def test_an_annotated_tag_is_read_at_its_commit(self) -> None:
        commit, tag_object = "c" * 40, "d" * 40
        name = _tag_name("20261016093000", commit)
        text = f"{tag_object}\trefs/tags/{name}\n{commit}\trefs/tags/{name}^{{}}\n"
        assert [t.sha for t in mod.parse_dev_tags(text)] == [commit]

    def test_no_tags_is_empty(self) -> None:
        assert mod.parse_dev_tags("") == []


class TestCheckoutFollowsTheNewestDevTag:
    def test_the_branch_is_replaced_by_its_newest_tag(self, tmp_path: Path) -> None:
        origin, (first, second, head) = _origin_and_clone(tmp_path)
        _git(origin, "tag", _tag_name("20261016080000", first), first)
        _git(origin, "tag", _tag_name("20261016090000", second), second)

        result = mod.clean_checkout(
            tmp_path / "omnimarket",
            "origin/dev",
            follow_dev_tag=True,
            worktree_root=tmp_path / "trees",
        )

        # dev's head is one merge past the newest tag; the tag is what is built
        assert head != second
        assert result.expected_sha == second
        assert result.head_sha == second
        assert result.ref == _tag_name("20261016090000", second)
        assert result.dev_tag_from == "origin/dev"

    def test_the_fallback_branch_is_replaced_too(self, tmp_path: Path) -> None:
        origin, (first, _second, _head) = _origin_and_clone(tmp_path)
        _git(origin, "tag", _tag_name("20261016080000", first), first)

        result = mod.clean_checkout(
            tmp_path / "omnimarket",
            "f" * 40,  # an omnibase_infra commit, no ref here
            fallback_ref="origin/dev",
            follow_dev_tag=True,
            worktree_root=tmp_path / "trees",
        )

        assert result.expected_sha == first
        assert result.fallback_from == "f" * 40
        assert result.dev_tag_from == "origin/dev"

    def test_an_explicit_sha_is_never_replaced(self, tmp_path: Path) -> None:
        origin, (first, second, _head) = _origin_and_clone(tmp_path)
        _git(origin, "tag", _tag_name("20261016090000", second), second)

        result = mod.clean_checkout(
            tmp_path / "omnimarket",
            first,
            follow_dev_tag=True,
            worktree_root=tmp_path / "trees",
        )

        assert result.expected_sha == first
        assert result.dev_tag_from == ""

    def test_a_repository_that_does_not_follow_builds_its_branch(
        self, tmp_path: Path
    ) -> None:
        origin, (_first, second, head) = _origin_and_clone(tmp_path)
        _git(origin, "tag", _tag_name("20261016090000", second), second)

        result = mod.clean_checkout(
            tmp_path / "omnimarket", "origin/dev", worktree_root=tmp_path / "trees"
        )

        assert result.expected_sha == head
        assert result.dev_tag_from == ""

    def test_no_tag_on_origin_builds_the_branch_and_says_so(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _origin, (_first, _second, head) = _origin_and_clone(tmp_path)

        result = mod.clean_checkout(
            tmp_path / "omnimarket",
            "origin/dev",
            follow_dev_tag=True,
            worktree_root=tmp_path / "trees",
        )

        assert result.expected_sha == head
        assert result.dev_tag_from == ""
        assert "has no refs/tags/dev-* tag" in capsys.readouterr().err

    def test_a_newest_tag_off_the_branch_is_refused(self, tmp_path: Path) -> None:
        origin, (first, _second, _head) = _origin_and_clone(tmp_path)
        _git(origin, "checkout", "-q", "-b", "side", first)
        (origin / "side.txt").write_text("side\n", encoding="utf-8")
        _git(origin, "add", "-A")
        _git(origin, "commit", "-q", "-m", "side")
        side = _git(origin, "rev-parse", "HEAD")
        _git(origin, "tag", _tag_name("20261016090000", side), side)
        _git(origin, "checkout", "-q", "dev")

        with pytest.raises(mod.DeploySourceRefError, match="is not on 'origin/dev'"):
            mod.clean_checkout(
                tmp_path / "omnimarket",
                "origin/dev",
                follow_dev_tag=True,
                worktree_root=tmp_path / "trees",
            )

    def test_the_cli_flag_reaches_the_manifest(self, tmp_path: Path) -> None:
        origin, (_first, second, _head) = _origin_and_clone(tmp_path)
        _git(origin, "tag", _tag_name("20261016090000", second), second)
        manifest = tmp_path / "refs.json"

        rc = mod.main(
            [
                "checkout",
                "--repo",
                f"omnimarket={tmp_path / 'omnimarket'}",
                "--ref",
                "origin/dev",
                "--follow-dev-tag",
                "omnimarket",
                "--worktree-root",
                str(tmp_path / "trees"),
                "--output",
                str(manifest),
            ]
        )

        assert rc == 0
        import json

        row = json.loads(manifest.read_text(encoding="utf-8"))["repos"]["omnimarket"]
        assert row["expected_sha"] == second
        assert row["dev_tag_from"] == "origin/dev"


class TestTheLabCallersNameOmnimarket:
    """The lab paths set the follow list themselves; nobody has to remember it."""

    def test_stage_workspace_passes_the_follow_list(self) -> None:
        text = (REPO_ROOT / "scripts/runtime_build/stage_workspace.sh").read_text()
        assert "${DEPLOY_FOLLOW_DEV_TAG_REPOS:-}" in text
        assert "--follow-dev-tag" in text

    def test_the_stability_refresh_follows_omnimarket(self) -> None:
        text = (
            REPO_ROOT / "scripts/runtime_build/refresh_stability_lane.sh"
        ).read_text()
        assert 'export DEPLOY_FOLLOW_DEV_TAG_REPOS="omnimarket"' in text

    def test_the_deploy_agent_follows_omnimarket(self) -> None:
        tracking = (
            REPO_ROOT / "scripts/deploy-agent/deploy_agent/tracking_ref.py"
        ).read_text()
        executor = (
            REPO_ROOT / "scripts/deploy-agent/deploy_agent/executor.py"
        ).read_text()
        assert 'DEV_TAG_FOLLOWED_REPOSITORIES: Final = ("omnimarket",)' in tracking
        assert (
            "staging_env[ENV_FOLLOW_DEV_TAG_REPOS] = "
            '" ".join(DEV_TAG_FOLLOWED_REPOSITORIES)' in executor
        )
