# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Safety of the destructive phases of `scripts/git-maintenance.sh` (OMN-19396).

On 2026-09-24 a run of this script with `--execute --prune-worktrees` and no
WORKTREE_ROOT fell back to a derived default root that existed on the host, and
removed 119 ticket directories. Its worktree phase called a tree "clean" when
`git status --porcelain` printed nothing (a FAILING status printed nothing too,
and ignored files never counted), looked at no push state, pull request, ledger
CLAIM or stash, and finished every ticket with a recursive delete of the whole
ticket directory, non-git content included. Its branch phase deleted every
remote branch merged into origin/main, `dev` included when dev equals main.

Every test here drives the REAL script against a throwaway registry under
pytest's `tmp_path`. `Sandbox.run` asserts, before it starts the script, that
OMNI_HOME, WORKTREE_ROOT and ONEX_LEDGER_PATH all resolve inside `tmp_path`,
and it refuses `--execute` unless WORKTREE_ROOT is set there explicitly, so no
version of the script, old or new, can be pointed at a real directory by a
test in this file. `test_control_*` are the positive controls: without them a
script that removed nothing at all would pass every "is kept" assertion.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "git-maintenance.sh"

FAKE_GH = """#!/usr/bin/env bash
# Test double for `gh pr list --head <branch> --state open --json number --jq length`.
# Prints 1 when the branch is listed in $FAKE_GH_OPEN_PRS, else 0. Exits with
# $FAKE_GH_EXIT when that is set, to model gh being unable to answer.
if [ -n "${FAKE_GH_EXIT:-}" ]; then
  echo "fake gh: failing on purpose" >&2
  exit "$FAKE_GH_EXIT"
fi
branch=""
while [ $# -gt 0 ]; do
  case "$1" in --head) branch="$2"; shift ;; esac
  shift
done
if grep -qxF -- "$branch" "$FAKE_GH_OPEN_PRS"; then echo 1; else echo 0; fi
"""


FAKE_SNAPSHOT_HELPER = """import json, os, sys
wt = sys.argv[1]
if os.environ.get("FAKE_SNAPSHOT_FAIL"):
    print(json.dumps({"ok": False})); sys.exit(3)
d = os.path.join(os.environ["OMNI_HOME"], ".onex_state", "worktree-removal-snapshots",
                 os.path.basename(os.path.dirname(wt)))
os.makedirs(d)
print(json.dumps({"ok": True, "directory": d}))
"""


def _inside(path: str, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


class Sandbox:
    """A registry with one clone that has a bare `origin`, a worktrees root, a
    ledger and a fake `gh`, all under `tmp_path`."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.registry = tmp_path / "omni_home"
        self.worktrees = tmp_path / "omni_worktrees"
        self.origin = tmp_path / "origin.git"
        self.ledger = tmp_path / "ledger.md"
        self.open_prs = tmp_path / "open_prs.txt"
        bindir = tmp_path / "bin"
        for d in (self.registry, self.worktrees, bindir):
            d.mkdir()
        self.ledger.write_text("# ledger\n", encoding="utf-8")
        self.open_prs.write_text("", encoding="utf-8")
        gh = bindir / "gh"
        gh.write_text(FAKE_GH, encoding="utf-8")
        gh.chmod(0o755)

        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = os.devnull
        env["GIT_CONFIG_SYSTEM"] = os.devnull
        env["GIT_AUTHOR_NAME"] = "OMN-19396 Test"
        env["GIT_AUTHOR_EMAIL"] = "omn19396@example.invalid"
        env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
        env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
        env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
        env["FAKE_GH_OPEN_PRS"] = str(self.open_prs)
        env["OMNI_HOME"] = str(self.registry)
        env["WORKTREE_ROOT"] = str(self.worktrees)
        env["ONEX_LEDGER_PATH"] = str(self.ledger)
        for key in ("GIT_PREFIX", "FAKE_GH_EXIT"):
            env.pop(key, None)
        # git exports repo-scoping variables into hook processes and they
        # override both cwd= and git -C (OMN-14891 / OMN-18434).
        self.env = scrub_git_location_env(env)

        # The shared pre-removal save (omniclaude worktree_removal_snapshot.py,
        # OMN-19539), reduced to its contract: exit 0 and a JSON line naming the
        # saved directory, or exit 3 when FAKE_SNAPSHOT_FAIL is set. Its own
        # behaviour is tested where it lives.
        helper = (
            self.registry / "omniclaude" / "scripts" / "worktree_removal_snapshot.py"
        )
        helper.parent.mkdir(parents=True)
        helper.write_text(FAKE_SNAPSHOT_HELPER, encoding="utf-8")
        self.snapshots = self.registry / ".onex_state" / "worktree-removal-snapshots"
        self.env.pop("FAKE_SNAPSHOT_FAIL", None)

        self.git("init", "-q", "--bare", "-b", "main", str(self.origin), cwd=tmp_path)
        self.clone = self.registry / "fixture_repo"
        self.clone.mkdir()
        self.git("init", "-q", "-b", "main", cwd=self.clone)
        (self.clone / ".gitignore").write_text(".env\n.venv/\n", encoding="utf-8")
        (self.clone / "root.txt").write_text("root\n", encoding="utf-8")
        self.git("add", "-A", cwd=self.clone)
        self.git("commit", "-q", "-m", "seed", cwd=self.clone)
        self.git("remote", "add", "origin", str(self.origin), cwd=self.clone)
        self.git("push", "-q", "-u", "origin", "main", cwd=self.clone)
        self.git("remote", "set-head", "origin", "main", cwd=self.clone)

    def git(self, *args: str, cwd: Path) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=scrub_git_location_env(self.env),
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"git {args}: {result.stderr}"
        return result.stdout

    def add_worktree(self, ticket: str, *, push: bool = True) -> Path:
        path = self.worktrees / ticket / "fixture_repo"
        branch = f"jonah/{ticket.lower()}-fixture"
        self.git(
            "worktree", "add", "-q", str(path), "-b", branch, "HEAD", cwd=self.clone
        )
        if push:
            self.git("push", "-q", "-u", "origin", branch, cwd=path)
        return path

    def branch_of(self, worktree: Path) -> str:
        return self.git("symbolic-ref", "--short", "HEAD", cwd=worktree).strip()

    def run(
        self, *args: str, **env_overrides: str | None
    ) -> subprocess.CompletedProcess[str]:
        env = dict(self.env)
        for key, value in env_overrides.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        # The sandbox assertion, made BEFORE the script starts (OMN-19396).
        assert _inside(env["OMNI_HOME"], self.tmp), env["OMNI_HOME"]
        if "WORKTREE_ROOT" in env:
            assert _inside(env["WORKTREE_ROOT"], self.tmp), env["WORKTREE_ROOT"]
        elif "--execute" in args:
            pytest.fail(
                "--execute without an explicit sandbox WORKTREE_ROOT is never run"
            )
        if "ONEX_LEDGER_PATH" in env:
            assert _inside(env["ONEX_LEDGER_PATH"], self.tmp), env["ONEX_LEDGER_PATH"]
        return subprocess.run(
            ["bash", str(SCRIPT), *args],
            env=scrub_git_location_env(env),
            capture_output=True,
            text=True,
            check=False,
        )

    def remote_branches(self) -> set[str]:
        out = self.git(
            "for-each-ref", "--format=%(refname:short)", "refs/heads", cwd=self.origin
        )
        return set(out.split())


@pytest.fixture
def box(tmp_path: Path) -> Sandbox:
    return Sandbox(tmp_path)


# --- (1) an explicit root, every time ---------------------------------------


def test_prune_refuses_without_explicit_worktree_root(box: Sandbox) -> None:
    """No derived default for a destructive step. DRY-RUN ONLY on purpose: a
    RED run of this test against any older script can delete nothing."""
    derived = box.registry / "omni_worktrees" / "OMN-DERIVED" / "fixture_repo"
    box.git(
        "worktree", "add", "-q", str(derived), "-b", "wt/derived", "HEAD", cwd=box.clone
    )

    result = box.run("--dry-run", "--prune-worktrees", WORKTREE_ROOT=None)

    assert result.returncode != 0, result.stdout
    assert "WORKTREE_ROOT" in result.stderr, result.stderr
    assert "=== Worktree Cleanup ===" not in result.stdout, result.stdout
    assert "OMN-DERIVED" not in result.stdout, result.stdout
    assert derived.exists()


def test_prune_refuses_without_a_ledger(box: Sandbox) -> None:
    """Open CLAIMs cannot be honoured without the ledger, so the phase refuses."""
    worktree = box.add_worktree("OMN-NOLEDGER")
    result = box.run("--execute", "--prune-worktrees", ONEX_LEDGER_PATH=None)
    assert result.returncode != 0, result.stdout
    assert "ONEX_LEDGER_PATH" in result.stderr, result.stderr
    assert worktree.exists()


def test_control_explicit_worktree_root_is_honoured(box: Sandbox) -> None:
    worktree = box.add_worktree("OMN-EXPLICIT")
    result = box.run("--dry-run", "--prune-worktrees")
    assert result.returncode == 0, result.stderr
    assert "[would remove] OMN-EXPLICIT" in result.stdout, result.stdout
    assert worktree.exists(), "a dry run must not remove anything"


# --- (2) a worktree is removed only when every check passes -----------------


def test_unpushed_commits_are_kept(box: Sandbox) -> None:
    worktree = box.add_worktree("OMN-UNPUSHED")
    (worktree / "work.txt").write_text("work\n", encoding="utf-8")
    box.git("add", "work.txt", cwd=worktree)
    box.git("commit", "-q", "-m", "unpushed work", cwd=worktree)

    result = box.run("--execute", "--prune-worktrees")

    assert result.returncode == 0, result.stderr
    assert worktree.exists(), result.stdout
    assert "unpushed" in result.stdout, result.stdout
    assert "[skip] OMN-UNPUSHED" in result.stdout, result.stdout


def test_failing_git_status_is_kept_and_never_read_as_clean(box: Sandbox) -> None:
    """A corrupt index makes `git status` fail. That must keep the tree, with
    git's own error shown, and no removal may even be attempted."""
    worktree = box.add_worktree("OMN-STATUSFAIL")
    admin = Path(box.git("rev-parse", "--absolute-git-dir", cwd=worktree).strip())
    (admin / "index").write_bytes(b"not an index")
    (worktree / "notes.txt").write_text("keep me\n", encoding="utf-8")

    result = box.run("--execute", "--prune-worktrees")
    out = result.stdout

    assert worktree.exists(), out
    assert (worktree / "notes.txt").exists(), out
    assert "git status failed" in out, out
    assert "[remove-failed]" not in out, out
    assert "[removed]" not in out, out


def test_non_git_content_is_kept(box: Sandbox) -> None:
    """Nothing is deleted recursively: files, hidden dirs and non-git dirs in a
    ticket dir survive, and a ticket dir holding no worktree is untouched."""
    worktree = box.add_worktree("OMN-MIXED")
    mixed = worktree.parent
    (mixed / "notes.md").write_text("notes\n", encoding="utf-8")
    (mixed / ".scratch").mkdir()
    (mixed / ".scratch" / "draft.txt").write_text("draft\n", encoding="utf-8")
    (mixed / "plain_dir").mkdir()
    (mixed / "plain_dir" / "data.json").write_text("{}\n", encoding="utf-8")
    files_only = box.worktrees / "OMN-FILES"
    files_only.mkdir()
    (files_only / "handoff.md").write_text("handoff\n", encoding="utf-8")

    result = box.run("--execute", "--prune-worktrees")
    out = result.stdout

    assert result.returncode == 0, result.stderr
    assert (mixed / "notes.md").exists(), out
    assert (mixed / ".scratch" / "draft.txt").exists(), out
    assert (mixed / "plain_dir" / "data.json").exists(), out
    assert (files_only / "handoff.md").exists(), out
    assert "[removed] OMN-FILES" not in out, out
    # Positive control inside the same run: the clean, pushed worktree itself
    # WAS removed, through git, and the leftovers were reported.
    assert not worktree.exists(), out
    assert f"[removed] {worktree}" in out, out
    assert "[worktrees-removed] OMN-MIXED" in out, out


def test_open_pull_request_is_kept(box: Sandbox) -> None:
    worktree = box.add_worktree("OMN-OPENPR")
    box.open_prs.write_text(box.branch_of(worktree) + "\n", encoding="utf-8")
    result = box.run("--execute", "--prune-worktrees")
    assert worktree.exists(), result.stdout
    assert "has an open pull request" in result.stdout, result.stdout


def test_gh_that_cannot_answer_keeps_the_worktree(box: Sandbox) -> None:
    worktree = box.add_worktree("OMN-GHDOWN")
    result = box.run("--execute", "--prune-worktrees", FAKE_GH_EXIT="1")
    assert worktree.exists(), result.stdout
    assert "cannot check for an open pull request" in result.stdout, result.stdout


def test_open_ledger_claim_is_kept_and_a_closed_one_is_not(box: Sandbox) -> None:
    held = box.add_worktree("OMN-90001")
    released = box.add_worktree("OMN-90002")
    box.ledger.write_text(
        "2026-09-24T10:00:00Z | CLAIM | lane=held-lane | ticket=OMN-90001 | scope=x\n"
        "2026-09-24T10:00:01Z | CLAIM | lane=done-lane | ticket=OMN-90002 | scope=y\n"
        "2026-09-24T11:00:00Z | TERMINAL | lane=done-lane | ticket=OMN-90002 | "
        "closes-CLAIM=LCT1-1-2-abc-2026-09-24T10:00:01Z | friction=none | done\n",
        encoding="utf-8",
    )
    result = box.run("--execute", "--prune-worktrees")
    out = result.stdout
    assert held.exists(), out
    assert "open ledger CLAIM" in out, out
    assert not released.exists(), out
    assert "[removed] OMN-90002" in out, out


def test_lane_terminal_without_claim_id_closes_all_earlier_claims(
    box: Sandbox,
) -> None:
    """Canonical ledger semantics close every earlier CLAIM for the lane."""
    first = box.add_worktree("OMN-90003")
    second = box.add_worktree("OMN-90004")
    box.ledger.write_text(
        "2026-09-24T10:00:00Z | CLAIM | lane=done-lane | ticket=OMN-90003 | "
        "scope=first\n"
        "2026-09-24T10:00:01Z | CLAIM | lane=done-lane | ticket=OMN-90004 | "
        "scope=second\n"
        "2026-09-24T11:00:00Z | TERMINAL | lane=done-lane | friction=none | "
        "done\n",
        encoding="utf-8",
    )

    result = box.run("--execute", "--prune-worktrees")

    assert result.returncode == 0, result.stderr
    assert not first.exists(), result.stdout
    assert not second.exists(), result.stdout
    assert "[removed] OMN-90003" in result.stdout, result.stdout
    assert "[removed] OMN-90004" in result.stdout, result.stdout


def test_stash_for_the_branch_is_kept(box: Sandbox) -> None:
    worktree = box.add_worktree("OMN-STASHED")
    (worktree / "root.txt").write_text("edited\n", encoding="utf-8")
    box.git("stash", "push", "-q", "-m", "parked", cwd=worktree)
    result = box.run("--execute", "--prune-worktrees")
    assert worktree.exists(), result.stdout
    assert "holds a stash" in result.stdout, result.stdout


def test_ignored_file_that_git_cannot_restore_is_kept(box: Sandbox) -> None:
    worktree = box.add_worktree("OMN-DOTENV")
    (worktree / ".env").write_text("LOCAL=1\n", encoding="utf-8")
    result = box.run("--execute", "--prune-worktrees")
    assert (worktree / ".env").exists(), result.stdout
    assert "holds ignored file" in result.stdout, result.stdout


@pytest.mark.parametrize(
    "secret_name",
    [
        ".env",
        "service-account-prod.json",
        "terraform.tfstate",
        ".infisical-admin-token",
        ".monitor-env",
    ],
)
def test_secret_bearing_file_inside_ignored_dist_is_kept(
    box: Sandbox, secret_name: str
) -> None:
    """Ignored build output must not hide credential-bearing local files."""
    (box.clone / ".gitignore").write_text(".env\n.venv/\ndist/\n", encoding="utf-8")
    box.git("commit", "-q", "-am", "ignore dist", cwd=box.clone)
    box.git("push", "-q", "origin", "main", cwd=box.clone)
    worktree = box.add_worktree("OMN-DISTSECRET")
    dist = worktree / "dist"
    dist.mkdir()
    (dist / "bundle.js").write_text("built\n", encoding="utf-8")
    secret = dist / secret_name
    secret.write_text("credential-shaped local data\n", encoding="utf-8")

    result = box.run("--execute", "--prune-worktrees")

    assert result.returncode == 0, result.stderr
    assert secret.exists(), result.stdout
    assert "holds ignored file not recoverable from git" in result.stdout, result.stdout


def test_control_ignored_dist_without_secrets_does_not_block_removal(
    box: Sandbox,
) -> None:
    (box.clone / ".gitignore").write_text(".env\n.venv/\ndist/\n", encoding="utf-8")
    box.git("commit", "-q", "-am", "ignore dist", cwd=box.clone)
    box.git("push", "-q", "origin", "main", cwd=box.clone)
    worktree = box.add_worktree("OMN-DISTONLY")
    dist = worktree / "dist"
    dist.mkdir()
    (dist / "bundle.js").write_text("built\n", encoding="utf-8")

    result = box.run("--execute", "--prune-worktrees")

    assert result.returncode == 0, result.stderr
    assert not worktree.exists(), result.stdout
    assert "[removed] OMN-DISTONLY" in result.stdout, result.stdout


def test_control_regenerable_cache_does_not_block_removal(box: Sandbox) -> None:
    worktree = box.add_worktree("OMN-CACHE")
    (worktree / ".venv").mkdir()
    (worktree / ".venv" / "pyvenv.cfg").write_text("x\n", encoding="utf-8")
    result = box.run("--execute", "--prune-worktrees")
    assert result.returncode == 0, result.stderr
    assert not worktree.exists(), result.stdout
    assert "[removed] OMN-CACHE" in result.stdout, result.stdout
    assert not (box.worktrees / "OMN-CACHE").exists()


def test_an_env_inside_an_ignored_regenerable_dir_is_kept(box: Sandbox) -> None:
    """OMN-19539: `git status --ignored` collapses an ignored `dist/` to one
    line, so a name-only check read the `.env` inside it as build output."""
    (box.clone / ".gitignore").write_text(".env\n.venv/\ndist/\n", encoding="utf-8")
    box.git("commit", "-q", "-am", "ignore dist", cwd=box.clone)
    box.git("push", "-q", "origin", "main", cwd=box.clone)
    worktree = box.add_worktree("OMN-DISTENV")
    (worktree / "dist").mkdir()
    (worktree / "dist" / "bundle.js").write_text("built\n", encoding="utf-8")
    (worktree / "dist" / ".env").write_text("TOKEN=x\n", encoding="utf-8")

    result = box.run("--execute", "--prune-worktrees")

    assert (worktree / "dist" / ".env").exists(), result.stdout
    assert "secrets-shaped file inside dist/" in result.stdout, result.stdout


def test_control_an_ignored_dist_without_secrets_is_regenerable(box: Sandbox) -> None:
    (box.clone / ".gitignore").write_text(".env\n.venv/\ndist/\n", encoding="utf-8")
    box.git("commit", "-q", "-am", "ignore dist", cwd=box.clone)
    box.git("push", "-q", "origin", "main", cwd=box.clone)
    worktree = box.add_worktree("OMN-DISTONLY")
    (worktree / "dist").mkdir()
    (worktree / "dist" / "bundle.js").write_text("built\n", encoding="utf-8")

    result = box.run("--execute", "--prune-worktrees")

    assert not worktree.exists(), result.stdout
    assert "[removed] OMN-DISTONLY" in result.stdout, result.stdout


def test_every_removal_is_saved_first(box: Sandbox) -> None:
    worktree = box.add_worktree("OMN-SAVED")

    result = box.run("--execute", "--prune-worktrees")

    assert not worktree.exists(), result.stdout
    assert (box.snapshots / "OMN-SAVED").is_dir(), result.stdout
    assert result.stdout.index("[saved]") < result.stdout.index("[removed] OMN-SAVED")


def test_a_failed_save_keeps_the_worktree(box: Sandbox) -> None:
    worktree = box.add_worktree("OMN-UNSAVED")

    result = box.run("--execute", "--prune-worktrees", FAKE_SNAPSHOT_FAIL="1")

    assert worktree.exists(), result.stdout
    assert "pre-removal snapshot failed" in result.stdout, result.stdout
    assert "[incomplete] OMN-UNSAVED" in result.stdout, result.stdout


def test_a_missing_helper_keeps_the_worktree(box: Sandbox) -> None:
    worktree = box.add_worktree("OMN-NOHELPER")
    (box.registry / "omniclaude" / "scripts" / "worktree_removal_snapshot.py").unlink()

    result = box.run("--execute", "--prune-worktrees")

    assert worktree.exists(), result.stdout
    assert "snapshot helper missing" in result.stdout, result.stdout


def test_script_never_deletes_recursively() -> None:
    """The falsifier for the recursive delete, pinned in source."""
    code = [
        line
        for line in SCRIPT.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    assert not [line for line in code if "rm -rf" in line or "rm -fr" in line]


# --- (3) the remote-branch phase is gated the same way ----------------------


def _branch_world(box: Sandbox) -> Path:
    """origin holds main, dev at main's commit, a merged branch no one uses,
    and a merged branch checked out in a live worktree."""
    box.git("push", "-q", "origin", "main:dev", cwd=box.clone)
    box.git("push", "-q", "origin", "main:jonah/omn-1-merged", cwd=box.clone)
    return box.add_worktree("OMN-LIVE")


def test_execute_without_the_branch_flag_deletes_no_remote_branch(box: Sandbox) -> None:
    _branch_world(box)
    before = box.remote_branches()
    result = box.run("--execute")
    assert result.returncode == 0, result.stderr
    assert box.remote_branches() == before, result.stdout


def test_branch_phase_keeps_dev_and_checked_out_branches(box: Sandbox) -> None:
    live = _branch_world(box)
    live_branch = box.branch_of(live)
    result = box.run("--execute", "--delete-remote-branches")
    after = box.remote_branches()
    assert result.returncode == 0, result.stderr
    assert "dev" in after, result.stdout
    assert "main" in after, result.stdout
    assert live_branch in after, result.stdout
    # Positive control: the merged branch nobody holds is deleted.
    assert "jonah/omn-1-merged" not in after, result.stdout
    assert "[deleted] origin/jonah/omn-1-merged" in result.stdout, result.stdout


def test_branch_phase_keeps_a_branch_with_an_open_claim(box: Sandbox) -> None:
    _branch_world(box)
    box.ledger.write_text(
        "2026-09-24T10:00:00Z | CLAIM | lane=l | ticket=OMN-1 | scope=x\n",
        encoding="utf-8",
    )
    result = box.run("--execute", "--delete-remote-branches")
    assert "jonah/omn-1-merged" in box.remote_branches(), result.stdout
    assert "open ledger CLAIM" in result.stdout, result.stdout
