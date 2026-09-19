# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Real-trigger behaviour of the sanctioned orphan-branch deletion tool
(OMN-18370, `scripts/delete_orphan_branch.py`).

Every test drives the REAL tool against a throwaway registry whose clone has the
canonical-clone guard installed exactly as a host installs it. None of them
stubs the guard out: a tool that passes with the guard absent is precisely the
tool this ticket exists to stop someone writing, because the guard is the thing
it has to satisfy.

`gh` is shimmed onto PATH rather than monkeypatched, so the open-pull-request
clause is exercised through the same subprocess boundary the real run uses, and
a shim that FAILS is a distinguishable case from one that returns no rows.

`test_control_*` are the positive controls: without them a tool that refused
everything unconditionally would satisfy every refusal assertion here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOL = REPO_ROOT / "scripts" / "delete_orphan_branch.py"
HOOKS_DIR = REPO_ROOT / "scripts" / "git-hooks" / "canonical-clone"

CONSENT_ROW = (
    "2026-09-16T14:34:06Z | OPERATOR-CONSENT | lane=fixture | approved_by=operator "
    '| "Seven days is fine" | APPROVED SCOPE: removal of rescue-only worktrees and '
    "deletion of their local branch | OUT OF SCOPE: anything with an open pull "
    "request | This row is the durable authorization evidence"
)


def _git(
    *args: str, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    # Scrubbed at the CALL (OMN-14891 / OMN-18434): git exports the repo-scoping
    # variables into every hook process and they OVERRIDE both `cwd=` and
    # `git -C`, so a fixture that misses one rewrites the real worktree.
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=scrub_git_location_env(env),
        capture_output=True,
        text=True,
        check=False,
    )


def _base_env(registry: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_AUTHOR_NAME"] = "OMN-18370 Test"
    env["GIT_AUTHOR_EMAIL"] = "omn18370@example.invalid"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    env["OMNI_HOME"] = str(registry)
    for leaked in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
        "ALLOW_CANONICAL_CLONE_COMMIT",
        "ONEX_CANONICAL_CONVERGE",
        "ONEX_WORKTREES_ROOT",
        "ONEX_BRANCH_DELETE_CONSENT",
        "ONEX_BRANCH_DELETE_REFS",
    ):
        env.pop(leaked, None)
    return env


def _commit(repo: Path, env: dict[str, str], name: str) -> str:
    (repo / name).write_text(name, encoding="utf-8")
    assert _git("add", name, cwd=repo, env=env).returncode == 0
    seed = dict(env)
    seed["ALLOW_CANONICAL_CLONE_COMMIT"] = "1"
    assert _git("commit", "-q", "-m", name, cwd=repo, env=seed).returncode == 0
    return _git("rev-parse", "HEAD", cwd=repo, env=env).stdout.strip()


def _gh_shim(
    bin_dir: Path,
    *,
    prs: str = "[]",
    exit_code: int = 0,
    merged: str = "[]",
    merged_exit_code: int = 0,
) -> None:
    """A `gh` on PATH that answers `pr list` and nothing else.

    It reads `--state` out of its own argv and answers the OPEN and the MERGED
    query differently, because the tool asks two distinct questions and a shim
    that conflated them could not tell a merged pull request from an open one.
    Each state carries its own exit code so a failure on one lookup is
    distinguishable from a failure on the other.
    """
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "gh"
    shim.write_text(
        "#!/usr/bin/env bash\n"
        "state=open\n"
        "prev=\n"
        'for arg in "$@"; do\n'
        '  if [[ "$prev" == "--state" ]]; then state="$arg"; fi\n'
        '  prev="$arg"\n'
        "done\n"
        'if [[ "$state" == "merged" ]]; then\n'
        f"  if [[ {merged_exit_code} -ne 0 ]]; then\n"
        '    echo "gh: simulated merged-lookup failure" >&2\n'
        f"    exit {merged_exit_code}\n"
        "  fi\n"
        f"  printf '%s' '{merged}'\n"
        "  exit 0\n"
        "fi\n"
        f"if [[ {exit_code} -ne 0 ]]; then\n"
        '  echo "gh: simulated failure" >&2\n'
        f"  exit {exit_code}\n"
        "fi\n"
        f"printf '%s' '{prs}'\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)


@pytest.fixture
def world(tmp_path: Path) -> dict[str, object]:
    """A registry holding an upstream, a guarded canonical clone with `dev` and
    two branches -- `merged` (an ancestor of the upstream default) and
    `orphan` (divergent, unmerged) -- plus a ledger whose line 2 is a consent
    row and whose line 1 deliberately is not."""
    registry = tmp_path / "omni_home"
    (registry / "omni_worktrees").mkdir(parents=True)
    env = _base_env(registry)

    upstream = tmp_path / "upstream"
    upstream.mkdir()
    assert _git("init", "-q", "-b", "dev", cwd=upstream, env=env).returncode == 0
    _commit(upstream, env, "one.txt")
    assert _git("checkout", "-q", "-b", "orphan", cwd=upstream, env=env).returncode == 0
    orphan_tip = _commit(upstream, env, "orphan.txt")
    assert _git("checkout", "-q", "dev", cwd=upstream, env=env).returncode == 0
    _commit(upstream, env, "two.txt")

    clone = registry / "some_repo"
    assert (
        _git("clone", "-q", str(upstream), str(clone), cwd=registry, env=env).returncode
        == 0
    )
    assert (
        _git("config", "core.hooksPath", str(HOOKS_DIR), cwd=clone, env=env).returncode
        == 0
    )
    # `merged` sits one commit behind the upstream default, so it is a genuine
    # ancestor; `orphan` is on the divergent lineage and is not.
    assert _git("branch", "merged", "HEAD~1", cwd=clone, env=env).returncode == 0
    assert _git("branch", "orphan", "origin/orphan", cwd=clone, env=env).returncode == 0

    ledger = registry / "docs" / "tracking" / "LEDGER.md"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "2026-09-16T14:00:00Z | NOTE | lane=fixture | not a consent row\n"
        + CONSENT_ROW
        + "\n",
        encoding="utf-8",
    )

    bin_dir = tmp_path / "bin"
    _gh_shim(bin_dir)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"

    dev_tip = _git("rev-parse", "origin/dev", cwd=clone, env=env).stdout.strip()

    return {
        "registry": registry,
        "clone": clone,
        "env": env,
        "bin": bin_dir,
        "ledger": ledger,
        "orphan_tip": orphan_tip,
        "dev_tip": dev_tip,
    }


def _descendant_commit(world: dict[str, object], parent: str) -> str:
    """A commit object in the CLONE that has `parent` as its parent.

    Built with plumbing so the working tree is never touched: it stands in for
    the extra commits a pull request head can carry beyond what a local branch
    holds, and it has to exist in the clone's object store for the reachability
    check to be answerable at all.
    """
    tree = _git(
        "rev-parse",
        f"{parent}^{{tree}}",
        cwd=world["clone"],  # type: ignore[arg-type]
        env=world["env"],  # type: ignore[arg-type]
    ).stdout.strip()
    result = _git(
        "commit-tree",
        tree,
        "-p",
        parent,
        "-m",
        "pushed after the local tip",
        cwd=world["clone"],  # type: ignore[arg-type]
        env=world["env"],  # type: ignore[arg-type]
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _merged_row(branch: str, oid: str, *, number: int = 4242) -> str:
    """One row shaped like `gh pr list --state merged --json ...` returns."""
    return json.dumps(
        [
            {
                "number": number,
                "headRefName": branch,
                "headRefOid": oid,
                "mergedAt": "2026-09-18T09:14:07Z",
            }
        ]
    )


def _run_tool(world: dict[str, object], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo",
            str(world["clone"]),
            "--consent",
            "docs/tracking/LEDGER.md:2",
            "--json",
            *args,
        ],
        env=scrub_git_location_env(world["env"]),  # type: ignore[arg-type]
        capture_output=True,
        text=True,
        check=False,
    )


def _report(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _branch_exists(world: dict[str, object], name: str) -> bool:
    return (
        _git(
            "rev-parse",
            "--verify",
            "--quiet",
            f"refs/heads/{name}",
            cwd=world["clone"],  # type: ignore[arg-type]
            env=world["env"],  # type: ignore[arg-type]
        ).returncode
        == 0
    )


# ---------------------------------------------------------------------------
# Positive controls
# ---------------------------------------------------------------------------


def test_control_a_bare_delete_is_still_refused_by_the_guard(
    world: dict[str, object],
) -> None:
    """The guard's blanket refusal is unchanged. Every permitted deletion below
    is the declared door opening, not the refusal having quietly died."""
    result = _git(
        "branch",
        "-D",
        "merged",
        cwd=world["clone"],  # type: ignore[arg-type]
        env=world["env"],  # type: ignore[arg-type]
    )
    assert result.returncode == 128, result.stderr
    assert "refused deleting a branch" in result.stderr
    assert _branch_exists(world, "merged")


def test_control_a_merged_branch_is_deleted_through_the_tool(
    world: dict[str, object],
) -> None:
    """The control for every refusal: the tool CAN delete, so a refusal
    elsewhere is a decision rather than a tool that never works."""
    report = _report(_run_tool(world, "--branch", "merged", "--execute"))

    assert report["deleted"] == 1
    row = report["branches"][0]  # type: ignore[index]
    assert row["eligible"] is True
    assert row["deleted"] is True
    assert row["reason"] == "merged_into_upstream_default"
    assert not _branch_exists(world, "merged")


# ---------------------------------------------------------------------------
# The eligibility bar
# ---------------------------------------------------------------------------


def test_dry_is_the_default_and_deletes_nothing(world: dict[str, object]) -> None:
    report = _report(_run_tool(world, "--branch", "merged"))

    assert report["executed"] is False
    assert report["eligible"] == 1
    assert report["deleted"] == 0
    assert _branch_exists(world, "merged")


def test_a_branch_with_an_open_pull_request_is_refused(
    world: dict[str, object],
) -> None:
    """An open pull request means the branch is live work, whatever its age."""
    _gh_shim(world["bin"], prs='[{"number":123}]')  # type: ignore[arg-type]

    report = _report(_run_tool(world, "--branch", "merged", "--execute"))

    assert report["deleted"] == 0
    assert report["branches"][0]["reason"] == "open_pull_request"  # type: ignore[index]
    assert _branch_exists(world, "merged")


def test_a_gh_failure_refuses_rather_than_reading_as_no_pull_request(
    world: dict[str, object],
) -> None:
    """An unanswerable question is not a pass. A `gh` that errors and an empty
    result set are indistinguishable to a tool that ignores the exit status --
    that is the shape of CLAUDE.md rule 16's false zero."""
    _gh_shim(world["bin"], exit_code=1)  # type: ignore[arg-type]

    report = _report(_run_tool(world, "--branch", "merged", "--execute"))

    assert report["deleted"] == 0
    assert "refused" in report["branches"][0]["reason"]  # type: ignore[index]
    assert _branch_exists(world, "merged")


def test_an_unmerged_branch_whose_tip_is_not_recorded_is_refused(
    world: dict[str, object],
) -> None:
    """The rescue-only class is unmerged by construction. Deleting it with no
    record anywhere of what it pointed at is the objection the guard's blanket
    refusal was built on, and it still holds here."""
    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 0
    assert (
        report["branches"][0]["reason"] == "unmerged_and_tip_not_recorded"  # type: ignore[index]
    )
    assert _branch_exists(world, "orphan")


def test_an_unmerged_branch_whose_tip_is_recorded_in_the_ledger_is_deleted(
    world: dict[str, object],
) -> None:
    """Recording the tip in the cited ledger is what makes an unmerged deletion
    survivable: the pointer goes, the oid stays resolvable in the same artifact
    that carries the authorisation."""
    ledger: Path = world["ledger"]  # type: ignore[assignment]
    ledger.write_text(
        ledger.read_text(encoding="utf-8")
        + f"2026-09-16T16:43:26Z | TERMINAL | lane=fixture | tip {world['orphan_tip']}\n",
        encoding="utf-8",
    )

    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 1
    assert report["branches"][0]["reason"] == "tip_recorded_in_ledger"  # type: ignore[index]
    assert not _branch_exists(world, "orphan")


def test_a_branch_a_live_worktree_still_holds_is_refused(
    world: dict[str, object], tmp_path: Path
) -> None:
    """A branch a worktree still holds is not an orphan. This clause is what
    keeps the tool from racing a prune that has not finished."""
    linked = world["registry"] / "omni_worktrees" / "OMN-1" / "some_repo"  # type: ignore[operator]
    assert (
        _git(
            "worktree",
            "add",
            str(linked),
            "merged",
            cwd=world["clone"],  # type: ignore[arg-type]
            env=world["env"],  # type: ignore[arg-type]
        ).returncode
        == 0
    )

    report = _report(_run_tool(world, "--branch", "merged", "--execute"))

    assert report["deleted"] == 0
    assert (
        report["branches"][0]["reason"] == "worktree_still_holds_branch"  # type: ignore[index]
    )
    assert _branch_exists(world, "merged")


# ---------------------------------------------------------------------------
# The merged pull request path (OMN-18825)
# ---------------------------------------------------------------------------
#
# This org squash-merges. A squash merge writes a NEW commit onto the base and
# leaves the branch tip on a lineage the base never absorbs, so `merged into the
# upstream default` is false for almost every branch whose work has in fact
# landed -- measured 2026-09-19, 274 of 295 confirmed-merged branches were
# refused by the two original clauses. The merged pull request is the evidence
# that actually exists for those.
#
# The reachability half is not decoration. A name-only merged-PR match deleted
# five unmerged branches once already (OMN-16564): the branch NAME says the work
# landed, and says nothing at all about commits that were never pushed. So the
# bar is the pull request's own head oid CONTAINING the local tip.


def test_a_merged_pull_request_whose_head_is_the_local_tip_admits_the_branch(
    world: dict[str, object],
) -> None:
    """The ordinary squash-merge case: the branch was pushed, the pull request
    merged, and the tip is exactly what the pull request carried."""
    _gh_shim(
        world["bin"],  # type: ignore[arg-type]
        merged=_merged_row("orphan", world["orphan_tip"]),  # type: ignore[arg-type]
    )

    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 1
    row = report["branches"][0]  # type: ignore[index]
    assert row["eligible"] is True
    assert row["reason"] == "merged_pull_request"
    assert not _branch_exists(world, "orphan")


def test_a_merged_pull_request_head_ahead_of_the_local_tip_admits_the_branch(
    world: dict[str, object],
) -> None:
    """The local tip need not EQUAL the pull request head, only be contained by
    it: a branch left behind the ref that was actually merged has lost nothing
    by being deleted."""
    ahead = _descendant_commit(world, world["orphan_tip"])  # type: ignore[arg-type]
    _gh_shim(world["bin"], merged=_merged_row("orphan", ahead))  # type: ignore[arg-type]

    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 1
    assert report["branches"][0]["reason"] == "merged_pull_request"  # type: ignore[index]
    assert not _branch_exists(world, "orphan")


def test_a_merged_pull_request_not_containing_the_local_tip_refuses(
    world: dict[str, object],
) -> None:
    """OMN-16564's defect, as a test. The pull request merged, and the local
    branch still carries commits it never contained -- deleting it would destroy
    them with no record anywhere."""
    _gh_shim(
        world["bin"],  # type: ignore[arg-type]
        merged=_merged_row("orphan", world["dev_tip"]),  # type: ignore[arg-type]
    )

    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 0
    assert (
        report["branches"][0]["reason"]  # type: ignore[index]
        == "merged_pr_head_does_not_contain_local_tip"
    )
    assert _branch_exists(world, "orphan")


def test_a_merged_pull_request_head_absent_from_the_clone_refuses(
    world: dict[str, object],
) -> None:
    """An oid the clone has never seen cannot be shown to contain the tip. An
    unanswerable reachability question refuses, exactly as an unanswerable
    pull-request question does."""
    _gh_shim(
        world["bin"],  # type: ignore[arg-type]
        merged=_merged_row("orphan", "0" * 40),
    )

    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 0
    assert (
        report["branches"][0]["reason"]  # type: ignore[index]
        == "merged_pr_head_does_not_contain_local_tip"
    )
    assert _branch_exists(world, "orphan")


def test_a_merged_pull_request_for_a_different_head_ref_does_not_admit(
    world: dict[str, object],
) -> None:
    """`--head` is a server-side filter the tool does not get to trust. A row
    naming another ref is discarded even when its oid would have matched, so a
    filter that ever loosened could not silently widen the deletion set."""
    _gh_shim(
        world["bin"],  # type: ignore[arg-type]
        merged=_merged_row("some-other-branch", world["orphan_tip"]),  # type: ignore[arg-type]
    )

    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 0
    assert (
        report["branches"][0]["reason"] == "unmerged_and_tip_not_recorded"  # type: ignore[index]
    )
    assert _branch_exists(world, "orphan")


def test_an_open_pull_request_refuses_before_the_merged_lookup(
    world: dict[str, object],
) -> None:
    """A branch can carry both a merged pull request and a newer open one. The
    open one wins: the branch is live work whatever else happened to it."""
    _gh_shim(
        world["bin"],  # type: ignore[arg-type]
        prs='[{"number":777}]',
        merged=_merged_row("orphan", world["orphan_tip"]),  # type: ignore[arg-type]
    )

    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 0
    assert report["branches"][0]["reason"] == "open_pull_request"  # type: ignore[index]
    assert _branch_exists(world, "orphan")


def test_a_closed_unmerged_pull_request_does_not_admit_the_branch(
    world: dict[str, object],
) -> None:
    """A closed-unmerged pull request answers the merged query with no rows.
    Nothing landed, so nothing is evidence that the tip survives deletion."""
    _gh_shim(world["bin"], merged="[]")  # type: ignore[arg-type]

    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 0
    assert (
        report["branches"][0]["reason"] == "unmerged_and_tip_not_recorded"  # type: ignore[index]
    )
    assert _branch_exists(world, "orphan")


def test_a_gh_failure_on_the_merged_lookup_refuses(
    world: dict[str, object],
) -> None:
    """Rule 16 again, on the new call. An errored merged lookup and a branch
    with no merged pull request return the same empty list to a tool that
    ignores the exit status."""
    _gh_shim(world["bin"], merged_exit_code=1)  # type: ignore[arg-type]

    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 0
    assert "refused" in report["branches"][0]["reason"]  # type: ignore[index]
    assert "merged" in report["branches"][0]["reason"]  # type: ignore[index]
    assert _branch_exists(world, "orphan")


def test_the_merged_path_never_overrides_the_live_worktree_refusal(
    world: dict[str, object],
) -> None:
    """The strongest evidence a branch has landed still does not make a branch
    a live worktree is sitting on an orphan."""
    linked = world["registry"] / "omni_worktrees" / "OMN-2" / "some_repo"  # type: ignore[operator]
    assert (
        _git(
            "worktree",
            "add",
            str(linked),
            "orphan",
            cwd=world["clone"],  # type: ignore[arg-type]
            env=world["env"],  # type: ignore[arg-type]
        ).returncode
        == 0
    )
    _gh_shim(
        world["bin"],  # type: ignore[arg-type]
        merged=_merged_row("orphan", world["orphan_tip"]),  # type: ignore[arg-type]
    )

    report = _report(_run_tool(world, "--branch", "orphan", "--execute"))

    assert report["deleted"] == 0
    assert (
        report["branches"][0]["reason"] == "worktree_still_holds_branch"  # type: ignore[index]
    )
    assert _branch_exists(world, "orphan")


def test_the_merged_path_still_requires_a_resolving_consent_citation(
    world: dict[str, object],
) -> None:
    """A merged pull request is evidence the work landed. It is not
    authorisation, and the tool does not let it stand in for one."""
    _gh_shim(
        world["bin"],  # type: ignore[arg-type]
        merged=_merged_row("orphan", world["orphan_tip"]),  # type: ignore[arg-type]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo",
            str(world["clone"]),
            "--consent",
            "docs/tracking/LEDGER.md:1",
            "--branch",
            "orphan",
            "--execute",
        ],
        env=scrub_git_location_env(world["env"]),  # type: ignore[arg-type]
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 3, result.stdout
    assert _branch_exists(world, "orphan")


def test_the_merged_path_is_dry_by_default(world: dict[str, object]) -> None:
    _gh_shim(
        world["bin"],  # type: ignore[arg-type]
        merged=_merged_row("orphan", world["orphan_tip"]),  # type: ignore[arg-type]
    )

    report = _report(_run_tool(world, "--branch", "orphan"))

    assert report["executed"] is False
    assert report["eligible"] == 1
    assert report["deleted"] == 0
    assert _branch_exists(world, "orphan")


# ---------------------------------------------------------------------------
# The consent citation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("citation", "why"),
    [
        ("docs/tracking/LEDGER.md:1", "line 1 is a NOTE row, not a consent row"),
        ("docs/tracking/LEDGER.md:99", "past the end of the file"),
        ("docs/tracking/NOPE.md:2", "the cited file does not exist"),
        ("docs/tracking/LEDGER.md", "no line number"),
        ("docs/tracking/LEDGER.md:0", "line numbers are one-based"),
    ],
)
def test_an_invalid_consent_citation_refuses_the_whole_run(
    world: dict[str, object], citation: str, why: str
) -> None:
    """The citation is checked ONCE, before any branch is touched. A run whose
    authorisation does not resolve deletes nothing at all, rather than deleting
    until it notices."""
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--repo",
            str(world["clone"]),
            "--consent",
            citation,
            "--branch",
            "merged",
            "--execute",
        ],
        env=scrub_git_location_env(world["env"]),  # type: ignore[arg-type]
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 3, why
    assert "REFUSED" in result.stderr, why
    assert _branch_exists(world, "merged"), why


def test_an_absent_branch_is_reported_not_crashed_on(
    world: dict[str, object],
) -> None:
    report = _report(_run_tool(world, "--branch", "no-such-branch", "--execute"))

    assert report["deleted"] == 0
    assert report["branches"][0]["reason"] == "branch_absent"  # type: ignore[index]


def test_branches_file_and_flags_combine(world: dict[str, object]) -> None:
    """The 171-branch run reads its list from a file. Blank lines and comments
    are ignored so the file can carry its own provenance header."""
    listing = world["registry"] / "branches.txt"  # type: ignore[operator]
    listing.write_text("# from the candidate table\n\nmerged\n", encoding="utf-8")

    report = _report(_run_tool(world, "--branches-file", str(listing), "--execute"))

    assert report["deleted"] == 1
    assert not _branch_exists(world, "merged")
