#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""RT-1 (OMN-14438): deploy from a CLEAN CHECKOUT of a named ref, not the ambient
host tree, and HARD-ASSERT the vendored-SHA manifest equals that ref.

The disease (mechanical-release-trains plan, instance #3): ``stage_workspace.sh``
``rsync``s each sibling from ``${OMNI_HOME}/<repo>`` on the ``.201`` host -- an
UNMANAGED working copy that is routinely detached, behind, and dirty. Merging a
PR does not change what gets built, so every "deployed" claim is unfalsifiable.
A vendored-SHA manifest *is* emitted (``sibling-vcs-provenance.json``) but nobody
asserts it equals the intended ref, so a behind/dirty clone silently leaks a
stale SHA into the image.

This module fixes both halves at the root:

  * ``checkout`` -- before staging, bring each sibling clone to a clean checkout
    of ``<ref>`` (``git fetch --prune`` + ``git checkout`` + ``git reset --hard``
    + ``git clean -ffdx``), where ``<ref>`` is a dev tag, a branch, or dev HEAD.
    Build from *that*, never the ambient tree. Emits an expected-refs manifest
    recording the resolved SHA for every sibling. Fails closed (a checkout that
    does not land at ``<ref>``, or leaves a dirty tree, is an unverifiable build).

  * ``assert`` -- after staging, hard-assert the vendored-SHA manifest
    (``sibling-vcs-provenance.json``) equals ``<ref>``'s SHA for every sibling.
    A vendored SHA that does not equal the intended ref -- the exists-but-WRONG
    failure where a behind/dirty clone leaked a stale SHA -- FAILS the build
    (non-zero exit), never a silent pass.

  * ``reconcile`` (OMN-17291) -- the standing reconciler for the deploy-source
    clones themselves (``/data/omninode/omni_home`` on ``.201``), which no
    reconciler covered: fetch, fast-forward the tracked branch, refuse loudly
    (naming the repo) on a dirty or non-fast-forwardable clone, and VERIFY HEAD
    actually landed on the fetched tip. Emits a receipt naming each repo and its
    old -> new SHA. Unlike ``checkout`` it never runs ``reset --hard`` /
    ``clean -ffdx``: this runs against long-lived host clones, so it refuses
    rather than destroys.

``--hotpatch`` deploys a deliberately-dirty tree: the checkout does NOT reset/clean
(that would destroy the patch); the manifest records the base SHA plus a
``hotpatch: true`` / ``dirty: true`` label. A hot-patch is labelled, never
laundered.

Exit codes:
  0  success
  2  usage / argument error
  3  checkout failed (git error, ref unresolvable, HEAD not at ref, or a clean
     checkout left the tree dirty -- an unverifiable build)
  4  manifest assertion failed (vendored SHA != intended ref, or an unlabelled
     dirty tree)
  5  sync did not converge -- a fetch succeeded but HEAD is not at the fetched
     tip afterwards (OMN-17291: a clean fetch is NOT evidence of sync)

This script is stdlib-only so it can run standalone on the ``.201`` host and
inside the build context without the project venv.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

USAGE_ERROR = 2
CHECKOUT_FAILED = 3
ASSERT_FAILED = 4
SYNC_NOT_CONVERGED = 5

# Paths the BUILD ITSELF writes into its own build context on every run, and
# which the reconciler therefore must not read as operator work (OMN-17291).
# Measured on .201 2026-08-31: `git status --porcelain` on the omnibase_infra
# deploy clone reported ` M workspace/sibling-pin-comparison.json`,
# ` M workspace/sibling-vcs-provenance.json`, `?? workspace/deploy-source-refs.json`
# -- outputs of check_sibling_lock_pins.py / stage_workspace.sh / this very
# script, two of which are TRACKED committed placeholders the build overwrites
# by design. A reconciler that refuses on those refuses forever on the one clone
# it most needs to cover. Everything OUTSIDE this prefix still blocks.
BUILD_SCRATCH_PREFIXES: tuple[str, ...] = ("workspace/",)


class DeploySourceRefError(RuntimeError):
    """Raised on a checkout or assertion failure; carries the process exit code."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


@dataclass(frozen=True)
class RepoRefResult:
    """Per-sibling result of a clean-ref checkout (one row of the expected-refs
    manifest)."""

    repo: str
    path: str
    ref: str
    expected_sha: str  # the SHA the ref resolves to (== HEAD after clean checkout)
    head_sha: str  # actual HEAD after the operation
    dirty: bool  # working tree dirty after the operation
    hotpatch: bool  # deliberately-dirty deploy, labelled not laundered
    before_sha: str = ""  # HEAD before the operation (OMN-17291: old -> new receipt)


@dataclass(frozen=True)
class RepoSyncResult:
    """Per-clone result of a fast-forward reconcile (one row of the receipt).

    ``before_sha`` -> ``after_sha`` is the durable old -> new evidence AC1 asks
    for; ``target_sha`` is the fetched tip both are checked against, so a receipt
    can never claim a sync that did not land.
    """

    repo: str
    path: str
    branch: str
    before_sha: str
    after_sha: str
    target_sha: str
    moved: bool


def _git(repo_path: Path, *args: str) -> str:
    """Run a git command in ``repo_path`` and return trimmed stdout.

    Raises ``DeploySourceRefError`` (CHECKOUT_FAILED) on a non-zero exit so every
    git failure is loud, not swallowed.

    OMN-14900: ``-c safe.directory=<repo_path>`` is injected on every call so a
    uid-mismatched invoker (the deploy runner container driving RT-1 against
    the OMNI_HOME clones) is never rejected with "detected dubious ownership".
    Scoped per process + path; removes the fragile dependency on the container
    env GIT_CONFIG_* pass-through, which subprocess env overrides can drop.
    """
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo_path}", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )
    if result.returncode != 0:
        raise DeploySourceRefError(
            f"git {' '.join(args)} failed in {repo_path} (rc={result.returncode}): "
            f"{result.stderr.strip()}",
            CHECKOUT_FAILED,
        )
    return result.stdout.strip()


def _git_raw(repo_path: Path, *args: str) -> str:
    """Like :func:`_git` but returns stdout UNSTRIPPED.

    ``git status --porcelain`` encodes state in the first two columns, so an
    unstaged modification arrives as ``" M path"``. :func:`_git` strips stdout,
    which eats that leading space on the first line and shifts every subsequent
    path parse by one character. Anything that reads porcelain columns must use
    this variant.
    """
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo_path}", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )
    if result.returncode != 0:
        raise DeploySourceRefError(
            f"git {' '.join(args)} failed in {repo_path} (rc={result.returncode}): "
            f"{result.stderr.strip()}",
            CHECKOUT_FAILED,
        )
    return result.stdout


def _assert_git_repo(repo_path: Path) -> None:
    """Assert ``repo_path`` is a git clone this tool can actually CHECK OUT into.

    OMN-17291: ``git rev-parse --is-inside-work-tree`` exits 0 and prints
    ``false`` on a ``core.bare=true`` clone -- so a check that only reads the
    exit status waves through exactly the corruption found on ``.201``: a clone
    with a full working tree on disk whose ``core.bare`` is ``true``. In that
    shape ``git fetch`` exits 0 forever while every ``checkout``/``status``
    exits 128, so a sync loop reading the fetch status reports success while
    HEAD never moves. Diagnose it here, by name, with the one-command repair --
    never leave it as git's generic "must be run in a work tree" fatal three
    calls later.
    """
    if not repo_path.exists():
        raise DeploySourceRefError(
            f"{repo_path}: sibling clone path does not exist", CHECKOUT_FAILED
        )
    # rev-parse raises (CHECKOUT_FAILED) if this is not a git repository at all.
    inside_work_tree = _git(repo_path, "rev-parse", "--is-inside-work-tree")
    if inside_work_tree == "true":
        return

    # Not a work tree. A GENUINE bare repo has no nested `.git`; it IS the git
    # directory. A `.git` entry beside the tracked files means this is a normal
    # clone whose core.bare flag is wrong -- repairable, and never a reason to
    # report a no-op sync as success.
    if (repo_path / ".git").exists():
        raise DeploySourceRefError(
            f"{repo_path.name}: clone at {repo_path} has core.bare=true but a "
            f"working tree on disk -- git fetch will keep exiting 0 while every "
            f"checkout fails, so this clone can NEVER advance and any sync that "
            f"reports success is lying. Repair with: "
            f"git -C {repo_path} config core.bare false && "
            f"git -C {repo_path} reset --hard HEAD",
            CHECKOUT_FAILED,
        )
    raise DeploySourceRefError(
        f"{repo_path.name}: {repo_path} is a bare repository (no working tree); "
        f"a deploy-source clone must be a normal checkout",
        CHECKOUT_FAILED,
    )


def _resolve_commit(repo_path: Path, ref: str) -> str:
    """Resolve ``ref`` (branch, tag, or sha) to a full commit SHA, or fail loud."""
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_path}",
            "-C",
            str(repo_path),
            "rev-parse",
            "--verify",
            "--quiet",
            f"{ref}^{{commit}}",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )
    sha = result.stdout.strip()
    if result.returncode != 0 or not sha:
        raise DeploySourceRefError(
            f"{repo_path.name}: cannot resolve ref {ref!r} to a commit "
            f"(pass a branch like 'origin/dev', a tag, or a full SHA)",
            CHECKOUT_FAILED,
        )
    return sha


def _head_or_empty(repo_path: Path) -> str:
    """HEAD's SHA, or ``""`` when HEAD is unborn (a clone with no commits yet).

    Used only to record the *before* side of the old -> new receipt, so an
    unborn HEAD must not abort an otherwise valid reconcile.
    """
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_path}",
            "-C",
            str(repo_path),
            "rev-parse",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _is_ancestor(repo_path: Path, ancestor: str, descendant: str) -> bool:
    """True when ``ancestor`` is reachable from ``descendant`` (i.e. moving from
    ``ancestor`` to ``descendant`` is a fast-forward)."""
    result = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repo_path}",
            "-C",
            str(repo_path),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        check=False,
    )
    return result.returncode == 0


def clean_checkout(
    repo_path: Path,
    ref: str | None,
    *,
    hotpatch: bool = False,
    fetch: bool = True,
) -> RepoRefResult:
    """Bring a sibling clone to a clean checkout of ``ref`` (or snapshot it, when
    ``hotpatch``).

    Non-hotpatch: ``git fetch --prune`` (only if an ``origin`` remote exists),
    resolve ``ref`` -> SHA, ``git checkout --force`` + ``git reset --hard`` +
    ``git clean -ffdx``. HEAD must then equal the resolved SHA and the tree must be
    clean, or the checkout is treated as unverifiable and fails closed.

    Hotpatch: the tree is deployed AS-IS -- no reset/clean (which would destroy the
    deliberate patch). The recorded base is the current HEAD; ``dirty`` is expected
    and labelled ``hotpatch: true``.
    """
    repo_path = Path(repo_path)
    _assert_git_repo(repo_path)
    before_sha = _head_or_empty(repo_path)

    if hotpatch:
        head = _git(repo_path, "rev-parse", "HEAD")
        branch = _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
        dirty = bool(_git(repo_path, "status", "--porcelain"))
        recorded_ref = ref or (branch if branch != "HEAD" else head)
        return RepoRefResult(
            repo=repo_path.name,
            path=str(repo_path),
            ref=recorded_ref,
            expected_sha=head,
            head_sha=head,
            dirty=dirty,
            hotpatch=True,
            before_sha=before_sha,
        )

    if not ref:
        raise DeploySourceRefError(
            f"{repo_path.name}: no ref given for a clean checkout "
            f"(pass a ref, or --hotpatch to deploy the dirty tree deliberately)",
            USAGE_ERROR,
        )

    if fetch and "origin" in _git(repo_path, "remote").split():
        # An origin remote exists -> a fetch failure is a real (network) error, not
        # the local-clone-with-no-remote case; let it raise.
        _git(repo_path, "fetch", "--prune", "--tags", "origin")

    sha = _resolve_commit(repo_path, ref)
    _git(repo_path, "checkout", "--force", sha)
    _git(repo_path, "reset", "--hard", sha)
    _git(repo_path, "clean", "-ffdx")

    head = _git(repo_path, "rev-parse", "HEAD")
    if head != sha:
        raise DeploySourceRefError(
            f"{repo_path.name}: HEAD {head} != resolved ref {ref} ({sha}) after "
            f"clean checkout -- the checkout did not land where it claimed",
            CHECKOUT_FAILED,
        )
    status = _git(repo_path, "status", "--porcelain")
    if status:
        raise DeploySourceRefError(
            f"{repo_path.name}: tree still dirty after clean checkout of {ref}:\n"
            f"{status}",
            CHECKOUT_FAILED,
        )
    return RepoRefResult(
        repo=repo_path.name,
        path=str(repo_path),
        ref=ref,
        expected_sha=sha,
        head_sha=head,
        dirty=False,
        hotpatch=False,
        before_sha=before_sha,
    )


def _partition_dirt(status_porcelain: str) -> tuple[list[str], list[str]]:
    """Split ``git status --porcelain`` output into (blocking, tolerated).

    Tolerated, because neither can destroy operator work:

      * UNTRACKED entries (``??``) anywhere -- a long-lived host clone always
        carries some (``.venv``, stray ``origin/`` dirs, build leftovers), and
        the checkout leaves them untouched.
      * anything under :data:`BUILD_SCRATCH_PREFIXES` -- the build's own outputs
        in its own build context. A TRACKED one here IS reset to the branch's
        content by the checkout; that is correct, because the next build
        regenerates it. Nothing an operator authored lives there.

    Everything else -- a tracked modification, a staged change, a deletion --
    BLOCKS. Those are exactly what a fast-forward checkout would clobber.
    """
    blocking: list[str] = []
    tolerated: list[str] = []
    for line in status_porcelain.splitlines():
        if not line.strip():
            continue
        # Porcelain v1: two status columns, a space, then the path.
        path = line[3:].strip().strip('"')
        # Rename/copy entries read "old -> new"; judge the destination path.
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        untracked = line.startswith("??")
        build_scratch = any(
            path.startswith(prefix) for prefix in BUILD_SCRATCH_PREFIXES
        )
        (tolerated if untracked or build_scratch else blocking).append(line)
    return blocking, tolerated


def reconcile_clone(
    repo_path: Path,
    branch: str,
    *,
    fetch: bool = True,
) -> RepoSyncResult:
    """Fast-forward one long-lived deploy-source clone onto ``origin/<branch>``.

    OMN-17291. This is the standing reconciler for the clones every lane image
    is built from (``/data/omninode/omni_home`` on ``.201``), which nothing
    covered. It differs from :func:`clean_checkout` on purpose:

      * it REFUSES on a dirty or non-fast-forwardable clone instead of
        ``reset --hard`` + ``clean -ffdx``. These are host clones an operator may
        be mid-edit in; destroying that work to make a sync "succeed" is the
        wrong trade for a loop that runs unattended.
      * it VERIFIES HEAD landed on the fetched tip afterwards. A clean ``fetch``
        is not evidence of sync -- on the ``core.bare=true`` clone found on
        ``.201`` the fetch exited 0 forever while HEAD never moved. That
        post-condition, not the fetch's exit status, is what proves progress.

    A detached clone (the state omnimarket and omnibase_core were actually in)
    is recovered onto ``branch``, not merely left behind on a detached tip.
    """
    repo_path = Path(repo_path)
    _assert_git_repo(repo_path)

    blocking, tolerated = _partition_dirt(_git_raw(repo_path, "status", "--porcelain"))
    if blocking:
        raise DeploySourceRefError(
            f"{repo_path.name}: clone at {repo_path} carries uncommitted tracked "
            f"changes -- refusing to reconcile it (a fast-forward checkout would "
            f"discard them). Commit, stash, or discard the DIRTY paths below, "
            f"then re-run:\n" + "\n".join(blocking),
            CHECKOUT_FAILED,
        )
    if tolerated:
        print(
            f"reconcile: {repo_path.name}: tolerating {len(tolerated)} untracked / "
            f"build-scratch path(s) -- untracked files are left untouched; tracked "
            f"build outputs under {', '.join(BUILD_SCRATCH_PREFIXES)} are reset to "
            f"branch content and regenerated by the next build",
            file=sys.stderr,
        )

    if fetch and "origin" in _git(repo_path, "remote").split():
        _git(repo_path, "fetch", "--prune", "--tags", "origin", branch)

    target_sha = _resolve_commit(repo_path, f"origin/{branch}")
    before_sha = _head_or_empty(repo_path)

    if (
        before_sha
        and before_sha != target_sha
        and not _is_ancestor(repo_path, before_sha, target_sha)
    ):
        raise DeploySourceRefError(
            f"{repo_path.name}: HEAD {before_sha[:12]} is not an ancestor of "
            f"origin/{branch} ({target_sha[:12]}) -- the clone carries local or "
            f"diverged commits and cannot be fast-forward reconciled. Resolve it "
            f"on the host (the reconciler will not discard commits).",
            CHECKOUT_FAILED,
        )

    # Idempotent: also re-attaches a DETACHED clone that already sits at the tip.
    _git(repo_path, "checkout", "--force", "-B", branch, target_sha)

    after_sha = _git(repo_path, "rev-parse", "HEAD")
    if after_sha != target_sha:
        raise DeploySourceRefError(
            f"{repo_path.name}: checkout reported success but HEAD is "
            f"{after_sha[:12]}, not the fetched tip origin/{branch} "
            f"({target_sha[:12]}) -- the clone did NOT advance. A clean fetch is "
            f"not evidence of sync; refusing to report this repo as synced.",
            SYNC_NOT_CONVERGED,
        )
    after_branch = _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    if after_branch != branch:
        raise DeploySourceRefError(
            f"{repo_path.name}: expected to land on branch {branch!r} but HEAD "
            f"reports {after_branch!r} -- refusing to report this repo as synced.",
            SYNC_NOT_CONVERGED,
        )

    return RepoSyncResult(
        repo=repo_path.name,
        path=str(repo_path),
        branch=branch,
        before_sha=before_sha,
        after_sha=after_sha,
        target_sha=target_sha,
        moved=bool(before_sha) and before_sha != after_sha,
    )


def write_sync_receipt(
    results: list[RepoSyncResult],
    problems: list[str],
    branch: str,
    output: Path,
) -> None:
    """Write the reconcile receipt: repo, old -> new SHA, and every refusal.

    Written on the failure path too -- an operator needs to see which clones DID
    advance and which refused. ``ok`` carries the verdict; the process exit code
    carries it as well, so the receipt can never be read as a success claim on
    its own.
    """
    manifest = {
        "generated_by": "deploy_source_ref.py reconcile",
        "branch": branch,
        "ok": not problems,
        "problems": problems,
        "repos": {r.repo: asdict(r) for r in results},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_expected_refs(results: list[RepoRefResult], output: Path) -> None:
    """Write the expected-refs manifest consumed by the ``assert`` command."""
    manifest = {
        "generated_by": "deploy_source_ref.py checkout",
        "ref_pinned": True,
        "repos": {r.repo: asdict(r) for r in results},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def assert_manifest_matches_refs(
    vcs_provenance_path: Path,
    expected_refs_path: Path,
) -> None:
    """Hard-assert the vendored-SHA manifest equals the intended ref for every
    sibling that was checked out.

    This is the load-bearing RT-1 gate. It fails closed (raises
    ``DeploySourceRefError`` with ASSERT_FAILED) when:

      * a checked-out sibling is missing from the vendored VCS provenance, or
      * the vendored SHA != the intended ref's SHA (the exists-but-WRONG stale-SHA
        leak), or
      * the vendored tree is dirty and NOT a labelled hot-patch.

    A hot-patch (labelled ``hotpatch: true``) is allowed to be dirty; the label,
    not the cleanliness, is what makes it acceptable.
    """
    if not expected_refs_path.exists():
        raise DeploySourceRefError(
            f"expected-refs manifest not found: {expected_refs_path} "
            f"(RT-1 checkout did not run -- refusing to assert an unpinned build)",
            ASSERT_FAILED,
        )
    if not vcs_provenance_path.exists():
        raise DeploySourceRefError(
            f"vendored VCS provenance manifest not found: {vcs_provenance_path}",
            ASSERT_FAILED,
        )

    vcs = json.loads(vcs_provenance_path.read_text(encoding="utf-8"))
    expected = json.loads(expected_refs_path.read_text(encoding="utf-8"))
    siblings = vcs.get("siblings", {})
    exp_repos = expected.get("repos", {})

    if not exp_repos:
        raise DeploySourceRefError(
            f"expected-refs manifest {expected_refs_path} lists no repos", ASSERT_FAILED
        )

    problems: list[str] = []
    for repo, intended in exp_repos.items():
        intended_sha = intended.get("expected_sha")
        ref = intended.get("ref")
        hotpatch = bool(intended.get("hotpatch", False))

        vendored = siblings.get(repo)
        if vendored is None:
            problems.append(
                f"{repo}: checked out at {ref} ({intended_sha}) but ABSENT from the "
                f"vendored VCS provenance manifest"
            )
            continue

        vendored_sha = vendored.get("vcs_ref")
        vendored_dirty = bool(vendored.get("vcs_dirty"))

        if vendored_sha != intended_sha:
            problems.append(
                f"{repo}: vendored SHA {vendored_sha} != intended ref {ref} "
                f"({intended_sha}) -- a stale/behind clone leaked the wrong commit "
                f"into the image"
            )
            continue

        if vendored_dirty and not hotpatch:
            problems.append(
                f"{repo}: vendored tree is DIRTY but not a labelled hot-patch "
                f"(ref {ref}) -- a clean-ref build must vendor a clean tree"
            )
            continue

        if hotpatch and not vendored_dirty:
            # A hot-patch that vendored a clean tree is a mislabel, not a hazard;
            # surface it so the label is not silently meaningless, but do not fail.
            print(
                f"RT-1 note: {repo} labelled hotpatch but vendored a clean tree "
                f"(nothing was actually patched)",
                file=sys.stderr,
            )

    if problems:
        raise DeploySourceRefError(
            "RT-1 manifest assertion FAILED -- vendored siblings do not match the "
            "intended ref:\n  " + "\n  ".join(problems),
            ASSERT_FAILED,
        )


def _parse_name_value(items: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise DeploySourceRefError(
                f"{flag} expects NAME=VALUE, got {item!r}", USAGE_ERROR
            )
        name, _, value = item.partition("=")
        if not name or not value:
            raise DeploySourceRefError(
                f"{flag} expects a non-empty NAME=VALUE, got {item!r}", USAGE_ERROR
            )
        out[name] = value
    return out


def _cmd_checkout(args: argparse.Namespace) -> int:
    repos = _parse_name_value(args.repo, "--repo")
    repo_refs = _parse_name_value(args.repo_ref or [], "--repo-ref")

    results: list[RepoRefResult] = []
    for name, path in repos.items():
        ref = repo_refs.get(name, args.ref)
        if not args.hotpatch and not ref:
            raise DeploySourceRefError(
                f"no ref for {name}: pass --ref, --repo-ref {name}=<ref>, or --hotpatch",
                USAGE_ERROR,
            )
        result = clean_checkout(
            Path(path), ref, hotpatch=args.hotpatch, fetch=not args.no_fetch
        )
        print(
            f"RT-1 checkout: {name} -> {result.ref} @ {result.expected_sha[:12]}"
            + (" [HOTPATCH/dirty]" if result.hotpatch and result.dirty else ""),
            file=sys.stderr,
        )
        results.append(result)

    write_expected_refs(results, Path(args.output))
    return 0


def _cmd_reconcile(args: argparse.Namespace) -> int:
    repos = _parse_name_value(args.repo, "--repo")

    results: list[RepoSyncResult] = []
    problems: list[str] = []
    worst_code = 0
    for name, path in repos.items():
        try:
            result = reconcile_clone(Path(path), args.branch, fetch=not args.no_fetch)
        except DeploySourceRefError as exc:
            # Every clone is attempted: one corrupt repo must not hide the state
            # of the other five. All refusals are reported together at the end.
            problems.append(f"{name}: {exc}")
            worst_code = max(worst_code, exc.exit_code)
            print(f"reconcile: {name} REFUSED -- {exc}", file=sys.stderr)
            continue
        movement = (
            f"{result.before_sha[:12]} -> {result.after_sha[:12]}"
            if result.moved
            else f"already at {result.after_sha[:12]}"
        )
        print(f"reconcile: {name} @ origin/{args.branch} {movement}", file=sys.stderr)
        results.append(result)

    write_sync_receipt(results, problems, args.branch, Path(args.output))

    if problems:
        print(
            f"ERROR: {len(problems)} of {len(repos)} deploy-source clone(s) did not "
            f"reconcile to origin/{args.branch} (OMN-17291):",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return worst_code or CHECKOUT_FAILED

    print(
        f"reconcile: all {len(results)} deploy-source clone(s) at "
        f"origin/{args.branch}; receipt written to {args.output}",
        file=sys.stderr,
    )
    return 0


def _cmd_assert(args: argparse.Namespace) -> int:
    assert_manifest_matches_refs(Path(args.vcs_provenance), Path(args.expected_refs))
    print("RT-1 assert: every sibling vendored at its intended ref.", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deploy_source_ref.py",
        description="RT-1 (OMN-14438): clean-ref deploy source + vendored-SHA assertion.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_checkout = sub.add_parser(
        "checkout",
        help="bring sibling clones to a clean checkout of a named ref and emit the "
        "expected-refs manifest",
    )
    p_checkout.add_argument(
        "--repo",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a sibling repo to check out (repeatable)",
    )
    p_checkout.add_argument(
        "--ref",
        default=None,
        help="default ref (branch/tag/sha) for every repo; prefer 'origin/<branch>' "
        "for the freshest branch tip",
    )
    p_checkout.add_argument(
        "--repo-ref",
        action="append",
        default=[],
        metavar="NAME=REF",
        help="per-repo ref override (repeatable); wins over --ref",
    )
    p_checkout.add_argument(
        "--hotpatch",
        action="store_true",
        help="deploy the current (possibly dirty) tree AS-IS; no reset/clean, "
        "labelled hotpatch in the manifest",
    )
    p_checkout.add_argument(
        "--no-fetch",
        action="store_true",
        help="skip git fetch (offline / already-fetched clones)",
    )
    p_checkout.add_argument(
        "--output",
        required=True,
        help="path to write the expected-refs manifest",
    )
    p_checkout.set_defaults(func=_cmd_checkout)

    p_reconcile = sub.add_parser(
        "reconcile",
        help="fast-forward the deploy-source clones onto origin/<branch>, "
        "verifying HEAD actually landed on the fetched tip (OMN-17291)",
    )
    p_reconcile.add_argument(
        "--repo",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="a deploy-source clone to reconcile (repeatable)",
    )
    p_reconcile.add_argument(
        "--branch",
        default="dev",
        help="tracked branch to reconcile onto (default: dev)",
    )
    p_reconcile.add_argument(
        "--no-fetch",
        action="store_true",
        help="skip git fetch (offline / already-fetched clones)",
    )
    p_reconcile.add_argument(
        "--output",
        required=True,
        help="path to write the reconcile receipt (repo, old -> new SHA)",
    )
    p_reconcile.set_defaults(func=_cmd_reconcile)

    p_assert = sub.add_parser(
        "assert",
        help="hard-assert the vendored-SHA manifest equals the intended ref for "
        "every checked-out sibling",
    )
    p_assert.add_argument(
        "--vcs-provenance",
        required=True,
        help="path to the vendored VCS provenance manifest "
        "(sibling-vcs-provenance.json)",
    )
    p_assert.add_argument(
        "--expected-refs",
        required=True,
        help="path to the expected-refs manifest emitted by 'checkout'",
    )
    p_assert.set_defaults(func=_cmd_assert)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DeploySourceRefError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    sys.exit(main())
