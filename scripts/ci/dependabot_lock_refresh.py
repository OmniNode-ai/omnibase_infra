#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regenerate ``docker/runners/runner-image.lock.json`` on open Dependabot PRs.

OMN-16553. Any PR that touches ``pyproject.toml``/``uv.lock`` flips the
runner-image identity binding (OMN-12567) and must regenerate the lock file
on its own branch (``runner_image_identity.py --mode generate``) or the
``runner-image-build-smoke`` gate fails. A human/agent PR author can do that
as part of their own commit; Dependabot cannot, since it never runs repo
scripts. This module is the missing remediation step, driven by a scheduled
workflow rather than a ``pull_request_target`` trigger -- this repo's runner
routing policy (``config/runner_routing_policy.yaml`` /
``scripts/audit-runner-routing.py``) prohibits ``pull_request_target``
outright, so the fix cannot be event-triggered off the PR itself. Instead a
scheduled tick lists open Dependabot PRs and inspects/repairs each one.

Trust boundary: the *script logic* driving the regen (this file,
``runner_image_identity.py``, ``ci_env_digest.py``, and the two non-manifest
shared-env inputs) always comes from the trusted checkout this process is
running in -- never from a PR branch. Only the two PR-authored data files
(``pyproject.toml``, ``uv.lock``) are read from the PR ref, via ``git show``
(no PR code is ever executed). The regenerated lock is then written into a
throwaway ``git worktree`` checked out at the PR branch tip and pushed --
the main checkout's ``HEAD`` is never moved.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runner_image_identity

# Inputs whose bytes must always come from the trusted base checkout, never
# from a PR branch -- mirrors ci_env_digest.DEFAULT_ENV_INPUTS minus the two
# manifest files, which are the only PR-authored inputs.
TRUSTED_ENV_INPUTS: tuple[str, ...] = (
    ".github/actions/setup-python-uv/action.yml",
    "scripts/ci/ci_env_digest.py",
    "scripts/ci/ensure_ci_env.sh",
)
PR_MANIFEST_INPUTS: tuple[str, ...] = ("pyproject.toml", "uv.lock")
LOCK_RELATIVE = "docker/runners/runner-image.lock.json"
COMMIT_MESSAGE = (
    "chore(OMN-16553): regenerate runner image lock for dependency bump [bot]"
)
COMMENT_BODY = (
    "Regenerated `docker/runners/runner-image.lock.json` (OMN-16553): this PR's "
    "dependency-manifest bump flipped the runner-image identity binding, and "
    "Dependabot can't run repo scripts to fix it itself. Pushed the regenerated "
    "lock so `runner-image-build-smoke` re-runs green."
)


def _run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _gh_json(args: list[str]) -> Any:
    result = subprocess.run(["gh", *args], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def list_dependabot_prs(repo: str) -> list[dict[str, Any]]:
    """Return every open PR authored by Dependabot in ``repo``."""
    payload = _gh_json(
        [
            "pr",
            "list",
            "--repo",
            repo,
            "--author",
            "app/dependabot",
            "--state",
            "open",
            "--json",
            "number,headRefName,headRepository,headRepositoryOwner",
        ]
    )
    assert isinstance(payload, list)
    return payload


def get_pr(repo: str, number: int) -> dict[str, Any]:
    payload = _gh_json(
        [
            "pr",
            "view",
            str(number),
            "--repo",
            repo,
            "--json",
            "number,headRefName,headRepository,headRepositoryOwner",
        ]
    )
    assert isinstance(payload, dict)
    return payload


def is_same_repo_dependabot_branch(pr: dict[str, Any], repo: str) -> bool:
    """Return True iff ``pr`` is a same-repo (never a fork) ``dependabot/*`` branch.

    Defense in depth independent of the caller's own author filter: this
    check runs again immediately before any push, so a manually-supplied
    ``--pr`` can never target a fork PR or a non-Dependabot branch.
    """
    owner = pr.get("headRepositoryOwner", {}).get("login")
    name = pr.get("headRepository", {}).get("name")
    if not owner or not name:
        return False
    head_ref = pr.get("headRefName", "")
    return f"{owner}/{name}" == repo and head_ref.startswith("dependabot/")


def regenerate_for_ref(repo_root: Path, head_ref: str) -> bytes | None:
    """Return the regenerated lock file bytes for ``head_ref``, or None if unchanged.

    Fetches ``head_ref`` into the local object database (no working-tree
    mutation of ``repo_root``), overlays the trusted shared-env inputs from
    ``repo_root`` and the two PR-authored manifest files (read via
    ``git show``, never executed) into a scratch directory, and runs the
    existing ``runner_image_identity.generate_lock`` against that overlay.
    """
    _run(
        [
            "git",
            "fetch",
            "origin",
            f"+refs/heads/{head_ref}:refs/remotes/origin/{head_ref}",
        ],
        cwd=repo_root,
    )

    with tempfile.TemporaryDirectory() as tmp:
        overlay = Path(tmp)
        for relative in TRUSTED_ENV_INPUTS:
            dest = overlay / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo_root / relative, dest)
        for relative in PR_MANIFEST_INPUTS:
            dest = overlay / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            blob = _run(["git", "show", f"origin/{head_ref}:{relative}"], cwd=repo_root)
            dest.write_text(blob.stdout, encoding="utf-8")

        lock_dest = overlay / LOCK_RELATIVE
        lock_dest.parent.mkdir(parents=True, exist_ok=True)
        lock_blob = _run(
            ["git", "show", f"origin/{head_ref}:{LOCK_RELATIVE}"], cwd=repo_root
        )
        lock_dest.write_text(lock_blob.stdout, encoding="utf-8")

        before = lock_dest.read_bytes()
        runner_image_identity.generate_lock(overlay, lock_dest)
        after = lock_dest.read_bytes()
        return None if before == after else after


def push_refresh(
    repo_root: Path, repo: str, number: int, head_ref: str, new_lock: bytes
) -> None:
    """Write ``new_lock`` onto ``head_ref`` via a throwaway worktree and push it."""
    _run(
        [
            "git",
            "fetch",
            "origin",
            f"+refs/heads/{head_ref}:refs/remotes/origin/{head_ref}",
        ],
        cwd=repo_root,
    )
    with tempfile.TemporaryDirectory() as tmp:
        worktree = Path(tmp) / "wt"
        _run(
            ["git", "worktree", "add", "--detach", str(worktree), f"origin/{head_ref}"],
            cwd=repo_root,
        )
        try:
            (worktree / LOCK_RELATIVE).write_bytes(new_lock)
            _run(["git", "add", LOCK_RELATIVE], cwd=worktree)
            _run(
                [
                    "git",
                    "-c",
                    "user.name=omninode-bot",
                    "-c",
                    "user.email=bot@omninode.ai",
                    "commit",
                    "-m",
                    COMMIT_MESSAGE,
                ],
                cwd=worktree,
            )
            _run(["git", "push", "origin", f"HEAD:refs/heads/{head_ref}"], cwd=worktree)
        finally:
            _run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo_root)
    subprocess.run(
        ["gh", "pr", "comment", str(number), "--repo", repo, "--body", COMMENT_BODY],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo", required=True, help="owner/name, e.g. OmniNode-ai/omnibase_infra"
    )
    parser.add_argument(
        "--pr", type=int, default=None, help="Limit to one PR number (manual backfill)"
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    prs = (
        [get_pr(args.repo, args.pr)]
        if args.pr is not None
        else list_dependabot_prs(args.repo)
    )

    exit_code = 0
    for pr in prs:
        number = pr["number"]
        head_ref = pr.get("headRefName", "")
        if not is_same_repo_dependabot_branch(pr, args.repo):
            print(
                f"PR #{number}: skipping ({head_ref!r} is not a same-repo dependabot/* branch)"
            )
            continue
        try:
            new_lock = regenerate_for_ref(repo_root, head_ref)
        except subprocess.CalledProcessError as exc:
            print(f"::error::PR #{number}: regen failed: {exc.stderr}")
            exit_code = 1
            continue
        if new_lock is None:
            print(f"PR #{number}: runner-image.lock.json already fresh")
            continue
        try:
            push_refresh(repo_root, args.repo, number, head_ref, new_lock)
        except subprocess.CalledProcessError as exc:
            print(f"::error::PR #{number}: push failed: {exc.stderr}")
            exit_code = 1
            continue
        print(f"PR #{number}: pushed regenerated lock to {head_ref}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
