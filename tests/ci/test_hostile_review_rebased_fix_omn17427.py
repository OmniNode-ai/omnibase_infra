# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A rebase rewrites a PR's shas; a ``fixed-in`` disposition must survive it."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.handlers.handler_git_fix_verifier import (
    CitedThreadLines,
    GitFixVerifier,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ),
    ).stdout.strip()


def _build(tmp_path: Path) -> tuple[Path, str, str, str]:
    """A PR with a reviewed commit and a fix, then dev moves and the PR rebases.

    Returns (repo, new_base, old_fix, new_head).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "ci")
    (repo / "other.py").write_text("base\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "pr")
    target = repo / "src.py"
    target.write_text("".join(f"line {i}\n" for i in range(1, 31)))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "reviewed")
    lines = target.read_text().splitlines(keepends=True)
    lines[19] = "line 20 fixed\n"
    target.write_text("".join(lines))
    _git(repo, "commit", "-q", "-am", "fix")
    old_fix = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "dev")
    (repo / "other.py").write_text("base\nmoved on\n")
    _git(repo, "commit", "-q", "-am", "dev moves")
    new_base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "pr")
    _git(repo, "rebase", "-q", "dev")
    return repo, new_base, old_fix, _git(repo, "rev-parse", "HEAD")


def test_rebased_fix_sha_is_accepted_by_patch_identity(tmp_path: Path) -> None:
    repo, base, old_fix, head = _build(tmp_path)
    assert old_fix != head  # the rebase rewrote the sha
    assert (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", old_fix, head],
            cwd=repo,
            env=scrub_git_location_env(os.environ),
            check=False,
        ).returncode
        != 0
    )
    cited = CitedThreadLines("src.py", "", 20, 20)
    ok, why = GitFixVerifier(repo, head, base)(old_fix, cited)
    assert ok, why
    assert "changes src.py:20-20" in why


def test_fix_absent_from_the_pr_is_still_refused(tmp_path: Path) -> None:
    repo, base, _old_fix, head = _build(tmp_path)
    _git(repo, "checkout", "-q", "-b", "elsewhere", base)
    (repo / "src.py").write_text("".join(f"stranger {i}\n" for i in range(1, 31)))
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "a change the PR never made")
    stranger = _git(repo, "rev-parse", "HEAD")
    cited = CitedThreadLines("src.py", "", 20, 20)
    ok, why = GitFixVerifier(repo, head, base)(stranger, cited)
    assert not ok
    assert "is not in this PR (not an ancestor of the head)" in why


def test_fix_cherry_picked_to_base_then_rebased_is_refused(tmp_path: Path) -> None:
    repo, _base, old_fix, _head = _build(tmp_path)
    _git(repo, "checkout", "-q", "dev")
    _git(repo, "cherry-pick", f"{old_fix}^", old_fix)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "pr")
    _git(repo, "rebase", "-q", "dev")
    head = _git(repo, "rev-parse", "HEAD")
    ok, why = GitFixVerifier(repo, head, base)(
        old_fix, CitedThreadLines("src.py", "", 20, 20)
    )
    assert not ok, why
    assert "not in this PR" in why or "already on the base branch" in why


def test_squashed_fix_with_a_different_patch_is_refused(tmp_path: Path) -> None:
    repo, base, old_fix, head = _build(tmp_path)
    _git(repo, "reset", "--soft", f"{head}^")
    (repo / "other.py").write_text("squashed unrelated change\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "squashed fix")
    ok, why = GitFixVerifier(repo, _git(repo, "rev-parse", "HEAD"), base)(
        old_fix, CitedThreadLines("src.py", "", 20, 20)
    )
    assert not ok
    assert "not an ancestor of the head" in why


def test_split_fix_with_no_identical_patch_is_refused(tmp_path: Path) -> None:
    repo, base, _old_fix, _head = _build(tmp_path)
    target = repo / "src.py"
    parent = _git(repo, "rev-parse", "HEAD")
    original = target.read_text()
    combined = original.replace("line 3\n", "line 3 fixed\n").replace(
        "line 25\n", "line 25 fixed\n"
    )
    target.write_text(combined)
    _git(repo, "commit", "-q", "-am", "combined fix")
    old_fix = _git(repo, "rev-parse", "HEAD")
    _git(repo, "reset", "--hard", parent)
    target.write_text(original.replace("line 3\n", "line 3 fixed\n"))
    _git(repo, "commit", "-q", "-am", "first half")
    target.write_text(combined)
    _git(repo, "commit", "-q", "-am", "second half")
    ok, why = GitFixVerifier(repo, _git(repo, "rev-parse", "HEAD"), base)(
        old_fix, CitedThreadLines("src.py", "", 3, 3)
    )
    assert not ok
    assert "not an ancestor of the head" in why


def test_conflict_resolved_rebase_with_changed_patch_is_refused(tmp_path: Path) -> None:
    repo, _base, old_fix, _head = _build(tmp_path)
    _git(repo, "checkout", "-q", "dev")
    _git(repo, "cherry-pick", f"{old_fix}^")
    target = repo / "src.py"
    target.write_text(target.read_text().replace("line 20\n", "line 20 on dev\n"))
    _git(repo, "commit", "-q", "-am", "conflicting base change")
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "pr")
    conflict = subprocess.run(
        ["git", "rebase", "dev"],
        cwd=repo,
        env=scrub_git_location_env(os.environ),
        capture_output=True,
        check=False,
    )
    assert conflict.returncode != 0
    lines = [f"line {i}\n" for i in range(1, 31)]
    lines[19] = "line 20 fixed after conflict\n"
    target.write_text("".join(lines))
    _git(repo, "add", "src.py")
    _git(repo, "-c", "core.editor=true", "rebase", "--continue")
    ok, why = GitFixVerifier(repo, _git(repo, "rev-parse", "HEAD"), base)(
        old_fix, CitedThreadLines("src.py", "", 20, 20)
    )
    assert not ok
    assert "not an ancestor of the head" in why


def test_patch_id_command_failure_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, base, old_fix, head = _build(tmp_path)
    real_run = subprocess.run

    def fail_patch_id(
        args: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        # Matched on the subcommand, not on argv[0]: the verifier prefixes every
        # call with ``env -u GIT_DIR ...`` so it reads no environment of its own.
        if "patch-id" in args:
            return subprocess.CompletedProcess(args, 1, "", "patch-id failed")
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_patch_id)
    ok, why = GitFixVerifier(repo, head, base)(
        old_fix, CitedThreadLines("src.py", "", 20, 20)
    )
    assert not ok
    assert "not an ancestor of the head" in why


def test_empty_commit_has_no_patch_identity(tmp_path: Path) -> None:
    repo, base, _old_fix, head = _build(tmp_path)
    _git(repo, "commit", "-q", "--allow-empty", "-m", "empty fix")
    empty = _git(repo, "rev-parse", "HEAD")
    ok, why = GitFixVerifier(repo, head, base)(
        empty, CitedThreadLines("src.py", "", 20, 20)
    )
    assert not ok
    assert "not an ancestor of the head" in why


@pytest.mark.parametrize("accepted", [True, False])
def test_contract_bound_handler_persists_typed_runtime_readback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, accepted: bool
) -> None:
    repo, base, old_fix, head = _build(tmp_path)
    contract = (
        _REPO
        / "src/omnibase_infra/nodes/node_hostile_review_fix_verifier_effect/contract.yaml"
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "sha": old_fix,
                "head": head,
                "base": base,
                "path": "src.py",
                "start": 20 if accepted else 3,
                "end": 20 if accepted else 3,
            }
        )
    )
    monkeypatch.chdir(repo)
    state_root = tmp_path / "runtime-readback"
    runtime = RuntimeLocal(
        workflow_path=contract,
        state_root=state_root,
        input_path=request,
        backend_overrides={"event_bus": "inmemory"},
        timeout=10,
    )
    runtime.run()
    readback = json.loads((state_root / "workflow_result.json").read_text())
    assert runtime.exit_code == readback["exit_code"] == 0
    assert readback["handler_result"]["accepted"] is accepted
    assert (
        "changes src.py:20-20" if accepted else "does not touch the cited lines"
    ) in (readback["handler_result"]["reason"])
