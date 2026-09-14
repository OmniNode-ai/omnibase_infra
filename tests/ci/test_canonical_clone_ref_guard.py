# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Real-trigger behaviour of the canonical-clone REFERENCE-TRANSACTION guard
(OMN-16497 layer 1).

## What this guard is for

`canonical-clone-guard.py` is a Claude Code PreToolUse hook, so it can only see
processes that are Claude Code tool calls. Measured 2026-08-22: a Claude
worker's Edit into the canonical omnimarket clone was DENIED at 13:59:06Z and
correctly moved to a worktree; 52 seconds later a Codex CLI session ran
`apply_patch` against the same path in the same clone and succeeded, because
Codex has no such hook. The commit/push guard beside this one does not close it
either -- no commit or push hook fires for `update-ref`, `branch -f`, or a
`fetch` into `refs/heads/*`.

A `reference-transaction` hook does, for every actor and every verb, because it
lives in git rather than in one agent's tool loop.

## How these tests are written

Every assertion drives a REAL git command against a throwaway registry with the
guard installed exactly as a host installs it -- `core.hooksPath` pointed at the
shared `canonical-clone/` directory. None of them invokes the script directly
with hand-written stdin: a hook that passes a synthetic transaction and fails a
real one is the failure mode this suite exists to prevent, and a `prepared`
stage that never aborts would satisfy a direct-invocation test perfectly.

Each refusal asserts the REF DID NOT MOVE, not merely that git exited non-zero.

`test_control_*` are the positive controls. They run the same commands against
the same clone with the guard NOT installed and assert they SUCCEED. Without
them, a guard that broke git entirely -- or a fixture that silently built an
unusable repo -- would make every refusal assertion pass for the wrong reason.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GIT_HOOKS = REPO_ROOT / "scripts" / "git-hooks"
REF_GUARD = GIT_HOOKS / "canonical_clone_ref_guard.sh"
COMMIT_GUARD = GIT_HOOKS / "canonical_clone_guard.sh"
SHARED_PATHS = GIT_HOOKS / "canonical_clone_paths.sh"
HOOKS_DIR = GIT_HOOKS / "canonical-clone"


def _git(
    *args: str, cwd: Path, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _base_env(registry: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    env["GIT_AUTHOR_NAME"] = "OMN-16497 Test"
    env["GIT_AUTHOR_EMAIL"] = "omn16497@example.invalid"
    env["GIT_COMMITTER_NAME"] = env["GIT_AUTHOR_NAME"]
    env["GIT_COMMITTER_EMAIL"] = env["GIT_AUTHOR_EMAIL"]
    env["OMNI_HOME"] = str(registry)
    env.pop("ALLOW_CANONICAL_CLONE_COMMIT", None)
    env.pop("ONEX_CANONICAL_CONVERGE", None)
    env.pop("ONEX_WORKTREES_ROOT", None)
    # git EXPORTS repo-scoping variables into hook processes and they OVERRIDE
    # both `-C` and the cwd for every descendant git call. When this suite runs
    # from inside a hook these leak in and every `git` below would operate on
    # the omnibase_infra worktree instead of the throwaway registry.
    for leaked in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_COMMON_DIR",
        "GIT_PREFIX",
    ):
        env.pop(leaked, None)
    return env


def _commit(repo: Path, env: dict[str, str], name: str, body: str) -> str:
    (repo / name).write_text(body, encoding="utf-8")
    assert _git("add", name, cwd=repo, env=env).returncode == 0
    seed_env = dict(env)
    # Committing in a canonical clone is the COMMIT guard's business, not this
    # one's. Use its documented override so these fixtures exercise the ref
    # guard rather than tripping its neighbour.
    seed_env["ALLOW_CANONICAL_CLONE_COMMIT"] = "1"
    result = _git("commit", "-q", "-m", name, cwd=repo, env=seed_env)
    assert result.returncode == 0, result.stderr
    return _git("rev-parse", "HEAD", cwd=repo, env=env).stdout.strip()


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    reg = tmp_path / "omni_home"
    (reg / "omni_worktrees").mkdir(parents=True)
    return reg


@pytest.fixture
def upstream(registry: Path, tmp_path: Path) -> Path:
    """A remote with two DIVERGENT lineages.

    `dev` runs c1 -> c2; `sidebranch` runs c1 -> c3. The divergence is the
    point: without it, every "rewind or divert" the guard is supposed to refuse
    is really a fast-forward, and the refusal tests pass for the wrong reason or
    not at all. This suite's first draft had a linear history and two of its
    refusal assertions were vacuous.
    """
    up = tmp_path / "upstream"
    up.mkdir()
    env = _base_env(registry)
    assert _git("init", "-q", "-b", "dev", cwd=up, env=env).returncode == 0
    _commit(up, env, "one.txt", "one\n")  # c1
    assert _git("checkout", "-q", "-b", "sidebranch", cwd=up, env=env).returncode == 0
    _commit(up, env, "side.txt", "side\n")  # c3
    assert _git("checkout", "-q", "dev", cwd=up, env=env).returncode == 0
    _commit(up, env, "two.txt", "two\n")  # c2
    return up


def _make_clone(registry: Path, upstream: Path, *, guarded: bool) -> Path:
    """A canonical clone at `<registry>/some_repo`, `dev` at the upstream tip."""
    env = _base_env(registry)
    repo = registry / "some_repo"
    assert (
        _git("clone", "-q", str(upstream), str(repo), cwd=registry, env=env).returncode
        == 0
    )
    if guarded:
        assert (
            _git(
                "config", "core.hooksPath", str(HOOKS_DIR), cwd=repo, env=env
            ).returncode
            == 0
        )
    return repo


@pytest.fixture
def clone(registry: Path, upstream: Path) -> Path:
    return _make_clone(registry, upstream, guarded=True)


@pytest.fixture
def unguarded_clone(registry: Path, upstream: Path) -> Path:
    return _make_clone(registry, upstream, guarded=False)


def _head_symref(repo: Path, env: dict[str, str]) -> str:
    return _git("symbolic-ref", "--quiet", "HEAD", cwd=repo, env=env).stdout.strip()


def _ref(repo: Path, env: dict[str, str], ref: str) -> str:
    return _git(
        "rev-parse", "--verify", "--quiet", ref, cwd=repo, env=env
    ).stdout.strip()


# ---------------------------------------------------------------------------
# Installation shape -- the composition point (OMN-16497, and the constraint
# OMN-18288 runs into)
# ---------------------------------------------------------------------------


def test_ref_guard_is_installed_as_a_sibling_hook_type_not_a_replacement() -> None:
    """The shared hooks directory composes by hook TYPE.

    `core.hooksPath` points every canonical clone at ONE directory. A second
    hook family therefore adds a new TYPE symlink pointing at its own script --
    it never replaces the directory and never overwrites a sibling. This test
    is what makes that contract mechanical: it fails if the commit guard's
    symlinks are repointed, or if `reference-transaction` is made to resolve to
    the commit guard.
    """
    assert REF_GUARD.is_file()
    assert os.access(REF_GUARD, os.X_OK), f"{REF_GUARD} must be executable"

    link = HOOKS_DIR / "reference-transaction"
    assert link.is_symlink(), f"{link} must be a symlink to the ref guard"
    assert link.resolve() == REF_GUARD.resolve()

    for hook_type in ("pre-commit", "pre-push", "commit-msg", "pre-merge-commit"):
        sibling = HOOKS_DIR / hook_type
        assert sibling.resolve() == COMMIT_GUARD.resolve(), (
            f"{sibling} must still resolve to the commit guard; the ref guard "
            "is a sibling, not a replacement"
        )


def test_both_guards_source_the_one_shared_position_library() -> None:
    """Two copies of "is this a canonical clone?" would drift, and a drifted
    copy fails in the dangerous direction: a guard that misreads a canonical
    clone as a worktree permits exactly what it was installed to refuse."""
    assert SHARED_PATHS.is_file()
    for script in (REF_GUARD, COMMIT_GUARD):
        text = script.read_text(encoding="utf-8")
        assert "canonical_clone_paths.sh" in text, (
            f"{script.name} must source the shared position library"
        )
        assert "canonical_clone_context" in text


# ---------------------------------------------------------------------------
# Positive controls -- the same commands SUCCEED without the guard
# ---------------------------------------------------------------------------


def test_control_ref_moves_succeed_without_the_guard(
    registry: Path, unguarded_clone: Path
) -> None:
    env = _base_env(registry)
    target = _ref(unguarded_clone, env, "HEAD~1")
    assert target, "fixture did not produce a second commit on dev"

    assert (
        _git(
            "update-ref", "refs/heads/dev", target, cwd=unguarded_clone, env=env
        ).returncode
        == 0
    )
    assert _ref(unguarded_clone, env, "refs/heads/dev") == target

    assert (
        _git(
            "checkout",
            "-q",
            "-b",
            "local-side",
            "origin/sidebranch",
            cwd=unguarded_clone,
            env=env,
        ).returncode
        == 0
    )
    assert _head_symref(unguarded_clone, env) == "refs/heads/local-side"


# ---------------------------------------------------------------------------
# Refusals -- every one asserts the ref DID NOT MOVE
# ---------------------------------------------------------------------------


def test_update_ref_plumbing_is_refused_and_the_ref_does_not_move(
    registry: Path, clone: Path
) -> None:
    """The exact verb the 2026-08-22 incident chain ended in. No commit or push
    hook fires for it."""
    env = _base_env(registry)
    before = _ref(clone, env, "refs/heads/dev")
    target = _ref(clone, env, "HEAD~1")
    assert target and before != target, "fixture did not produce a rewind target"

    result = _git("update-ref", "refs/heads/dev", target, cwd=clone, env=env)

    assert result.returncode != 0
    assert "refused" in result.stderr
    assert _ref(clone, env, "refs/heads/dev") == before


def test_branch_switch_is_refused_and_head_does_not_move(
    registry: Path, clone: Path
) -> None:
    """The half the friction report of 2026-09-13 asked for: the canonical
    knowledge-base clone was found sitting on a background-fleet feature
    branch, so every lane resolving it read a stale main."""
    env = _base_env(registry)
    assert (
        _git("branch", "sidebranch", "origin/sidebranch", cwd=clone, env=env).returncode
        == 0
    )
    before = _head_symref(clone, env)

    result = _git("checkout", "-q", "sidebranch", cwd=clone, env=env)

    assert result.returncode != 0
    assert "a branch switch" in result.stderr
    assert _head_symref(clone, env) == before


def test_switch_verb_is_refused_too(registry: Path, clone: Path) -> None:
    env = _base_env(registry)
    assert (
        _git("branch", "sidebranch", "origin/sidebranch", cwd=clone, env=env).returncode
        == 0
    )
    before = _head_symref(clone, env)

    result = _git("switch", "-q", "sidebranch", cwd=clone, env=env)

    assert result.returncode != 0
    assert _head_symref(clone, env) == before


def test_checkout_b_is_refused_and_head_does_not_move(
    registry: Path, clone: Path
) -> None:
    env = _base_env(registry)
    before = _head_symref(clone, env)

    result = _git("checkout", "-q", "-b", "brand-new", cwd=clone, env=env)

    assert result.returncode != 0
    assert _head_symref(clone, env) == before


def test_detaching_head_is_refused(registry: Path, clone: Path) -> None:
    """Detachment is what left the omnimarket clone unrepairable for two days
    (OMN-17313)."""
    env = _base_env(registry)
    before = _head_symref(clone, env)

    result = _git("checkout", "-q", "--detach", "HEAD", cwd=clone, env=env)

    assert result.returncode != 0
    assert "detaching HEAD" in result.stderr
    assert _head_symref(clone, env) == before


def test_non_fast_forward_branch_move_is_refused(registry: Path, clone: Path) -> None:
    env = _base_env(registry)
    assert (
        _git("branch", "sidebranch", "origin/sidebranch", cwd=clone, env=env).returncode
        == 0
    )
    before = _ref(clone, env, "refs/heads/sidebranch")
    diverged = _ref(clone, env, "refs/heads/dev")
    # Positive control on the FIXTURE, not the guard: assert the two really do
    # diverge, so this test cannot pass by asserting a refusal of something that
    # was never a fast-forward candidate in the first place.
    assert (
        _git(
            "merge-base", "--is-ancestor", before, diverged, cwd=clone, env=env
        ).returncode
        != 0
    ), "fixture lineages are not divergent; this test would be vacuous"

    result = _git("branch", "-f", "sidebranch", diverged, cwd=clone, env=env)

    assert result.returncode != 0
    assert "non-fast-forward" in result.stderr
    assert _ref(clone, env, "refs/heads/sidebranch") == before


def test_branch_deletion_is_refused(registry: Path, clone: Path) -> None:
    env = _base_env(registry)
    assert (
        _git("branch", "sidebranch", "origin/sidebranch", cwd=clone, env=env).returncode
        == 0
    )
    before = _ref(clone, env, "refs/heads/sidebranch")

    result = _git("branch", "-D", "sidebranch", cwd=clone, env=env)

    assert result.returncode != 0
    assert _ref(clone, env, "refs/heads/sidebranch") == before


def test_stash_is_refused(registry: Path, clone: Path) -> None:
    env = _base_env(registry)
    (clone / "one.txt").write_text("dirtied\n", encoding="utf-8")

    result = _git("stash", "push", "-q", cwd=clone, env=env)

    assert result.returncode != 0
    assert (clone / "one.txt").read_text(encoding="utf-8") == "dirtied\n"


# ---------------------------------------------------------------------------
# Permitted paths -- a guard that refused everything would pass every test above
# ---------------------------------------------------------------------------


def test_fetch_and_fast_forward_are_permitted(
    registry: Path, upstream: Path, clone: Path
) -> None:
    """The `pull --ff-only` path: remote-tracking refs move freely and the
    checked-out branch may advance to them.

    This is the single most important ALLOW in the policy. The canonical clones
    exist to be pulled; a guard that refused this would not be strict, it would
    be broken, and the first lane to hit it would route around the whole
    family.
    """
    env = _base_env(registry)
    before = _ref(clone, env, "refs/heads/dev")
    _commit(upstream, env, "three.txt", "three\n")

    assert _git("fetch", "-q", "origin", cwd=clone, env=env).returncode == 0
    ahead = _ref(clone, env, "refs/remotes/origin/dev")
    assert ahead != before, "fixture did not actually move the upstream"

    result = _git("merge", "-q", "--ff-only", "origin/dev", cwd=clone, env=env)

    assert result.returncode == 0, result.stderr
    assert _ref(clone, env, "refs/heads/dev") == ahead


def test_tags_are_permitted(registry: Path, clone: Path) -> None:
    env = _base_env(registry)
    result = _git("tag", "some-tag", cwd=clone, env=env)
    assert result.returncode == 0, result.stderr
    assert _ref(clone, env, "refs/tags/some-tag") != ""


def test_linking_a_worktree_is_permitted(registry: Path, clone: Path) -> None:
    """The sanctioned escape hatch (rule 9) has to keep working.

    git hands this hook a HEAD line for the NEW worktree that is byte-identical
    to a real branch switch -- same stage, same shape, and an empty environment
    in both cases. This test is the one that fails if the discriminator
    regresses, and it fails by making every lane unable to start work, so it is
    load-bearing rather than decorative.
    """
    env = _base_env(registry)
    worktree = registry / "omni_worktrees" / "OMN-16497" / "some_repo"
    before = _head_symref(clone, env)

    result = _git(
        "worktree",
        "add",
        "-q",
        "-b",
        "jonah/omn-16497-test",
        str(worktree),
        "HEAD",
        cwd=clone,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert worktree.is_dir()
    # The clone's own HEAD is untouched -- which is exactly why this is allowed.
    assert _head_symref(clone, env) == before


def test_ref_moves_inside_a_linked_worktree_are_permitted(
    registry: Path, clone: Path
) -> None:
    """Worktrees share the clone's config, and therefore this hooksPath, so
    every lane's ordinary work reaches this hook. Worktrees are mutable."""
    env = _base_env(registry)
    worktree = registry / "omni_worktrees" / "OMN-16497" / "some_repo"
    assert (
        _git(
            "worktree",
            "add",
            "-q",
            "-b",
            "jonah/omn-16497-work",
            str(worktree),
            "HEAD",
            cwd=clone,
            env=env,
        ).returncode
        == 0
    )

    # A branch switch inside the worktree: refused in the clone, fine here.
    assert _git("branch", "scratch", cwd=worktree, env=env).returncode == 0
    result = _git("checkout", "-q", "scratch", cwd=worktree, env=env)
    assert result.returncode == 0, result.stderr
    assert _head_symref(worktree, env) == "refs/heads/scratch"

    # And the plumbing verb the clone refuses.
    target = _ref(worktree, env, "HEAD")
    assert (
        _git(
            "update-ref", "refs/heads/scratch", target, cwd=worktree, env=env
        ).returncode
        == 0
    )


def test_the_sanctioned_convergence_env_var_permits_a_refused_move(
    registry: Path, clone: Path
) -> None:
    """`converge-canonical-clone.sh` preserves the tree as patches and appends a
    ledger row BEFORE it moves anything. That is the difference between a
    convergence and the drift this hook exists to stop -- so it gets a named,
    evidence-producing door. A hook with no door gets routed around."""
    env = _base_env(registry)
    assert (
        _git("branch", "sidebranch", "origin/sidebranch", cwd=clone, env=env).returncode
        == 0
    )
    refused = _git("checkout", "-q", "sidebranch", cwd=clone, env=env)
    assert refused.returncode != 0

    converge_env = dict(env)
    converge_env["ONEX_CANONICAL_CONVERGE"] = "1"
    permitted = _git("checkout", "-q", "sidebranch", cwd=clone, env=converge_env)

    assert permitted.returncode == 0, permitted.stderr
    assert _head_symref(clone, env) == "refs/heads/sidebranch"


def test_a_repo_outside_the_registry_is_untouched(
    registry: Path, upstream: Path, tmp_path: Path
) -> None:
    """The hook must be inert everywhere but a canonical clone. `core.hooksPath`
    is a host-wide install, so a false positive here would break ordinary work
    on every unrelated repository on the machine."""
    env = _base_env(registry)
    outside = tmp_path / "not_in_registry"
    assert (
        _git(
            "clone", "-q", str(upstream), str(outside), cwd=tmp_path, env=env
        ).returncode
        == 0
    )
    assert (
        _git(
            "config", "core.hooksPath", str(HOOKS_DIR), cwd=outside, env=env
        ).returncode
        == 0
    )

    result = _git("checkout", "-q", "-b", "anything", cwd=outside, env=env)

    assert result.returncode == 0, result.stderr
    assert _head_symref(outside, env) == "refs/heads/anything"


# ---------------------------------------------------------------------------
# Fail-closed and stage handling
# ---------------------------------------------------------------------------


def test_an_unparseable_transaction_line_fails_closed(
    registry: Path, clone: Path
) -> None:
    """Inside a canonical clone this hook has the whole transaction on stdin and
    no session to protect, so a line it cannot read is a mutation it cannot
    vouch for. This is the one assertion that must drive the script directly:
    git cannot be made to emit a malformed line."""
    env = _base_env(registry)
    result = subprocess.run(
        [str(REF_GUARD), "prepared"],
        cwd=clone,
        env=env,
        input="   \n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "unparseable" in result.stderr


@pytest.mark.parametrize("stage", ["committed", "aborted"])
def test_after_the_fact_stages_are_a_no_op(
    registry: Path, clone: Path, stage: str
) -> None:
    """Only `prepared` can abort a transaction, so neither of these may ever
    write to stderr or fail a command.

    `committed` reads nothing at all: doing so would cost a git invocation per
    ref of every fetch. `aborted` is no longer inert -- it is where the
    OMN-18358 restore runs -- but it stays silent and successful whenever no
    marker from this hook's own refusal is present, which is every abort with
    another cause."""
    env = _base_env(registry)
    result = subprocess.run(
        [str(REF_GUARD), stage],
        cwd=clone,
        env=env,
        input="0000000000000000000000000000000000000000 ref:refs/heads/x HEAD\n",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stderr == ""


# ---------------------------------------------------------------------------
# The half-apply (OMN-18358)
#
# git updates the working tree and the index BEFORE it opens the ref
# transaction that carries the HEAD symref move, so aborting at `prepared`
# refuses the LAST step of a checkout and keeps the rest. Measured on the
# canonical omniclaude clone 2026-09-14T07:0xZ: HEAD on dev, 420 staged paths,
# 0 worktree-modified, 0 untracked, and `git diff --cached --name-only
# origin/main` returning zero paths -- the index was byte-identical to the
# target while HEAD said otherwise. omnidash (149 staged) and omninode_infra
# (858 staged) carried the same signature.
#
# Every test below asserts the clone is BYTE-IDENTICAL after the refusal, not
# merely that the ref did not move. The suite above asserted only the ref, which
# is why the corruption shipped.
# ---------------------------------------------------------------------------


def _fingerprint(repo: Path, env: dict[str, str]) -> tuple[str, str, str]:
    """HEAD symref, HEAD oid, and the porcelain status -- the three facts a
    refusal must leave untouched.

    The status is included verbatim rather than as a count: `A  side.txt` and
    `?? side.txt` are different outcomes and a count cannot tell them apart.
    """
    return (
        _head_symref(repo, env),
        _ref(repo, env, "HEAD"),
        _git("status", "--porcelain", cwd=repo, env=env).stdout,
    )


def _refusal_log(repo: Path) -> Path:
    return repo / ".git" / "onex-canonical-clone-refusals.log"


@pytest.mark.parametrize(
    ("setup", "verb"),
    [
        (["branch", "sidebranch", "origin/sidebranch"], ["checkout", "sidebranch"]),
        (None, ["checkout", "sidebranch"]),  # the DWIM form pull-all.sh takes
        (["branch", "sidebranch", "origin/sidebranch"], ["switch", "sidebranch"]),
        (None, ["checkout", "-b", "brand-new", "origin/sidebranch"]),
        (None, ["checkout", "--detach", "origin/sidebranch"]),
    ],
    ids=["checkout-branch", "checkout-dwim", "switch", "checkout-b-start", "detach"],
)
def test_a_refused_switch_leaves_the_clone_byte_identical(
    registry: Path, clone: Path, setup: list[str] | None, verb: list[str]
) -> None:
    env = _base_env(registry)
    if setup:
        assert _git(*setup, cwd=clone, env=env).returncode == 0
    before = _fingerprint(clone, env)
    assert before[2] == "", "fixture clone is not clean; this test would be vacuous"

    result = _git(*verb, cwd=clone, env=env)

    assert result.returncode != 0, "the guard must still refuse"
    assert _fingerprint(clone, env) == before, (
        "a refusal that leaves the clone on the target tree is not a refusal; "
        "it is a checkout with its last step removed"
    )


def test_the_control_shows_the_verb_really_does_move_the_tree(
    registry: Path, unguarded_clone: Path
) -> None:
    """Positive control for the suite above.

    Without it, a fixture that silently produced a no-op checkout would make
    every byte-identical assertion pass for the wrong reason.
    """
    env = _base_env(registry)
    before = _fingerprint(unguarded_clone, env)

    result = _git("checkout", "-q", "sidebranch", cwd=unguarded_clone, env=env)

    assert result.returncode == 0, result.stderr
    assert _fingerprint(unguarded_clone, env) != before


def test_a_refused_reset_hard_leaves_the_clone_byte_identical(
    registry: Path, clone: Path
) -> None:
    """`reset --hard` backwards rewrites the tree before the branch-ref
    transaction, so it half-applies the same way a checkout does."""
    env = _base_env(registry)
    before = _fingerprint(clone, env)

    result = _git("reset", "--hard", "HEAD~1", cwd=clone, env=env)

    assert result.returncode != 0
    assert _fingerprint(clone, env) == before


def test_a_refusal_writes_a_durable_record(registry: Path, clone: Path) -> None:
    """A refusal that leaves no artifact is indistinguishable from a command
    nobody ran. git writes no reflog entry for an aborted transaction, so if
    this hook writes nothing, nothing anywhere records that the clone was
    touched -- which is how the half-apply stayed invisible for a day."""
    env = _base_env(registry)
    assert not _refusal_log(clone).exists()
    assert (
        _git("branch", "sidebranch", "origin/sidebranch", cwd=clone, env=env).returncode
        == 0
    )

    assert _git("checkout", "-q", "sidebranch", cwd=clone, env=env).returncode != 0

    log = _refusal_log(clone)
    assert log.is_file(), f"{log} must exist after a refusal"
    text = log.read_text(encoding="utf-8")
    assert "REFUSED" in text
    assert "HEAD" in text
    assert str(clone) in text
    assert "actor=" in text
    # The outcome of the restore is recorded too: a record that says a refusal
    # happened but not what became of the tree cannot answer the only question
    # anyone asks it. Matched as a delimited field, because "NOT_RESTORED"
    # contains "RESTORED" and a substring test would read a refusal to restore
    # as a successful one.
    verdicts = [line.split(" | ")[1] for line in text.splitlines() if " | " in line]
    assert verdicts == ["REFUSED", "RESTORED"], verdicts


def test_an_unknown_local_edit_is_preserved_rather_than_restored_over(
    registry: Path, clone: Path
) -> None:
    """The restore exists because canonical clones carry no uncommitted work by
    rule. When that premise is false, the safe move is to leave the tree alone
    and say so -- a guard that destroys a lane's in-flight work to tidy up is
    worse than the corruption it repairs."""
    env = _base_env(registry)
    assert (
        _git("branch", "sidebranch", "origin/sidebranch", cwd=clone, env=env).returncode
        == 0
    )
    (clone / "one.txt").write_text("locally edited\n", encoding="utf-8")
    assert _git("add", "one.txt", cwd=clone, env=env).returncode == 0

    result = _git("checkout", "-q", "sidebranch", cwd=clone, env=env)

    assert result.returncode != 0
    assert (clone / "one.txt").read_text(encoding="utf-8") == "locally edited\n", (
        "the guard destroyed content it could not account for"
    )
    text = _refusal_log(clone).read_text(encoding="utf-8")
    verdicts = [line.split(" | ")[1] for line in text.splitlines() if " | " in line]
    assert verdicts == ["REFUSED", "NOT_RESTORED"], verdicts
    assert "does not match the refused target" in text


def test_an_abort_this_guard_did_not_cause_does_not_trigger_a_restore(
    registry: Path, clone: Path
) -> None:
    """`aborted` fires for aborts with other causes -- measured on git 2.50.1,
    `reset --hard` emits two `AUTO_MERGE` aborts of its own. Restoring on every
    abort would make this hook reach into trees it never refused anything in."""
    env = _base_env(registry)
    worktree = registry / "omni_worktrees" / "OMN-18358" / "some_repo"
    assert (
        _git(
            "worktree",
            "add",
            "-q",
            "-b",
            "jonah/omn-18358-scratch",
            str(worktree),
            "HEAD",
            cwd=clone,
            env=env,
        ).returncode
        == 0
    )
    (worktree / "one.txt").write_text("work in progress\n", encoding="utf-8")

    # A real abort in the worktree, with no refusal from this guard behind it.
    result = subprocess.run(
        [str(REF_GUARD), "aborted"],
        cwd=worktree,
        env=env,
        input="0000000000000000000000000000000000000000 0000000000000000000000000000000000000000 AUTO_MERGE\n",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert (worktree / "one.txt").read_text(encoding="utf-8") == "work in progress\n"


def test_the_converge_door_still_leaves_no_refusal_record(
    registry: Path, clone: Path
) -> None:
    """The sanctioned path refuses nothing, so it must record nothing: a log
    that fills up with entries for permitted commands stops being read."""
    env = _base_env(registry)
    assert (
        _git("branch", "sidebranch", "origin/sidebranch", cwd=clone, env=env).returncode
        == 0
    )
    converge_env = dict(env)
    converge_env["ONEX_CANONICAL_CONVERGE"] = "1"

    assert (
        _git("checkout", "-q", "sidebranch", cwd=clone, env=converge_env).returncode
        == 0
    )

    assert not _refusal_log(clone).exists()
