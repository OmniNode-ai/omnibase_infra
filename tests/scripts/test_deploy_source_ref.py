# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RT-1 (OMN-14438): clean-ref deploy source + vendored-SHA assertion.

The load-bearing test is ``test_assert_red_on_exists_but_wrong_stale_sha``: the
sibling clone EXISTS and is a valid git tree, but it is BEHIND the intended ref
(HEAD at an older commit) -- the exists-but-WRONG failure the operator called out
(a green on ABSENCE is vacuous; see memory ``feedback_prove_red_against_exists_but_wrong``).
The vendored SHA fed to the assertion is read from that REAL behind clone, not a
hand-typed fake, and the assertion must go RED. The companion GREEN test then runs
the REAL clean checkout to convert the wrong tree into the correct one.
"""

from __future__ import annotations

import importlib.util
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
MODULE_PATH = REPO_ROOT / "scripts" / "runtime_build" / "deploy_source_ref.py"

_spec = importlib.util.spec_from_file_location(
    "deploy_source_ref_under_test", MODULE_PATH
)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
# Register before exec so dataclass field-annotation resolution (which looks the
# module up in sys.modules under __future__ string annotations) succeeds.
sys.modules[_spec.name] = mod
_spec.loader.exec_module(mod)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**scrub_git_location_env(), "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout.strip()


def _make_behind_dirty_repo(path: Path) -> tuple[str, str]:
    """Create a repo with commits A then B on branch ``dev`` (dev -> B), then leave
    the working tree BEHIND (detached at A) with an untracked dirty file.

    Returns (sha_A, sha_B). The clone thus EXISTS and is valid but is at the WRONG
    commit relative to the intended ref ``dev``.
    """
    path.mkdir(parents=True)
    _git(path, "init", "-q", "-b", "dev")
    _git(path, "config", "user.email", "t@t.t")
    _git(path, "config", "user.name", "t")
    (path / "f.txt").write_text("A\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "A")
    sha_a = _git(path, "rev-parse", "HEAD")
    (path / "f.txt").write_text("B\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-q", "-m", "B")
    sha_b = _git(path, "rev-parse", "HEAD")
    # Put the working tree BEHIND dev (detached at A) and make it dirty.
    _git(path, "checkout", "-q", "--detach", sha_a)
    (path / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    return sha_a, sha_b


@pytest.mark.unit
def test_immutable_checkout_resolves_all_pins_before_any_checkout(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    before_first, target_first = _make_behind_dirty_repo(first)
    _make_behind_dirty_repo(second)
    (first / "untracked.txt").unlink()
    (second / "untracked.txt").unlink()
    rc = mod.main(
        [
            "checkout",
            "--require-immutable-refs",
            "--no-fetch",
            "--repo",
            f"first={first}",
            "--repo",
            f"second={second}",
            "--repo-ref",
            f"first={target_first}",
            "--repo-ref",
            f"second={'f' * 40}",
            "--output",
            str(tmp_path / "refs.json"),
        ]
    )
    assert rc == mod.CHECKOUT_FAILED
    assert _git(first, "rev-parse", "HEAD") == before_first
    assert not (tmp_path / "refs.json").exists()


@pytest.mark.unit
def test_immutable_checkout_never_uses_destructive_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    _before, target = _make_behind_dirty_repo(repo)
    (repo / "untracked.txt").unlink()
    original_git = mod._git
    calls: list[tuple[str, ...]] = []

    def record_git(path: Path, *args: str) -> str:
        calls.append(args)
        return original_git(path, *args)

    monkeypatch.setattr(mod, "_git", record_git)
    rc = mod.main(
        [
            "checkout",
            "--require-immutable-refs",
            "--no-fetch",
            "--repo",
            f"repo={repo}",
            "--repo-ref",
            f"repo={target}",
            "--output",
            str(tmp_path / "refs.json"),
        ]
    )
    assert rc == 0
    assert _git(repo, "rev-parse", "HEAD") == target
    assert not any(args[0] in {"reset", "clean"} for args in calls)
    assert not any("--force" in args for args in calls)


@pytest.mark.unit
@pytest.mark.parametrize("invalid", ["duplicate", "unknown", "missing", "branch"])
def test_immutable_selection_validation_precedes_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    calls: list[tuple[str, ...]] = []

    def no_git(_path: Path, *args: str) -> str:
        calls.append(args)
        raise AssertionError("invalid selections must not reach git")

    monkeypatch.setattr(mod, "_git", no_git)
    refs = ["repo=" + "a" * 40]
    if invalid == "duplicate":
        refs += refs
    elif invalid == "unknown":
        refs.append("other=" + "b" * 40)
    elif invalid == "missing":
        refs = []
    else:
        refs = ["repo=dev"]
    command = [
        "checkout",
        "--require-immutable-refs",
        "--repo",
        f"repo={tmp_path / 'repo'}",
        "--output",
        str(tmp_path / "refs.json"),
    ]
    for ref in refs:
        command.extend(["--repo-ref", ref])
    assert mod.main(command) == mod.USAGE_ERROR
    assert calls == []


# ---------------------------------------------------------------------------
# The load-bearing RED: exists-but-WRONG (behind + dirty) leaks a stale SHA.
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_assert_red_on_exists_but_wrong_stale_sha(tmp_path: Path) -> None:
    repo = tmp_path / "omnibase_core"
    sha_a, sha_b = _make_behind_dirty_repo(repo)

    # The clone EXISTS and is valid, but HEAD is A (behind the intended ref dev=B).
    ambient_head = _git(repo, "rev-parse", "HEAD")
    assert ambient_head == sha_a
    assert sha_a != sha_b

    # Intended ref is dev (== B); the vendored provenance carries the STALE A.
    expected_refs = tmp_path / "deploy-source-refs.json"
    expected_refs.write_text(
        json.dumps(
            {
                "repos": {
                    "omnibase_core": {
                        "path": str(repo),
                        "ref": "dev",
                        "expected_sha": sha_b,
                        "hotpatch": False,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    vcs = tmp_path / "sibling-vcs-provenance.json"
    vcs.write_text(
        json.dumps(
            {
                "siblings": {
                    "omnibase_core": {
                        "vcs_ref": ambient_head,  # the REAL stale SHA
                        "vcs_dirty": True,
                        "vcs_branch": "HEAD",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(mod.DeploySourceRefError) as exc:
        mod.assert_manifest_matches_refs(vcs, expected_refs)
    assert exc.value.exit_code == mod.ASSERT_FAILED
    assert sha_a[:12] in str(exc.value)  # the stale SHA is named in the failure
    assert "!=" in str(exc.value)


@pytest.mark.unit
def test_clean_checkout_converts_wrong_tree_then_asserts_green(tmp_path: Path) -> None:
    repo = tmp_path / "omnibase_core"
    sha_a, sha_b = _make_behind_dirty_repo(repo)
    assert _git(repo, "rev-parse", "HEAD") == sha_a  # behind + dirty precondition

    # The REAL clean checkout to ref dev (== B) fixes the tree.
    result = mod.clean_checkout(repo, "dev")
    assert result.expected_sha == sha_b
    assert result.head_sha == sha_b
    assert result.dirty is False
    assert result.hotpatch is False
    assert _git(repo, "rev-parse", "HEAD") == sha_b
    assert _git(repo, "status", "--porcelain") == ""
    assert not (repo / "untracked.txt").exists()  # clean -ffdx removed it

    expected_refs = tmp_path / "deploy-source-refs.json"
    mod.write_expected_refs([result], expected_refs)
    vcs = tmp_path / "sibling-vcs-provenance.json"
    vcs.write_text(
        json.dumps(
            {
                "siblings": {
                    "omnibase_core": {
                        "vcs_ref": sha_b,
                        "vcs_dirty": False,
                        "vcs_branch": "HEAD",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    # No raise == GREEN.
    mod.assert_manifest_matches_refs(vcs, expected_refs)


@pytest.mark.unit
def test_assert_red_on_unlabelled_dirty_tree(tmp_path: Path) -> None:
    # Vendored SHA matches the ref, but the tree is DIRTY and NOT a hot-patch.
    sha = "a" * 40
    expected_refs = tmp_path / "refs.json"
    expected_refs.write_text(
        json.dumps(
            {
                "repos": {
                    "omnimarket": {"ref": "dev", "expected_sha": sha, "hotpatch": False}
                }
            }
        ),
        encoding="utf-8",
    )
    vcs = tmp_path / "vcs.json"
    vcs.write_text(
        json.dumps(
            {
                "siblings": {
                    "omnimarket": {
                        "vcs_ref": sha,
                        "vcs_dirty": True,
                        "vcs_branch": "dev",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(mod.DeploySourceRefError) as exc:
        mod.assert_manifest_matches_refs(vcs, expected_refs)
    assert exc.value.exit_code == mod.ASSERT_FAILED
    assert "DIRTY" in str(exc.value)


@pytest.mark.unit
def test_hotpatch_labels_dirty_and_passes(tmp_path: Path) -> None:
    repo = tmp_path / "omnimarket"
    _sha_a, sha_b = _make_behind_dirty_repo(repo)
    # Move HEAD to the intended base (B) but leave the tree dirty -- a deliberate
    # hot-patch on top of the named ref.
    _git(repo, "checkout", "-q", "--detach", sha_b)
    (repo / "patch.txt").write_text("hotfix\n", encoding="utf-8")
    assert _git(repo, "status", "--porcelain") != ""  # dirty

    result = mod.clean_checkout(repo, "dev", hotpatch=True)
    assert result.hotpatch is True
    assert result.dirty is True
    assert result.expected_sha == sha_b
    # The patch was NOT laundered away.
    assert _git(repo, "rev-parse", "HEAD") == sha_b
    assert (repo / "patch.txt").exists()

    expected_refs = tmp_path / "refs.json"
    mod.write_expected_refs([result], expected_refs)
    manifest = json.loads(expected_refs.read_text(encoding="utf-8"))
    entry = manifest["repos"]["omnimarket"]
    # The label is present AND the SHA is marked non-clean.
    assert entry["hotpatch"] is True
    assert entry["dirty"] is True

    vcs = tmp_path / "vcs.json"
    vcs.write_text(
        json.dumps(
            {
                "siblings": {
                    "omnimarket": {
                        "vcs_ref": sha_b,
                        "vcs_dirty": True,
                        "vcs_branch": "HEAD",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    # A labelled hot-patch with a dirty tree passes (label, not cleanliness, is
    # what makes it acceptable).
    mod.assert_manifest_matches_refs(vcs, expected_refs)


@pytest.mark.unit
def test_checkout_unresolvable_ref_fails_closed(tmp_path: Path) -> None:
    repo = tmp_path / "omnibase_core"
    _make_behind_dirty_repo(repo)
    with pytest.raises(mod.DeploySourceRefError) as exc:
        mod.clean_checkout(repo, "no-such-ref")
    assert exc.value.exit_code == mod.CHECKOUT_FAILED
    assert "cannot resolve" in str(exc.value)


@pytest.mark.unit
def test_cli_checkout_then_assert_green_and_red(tmp_path: Path) -> None:
    repo = tmp_path / "omnibase_core"
    sha_a, sha_b = _make_behind_dirty_repo(repo)
    out = tmp_path / "deploy-source-refs.json"

    rc = mod.main(
        [
            "checkout",
            "--repo",
            f"omnibase_core={repo}",
            "--ref",
            "dev",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert _git(repo, "rev-parse", "HEAD") == sha_b  # checkout moved the tree

    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest["repos"]["omnibase_core"]["expected_sha"] == sha_b

    vcs = tmp_path / "sibling-vcs-provenance.json"
    vcs.write_text(
        json.dumps(
            {
                "siblings": {
                    "omnibase_core": {
                        "vcs_ref": sha_b,
                        "vcs_dirty": False,
                        "vcs_branch": "HEAD",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert (
        mod.main(["assert", "--vcs-provenance", str(vcs), "--expected-refs", str(out)])
        == 0
    )

    # Poison the vendored provenance with the real OLD (behind, exists-but-wrong) SHA.
    vcs.write_text(
        json.dumps(
            {
                "siblings": {
                    "omnibase_core": {
                        "vcs_ref": sha_a,
                        "vcs_dirty": False,
                        "vcs_branch": "HEAD",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    rc = mod.main(["assert", "--vcs-provenance", str(vcs), "--expected-refs", str(out)])
    assert rc == mod.ASSERT_FAILED


def _make_canonical_clone(registry: Path) -> tuple[Path, str, str]:
    """A canonical clone ``<registry>/omnimarket``: commits A then B on ``dev``,
    HEAD ON the branch at B, with untracked operator work beside it.

    Returns (clone, sha_A, sha_B).
    """
    clone = registry / "omnimarket"
    sha_a, sha_b = _make_behind_dirty_repo(clone)
    (clone / "untracked.txt").unlink()
    _git(clone, "checkout", "-q", "dev")
    (clone / "operator.txt").write_text("work in progress\n", encoding="utf-8")
    return clone, sha_a, sha_b


def _assert_canonical_untouched(clone: Path, sha_b: str) -> None:
    assert _git(clone, "symbolic-ref", "-q", "HEAD") == "refs/heads/dev"
    assert _git(clone, "rev-parse", "HEAD") == sha_b
    assert (clone / "operator.txt").read_text(encoding="utf-8") == (
        "work in progress\n"
    )


@pytest.mark.unit
def test_checkout_refuses_to_detach_a_canonical_clone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-20263 falsifier: the deploy agent's RT-1 step checked a pinned SHA out
    IN ``$OMNI_HOME/omnimarket``, leaving the shared clone on a detached HEAD
    (46 times in 30 hours on h202). Pre-fix this call detaches the clone at A and
    ``clean -ffdx`` deletes ``operator.txt``; post-fix it refuses first."""
    registry = tmp_path / "omni_home"
    clone, sha_a, sha_b = _make_canonical_clone(registry)
    monkeypatch.setenv("OMNI_HOME", str(registry))
    monkeypatch.delenv("ONEX_REGISTRY_ROOTS", raising=False)

    with pytest.raises(mod.DeploySourceRefError) as excinfo:
        mod.clean_checkout(clone, sha_a, fetch=False)

    assert excinfo.value.exit_code == mod.CHECKOUT_FAILED
    assert "canonical clone" in str(excinfo.value)
    _assert_canonical_untouched(clone, sha_b)


@pytest.mark.unit
@pytest.mark.parametrize("immutable", [False, True])
def test_cli_checkout_pins_a_worktree_and_leaves_the_canonical_clone_on_its_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, immutable: bool
) -> None:
    """OMN-20263: with ``--worktree-root`` the pinned SHA lands in a detached
    worktree of the clone, the manifest names that worktree, and the canonical
    clone keeps its branch, HEAD and untracked work. A second run replaces the
    worktree (the deploy agent stages every rebuild)."""
    registry = tmp_path / "omni_home"
    clone, sha_a, sha_b = _make_canonical_clone(registry)
    monkeypatch.setenv("OMNI_HOME", str(registry))
    monkeypatch.delenv("ONEX_REGISTRY_ROOTS", raising=False)
    trees = tmp_path / "state" / "deploy_source_trees"
    out = tmp_path / "deploy-source-refs.json"
    pin = (
        ["--require-immutable-refs", "--repo-ref", f"omnimarket={sha_a}"]
        if immutable
        else ["--ref", sha_a]
    )
    argv = [
        "checkout",
        "--no-fetch",
        "--repo",
        f"omnimarket={clone}",
        *pin,
        "--worktree-root",
        str(trees),
        "--output",
        str(out),
    ]

    for _run in range(2):
        assert mod.main(argv) == 0
        row = json.loads(out.read_text(encoding="utf-8"))["repos"]["omnimarket"]
        tree = Path(row["path"])
        assert _git(tree, "rev-parse", "HEAD") == sha_a
        assert _git(tree, "status", "--porcelain") == ""
        assert trees.resolve() in tree.parents
        assert row["expected_sha"] == sha_a
        _assert_canonical_untouched(clone, sha_b)


@pytest.mark.unit
@pytest.mark.parametrize("location", ["legacy", "keyed"])
def test_pinned_worktree_refuses_non_git_entries(tmp_path: Path, location: str) -> None:
    """Quarantining foreign Git trees never authorizes moving arbitrary work."""
    clone = tmp_path / "repo"
    _old_sha, sha = _make_behind_dirty_repo(clone)
    root = tmp_path / "trees"
    tree = mod.pinned_worktree(clone, sha, root, force=False)
    _git(clone, "worktree", "remove", str(tree))
    target = root / clone.name if location == "legacy" else tree
    target.mkdir(parents=True)
    sentinel = target / "operator-work.txt"
    sentinel.write_text("must not move or delete\n")

    with pytest.raises(mod.DeploySourceRefError, match="refusing to overwrite"):
        mod.pinned_worktree(clone, sha, root, force=True)
    assert sentinel.read_text() == "must not move or delete\n"
    assert not list(target.parent.glob(f"{target.name}.foreign-*"))
