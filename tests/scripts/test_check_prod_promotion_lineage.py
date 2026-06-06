# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for scripts/check_prod_promotion_lineage.py (OMN-12626, R1).

The guard refuses to build/deploy the prod runtime image unless:
  (a) the build source working tree is clean (no uncommitted/untracked edits),
  (b) HEAD is an ancestor-of/equal-to origin/main (promoted lineage),
  (c) the build bakes org.opencontainers.image.revision=<git sha> +
      a build-provenance manifest, and
  (d) the deployed image is tagged with that revision so
      "prod == promoted main digest" is mechanically assertable.

These tests construct real git repositories on disk (no mocking of git) so the
lineage checks are exercised against real `git` behavior:

  - a clean, promoted HEAD PASSES,
  - a dirty tree FAILS,
  - an untracked-file tree FAILS,
  - a dev-only / non-promoted HEAD FAILS.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_prod_promotion_lineage.py"


def _load_module() -> Any:
    mod_name = "check_prod_promotion_lineage"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPT)
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


_mod = _load_module()

_GIT = shutil.which("git")
pytestmark = pytest.mark.skipif(_GIT is None, reason="git binary not available")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _init_repo(repo: Path) -> None:
    """Initialise a git repo with deterministic identity and an initial commit."""
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@omninode.ai")
    _git(repo, "config", "user.name", "Test Runner")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "file.txt").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "file.txt")
    _git(repo, "commit", "-q", "-m", "initial")


def _make_promoted_clone(tmp_path: Path) -> Path:
    """Return a clone whose HEAD == origin/main and whose tree is clean."""
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _git(origin, "init", "-q", "--bare", "--initial-branch=main")

    seed = tmp_path / "seed"
    _init_repo(seed)
    _git(seed, "branch", "-M", "main")
    _git(seed, "remote", "add", "origin", str(origin))
    _git(seed, "push", "-q", "origin", "main")

    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    # CI runners have no global git identity; set a deterministic committer in
    # the clone so tests that create a new commit (dev-only / second) succeed.
    _git(clone, "config", "user.email", "test@omninode.ai")
    _git(clone, "config", "user.name", "Test Runner")
    _git(clone, "config", "commit.gpgsign", "false")
    # Ensure origin/main is present in the clone's remote-tracking refs.
    _git(clone, "fetch", "-q", "origin", "main")
    return clone


# ---------------------------------------------------------------------------
# (a) clean tree
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_working_tree_clean_true_on_clean_clone(tmp_path: Path) -> None:
    clone = _make_promoted_clone(tmp_path)
    assert _mod.working_tree_clean(clone) is True


@pytest.mark.unit
def test_working_tree_clean_false_on_modified_tracked_file(tmp_path: Path) -> None:
    clone = _make_promoted_clone(tmp_path)
    (clone / "file.txt").write_text("dirty\n", encoding="utf-8")
    assert _mod.working_tree_clean(clone) is False


@pytest.mark.unit
def test_working_tree_clean_false_on_untracked_file(tmp_path: Path) -> None:
    clone = _make_promoted_clone(tmp_path)
    (clone / "untracked.txt").write_text("new\n", encoding="utf-8")
    assert _mod.working_tree_clean(clone) is False


# ---------------------------------------------------------------------------
# (b) promoted lineage
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_head_is_promoted_true_when_head_equals_origin_main(tmp_path: Path) -> None:
    clone = _make_promoted_clone(tmp_path)
    assert _mod.head_is_promoted(clone) is True


@pytest.mark.unit
def test_head_is_promoted_false_for_dev_only_commit(tmp_path: Path) -> None:
    """A commit that is not an ancestor of origin/main (dev-only) is rejected."""
    clone = _make_promoted_clone(tmp_path)
    # Create a dev-only commit on top of origin/main that was never promoted.
    (clone / "dev_only.txt").write_text("unpromoted\n", encoding="utf-8")
    _git(clone, "add", "dev_only.txt")
    _git(clone, "commit", "-q", "-m", "dev-only work")
    assert _mod.head_is_promoted(clone) is False


@pytest.mark.unit
def test_head_is_promoted_true_for_ancestor_of_origin_main(tmp_path: Path) -> None:
    """An older commit that is a strict ancestor of origin/main is promoted."""
    clone = _make_promoted_clone(tmp_path)
    # Advance origin/main one commit, leave the clone checked out at the parent.
    parent_sha = _git(clone, "rev-parse", "HEAD")
    (clone / "file.txt").write_text("v2\n", encoding="utf-8")
    _git(clone, "add", "file.txt")
    _git(clone, "commit", "-q", "-m", "second")
    _git(clone, "push", "-q", "origin", "HEAD:main")
    _git(clone, "fetch", "-q", "origin", "main")
    _git(clone, "checkout", "-q", parent_sha)
    assert _mod.head_is_promoted(clone) is True


# ---------------------------------------------------------------------------
# (a)+(b) combined assertion — the real guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assert_prod_build_promoted_passes_on_clean_promoted_head(
    tmp_path: Path,
) -> None:
    clone = _make_promoted_clone(tmp_path)
    sha = _mod.assert_prod_build_promoted(clone)
    assert sha == _git(clone, "rev-parse", "HEAD")
    assert len(sha) == 40


@pytest.mark.unit
def test_assert_prod_build_promoted_fails_on_dirty_tree(tmp_path: Path) -> None:
    clone = _make_promoted_clone(tmp_path)
    (clone / "file.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(_mod.ProdLineageError) as exc:
        _mod.assert_prod_build_promoted(clone)
    assert exc.value.reason == _mod.EnumProdLineageFailure.DIRTY_TREE


@pytest.mark.unit
def test_assert_prod_build_promoted_fails_on_untracked_file(tmp_path: Path) -> None:
    clone = _make_promoted_clone(tmp_path)
    (clone / "untracked.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(_mod.ProdLineageError) as exc:
        _mod.assert_prod_build_promoted(clone)
    assert exc.value.reason == _mod.EnumProdLineageFailure.DIRTY_TREE


@pytest.mark.unit
def test_assert_prod_build_promoted_fails_on_dev_only_head(tmp_path: Path) -> None:
    clone = _make_promoted_clone(tmp_path)
    (clone / "dev_only.txt").write_text("unpromoted\n", encoding="utf-8")
    _git(clone, "add", "dev_only.txt")
    _git(clone, "commit", "-q", "-m", "dev-only work")
    with pytest.raises(_mod.ProdLineageError) as exc:
        _mod.assert_prod_build_promoted(clone)
    assert exc.value.reason == _mod.EnumProdLineageFailure.NOT_PROMOTED


@pytest.mark.unit
def test_assert_prod_build_promoted_fails_when_origin_main_missing(
    tmp_path: Path,
) -> None:
    """Fail-fast (not silent default) when origin/main cannot be resolved."""
    repo = tmp_path / "no-remote"
    _init_repo(repo)
    with pytest.raises(_mod.ProdLineageError) as exc:
        _mod.assert_prod_build_promoted(repo)
    assert exc.value.reason == _mod.EnumProdLineageFailure.NO_PROMOTED_REF


# ---------------------------------------------------------------------------
# (c)+(d) image revision + provenance assertion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_assert_image_revision_matches_passes_when_label_equals_sha() -> None:
    sha = "0123456789abcdef0123456789abcdef01234567"
    inspect = {
        "Config": {
            "Labels": {
                "org.opencontainers.image.revision": sha,
                "com.omninode.workspace_provenance_manifest": "/app/build-provenance.json",
            }
        }
    }
    # Should not raise.
    _mod.assert_image_revision_matches(inspect, expected_sha=sha)


@pytest.mark.unit
def test_assert_image_revision_matches_accepts_short_sha_prefix() -> None:
    full = "0123456789abcdef0123456789abcdef01234567"
    short = full[:12]
    inspect = {
        "Config": {
            "Labels": {
                "org.opencontainers.image.revision": short,
                "com.omninode.workspace_provenance_manifest": "/app/build-provenance.json",
            }
        }
    }
    _mod.assert_image_revision_matches(inspect, expected_sha=full)


@pytest.mark.unit
def test_assert_image_revision_matches_fails_on_empty_revision() -> None:
    inspect = {"Config": {"Labels": {"org.opencontainers.image.revision": ""}}}
    with pytest.raises(_mod.ProdLineageError) as exc:
        _mod.assert_image_revision_matches(
            inspect, expected_sha="0123456789abcdef0123456789abcdef01234567"
        )
    assert exc.value.reason == _mod.EnumProdLineageFailure.MISSING_REVISION


@pytest.mark.unit
def test_assert_image_revision_matches_fails_on_unknown_sentinel() -> None:
    inspect = {"Config": {"Labels": {"org.opencontainers.image.revision": "unknown"}}}
    with pytest.raises(_mod.ProdLineageError) as exc:
        _mod.assert_image_revision_matches(
            inspect, expected_sha="0123456789abcdef0123456789abcdef01234567"
        )
    assert exc.value.reason == _mod.EnumProdLineageFailure.MISSING_REVISION


@pytest.mark.unit
def test_assert_image_revision_matches_fails_on_mismatch() -> None:
    inspect = {
        "Config": {
            "Labels": {
                "org.opencontainers.image.revision": "ffffffffffffffffffffffffffffffffffffffff",
                "com.omninode.workspace_provenance_manifest": "/app/build-provenance.json",
            }
        }
    }
    with pytest.raises(_mod.ProdLineageError) as exc:
        _mod.assert_image_revision_matches(
            inspect, expected_sha="0123456789abcdef0123456789abcdef01234567"
        )
    assert exc.value.reason == _mod.EnumProdLineageFailure.REVISION_MISMATCH


@pytest.mark.unit
def test_assert_image_revision_matches_fails_without_provenance_manifest() -> None:
    sha = "0123456789abcdef0123456789abcdef01234567"
    inspect = {"Config": {"Labels": {"org.opencontainers.image.revision": sha}}}
    with pytest.raises(_mod.ProdLineageError) as exc:
        _mod.assert_image_revision_matches(inspect, expected_sha=sha)
    assert exc.value.reason == _mod.EnumProdLineageFailure.MISSING_PROVENANCE


# ---------------------------------------------------------------------------
# build-arg emission (c) — the guard hands the build the revision + provenance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_promoted_build_args_bake_revision_and_provenance() -> None:
    sha = "0123456789abcdef0123456789abcdef01234567"
    args = _mod.promoted_build_args(sha)
    flat = " ".join(args)
    # Dockerfile bakes org.opencontainers.image.revision from these args.
    assert f"GIT_SHA={sha}" in flat
    assert f"VCS_REF={sha}" in flat
    assert f"RUNTIME_SOURCE_HASH={sha}" in flat
    # build-args must come in --build-arg KEY=VALUE token pairs.
    assert args.count("--build-arg") == 3


# ---------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_cli_exit_zero_on_clean_promoted_head(tmp_path: Path) -> None:
    clone = _make_promoted_clone(tmp_path)
    rc = _mod.main(["--repo", str(clone)])
    assert rc == 0


@pytest.mark.unit
def test_cli_exit_nonzero_on_dirty_tree(tmp_path: Path) -> None:
    clone = _make_promoted_clone(tmp_path)
    (clone / "file.txt").write_text("dirty\n", encoding="utf-8")
    rc = _mod.main(["--repo", str(clone)])
    assert rc != 0


@pytest.mark.unit
def test_cli_exit_nonzero_on_dev_only_head(tmp_path: Path) -> None:
    clone = _make_promoted_clone(tmp_path)
    (clone / "dev_only.txt").write_text("unpromoted\n", encoding="utf-8")
    _git(clone, "add", "dev_only.txt")
    _git(clone, "commit", "-q", "-m", "dev-only work")
    rc = _mod.main(["--repo", str(clone)])
    assert rc != 0
