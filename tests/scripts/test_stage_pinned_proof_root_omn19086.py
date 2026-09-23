# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19086: a proof source root is staged from a pinned snapshot, and a bad one refuses the build.

AC-1's falsifier is a canonical pull that fires in the middle of staging: every
staged repository must still come out clean and at its pinned sha. AC-2's is a
hand-dirtied staged repository: the precondition refuses, naming the repository
and both shas, and the same command proceeds once the file is reverted. Both run
here against real git repositories, through the same entrypoints a proof lane
uses: ``stage_pinned_proof_root.py`` and the dogfood leg of ``cut-lab-ref.sh``.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "runtime_build" / "stage_pinned_proof_root.py"
CUT_LAB_REF = REPO_ROOT / "scripts" / "runtime_build" / "cut-lab-ref.sh"

pytestmark = pytest.mark.unit


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stage_pinned_proof_root", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = _load()
REPOS: list[str] = MOD.required_repos()


def _env() -> dict[str, str]:
    return scrub_git_location_env(os.environ) | {"GIT_TERMINAL_PROMPT": "0"}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=scrub_git_location_env(os.environ) | {"GIT_TERMINAL_PROMPT": "0"},
    ).stdout.strip()


def _commit(repo: Path, name: str, content: str) -> str:
    (repo / name).write_text(content, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", f"write {name}")
    return _git(repo, "rev-parse", "HEAD")


def _make_upstream_and_canonical(tmp_path: Path) -> tuple[Path, Path]:
    """An upstream per repo, and a canonical clone of each tracking its ``dev``."""
    upstream = tmp_path / "upstream"
    canonical = tmp_path / "canonical"
    for repo in REPOS:
        up = upstream / repo
        up.mkdir(parents=True)
        _git(up, "init", "-q", "-b", "dev")
        _git(up, "config", "user.email", "t@t.t")
        _git(up, "config", "user.name", "t")
        for i in range(20):
            (up / f"module_{i}.py").write_text(f"VALUE = {i}\n", encoding="utf-8")
        _git(up, "add", "-A")
        _git(up, "commit", "-q", "-m", "init")
        subprocess.run(
            ["git", "clone", "-q", "--branch", "dev", str(up), str(canonical / repo)],
            check=True,
            capture_output=True,
            env=scrub_git_location_env(os.environ) | {"GIT_TERMINAL_PROMPT": "0"},
        )
    return upstream, canonical


def _advance_upstream(upstream: Path) -> None:
    """A new dev commit that rewrites every tracked file, as a big merge would."""
    for repo in REPOS:
        up = upstream / repo
        for i in range(20):
            (up / f"module_{i}.py").write_text(
                f"VALUE = {i + 1000}\n", encoding="utf-8"
            )
        _git(up, "commit", "-q", "-am", "advance")


def _pull_all(canonical: Path) -> None:
    for repo in REPOS:
        _git(canonical / repo, "pull", "-q", "--ff-only", "origin", "dev")


def _heads(root: Path) -> dict[str, str]:
    return {repo: _git(root / repo, "rev-parse", "HEAD") for repo in REPOS}


def _stage(
    dest: Path,
    canonical: Path,
    *,
    sources: Mapping[str, Path] | None = None,
    pins: Mapping[str, str] | None = None,
) -> dict[str, str]:
    result: dict[str, str] = MOD.stage(
        dest=dest,
        source_root=canonical,
        sources=dict(sources or {}),
        pins=dict(pins or {}),
        repos=REPOS,
    )
    return result


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
        env=_env(),
    )


def test_the_script_parses_as_python_3_9() -> None:
    """The .105 surface host runs this under its system Python 3.9.

    The first lab run there died on ``from datetime import UTC`` (3.11+), which a
    lint auto-fix had introduced. This pins the syntax floor; the runtime floor is
    the stdlib-only, no-3.10-name rule in the module docstring.
    """
    import ast

    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source, feature_version=(3, 9))
    assert "from datetime import" not in source


def test_required_set_is_read_from_the_owning_scripts() -> None:
    # spi is the repo that goes missing (OMN-15137) and the change-control repo is
    # the one only the hot-patch entrypoint names; both must be required.
    assert "omnibase_spi" in REPOS
    assert "onex_change_control" in REPOS
    assert "omnibase_infra" in REPOS
    assert len(REPOS) == len(set(REPOS))


def test_stage_produces_a_clean_root_at_the_source_heads(tmp_path: Path) -> None:
    _, canonical = _make_upstream_and_canonical(tmp_path)
    expected = {repo: _git(canonical / repo, "rev-parse", "HEAD") for repo in REPOS}
    root = tmp_path / "proof-root"

    verified = _stage(root, canonical)

    assert verified == expected
    assert _heads(root) == expected
    for repo in REPOS:
        assert _git(root / repo, "status", "--porcelain", "--untracked-files=all") == ""
    manifest = json.loads((root / "proof-root-pins.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "proof-root-pins.v1"
    assert {r: e["pin"] for r, e in manifest["repos"].items()} == expected


def test_ac1_a_pull_mid_stage_still_yields_every_repo_clean_at_its_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-1 falsifier: the canonical pull fires after the first repo is staged."""
    upstream, canonical = _make_upstream_and_canonical(tmp_path)
    pinned = {repo: _git(canonical / repo, "rev-parse", "HEAD") for repo in REPOS}
    _advance_upstream(upstream)

    real_clone: Callable[[Path, Path], subprocess.CompletedProcess[str]] = (
        MOD._clone_local
    )
    fired: list[bool] = []

    def clone_then_pull(src: Path, dest: Path) -> subprocess.CompletedProcess[str]:
        result = real_clone(src, dest)
        if not fired:
            fired.append(True)
            _pull_all(canonical)
        return result

    monkeypatch.setattr(MOD, "_clone_local", clone_then_pull)
    root = tmp_path / "proof-root"
    verified = _stage(root, canonical)

    # Positive control: the pull really fired and really moved every canonical clone.
    assert fired == [True]
    moved = {repo: _git(canonical / repo, "rev-parse", "HEAD") for repo in REPOS}
    assert all(moved[r] != pinned[r] for r in REPOS)

    assert verified == pinned
    assert _heads(root) == pinned
    for repo in REPOS:
        assert _git(root / repo, "status", "--porcelain", "--untracked-files=all") == ""
        assert (root / repo / "module_3.py").read_text(
            encoding="utf-8"
        ) == "VALUE = 3\n"


def test_the_measured_mixed_tree_shape_is_refused(tmp_path: Path) -> None:
    """The 2026-09-21 shape: git directory at the new commit, working tree at the old."""
    upstream, canonical = _make_upstream_and_canonical(tmp_path)
    root = tmp_path / "proof-root"
    pinned = _stage(root, canonical)
    _advance_upstream(upstream)
    _pull_all(canonical)
    infra = "omnibase_infra"
    shutil.rmtree(root / infra / ".git")
    shutil.copytree(canonical / infra / ".git", root / infra / ".git")
    new_head = _git(canonical / infra, "rev-parse", "HEAD")

    result = _cli("verify", "--root", str(root))

    assert result.returncode == 1, result.stderr
    assert f"REFUSE {infra}: HEAD {new_head} != pinned {pinned[infra]}" in result.stderr
    assert f"REFUSE {infra}: working tree not clean at HEAD {new_head}" in result.stderr
    assert "not building" in result.stderr


def test_ac2_a_dirty_staged_repo_refuses_and_the_revert_proceeds(
    tmp_path: Path,
) -> None:
    _, canonical = _make_upstream_and_canonical(tmp_path)
    root = tmp_path / "proof-root"
    pinned = _stage(root, canonical)
    repo = "omnibase_core"
    (root / repo / "module_0.py").write_text("VALUE = 'hand edit'\n", encoding="utf-8")

    refused = _cli("verify", "--root", str(root))
    assert refused.returncode == 1
    assert (
        f"REFUSE {repo}: working tree not clean at HEAD {pinned[repo]}, "
        f"pinned {pinned[repo]}" in refused.stderr
    )
    assert "module_0.py" in refused.stderr

    # Negative control: revert the file and the very same command proceeds.
    _git(root / repo, "checkout", "--", "module_0.py")
    proceeds = _cli("verify", "--root", str(root))
    assert proceeds.returncode == 0, proceeds.stderr
    summary = json.loads(proceeds.stdout)
    assert summary["verdict"] == "VERIFIED"
    assert summary["repos"] == pinned


def test_an_off_pin_staged_repo_refuses_naming_both_shas(tmp_path: Path) -> None:
    upstream, canonical = _make_upstream_and_canonical(tmp_path)
    root = tmp_path / "proof-root"
    pinned = _stage(root, canonical)
    _advance_upstream(upstream)
    repo = "omnimarket"
    _git(root / repo, "fetch", "-q", str(upstream / repo), "dev")
    _git(root / repo, "checkout", "-q", "--detach", "FETCH_HEAD")
    moved = _git(root / repo, "rev-parse", "HEAD")

    result = _cli("verify", "--root", str(root))

    assert result.returncode == 1
    assert f"REFUSE {repo}: HEAD {moved} != pinned {pinned[repo]}" in result.stderr


def test_an_untracked_file_refuses(tmp_path: Path) -> None:
    _, canonical = _make_upstream_and_canonical(tmp_path)
    root = tmp_path / "proof-root"
    _stage(root, canonical)
    (root / "omnibase_spi" / "stray.py").write_text("x = 1\n", encoding="utf-8")

    result = _cli("verify", "--root", str(root))

    assert result.returncode == 1
    assert "REFUSE omnibase_spi: working tree not clean" in result.stderr
    assert "stray.py" in result.stderr


def test_a_root_without_a_pin_manifest_refuses(tmp_path: Path) -> None:
    _, canonical = _make_upstream_and_canonical(tmp_path)
    result = _cli("verify", "--root", str(canonical))
    assert result.returncode == 1
    assert "proof-root-pins.json does not exist" in result.stderr


def test_a_manifest_missing_a_required_repo_refuses(tmp_path: Path) -> None:
    _, canonical = _make_upstream_and_canonical(tmp_path)
    root = tmp_path / "proof-root"
    _stage(root, canonical)
    path = root / "proof-root-pins.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["repos"]["omnibase_spi"]
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _cli("verify", "--root", str(root))

    assert result.returncode == 1
    assert "REFUSE omnibase_spi: the build needs it" in result.stderr


def test_explicit_pins_are_honoured_and_a_branch_name_is_refused(
    tmp_path: Path,
) -> None:
    upstream, canonical = _make_upstream_and_canonical(tmp_path)
    old = _git(canonical / "omnibase_infra", "rev-parse", "HEAD")
    _advance_upstream(upstream)
    _pull_all(canonical)

    root = tmp_path / "proof-root"
    verified = _stage(root, canonical, pins={"omnibase_infra": old})
    assert verified["omnibase_infra"] == old
    assert verified["omnibase_core"] == _git(
        canonical / "omnibase_core", "rev-parse", "HEAD"
    )

    refused = _cli(
        "stage",
        "--dest",
        str(tmp_path / "second-root"),
        "--source-root",
        str(canonical),
        "--pin",
        "omnibase_infra=origin/dev",
    )
    assert refused.returncode == 1
    assert "is not a full commit sha" in refused.stderr
    assert not (tmp_path / "second-root").exists()


def test_a_worktree_under_test_is_staged_at_its_own_commit(tmp_path: Path) -> None:
    _, canonical = _make_upstream_and_canonical(tmp_path)
    worktree = tmp_path / "ticket" / "omnibase_infra"
    _git(
        canonical / "omnibase_infra",
        "worktree",
        "add",
        "-q",
        "-b",
        "feature",
        str(worktree),
    )
    feature = _commit(worktree, "feature.py", "FEATURE = True\n")

    root = tmp_path / "proof-root"
    verified = _stage(root, canonical, sources={"omnibase_infra": worktree})

    assert verified["omnibase_infra"] == feature
    assert (root / "omnibase_infra" / "feature.py").is_file()

    (worktree / "feature.py").write_text("FEATURE = False\n", encoding="utf-8")
    with pytest.raises(MOD.RefusedError, match="has uncommitted changes"):
        _stage(tmp_path / "dirty-root", canonical, sources={"omnibase_infra": worktree})


def test_stage_refuses_a_non_empty_destination(tmp_path: Path) -> None:
    _, canonical = _make_upstream_and_canonical(tmp_path)
    root = tmp_path / "proof-root"
    root.mkdir()
    (root / "leftover").write_text("x", encoding="utf-8")
    with pytest.raises(MOD.RefusedError, match="not empty"):
        _stage(root, canonical)


def _run_cut_lab_ref(omni_home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(CUT_LAB_REF), *args],
        capture_output=True,
        text=True,
        check=False,
        env=_env() | {"OMNI_HOME": str(omni_home), "PROOF_ROOT_PYTHON": sys.executable},
    )


def test_cut_lab_ref_refuses_a_dogfood_build_from_an_unpinned_root(
    tmp_path: Path,
) -> None:
    _, canonical = _make_upstream_and_canonical(tmp_path)
    result = _run_cut_lab_ref(canonical, "--lane", "dogfood", "--hotpatch")
    assert result.returncode == 1
    assert "proof-root-pins.json does not exist" in result.stderr
    assert "not building" in result.stderr
    assert "deploy command" not in result.stderr


def test_cut_lab_ref_refuses_a_dirty_proof_root_and_proceeds_after_revert(
    tmp_path: Path,
) -> None:
    _, canonical = _make_upstream_and_canonical(tmp_path)
    root = tmp_path / "proof-root"
    pinned = _stage(root, canonical)
    (root / "omnibase_compat" / "module_1.py").write_text(
        "VALUE = -1\n", encoding="utf-8"
    )

    refused = _run_cut_lab_ref(root, "--lane", "dogfood", "--hotpatch")
    assert refused.returncode == 1
    assert (
        f"REFUSE omnibase_compat: working tree not clean at HEAD "
        f"{pinned['omnibase_compat']}, pinned {pinned['omnibase_compat']}"
        in refused.stderr
    )
    assert "deploy command" not in refused.stderr

    _git(root / "omnibase_compat", "checkout", "--", "module_1.py")
    proceeds = _run_cut_lab_ref(root, "--lane", "dogfood", "--hotpatch")
    assert proceeds.returncode == 0, proceeds.stderr
    assert "OMNIBASE_INFRA_COMPOSE_PROJECT=omnibase-infra-dogfood" in proceeds.stderr
    assert "dry-run" in proceeds.stderr


def test_cut_lab_ref_verifies_any_lane_whose_root_carries_a_manifest(
    tmp_path: Path,
) -> None:
    _, canonical = _make_upstream_and_canonical(tmp_path)
    root = tmp_path / "proof-root"
    _stage(root, canonical)
    (root / "omnibase_infra" / "module_2.py").write_text(
        "VALUE = -2\n", encoding="utf-8"
    )

    result = _run_cut_lab_ref(root, "--lane", "dev", "--hotpatch")

    assert result.returncode == 1
    assert "REFUSE omnibase_infra: working tree not clean" in result.stderr


def test_cut_lab_ref_refuses_a_ref_build_from_a_proof_root(tmp_path: Path) -> None:
    """A ref build fetches from the moving canonical clone and would discard the pins."""
    _, canonical = _make_upstream_and_canonical(tmp_path)
    root = tmp_path / "proof-root"
    _stage(root, canonical)

    for lane in ("dogfood", "dev"):
        result = _run_cut_lab_ref(root, "--lane", lane, "--ref", "origin/dev")
        assert result.returncode == 1, (lane, result.stderr)
        assert "builds with --hotpatch only" in result.stderr
        assert "deploy command" not in result.stderr
