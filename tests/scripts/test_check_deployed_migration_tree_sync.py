# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the deployed-migration-tree sync gate (OMN-13415).

Recurrence guard for the stability-promotion footgun where a bind-mounted
deployed forward-migration tree carried a stale 0016 (no 0018/0019) while the
lane looked "deployed" — silently applying the wrong migration SQL until an
out-of-band rsync from the canonical clone fixed it.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "check_deployed_migration_tree_sync.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_deployed_migration_tree_sync", _SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def clone(tmp_path: Path) -> Path:
    """A git clone with a forward-migration tree committed at HEAD."""
    root = tmp_path / "clone"
    tree = root / "docker" / "migrations" / "forward"
    tree.mkdir(parents=True)
    (tree / "0016_a.sql").write_text("-- 0016\nSELECT 1;\n")
    (tree / "0018_b.sql").write_text("-- 0018\nSELECT 2;\n")
    (tree / "nodes" / "savings").mkdir(parents=True)
    (tree / "nodes" / "savings" / "view.sql").write_text("-- savings\nSELECT 3;\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "migrations")
    return root


def _deployed_from(clone: Path, tmp_path: Path) -> Path:
    """An in-sync deployed tree copied byte-for-byte from the clone HEAD tree."""
    import shutil

    src = clone / "docker" / "migrations" / "forward"
    dst = tmp_path / "deployed" / "forward"
    shutil.copytree(src, dst)
    return dst


@pytest.mark.unit
def test_in_sync_passes(mod, clone, tmp_path):
    deployed = _deployed_from(clone, tmp_path)
    assert (
        mod.main(
            [
                "--deployed-tree",
                str(deployed),
                "--clone-root",
                str(clone),
                "--ref",
                "HEAD",
            ]
        )
        == 0
    )


@pytest.mark.unit
def test_stale_file_fails(mod, clone, tmp_path):
    """The literal OMN-13415 footgun: a deployed file with stale bytes."""
    deployed = _deployed_from(clone, tmp_path)
    (deployed / "0016_a.sql").write_text("-- STALE 0016\nSELECT 999;\n")
    assert (
        mod.main(
            [
                "--deployed-tree",
                str(deployed),
                "--clone-root",
                str(clone),
                "--ref",
                "HEAD",
            ]
        )
        == 1
    )


@pytest.mark.unit
def test_missing_file_fails(mod, clone, tmp_path):
    """A migration present in the clone but absent from the deployed tree (the
    missing-0018/0019 case)."""
    deployed = _deployed_from(clone, tmp_path)
    (deployed / "0018_b.sql").unlink()
    assert (
        mod.main(
            [
                "--deployed-tree",
                str(deployed),
                "--clone-root",
                str(clone),
                "--ref",
                "HEAD",
            ]
        )
        == 1
    )


@pytest.mark.unit
def test_extra_stale_leftover_fails(mod, clone, tmp_path):
    """A stale leftover file in the deployed tree absent from the clone @ ref."""
    deployed = _deployed_from(clone, tmp_path)
    (deployed / "0099_orphan.sql").write_text("-- orphan\n")
    assert (
        mod.main(
            [
                "--deployed-tree",
                str(deployed),
                "--clone-root",
                str(clone),
                "--ref",
                "HEAD",
            ]
        )
        == 1
    )


@pytest.mark.unit
def test_nested_node_migration_in_sync(mod, clone, tmp_path):
    """Nested node-owned migrations (docker/migrations/forward/nodes/<node>/) are
    compared too."""
    deployed = _deployed_from(clone, tmp_path)
    # mutate the nested file -> must fail
    (deployed / "nodes" / "savings" / "view.sql").write_text("-- drift\n")
    assert (
        mod.main(
            [
                "--deployed-tree",
                str(deployed),
                "--clone-root",
                str(clone),
                "--ref",
                "HEAD",
            ]
        )
        == 1
    )


@pytest.mark.unit
def test_bad_ref_is_config_error(mod, clone, tmp_path):
    deployed = _deployed_from(clone, tmp_path)
    assert (
        mod.main(
            [
                "--deployed-tree",
                str(deployed),
                "--clone-root",
                str(clone),
                "--ref",
                "deadbeef",
            ]
        )
        == 2
    )


@pytest.mark.unit
def test_missing_deployed_tree_is_config_error(mod, clone, tmp_path):
    assert (
        mod.main(
            [
                "--deployed-tree",
                str(tmp_path / "does-not-exist"),
                "--clone-root",
                str(clone),
                "--ref",
                "HEAD",
            ]
        )
        == 2
    )


@pytest.mark.unit
def test_non_git_clone_is_config_error(mod, clone, tmp_path):
    deployed = _deployed_from(clone, tmp_path)
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert (
        mod.main(
            [
                "--deployed-tree",
                str(deployed),
                "--clone-root",
                str(not_a_repo),
                "--ref",
                "HEAD",
            ]
        )
        == 2
    )


@pytest.mark.unit
def test_live_subprocess_smoke(clone, tmp_path):
    """Real subprocess run proves the script is importable + wired end to end."""
    import shutil

    src = clone / "docker" / "migrations" / "forward"
    dst = tmp_path / "deployed2" / "forward"
    shutil.copytree(src, dst)
    result = subprocess.run(
        [
            "python",
            str(_SCRIPT),
            "--deployed-tree",
            str(dst),
            "--clone-root",
            str(clone),
            "--ref",
            "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "byte-identical" in result.stdout
