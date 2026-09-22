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
import os
import subprocess
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "check_deployed_migration_tree_sync.py"
)
_DEPLOY_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy-runtime.sh"


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
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        env=scrub_git_location_env(os.environ),
    )


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


def _source_tree(clone: Path) -> Path:
    return clone / "docker" / "migrations" / "forward"


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
def test_frozen_source_tree_in_sync_passes_including_ledger(mod, clone, tmp_path):
    """A hotpatch attests the exact source tree, including its migration ledger."""
    source = _source_tree(clone)
    (source / "_ledger").mkdir()
    (source / "_ledger" / "application-migrations.tsv").write_text("0044\t3a108929\n")
    deployed = _deployed_from(clone, tmp_path)
    # _deployed_from ran before the ledger was added, so copy the frozen source
    # after adding the ledger to model a fresh source sync.
    import shutil

    shutil.rmtree(deployed)
    shutil.copytree(source, deployed)
    assert (
        mod.main(["--deployed-tree", str(deployed), "--source-tree", str(source)]) == 0
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("relative_path", "content"),
    [
        ("0016_a.sql", "-- tampered\n"),
        ("_ledger/application-migrations.tsv", "0044\twrong\n"),
    ],
)
def test_frozen_source_tree_stale_or_modified_file_fails(
    mod, clone, tmp_path, relative_path, content
):
    source = _source_tree(clone)
    (source / "_ledger").mkdir()
    (source / "_ledger" / "application-migrations.tsv").write_text("0044\tgood\n")
    import shutil

    deployed = tmp_path / "deployed" / "forward"
    shutil.copytree(source, deployed)
    (deployed / relative_path).write_text(content)
    assert (
        mod.main(["--deployed-tree", str(deployed), "--source-tree", str(source)]) == 1
    )


@pytest.mark.unit
def test_frozen_source_tree_missing_or_extra_file_fails(mod, clone, tmp_path):
    source = _source_tree(clone)
    deployed = _deployed_from(clone, tmp_path)
    (deployed / "0018_b.sql").unlink()
    (deployed / "0099_orphan.sql").write_text("-- orphan\n")
    assert (
        mod.main(["--deployed-tree", str(deployed), "--source-tree", str(source)]) == 1
    )


@pytest.mark.unit
def test_source_symlink_is_configuration_error(mod, clone, tmp_path):
    source = _source_tree(clone)
    deployed = _deployed_from(clone, tmp_path)
    (source / "escape.sql").symlink_to(tmp_path / "outside.sql")
    assert (
        mod.main(["--deployed-tree", str(deployed), "--source-tree", str(source)]) == 2
    )


@pytest.mark.unit
def test_deployed_symlink_is_drift(mod, clone, tmp_path):
    source = _source_tree(clone)
    deployed = _deployed_from(clone, tmp_path)
    (deployed / "escape.sql").symlink_to(tmp_path / "outside.sql")
    assert (
        mod.main(["--deployed-tree", str(deployed), "--source-tree", str(source)]) == 1
    )


@pytest.mark.unit
def test_deployed_non_regular_entry_is_drift(mod, clone, tmp_path):
    source = _source_tree(clone)
    deployed = _deployed_from(clone, tmp_path)
    fifo = deployed / "unexpected.pipe"
    os.mkfifo(fifo)
    assert (
        mod.main(["--deployed-tree", str(deployed), "--source-tree", str(source)]) == 1
    )


@pytest.mark.unit
def test_source_non_regular_entry_is_configuration_error(mod, clone, tmp_path):
    source = _source_tree(clone)
    deployed = _deployed_from(clone, tmp_path)
    os.mkfifo(source / "unexpected.pipe")
    assert (
        mod.main(["--deployed-tree", str(deployed), "--source-tree", str(source)]) == 2
    )


@pytest.mark.unit
def test_source_tree_cannot_be_deployed_destination(mod, clone):
    source = _source_tree(clone)
    assert mod.main(["--deployed-tree", str(source), "--source-tree", str(source)]) == 2


@pytest.mark.unit
def test_modes_are_mutually_exclusive(mod, clone, tmp_path):
    deployed = _deployed_from(clone, tmp_path)
    with pytest.raises(SystemExit):
        mod.main(
            [
                "--deployed-tree",
                str(deployed),
                "--clone-root",
                str(clone),
                "--source-tree",
                str(_source_tree(clone)),
            ]
        )


@pytest.mark.unit
def test_hotpatch_wiring_uses_frozen_source_and_clean_wiring_is_retained():
    """The hotpatch exception is explicit; the canonical SHA gate remains."""
    text = _DEPLOY_SCRIPT.read_text()
    start = text.index("assert_deployed_migration_tree_synced() {")
    end = text.index("\nsnapshot_migration_tree() {", start)
    function = text[start:end]
    assert '[[ "${DEPLOY_HOTPATCH:-0}" == "1" ]]' in function
    assert '--source-tree "${frozen_source_tree}"' in function
    assert '--clone-root "${repo_root}"' in function
    assert '--ref "${git_sha}"' in function


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
