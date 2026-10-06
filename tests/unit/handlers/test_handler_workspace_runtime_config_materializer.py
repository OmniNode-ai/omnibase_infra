# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A workspace's runtime config is materialised from ``origin/main`` (OMN-19212).

THE DEFECT. A bound workspace root resolved its transport from
``<root>/config/onex/runtime/runtime_config.yaml`` in the SHARED working tree.
That tree lags ``origin/main`` whenever a peer has work staged, so the file was
absent and every default ``onex delegate`` was refused.

THE SHAPE. The handler reads the file out of the git object database
(``git show origin/main:<path>``) and writes it, with a sha/time sidecar, under
``<root>/.onex_state/workspace-runtime/``. It never touches the index or the
working tree of the repository it reads (AC2).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from omnibase_core.validators.no_unguarded_git_subprocess import (
    scrub_git_location_env,
)
from omnibase_infra.handlers.handler_workspace_runtime_config_materializer import (
    MATERIALIZED_CONTRACTS_RELATIVE_PATH,
    MATERIALIZED_SIDECAR_NAME,
    SOURCE_PATH_IN_REPO,
    STALE_AFTER,
    HandlerWorkspaceRuntimeConfigMaterializer,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def config_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path))


_TIER1 = 'event_bus:\n  type: "kafka"\n  profile: "local"\n  lane: "dev"\n'


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=scrub_git_location_env(),
    )
    return result.stdout


def _lagging_workspace(root: Path) -> str:
    """A repo whose ``origin/main`` carries the config and whose HEAD does not.

    Returns the sha ``origin/main`` points at.
    """
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    # A registry workspace ignores its own state directory (.gitignore there).
    (root / ".git" / "info" / "exclude").write_text(".onex_state/\n", encoding="utf-8")
    (root / "notes.txt").write_text("base\n", encoding="utf-8")
    (root / "other.txt").write_text("base\n", encoding="utf-8")
    config = root / SOURCE_PATH_IN_REPO
    config.parent.mkdir(parents=True)
    config.write_text(_TIER1, encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "tier-1 config lands")
    sha = _git(root, "rev-parse", "HEAD").strip()
    _git(root, "update-ref", "refs/remotes/origin/main", sha)
    # The shared tree is behind: HEAD no longer carries the file.
    _git(root, "rm", "-q", SOURCE_PATH_IN_REPO)
    _git(root, "commit", "-q", "-m", "tree lags origin/main")
    assert not config.exists()
    return sha


def _tree_listing(root: Path) -> dict[str, str]:
    """Every file outside ``.git``, by relative path, with its content hash."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file() and ".git" not in path.relative_to(root).parts
    }


class TestMaterialize:
    """AC1: the file comes out of ``origin/main`` with a sha/time sidecar."""

    def test_the_config_is_written_from_origin_main_with_its_sha(
        self, tmp_path: Path
    ) -> None:
        sha = _lagging_workspace(tmp_path)
        outcome = HandlerWorkspaceRuntimeConfigMaterializer().materialize(tmp_path)
        assert outcome.ok is True
        assert outcome.sha == sha
        written = (
            tmp_path
            / MATERIALIZED_CONTRACTS_RELATIVE_PATH
            / "runtime"
            / "runtime_config.yaml"
        )
        assert written.read_text(encoding="utf-8") == _TIER1
        assert (tmp_path / MATERIALIZED_CONTRACTS_RELATIVE_PATH / "runtime").is_dir()

    def test_read_returns_the_copy_with_sha_and_a_fresh_time(
        self, tmp_path: Path
    ) -> None:
        sha = _lagging_workspace(tmp_path)
        handler = HandlerWorkspaceRuntimeConfigMaterializer()
        handler.materialize(tmp_path)
        copy = handler.read(tmp_path)
        assert copy is not None
        assert copy.sha == sha
        assert copy.contracts_dir == tmp_path / MATERIALIZED_CONTRACTS_RELATIVE_PATH
        assert copy.stale is False
        assert abs(datetime.now(UTC) - copy.materialized_at) < timedelta(minutes=5)

    def test_a_copy_older_than_the_window_reads_as_stale(self, tmp_path: Path) -> None:
        _lagging_workspace(tmp_path)
        handler = HandlerWorkspaceRuntimeConfigMaterializer()
        handler.materialize(tmp_path)
        later = datetime.now(UTC) + STALE_AFTER + timedelta(minutes=1)
        copy = handler.read(tmp_path, now=later)
        assert copy is not None
        assert copy.stale is True

    def test_a_second_run_follows_origin_main(self, tmp_path: Path) -> None:
        _lagging_workspace(tmp_path)
        handler = HandlerWorkspaceRuntimeConfigMaterializer()
        handler.materialize(tmp_path)
        _git(tmp_path, "checkout", "-q", "--detach")
        updated = _TIER1.replace('"dev"', '"stability-test"')
        (tmp_path / SOURCE_PATH_IN_REPO).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / SOURCE_PATH_IN_REPO).write_text(updated, encoding="utf-8")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-q", "-m", "lane moves")
        new_sha = _git(tmp_path, "rev-parse", "HEAD").strip()
        _git(tmp_path, "update-ref", "refs/remotes/origin/main", new_sha)
        outcome = handler.materialize(tmp_path)
        copy = handler.read(tmp_path)
        assert outcome.sha == new_sha
        assert copy is not None
        assert copy.sha == new_sha
        assert (copy.contracts_dir / "runtime" / "runtime_config.yaml").read_text(
            encoding="utf-8"
        ) == updated


class TestWhatItNeverDoes:
    """AC2: the index and the working tree of a repo with staged changes stay put."""

    def test_staged_unstaged_and_untracked_work_is_left_exactly_as_found(
        self, tmp_path: Path
    ) -> None:
        _lagging_workspace(tmp_path)
        # A peer's work in flight: one staged edit, one unstaged edit, one new file.
        (tmp_path / "notes.txt").write_text("staged by a peer\n", encoding="utf-8")
        _git(tmp_path, "add", "notes.txt")
        (tmp_path / "other.txt").write_text("unstaged by a peer\n", encoding="utf-8")
        (tmp_path / "untracked.txt").write_text("not added\n", encoding="utf-8")
        # Settle any stat refresh BEFORE the snapshot so the byte comparison of
        # the index below is a comparison of the handler's effect alone.
        status_before = _git(tmp_path, "status", "--porcelain=v1")
        cached_before = _git(tmp_path, "diff", "--cached")
        unstaged_before = _git(tmp_path, "diff")
        head_before = _git(tmp_path, "rev-parse", "HEAD")
        index_before = (tmp_path / ".git" / "index").read_bytes()
        tree_before = _tree_listing(tmp_path)

        outcome = HandlerWorkspaceRuntimeConfigMaterializer().materialize(tmp_path)

        assert outcome.ok is True
        assert (tmp_path / ".git" / "index").read_bytes() == index_before
        assert _git(tmp_path, "rev-parse", "HEAD") == head_before
        tree_after = _tree_listing(tmp_path)
        added = set(tree_after) - set(tree_before)
        assert added, "positive control: the handler wrote something"
        assert all(path.startswith(".onex_state/") for path in added), added
        assert {k: v for k, v in tree_after.items() if k in tree_before} == tree_before
        assert _git(tmp_path, "status", "--porcelain=v1") == status_before
        assert _git(tmp_path, "diff", "--cached") == cached_before
        assert _git(tmp_path, "diff") == unstaged_before
        # The positive control for the comparison itself: the staged edit is
        # really there to be disturbed.
        assert "staged by a peer" in cached_before


class TestWhatItReportsInsteadOfRaising:
    """A materialiser that cannot run reports why and leaves a prior copy alone."""

    def test_a_directory_that_is_not_a_repository(self, tmp_path: Path) -> None:
        outcome = HandlerWorkspaceRuntimeConfigMaterializer().materialize(tmp_path)
        assert outcome.ok is False
        assert outcome.detail
        assert not (tmp_path / MATERIALIZED_CONTRACTS_RELATIVE_PATH).exists()

    def test_a_repository_whose_origin_main_has_no_config(self, tmp_path: Path) -> None:
        tmp_path.mkdir(exist_ok=True)
        _git(tmp_path, "init", "-q", "-b", "main")
        (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
        _git(tmp_path, "add", "-A")
        _git(tmp_path, "commit", "-q", "-m", "no config here")
        _git(
            tmp_path,
            "update-ref",
            "refs/remotes/origin/main",
            _git(tmp_path, "rev-parse", "HEAD").strip(),
        )
        outcome = HandlerWorkspaceRuntimeConfigMaterializer().materialize(tmp_path)
        assert outcome.ok is False
        assert SOURCE_PATH_IN_REPO in outcome.detail

    def test_a_failed_run_keeps_the_earlier_copy(self, tmp_path: Path) -> None:
        _lagging_workspace(tmp_path)
        handler = HandlerWorkspaceRuntimeConfigMaterializer()
        assert handler.materialize(tmp_path).ok is True
        _git(tmp_path, "update-ref", "-d", "refs/remotes/origin/main")
        assert handler.materialize(tmp_path).ok is False
        copy = handler.read(tmp_path)
        assert copy is not None
        assert (copy.contracts_dir / "runtime" / "runtime_config.yaml").read_text(
            encoding="utf-8"
        ) == _TIER1

    def test_a_subdirectory_of_a_repository_is_not_the_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _lagging_workspace(tmp_path)
        inner = tmp_path / "not-the-root"
        inner.mkdir()
        monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(inner))
        outcome = HandlerWorkspaceRuntimeConfigMaterializer().materialize(inner)
        assert outcome.ok is False
        assert not (inner / MATERIALIZED_CONTRACTS_RELATIVE_PATH).exists()


class TestRead:
    def test_no_copy_reads_as_none(self, tmp_path: Path) -> None:
        assert HandlerWorkspaceRuntimeConfigMaterializer().read(tmp_path) is None

    def test_a_copy_with_an_unreadable_sidecar_is_not_attributable(
        self, tmp_path: Path
    ) -> None:
        _lagging_workspace(tmp_path)
        handler = HandlerWorkspaceRuntimeConfigMaterializer()
        handler.materialize(tmp_path)
        sidecar = (
            tmp_path
            / MATERIALIZED_CONTRACTS_RELATIVE_PATH
            / "runtime"
            / MATERIALIZED_SIDECAR_NAME
        )
        sidecar.write_text("{not json", encoding="utf-8")
        assert handler.read(tmp_path) is None

    def test_a_sidecar_with_no_config_beside_it_reads_as_none(
        self, tmp_path: Path
    ) -> None:
        _lagging_workspace(tmp_path)
        handler = HandlerWorkspaceRuntimeConfigMaterializer()
        handler.materialize(tmp_path)
        (
            tmp_path
            / MATERIALIZED_CONTRACTS_RELATIVE_PATH
            / "runtime"
            / "runtime_config.yaml"
        ).unlink()
        assert handler.read(tmp_path) is None


def test_config_is_read_from_owning_sibling_not_retiring_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OMN-19743: a poison config in the retiring root cannot answer."""
    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    workspace = tmp_path / "omni_home"
    owner = tmp_path / "omnibase_internal"
    workspace.mkdir()
    sha = _lagging_workspace(owner)
    legacy = workspace / SOURCE_PATH_IN_REPO
    legacy.parent.mkdir(parents=True)
    legacy.write_text(_TIER1.replace('"dev"', '"stability-test"'), encoding="utf-8")
    handler = HandlerWorkspaceRuntimeConfigMaterializer()
    outcome = handler.materialize(workspace)
    assert outcome.ok, outcome.detail
    assert outcome.sha == sha
    copy = handler.read(workspace)
    assert copy is not None
    assert copy.config_path.read_text(encoding="utf-8") == _TIER1


def test_missing_owner_does_not_materialize_retiring_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ONEX_WORKSPACE_CONFIG_ROOT", raising=False)
    workspace = tmp_path / "omni_home"
    _lagging_workspace(workspace)
    outcome = HandlerWorkspaceRuntimeConfigMaterializer().materialize(workspace)
    assert not outcome.ok
    assert "omnibase_internal" in outcome.detail


@pytest.mark.parametrize("owner", [None, "retired"])
def test_copy_from_retiring_owner_is_not_attributable(
    tmp_path: Path, owner: str | None
) -> None:
    _lagging_workspace(tmp_path)
    handler = HandlerWorkspaceRuntimeConfigMaterializer()
    assert handler.materialize(tmp_path).ok
    sidecar = (
        tmp_path
        / MATERIALIZED_CONTRACTS_RELATIVE_PATH
        / "runtime"
        / MATERIALIZED_SIDECAR_NAME
    )
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if owner is None:
        data.pop("source_repository")
    else:
        data["source_repository"] = str(tmp_path / owner)
    sidecar.write_text(json.dumps(data), encoding="utf-8")
    assert handler.read(tmp_path) is None
