# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20932: the agent resolves the private lane-overlay directory for compose.

Every lane compose file mounts
``${ONEX_LANE_OVERLAY_DIR:?...}/${COMPOSE_PROJECT_NAME}.bifrost.yaml`` and this
repository ships no lab lane overlay. ``_compose_env`` supplies the directory
through the runtime's workspace-config resolver, and leaves it unset when no
directory exists, so compose refuses with the mount's own message rather than
mounting anything else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from deploy_agent import executor


def _overlays(root: Path) -> Path:
    path = root / "config" / "lab" / "lane-overlays"
    path.mkdir(parents=True)
    return path


def test_explicit_directory_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "mine"
    explicit.mkdir()
    _overlays(tmp_path / "owner")
    monkeypatch.setattr(
        executor, "_workspace_config_root", lambda _: tmp_path / "owner"
    )
    env = {"ONEX_LANE_OVERLAY_DIR": str(explicit)}
    assert executor.resolve_lane_overlay_dir(env) == str(explicit.resolve())


def test_explicit_directory_that_does_not_exist_resolves_to_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _overlays(tmp_path / "owner")
    monkeypatch.setattr(
        executor, "_workspace_config_root", lambda _: tmp_path / "owner"
    )
    env = {"ONEX_LANE_OVERLAY_DIR": str(tmp_path / "missing")}
    assert executor.resolve_lane_overlay_dir(env) is None


def test_the_owner_of_the_clone_workspace_holds_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlays = _overlays(tmp_path / "owner")
    repo_dir = tmp_path / "workspace" / "infra-clone"
    seen: list[Path] = []

    def _owner(workspace: Path) -> Path:
        seen.append(workspace)
        return tmp_path / "owner"

    monkeypatch.setattr(executor, "_workspace_config_root", _owner)
    assert executor.resolve_lane_overlay_dir({}, str(repo_dir)) == str(
        overlays.resolve()
    )
    assert seen == [tmp_path / "workspace"]


def test_the_runtime_resolver_honours_the_config_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlays = _overlays(tmp_path / "owner")
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path / "owner"))
    repo_dir = tmp_path / "workspace" / "infra-clone"
    assert executor.resolve_lane_overlay_dir({}, str(repo_dir)) == str(
        overlays.resolve()
    )


def test_no_directory_resolves_to_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "_workspace_config_root", lambda _: tmp_path / "none")
    assert executor.resolve_lane_overlay_dir({}, str(tmp_path / "w" / "repo")) is None


def test_compose_env_carries_the_resolved_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlays = _overlays(tmp_path / "owner")
    monkeypatch.delenv("ONEX_LANE_OVERLAY_DIR", raising=False)
    monkeypatch.setattr(
        executor, "_workspace_config_root", lambda _: tmp_path / "owner"
    )
    assert executor._compose_env()["ONEX_LANE_OVERLAY_DIR"] == str(overlays.resolve())


def test_compose_env_leaves_the_directory_unset_when_none_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(tmp_path / "missing"))
    monkeypatch.setattr(executor, "_workspace_config_root", lambda _: tmp_path / "none")
    assert "ONEX_LANE_OVERLAY_DIR" not in executor._compose_env()
