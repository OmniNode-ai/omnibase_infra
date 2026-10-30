# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20932: the agent resolves the private lane-overlay directory for compose.

Every lane compose file mounts
``${ONEX_LANE_OVERLAY_DIR:?...}/${COMPOSE_PROJECT_NAME}.bifrost.yaml`` and this
repository ships no lab lane overlay. ``_compose_env`` supplies the directory
from ``ONEX_WORKSPACE_CONFIG_ROOT``, and leaves it unset when no
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
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path / "owner"))
    env = {"ONEX_LANE_OVERLAY_DIR": str(explicit)}
    assert executor.resolve_lane_overlay_dir(env) == str(explicit.resolve())


def test_explicit_directory_that_does_not_exist_resolves_to_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _overlays(tmp_path / "owner")
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path / "owner"))
    env = {"ONEX_LANE_OVERLAY_DIR": str(tmp_path / "missing")}
    assert executor.resolve_lane_overlay_dir(env) is None


def test_the_declared_config_root_holds_the_directory(tmp_path: Path) -> None:
    overlays = _overlays(tmp_path / "owner")
    env = {"ONEX_WORKSPACE_CONFIG_ROOT": str(tmp_path / "owner")}
    assert executor.resolve_lane_overlay_dir(env) == str(overlays.resolve())


def test_no_declared_root_adopts_nothing_beside_the_clone_or_in_the_process_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for neighbour in ("config", "neighbor", "workspace-config", "owner"):
        _overlays(tmp_path / neighbour)
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path / "owner"))
    assert executor.resolve_lane_overlay_dir({}) is None
    assert (
        executor.resolve_lane_overlay_dir({"ONEX_WORKSPACE_CONFIG_ROOT": " "}) is None
    )


def test_a_declared_root_without_the_directory_resolves_to_nothing(
    tmp_path: Path,
) -> None:
    env = {"ONEX_WORKSPACE_CONFIG_ROOT": str(tmp_path / "none")}
    assert executor.resolve_lane_overlay_dir(env) is None


def test_compose_env_carries_the_resolved_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    overlays = _overlays(tmp_path / "owner")
    monkeypatch.delenv("ONEX_LANE_OVERLAY_DIR", raising=False)
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path / "owner"))
    assert executor._compose_env()["ONEX_LANE_OVERLAY_DIR"] == str(overlays.resolve())


def test_compose_env_leaves_the_directory_unset_when_none_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ONEX_LANE_OVERLAY_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("ONEX_WORKSPACE_CONFIG_ROOT", str(tmp_path / "none"))
    assert "ONEX_LANE_OVERLAY_DIR" not in executor._compose_env()


def test_the_instance_env_template_names_the_config_root_the_agent_reads() -> None:
    """The agent takes the overlay directory from ONEX_WORKSPACE_CONFIG_ROOT in the
    environment it hands to compose, so each instance's env file must define it."""
    template = Path(executor.__file__).resolve().parents[1] / "deploy"
    names = {
        line.partition("=")[0]
        for line in (template / "dev-instance.env.template").read_text().splitlines()
        if line and not line.startswith("#")
    }
    assert {"ONEX_WORKSPACE_CONFIG_ROOT", "ONEX_LANE_OVERLAY_DIR"} <= names
