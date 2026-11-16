# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the contract hot-load lane overlay (OMN-21318)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.models.model_contract_hotload_config import (
    ModelContractHotloadConfig,
)
from omnibase_infra.models.model_contract_hotload_lane_overlay import (
    ModelContractHotloadLaneOverlay,
)

pytestmark = pytest.mark.unit


def _write_overlay(root: Path, lanes: dict[str, object]) -> None:
    """Write a lane overlay file under ``root``."""
    overlay_dir = root / "runtime_contract_hotload"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (overlay_dir / "overlay.yaml").write_text(
        yaml.safe_dump(
            {"schema_version": 1, "lanes": lanes},
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_no_overlay_root_gives_none(tmp_path: Path) -> None:
    """No ``ONEX_SKILL_OVERLAY_ROOTS`` means no roots and no lane row."""
    assert ModelContractHotloadLaneOverlay.for_lane({}, "lane-a") is None


def test_lane_without_row_gives_none(tmp_path: Path) -> None:
    """A lane with no row in the overlay file gives None, not another lane's row."""
    _write_overlay(
        tmp_path,
        {
            "lane-a": {
                "enabled": True,
                "mode": "on_demand",
                "allowed_environments": ["env-x"],
            }
        },
    )
    environ = {"ONEX_SKILL_OVERLAY_ROOTS": str(tmp_path)}
    assert ModelContractHotloadLaneOverlay.for_lane(environ, "lane-b") is None


def test_lane_row_is_returned(tmp_path: Path) -> None:
    """A lane's row is returned as a validated ``ModelContractHotloadConfig``."""
    _write_overlay(
        tmp_path,
        {
            "lane-a": {
                "enabled": True,
                "mode": "on_demand",
                "allowed_environments": ["env-x"],
            }
        },
    )
    environ = {"ONEX_SKILL_OVERLAY_ROOTS": str(tmp_path)}
    result = ModelContractHotloadLaneOverlay.for_lane(environ, "lane-a")
    assert result == ModelContractHotloadConfig(
        enabled=True, mode="on_demand", allowed_environments=("env-x",)
    )


def test_first_root_wins(tmp_path: Path) -> None:
    """The earlier root's file answers when both roots hold one."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_overlay(
        root_a,
        {
            "lane-a": {
                "enabled": True,
                "mode": "on_demand",
                "allowed_environments": ["env-x"],
            }
        },
    )
    _write_overlay(
        root_b,
        {
            "lane-a": {
                "enabled": True,
                "mode": "full_load",
                "allowed_environments": ["env-y"],
            }
        },
    )
    environ = {"ONEX_SKILL_OVERLAY_ROOTS": os.pathsep.join([str(root_a), str(root_b)])}
    result = ModelContractHotloadLaneOverlay.for_lane(environ, "lane-a")
    assert result == ModelContractHotloadConfig(
        enabled=True, mode="on_demand", allowed_environments=("env-x",)
    )


def test_root_without_file_is_skipped(tmp_path: Path) -> None:
    """A root with no overlay file is skipped; the next root's file is read."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    _write_overlay(
        root_b,
        {
            "lane-a": {
                "enabled": True,
                "mode": "on_demand",
                "allowed_environments": ["env-x"],
            }
        },
    )
    environ = {"ONEX_SKILL_OVERLAY_ROOTS": os.pathsep.join([str(root_a), str(root_b)])}
    result = ModelContractHotloadLaneOverlay.for_lane(environ, "lane-a")
    assert result == ModelContractHotloadConfig(
        enabled=True, mode="on_demand", allowed_environments=("env-x",)
    )


def test_invalid_file_raises(tmp_path: Path) -> None:
    """A lane row with enabled true and no mode raises ``ProtocolConfigurationError``."""
    _write_overlay(tmp_path, {"lane-a": {"enabled": True}})
    environ = {"ONEX_SKILL_OVERLAY_ROOTS": str(tmp_path)}
    with pytest.raises(ProtocolConfigurationError):
        ModelContractHotloadLaneOverlay.for_lane(environ, "lane-a")


def test_unknown_field_raises(tmp_path: Path) -> None:
    """An extra key in a lane row raises ``ProtocolConfigurationError``."""
    _write_overlay(
        tmp_path,
        {
            "lane-a": {
                "enabled": True,
                "mode": "on_demand",
                "allowed_environments": ["env-x"],
                "bogus": 1,
            }
        },
    )
    environ = {"ONEX_SKILL_OVERLAY_ROOTS": str(tmp_path)}
    with pytest.raises(ProtocolConfigurationError):
        ModelContractHotloadLaneOverlay.for_lane(environ, "lane-a")


def test_blank_lane_gives_none(tmp_path: Path) -> None:
    """A blank lane name gives None without reading any overlay."""
    _write_overlay(
        tmp_path,
        {
            "lane-a": {
                "enabled": True,
                "mode": "on_demand",
                "allowed_environments": ["env-x"],
            }
        },
    )
    environ = {"ONEX_SKILL_OVERLAY_ROOTS": str(tmp_path)}
    assert ModelContractHotloadLaneOverlay.for_lane(environ, "   ") is None
