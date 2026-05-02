# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed runner fleet configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelRunnerFleetConfig(BaseModel):
    """Authoritative configuration for the self-hosted runner fleet."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    version: str = Field(..., description="Runner fleet config schema version")
    github_org: str = Field(..., min_length=1)
    runner_host: str = Field(..., min_length=1)
    runner_group: str = Field(..., min_length=1)
    runner_name_prefix: str = Field(..., min_length=1)
    expected_count: int = Field(..., ge=1)


def default_runner_fleet_config_path() -> Path:
    """Return the default repo-local runner fleet config path."""
    env_path = os.environ.get("RUNNER_FLEET_CONFIG_PATH", "")
    if env_path:
        return Path(env_path).expanduser()
    return Path.cwd() / "config" / "runner_fleet.yaml"


def load_runner_fleet_config(path: Path | None = None) -> ModelRunnerFleetConfig:
    """Load and validate runner fleet config.

    The config file is required; missing config is a deployment error, not a
    signal to fall back to embedded lab values.
    """
    config_path = path or default_runner_fleet_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"Runner fleet config not found: {config_path}")

    raw = cast("object", yaml.safe_load(config_path.read_text(encoding="utf-8")) or {})
    return ModelRunnerFleetConfig.model_validate(raw)


__all__ = [
    "ModelRunnerFleetConfig",
    "default_runner_fleet_config_path",
    "load_runner_fleet_config",
]
