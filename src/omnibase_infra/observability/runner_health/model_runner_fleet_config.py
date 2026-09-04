# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed runner fleet configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.runner_health.model_dns_cache_config import (
    ModelDnsCacheConfig,
)
from omnibase_infra.observability.runner_health.model_git_mirror_config import (
    ModelGitMirrorConfig,
)
from omnibase_infra.observability.runner_health.model_model_review_capability_config import (
    ModelModelReviewCapabilityConfig,
)
from omnibase_infra.observability.runner_health.model_pypi_cache_config import (
    ModelPyPICacheConfig,
)
from omnibase_infra.observability.runner_health.model_tool_cache_config import (
    ModelToolCacheConfig,
)


class ModelRunnerFleetConfig(BaseModel):
    """Authoritative configuration for the self-hosted runner fleet."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    version: str = Field(..., description="Runner fleet config schema version")
    github_org: str = Field(..., min_length=1)
    runner_host: str = Field(..., min_length=1)
    runner_group: str = Field(..., min_length=1)
    runner_name_prefix: str = Field(..., min_length=1)
    expected_count: int = Field(..., ge=1)
    burst_count: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum runner count enabled only by the compose burst profile.",
    )
    network_pool_capacity: int = Field(
        default=31,
        gt=0,
        description=(
            "Max Docker networks the runner host's address pool can subnet "
            "before exhaustion (OMN-12566). Drives subnet-pool alerting."
        ),
    )
    network_pool_warn_ratio: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        description=(
            "Fraction of network_pool_capacity at which to alert before the "
            "subnet pool is exhausted (OMN-12566)."
        ),
    )
    wedge_queue_age_seconds: int = Field(
        default=600,
        ge=0,
        description="Queued-run age threshold for runner-fleet wedge classification.",
    )
    codeload_scan_limit: int = Field(
        default=5,
        ge=1,
        description="Recent failed runs per watched repo scanned for codeload throttling.",
    )
    watch_repos: tuple[str, ...] = Field(
        default=(),
        description="Repos watched for queued/zombie runs; empty uses the built-in OmniNode defaults.",
    )
    pypi_cache: ModelPyPICacheConfig | None = Field(
        default=None,
        description=(
            "OMN-14027 C1 — PyPI pull-through cache endpoint (devpi). Optional "
            "and inert until the soak-gated rollout sets active=True and wires "
            "the runner env. Absent in configs predating the egress-cache work."
        ),
    )
    git_mirror: ModelGitMirrorConfig | None = Field(
        default=None,
        description=(
            "OMN-16053 (OMN-14027 C2) — host-local bare git mirrors + fail-open "
            "job-workspace pre-seed. Optional; absent in configs predating the "
            "git-transport egress work."
        ),
    )
    tool_cache: ModelToolCacheConfig | None = Field(
        default=None,
        description=(
            "OMN-16053 (OMN-14027 C2) — Actions tool-cache durability record. "
            "Optional; absent in configs predating the git-transport egress work."
        ),
    )
    dns_cache: ModelDnsCacheConfig | None = Field(
        default=None,
        description=(
            "OMN-15736 — local caching DNS resolver (unbound) endpoint. "
            "Optional and inert until the operator-gated rollout sets "
            "active=True and repoints canary runners' `dns:` directive. "
            "Absent in configs predating this work."
        ),
    )
    model_review: ModelModelReviewCapabilityConfig | None = Field(
        default=None,
        description=(
            "OMN-17855 — opaque model-review runner-overlay contract. Optional "
            "and inactive by default; this record does not provision or activate "
            "any runner."
        ),
    )


def default_runner_fleet_config_path() -> Path:
    """Return the default repo-local runner fleet config path."""
    env_path = os.environ.get("RUNNER_FLEET_CONFIG_PATH", "")
    if env_path:
        return Path(env_path).expanduser()
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "config" / "runner_fleet.yaml"


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
