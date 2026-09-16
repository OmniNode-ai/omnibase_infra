# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed runner fleet configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

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
from omnibase_infra.observability.runner_health.model_runner_fleet_host import (
    ModelRunnerFleetHost,
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
    hosts: tuple[ModelRunnerFleetHost, ...] = Field(
        default=(),
        description=(
            "OMN-17477 — declared multi-host inventory. The scalar runner_host / "
            "runner_name_prefix / expected_count fields above remain the PRIMARY "
            "host's values and are not superseded, so a consumer that reads them "
            "behaves identically to before this field existed. Empty means a "
            "config predating the inventory, which is a single-host fleet."
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

    @model_validator(mode="after")
    def _validate_host_inventory(self) -> ModelRunnerFleetConfig:
        """Reject an inventory that cannot be acted on.

        Both checks below describe failures that are SILENT at runtime, which is
        why they are refused at load rather than discovered on a host.
        """
        if not self.hosts:
            return self

        prefixes = [host.runner_name_prefix for host in self.hosts]
        duplicates = {p for p in prefixes if prefixes.count(p) > 1}
        if duplicates:
            raise ValueError(
                f"runner_name_prefix must be unique across hosts; duplicated: "
                f"{sorted(duplicates)}"
            )
        # A prefix that is a prefix of another is the same collision one step
        # removed: `omninode-runner` matches `omninode-runner-101-1` under the
        # `<prefix>-<N>` pattern every consumer uses, so one host would count
        # another host's runners as its own.
        for outer in prefixes:
            for inner in prefixes:
                if outer != inner and inner.startswith(f"{outer}-"):
                    raise ValueError(
                        f"runner_name_prefix {inner!r} is nested under {outer!r}; "
                        "the `<prefix>-<N>` pattern would match both"
                    )

        hosts_by_name = {host.host: host for host in self.hosts}
        if len(hosts_by_name) != len(self.hosts):
            raise ValueError("each host may appear in the inventory exactly once")

        primary = hosts_by_name.get(self.runner_host)
        if primary is None:
            raise ValueError(
                f"runner_host {self.runner_host!r} is not declared in hosts; the "
                "scalar fields are the primary host's values, so the primary "
                "host must be in the inventory"
            )
        if (
            primary.runner_name_prefix != self.runner_name_prefix
            or primary.expected_count != self.expected_count
        ):
            raise ValueError(
                "the primary host's inventory row must agree with the scalar "
                "runner_name_prefix/expected_count fields; two sources of truth "
                "for one host make a consumer's answer depend on which it reads"
            )
        return self

    def primary_host(self) -> ModelRunnerFleetHost:
        """The host the scalar fields describe.

        Raises rather than inventing a row: a config with an inventory that
        omits its own primary host is refused at validation, so reaching here
        without one is a programming error, not a state to paper over.
        """
        for host in self.hosts:
            if host.host == self.runner_host:
                return host
        raise KeyError(f"primary host {self.runner_host!r} is not in the inventory")

    def declared_total(self, runner_class: str) -> int:
        """Summed declared runner count across hosts carrying ``runner_class``.

        Summed PER CLASS and never across the whole inventory. Capacity is not
        fungible -- a verify-class runner cannot pick up an action-class job --
        so a total that mixed classes would tell the router's degraded floor
        that capacity exists which can never satisfy the jobs the floor guards.

        A class nothing declares returns 0. That is the honest answer and it is
        also the safe one: a floor computed from 0 never reads a missing fleet
        as healthy.
        """
        if not self.hosts:
            # Pre-inventory config: one host, all classes, the scalar count.
            return self.expected_count
        return sum(
            host.expected_count for host in self.hosts if runner_class in host.classes
        )

    def hosts_for_arch(self, arch: str) -> tuple[ModelRunnerFleetHost, ...]:
        """Inventory rows on a given CPU architecture."""
        return tuple(host for host in self.hosts if host.arch.value == arch)


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
    "ModelRunnerFleetHost",
    "default_runner_fleet_config_path",
    "load_runner_fleet_config",
]
