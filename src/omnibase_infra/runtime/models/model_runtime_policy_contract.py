# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime policy contract model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from omnibase_infra.event_bus.models.config.model_kafka_consumer_fetch_budget import (
    ModelKafkaConsumerFetchBudget,
)
from omnibase_infra.runtime.models.model_runtime_profile_policy import (
    ModelRuntimeProfilePolicy,
)

# OMN-17150: ``lakshman`` is a fifth, COLLABORATOR-class lane. It is dev-class
# (fully mutable, owned by one external collaborator), never stability-class:
# it is deliberately absent from ``preflight_lane_deploy_attribution``'s
# GOVERNED_LANES and GRANT_INTERLOCK_LANES, and from omni_home's
# no-raw-prod-bypass matcher. No promotion grant may resolve against it.
# tests/ci/test_lakshman_lane_governance_boundary.py pins that, both ways.
RuntimeProfileName = Literal["dev", "stability-test", "judge", "prod", "lakshman"]


class ModelRuntimePolicyContract(BaseModel):
    """Contract-owned runtime policy rendered into deployment env."""

    model_config = ConfigDict(extra="forbid", frozen=True, from_attributes=True)

    name: Literal["runtime_policy"]
    version: int = Field(ge=1)
    active_runtime_packages: tuple[str, ...] = Field(min_length=1)
    llm_cloud_endpoint_host_allowlist: tuple[str, ...] = Field(default=())
    bifrost_vertex_gemini_endpoint_url: str = Field(min_length=1)
    google_cloud_project: str = Field(min_length=1)
    google_cloud_location: str = Field(min_length=1)
    omnimemory_memgraph_port: int = Field(ge=1, le=65535)
    # OMN-14297: arch-graph query/populate EFFECT nodes read a single flat
    # ${env.ARCH_GRAPH_BOLT_URI} (no per-lane prefix — see contract.yaml of
    # node_architecture_graph_{populate,query}_effect), so this is rendered
    # as one top-level var, matching omnimemory_memgraph_port's pattern
    # rather than the per-profile omnimemory_memgraph_host pattern.
    arch_graph_bolt_uri: str = Field(min_length=1)
    auxiliary_services_omnimemory_enabled: bool
    # OMN-17888: the aggregate Kafka consumer fetch-memory bound. Lane-invariant
    # today because every runtime lane runs under the same 1536M container
    # limit, and the bound divides that limit rather than restating it -- the
    # limit itself is read live from the cgroup, so a lane whose compose limit
    # changes gets a correspondingly smaller/larger bound with no edit here.
    # Rendered as ONEX_KAFKA_CONSUMER_FETCH_BUDGET_JSON and resolved fail-closed
    # by backends.auto_configure.select_event_bus.
    kafka_consumer_fetch_budget: ModelKafkaConsumerFetchBudget
    profiles: dict[RuntimeProfileName, ModelRuntimeProfilePolicy]

    @field_validator("active_runtime_packages")
    @classmethod
    def _packages_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            msg = "active runtime packages must be unique"
            raise ValueError(msg)
        return value

    @field_validator("llm_cloud_endpoint_host_allowlist")
    @classmethod
    def _cloud_hosts_are_exact_hostnames(
        cls, value: tuple[str, ...]
    ) -> tuple[str, ...]:
        normalized: list[str] = []
        for hostname in value:
            host = hostname.strip().lower().rstrip(".")
            if not host or "://" in host or "/" in host or ":" in host:
                msg = (
                    "llm_cloud_endpoint_host_allowlist entries must be bare "
                    f"hostnames, got {hostname!r}"
                )
                raise ValueError(msg)
            normalized.append(host)
        if len(set(normalized)) != len(normalized):
            msg = "llm cloud endpoint host allowlist entries must be unique"
            raise ValueError(msg)
        return tuple(normalized)

    @model_validator(mode="after")
    def _requires_runtime_profiles(self) -> ModelRuntimePolicyContract:
        required = {"dev", "stability-test", "judge", "prod", "lakshman"}
        observed = set(self.profiles)
        if observed != required:
            msg = f"runtime policy profiles must be {sorted(required)}, got {sorted(observed)}"
            raise ValueError(msg)

        addresses: set[str] = set()
        for profile in self.profiles.values():
            for process in profile.processes.values():
                if process.runtime_address in addresses:
                    msg = f"duplicate runtime address {process.runtime_address}"
                    raise ValueError(msg)
                addresses.add(process.runtime_address)
        return self
