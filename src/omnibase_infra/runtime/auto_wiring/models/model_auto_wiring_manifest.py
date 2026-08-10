# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Auto-wiring manifest produced by contract discovery (OMN-7653)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.auto_wiring.models.model_discovered_contract import (
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.models.model_discovery_error import (
    ModelDiscoveryError,
)
from omnibase_infra.runtime.auto_wiring.models.model_runtime_build_sha import (
    ModelRuntimeBuildSha,
)


def _unbound_build_sha() -> ModelRuntimeBuildSha:
    """Default for legacy/discovery-only construction (OMN-10856).

    Distinct from an env-lookup miss: this manifest was constructed before
    any identity binding was attempted at all (e.g. ``discover_contracts()``
    or one of the ~50 pre-existing test call sites that only care about
    ``contracts``/``errors``). ``bind_introspection_manifest_identity()`` is
    the single place that resolves the real SHA-from-env values.
    """
    return ModelRuntimeBuildSha(
        value=None,
        absent_reason="manifest constructed without a build-identity binding",
    )


class ModelAutoWiringManifest(BaseModel):
    """Complete manifest produced by contract auto-discovery.

    Contains all successfully discovered contracts and any errors
    encountered during scanning, plus the runtime build identity (OMN-10856)
    that binds this reported topology to a specific deployed process:
    which runtime profile produced it, and which image/deployment build it
    came from. Pure data — no side effects; identity resolution (env var
    reads) happens in ``bind_introspection_manifest_identity``, not here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contracts: tuple[ModelDiscoveredContract, ...] = Field(
        default_factory=tuple,
        description="Successfully discovered contracts",
    )
    errors: tuple[ModelDiscoveryError, ...] = Field(
        default_factory=tuple,
        description="Errors encountered during discovery",
    )
    runtime_profile: str = Field(
        default="",
        description=(
            "RUNTIME_PROFILE identity this manifest was built for (e.g. "
            "'workers', 'effects', 'main'). Empty string means the manifest "
            "was constructed before profile binding (discovery-only)."
        ),
    )
    image_sha: ModelRuntimeBuildSha = Field(
        default_factory=_unbound_build_sha,
        description=(
            "Container image SHA/digest this runtime was built from, or an "
            "explicit absent-with-reason marker (OMN-10856)."
        ),
    )
    deployment_sha: ModelRuntimeBuildSha = Field(
        default_factory=_unbound_build_sha,
        description=(
            "Deployment/source revision SHA this runtime was deployed from, "
            "or an explicit absent-with-reason marker (OMN-10856)."
        ),
    )

    @property
    def total_discovered(self) -> int:
        return len(self.contracts)

    @property
    def total_errors(self) -> int:
        return len(self.errors)

    def get_by_node_type(self, node_type: str) -> tuple[ModelDiscoveredContract, ...]:
        """Filter discovered contracts by node type."""
        return tuple(c for c in self.contracts if c.node_type == node_type)

    def get_all_subscribe_topics(self) -> frozenset[str]:
        """Collect all subscribe topics across discovered contracts."""
        topics: set[str] = set()
        for c in self.contracts:
            if c.event_bus:
                topics.update(c.event_bus.subscribe_topics)
        return frozenset(topics)

    def all_subscribe_topics(self) -> frozenset[str]:
        """Alias satisfying ProtocolAutoWiringManifestLike (OMN-8854)."""
        return self.get_all_subscribe_topics()

    def get_all_publish_topics(self) -> frozenset[str]:
        """Collect all publish topics across discovered contracts."""
        topics: set[str] = set()
        for c in self.contracts:
            if c.event_bus:
                topics.update(c.event_bus.publish_topics)
        return frozenset(topics)
