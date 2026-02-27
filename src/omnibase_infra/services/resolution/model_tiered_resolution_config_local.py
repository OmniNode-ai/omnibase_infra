# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Local tiered resolution configuration model for contract YAML loader.

Mirrors the essential fields from ``ModelTieredResolutionConfig`` in
omnibase_core (created in PR #576). Exists as a local schema so the
infra loader can operate independently of the core package release cycle.

This model represents the ``tiered_resolution`` block within a dependency
entry in contract.yaml::

    dependencies:
      - alias: "db"
        capability: "database.relational"
        tiered_resolution:
          min_tier: "local_exact"
          max_tier: "org_trusted"
          require_proofs: ["node_identity", "capability_attested"]
          classification: "internal"

TODO(OMN-2896): Replace this local model with the canonical import once
    omnibase_core PR #576 merges and a release containing
    ``ModelTieredResolutionConfig`` is available as a dependency.

Related:
    - OMN-2896: Contract YAML Integration (Phase 7 of OMN-2897 epic)
    - ModelTieredResolutionConfig in omnibase_core (PR #576)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# Valid resolution tier names matching EnumResolutionTier from omnibase_core.
VALID_RESOLUTION_TIERS: frozenset[str] = frozenset(
    {
        "local_exact",
        "local_compatible",
        "org_trusted",
        "federated_trusted",
        "quarantine",
    }
)

# Valid classification labels matching EnumClassification from omnibase_core.
VALID_CLASSIFICATIONS: frozenset[str] = frozenset(
    {
        "public",
        "internal",
        "confidential",
        "restricted",
    }
)

# Valid proof type identifiers matching EnumProofType from omnibase_core.
VALID_PROOF_TYPES: frozenset[str] = frozenset(
    {
        "node_identity",
        "capability_attested",
        "org_membership",
        "bus_membership",
        "policy_compliance",
    }
)


class ModelTieredResolutionConfigLocal(BaseModel):
    """Local tiered resolution configuration from contract YAML.

    Represents the ``tiered_resolution`` section within a dependency block
    in a contract YAML file. All fields are optional to maintain backward
    compatibility with existing contracts that do not declare tiered
    resolution constraints.

    Attributes:
        min_tier: Minimum resolution tier to attempt. Resolution will not
            try tiers below this level.
        max_tier: Maximum resolution tier to attempt. Resolution will not
            escalate beyond this level.
        require_proofs: Proof types required for resolution at tiers beyond
            ``local_exact``. Each entry must be a valid proof type identifier.
        classification: Data classification label constraining which tiers
            and buses may be used for this dependency.

    Example:
        >>> config = ModelTieredResolutionConfigLocal(
        ...     min_tier="local_exact",
        ...     max_tier="org_trusted",
        ...     require_proofs=("node_identity", "capability_attested"),
        ...     classification="internal",
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    min_tier: str | None = Field(
        default=None,
        description="Minimum resolution tier to attempt.",
    )
    max_tier: str | None = Field(
        default=None,
        description="Maximum resolution tier to attempt.",
    )
    require_proofs: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Proof types required for resolution beyond local_exact.",
    )
    classification: str | None = Field(
        default=None,
        description="Data classification label for tier/bus constraints.",
    )


__all__: list[str] = [
    "ModelTieredResolutionConfigLocal",
    "VALID_CLASSIFICATIONS",
    "VALID_PROOF_TYPES",
    "VALID_RESOLUTION_TIERS",
]
