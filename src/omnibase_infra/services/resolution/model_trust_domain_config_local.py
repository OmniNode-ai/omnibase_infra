# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Local trust domain configuration model for contract YAML loader.

Mirrors the essential fields from ``ModelTrustDomainConfig`` in
omnibase_core (created in PR #576). Exists as a local schema so the
infra loader can operate independently of the core package release cycle.

This model represents entries in the ``trust_domains`` top-level section
of a contract YAML file::

    trust_domains:
      - domain_id: "local.default"
        tier: "local_exact"
      - domain_id: "org.omninode"
        tier: "org_trusted"
        trust_root_ref: "secrets://keys/org-omninode-trust-root"

TODO(OMN-2896): Replace this local model with the canonical import once
    omnibase_core PR #576 merges and a release containing
    ``ModelTrustDomainConfig`` is available as a dependency.

Related:
    - OMN-2896: Contract YAML Integration (Phase 7 of OMN-2897 epic)
    - ModelTrustDomainConfig in omnibase_core (PR #576)
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelTrustDomainConfigLocal(BaseModel):
    """Local trust domain configuration from contract YAML.

    Represents a single entry in the ``trust_domains`` section of a
    contract YAML file. Maps a trust domain identifier to a resolution
    tier and optionally references a trust root key.

    Attributes:
        domain_id: Unique identifier for the trust domain
            (e.g., ``"local.default"``, ``"org.omninode"``).
        tier: Resolution tier associated with this domain. Must be a
            valid ``EnumResolutionTier`` value.
        trust_root_ref: Optional reference to the trust root key for
            this domain. Uses the ``secrets://`` URI scheme for
            Infisical-managed keys.

    Example:
        >>> domain = ModelTrustDomainConfigLocal(
        ...     domain_id="org.omninode",
        ...     tier="org_trusted",
        ...     trust_root_ref="secrets://keys/org-omninode-trust-root",
        ... )
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    domain_id: str = Field(
        ...,
        description="Unique identifier for the trust domain.",
        min_length=1,
    )
    tier: str = Field(
        ...,
        description="Resolution tier associated with this domain.",
        min_length=1,
    )
    trust_root_ref: str | None = Field(
        default=None,
        description="Optional reference to the trust root key for this domain.",
    )


__all__: list[str] = ["ModelTrustDomainConfigLocal"]
