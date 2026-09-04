# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Non-secret availability facts for inert RSD delegation preflight."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.runtime.models.rsd_live_delegation_schema import (
    _RFC3339_UTC,
    CanonicalCapabilityRef,
    CanonicalSha256,
    CanonicalUuid4,
)


class ModelRsdLiveDelegationObservation(BaseModel):
    """Injected non-secret facts consumed by the resolver-free preflight."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    installed_public_rsd_revision_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    present_capability_refs: frozenset[CanonicalCapabilityRef] = Field(
        default_factory=frozenset
    )
    healthy_capability_refs: frozenset[CanonicalCapabilityRef] = Field(
        default_factory=frozenset
    )
    sealed_root_provider_verified: bool = False
    authority_checked_at: str | None = Field(default=None, pattern=_RFC3339_UTC)
    verified_result_attestor_key_id: CanonicalUuid4 | None = None
    verified_result_attestor_public_key_fingerprint_sha256: CanonicalSha256 | None = (
        None
    )
    verified_overlay_digest_sha256: CanonicalSha256 | None = None
    observed_model_id: Literal["Qwen3.6-35B-A3B"] | None = None
    observed_model_attestation_sha256: CanonicalSha256 | None = None
    observed_model_match_status: Literal["model_id_mismatch"] | None = None

    @model_validator(mode="after")
    def validate_health_is_present(self) -> Self:
        if not self.healthy_capability_refs <= self.present_capability_refs:
            raise ValueError("healthy capability references must be present")
        return self


__all__ = ["ModelRsdLiveDelegationObservation"]
