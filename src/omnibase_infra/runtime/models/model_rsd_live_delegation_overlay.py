# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fixed operator facts for the inert RSD live delegation overlay."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from omnibase_infra.runtime.models.model_rsd_live_delegation_authority_envelope import (
    ModelRsdLiveDelegationAuthorityEnvelope,
)
from omnibase_infra.runtime.models.rsd_live_delegation_schema import (
    CanonicalCapabilityRef,
    CanonicalSha256,
    CanonicalUuid4,
)


class ModelRsdLiveDelegationOverlay(BaseModel):
    """Fixed operator facts. This record cannot enable public RSD execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["rsd_live_delegation_overlay.v1"]
    lane: Literal["dev"]
    locale: Literal["lab"]
    execute_enabled: Literal[False]
    rsd_distribution_ref: Literal["omninode-rsd/0.1.0"]
    public_rsd_revision_sha: Literal["3177c31ed7db4e5551e93dadfd0f9a034728b8ab"]
    delegation_policy_schema: Literal["rsd.delegation-overlay.v1"]
    dispatch_outcome_schema: Literal["rsd.dispatch-outcome-attestation.v1"]
    claim_binding_schema: Literal["rsd.delegation-claim-binding.v1"]
    route_ref: Literal["logical://delegation/qwen3.8-27b"]
    backend_id: Literal["local-coder"]
    model_id: Literal["Qwen/Qwen3.8-27B"]
    dispatch_policy: Literal["backend-pinned-single-attempt.v1"]
    one_shot_endpoint_capability_ref: CanonicalCapabilityRef
    root_authority_capability_ref: CanonicalCapabilityRef
    root_authority_key_id: CanonicalUuid4
    root_authority_public_key_fingerprint_sha256: CanonicalSha256
    result_attestor_signer_capability_ref: CanonicalCapabilityRef
    result_attestor_key_capability_ref: CanonicalCapabilityRef
    result_attestor_fingerprint_capability_ref: CanonicalCapabilityRef
    result_attestor_key_id: CanonicalUuid4
    result_attestor_public_key_fingerprint_sha256: CanonicalSha256
    postgres_capability_ref: Literal["capability://rsd/postgres/acceptance"]
    observed_model_attestation_capability_ref: CanonicalCapabilityRef
    observed_model_id: Literal["Qwen3.6-35B-A3B"]
    observed_model_attestation_sha256: CanonicalSha256
    model_match_status: Literal["model_id_mismatch"]
    authority_envelope: ModelRsdLiveDelegationAuthorityEnvelope

    @model_validator(mode="after")
    def validate_inert_mismatch_and_distinct_authorities(self) -> Self:
        """Make activation unavailable and capability roles unambiguous."""
        if self.model_id == self.observed_model_id:
            raise ValueError(
                "model_id_mismatch requires distinct requested and observed models"
            )
        references = (
            self.one_shot_endpoint_capability_ref,
            self.root_authority_capability_ref,
            self.result_attestor_signer_capability_ref,
            self.result_attestor_key_capability_ref,
            self.result_attestor_fingerprint_capability_ref,
            self.postgres_capability_ref,
            self.observed_model_attestation_capability_ref,
        )
        if len(references) != len(set(references)):
            raise ValueError(
                "RSD live delegation capability references must be distinct"
            )
        if (
            self.authority_envelope.observed_model_attestation_sha256
            != self.observed_model_attestation_sha256
        ):
            raise ValueError(
                "authority envelope must pin the exact observed model attestation"
            )
        if (
            self.authority_envelope.authority_root_id != self.root_authority_key_id
            or self.authority_envelope.authority_root_public_key_fingerprint_sha256
            != self.root_authority_public_key_fingerprint_sha256
            or self.authority_envelope.attestor_key_id != self.result_attestor_key_id
            or self.authority_envelope.attestor_public_key_fingerprint_sha256
            != self.result_attestor_public_key_fingerprint_sha256
        ):
            raise ValueError(
                "authority envelope must pin the exact operator key identities"
            )
        return self


__all__ = ["ModelRsdLiveDelegationOverlay"]
