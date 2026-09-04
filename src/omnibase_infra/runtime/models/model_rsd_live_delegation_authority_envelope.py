# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Root-signed authority facts for one inert RSD lane overlay."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.models.rsd_live_delegation_schema import (
    _RFC3339_UTC,
    CanonicalEd25519Signature,
    CanonicalSha256,
    CanonicalUuid4,
)


class ModelRsdLiveDelegationAuthorityEnvelope(BaseModel):
    """Root-signed, topology-free authorization for one exact lane overlay."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    schema_version: Literal["rsd.live-delegation-authority-envelope.v1"]
    authority_domain: Literal["omninode-rsd.inert-live-delegation.v1"]
    authority_root_id: CanonicalUuid4
    authority_root_public_key_fingerprint_sha256: CanonicalSha256
    overlay_schema_version: Literal["rsd_live_delegation_overlay.v1"]
    overlay_digest_sha256: CanonicalSha256
    delegation_policy_schema: Literal["rsd.delegation-overlay.v1"]
    dispatch_outcome_schema: Literal["rsd.dispatch-outcome-attestation.v1"]
    claim_binding_schema: Literal["rsd.delegation-claim-binding.v1"]
    route_ref: Literal["logical://delegation/qwen3.8-27b"]
    backend_id: Literal["local-coder"]
    model_id: Literal["Qwen/Qwen3.8-27B"]
    attestor_key_id: CanonicalUuid4
    attestor_public_key_fingerprint_sha256: CanonicalSha256
    observed_model_id: Literal["Qwen3.6-35B-A3B"]
    observed_model_attestation_sha256: CanonicalSha256
    issued_at: str = Field(pattern=_RFC3339_UTC)
    expires_at: str = Field(pattern=_RFC3339_UTC)
    signature_base64: CanonicalEd25519Signature


__all__ = ["ModelRsdLiveDelegationAuthorityEnvelope"]
