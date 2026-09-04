# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Derived result-attestation trust facts for inert RSD preflight."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.runtime.models.rsd_live_delegation_schema import (
    CanonicalSha256,
    CanonicalUuid4,
)


class ModelRsdLiveDelegationResultAnchor(BaseModel):
    """Result trust facts derived solely from a sealed authority capability."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    signer_key_id: CanonicalUuid4
    signer_public_key_fingerprint_sha256: CanonicalSha256
    dispatch_outcome_schema: Literal["rsd.dispatch-outcome-attestation.v1"]
    claim_binding_schema: Literal["rsd.delegation-claim-binding.v1"]


__all__ = ["ModelRsdLiveDelegationResultAnchor"]
