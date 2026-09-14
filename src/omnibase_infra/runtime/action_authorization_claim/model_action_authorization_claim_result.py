# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed, redacted results for registration and atomic nonce claiming."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.action_authorization_claim.enum_action_authorization_claim_outcome import (
    EnumActionAuthorizationClaimOutcome,
)
from omnibase_infra.runtime.action_authorization_claim.enum_action_authorization_claim_state import (
    EnumActionAuthorizationClaimState,
)
from omnibase_infra.runtime.action_authorization_claim.model_action_authorization_claim_request import (
    Sha256Digest,
)


class ModelActionAuthorizationClaimResult(BaseModel):
    """One terminal result of attempting an atomic durable claim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: EnumActionAuthorizationClaimOutcome
    state: EnumActionAuthorizationClaimState | None = None
    version: int | None = Field(default=None, ge=0)
    redacted_receipt_digest: Sha256Digest | None = None


__all__ = ["ModelActionAuthorizationClaimResult"]
