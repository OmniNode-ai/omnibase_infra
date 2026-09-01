# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Payload-free durable row returned by the verified-grant staging port."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.runtime.first_effect_ledger.enum_verified_first_effect_grant_state import (
    EnumVerifiedFirstEffectGrantState,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_authorization_request import (
    Sha256Digest,
)
from omnibase_infra.runtime.first_effect_ledger.verified_first_effect_grant_types import (
    CanonicalModelEventClass,
    FirstEffectRetryDisposition,
    OpaqueIdentifier,
    VerifiedGrantExpectedOutputTopic,
)


class ModelVerifiedFirstEffectGrantRecord(BaseModel):
    """Immutable causal projection plus its durable lifecycle state.

    The absence of a payload field is intentional: the workflow row remains the
    sole outbox payload owner.  This record is not issuer proof and must never be
    rehydrated by a caller to authorize an effect.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    authorization_digest: Sha256Digest
    grant_id: UUID
    grant_envelope_id: UUID
    nonce_digest: Sha256Digest
    request_digest: Sha256Digest
    correlation_id: UUID
    tenant_id: OpaqueIdentifier
    backend_id: OpaqueIdentifier
    rendered_contract_sha256: Sha256Digest
    issuer_key_fingerprint_sha256: Sha256Digest
    retry_disposition: FirstEffectRetryDisposition
    expected_output_topic: VerifiedGrantExpectedOutputTopic
    expected_output_event_class: CanonicalModelEventClass
    expected_output_event_index: Literal[0]
    state: EnumVerifiedFirstEffectGrantState
    outbox_envelope_id: UUID | None = None
    outbox_body_sha256: Sha256Digest | None = None
    outbox_topic: VerifiedGrantExpectedOutputTopic | None = None
    outbox_event_class: CanonicalModelEventClass | None = None
    outbox_event_index: int | None = Field(default=None, ge=0)
    workflow_version: int | None = Field(default=None, ge=0)
    verified_at: datetime
    staged_at: datetime | None = None
    publishing_at: datetime | None = None
    claimed_at: datetime | None = None
    published_unknown_at: datetime | None = None
    terminal_at: datetime | None = None
    updated_at: datetime
    version: int = Field(ge=0)


__all__ = ["ModelVerifiedFirstEffectGrantRecord"]
