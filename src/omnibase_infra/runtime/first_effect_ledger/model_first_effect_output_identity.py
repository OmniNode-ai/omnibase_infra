# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exact direct-message output identity required by durable transitions."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.runtime.first_effect_ledger.model_first_effect_authorization_request import (
    Sha256Digest,
)
from omnibase_infra.runtime.first_effect_ledger.verified_first_effect_grant_types import (
    CanonicalModelEventClass,
    VerifiedGrantExpectedOutputTopic,
)


class ModelFirstEffectOutputIdentity(BaseModel):
    """Exact output pin; no caller can claim by authorization digest alone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    authorization_digest: Sha256Digest
    outbox_envelope_id: UUID
    outbox_body_sha256: Sha256Digest
    output_topic: VerifiedGrantExpectedOutputTopic
    output_event_class: CanonicalModelEventClass
    output_event_index: Literal[0]


__all__ = ["ModelFirstEffectOutputIdentity"]
