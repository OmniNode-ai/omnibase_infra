# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Value-redacted request contract for the first-effect recording scaffold."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

Sha256Digest = Annotated[
    str,
    Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="Lowercase SHA-256 digest; source values are never persisted here.",
    ),
]


class ModelFirstEffectAuthorizationRequest(BaseModel):
    """Immutable redacted recording identity; it confers no effect permission.

    ``authorization_digest`` remains a factual schema identifier, not a grant.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    authorization_digest: Sha256Digest
    nonce_digest: Sha256Digest
    correlation_id: UUID
    request_digest: Sha256Digest
    manifest_hash: Sha256Digest


__all__: list[str] = ["ModelFirstEffectAuthorizationRequest", "Sha256Digest"]
