# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Value-redacted durable record from the first-effect recording scaffold."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from omnibase_infra.runtime.first_effect_ledger.enum_first_effect_ledger_state import (
    EnumFirstEffectLedgerState,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_authorization_request import (
    ModelFirstEffectAuthorizationRequest,
    Sha256Digest,
)


class ModelFirstEffectLedgerRecord(ModelFirstEffectAuthorizationRequest):
    """A durable observation row that confers no effect permission."""

    state: EnumFirstEffectLedgerState
    publish_evidence_hash: Sha256Digest | None = None
    terminal_receipt_hash: Sha256Digest | None = None
    issued_at: datetime
    preflight_consumed_at: datetime | None = None
    publishing_at: datetime | None = None
    published_unknown_at: datetime | None = None
    terminal_observed_at: datetime | None = None
    blocked_at: datetime | None = None
    updated_at: datetime
    version: int = Field(ge=0)


__all__: list[str] = ["ModelFirstEffectLedgerRecord"]
