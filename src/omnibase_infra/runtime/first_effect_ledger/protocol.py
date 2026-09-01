# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Narrow port for recording one irreversible first-effect lifecycle."""

from __future__ import annotations

from typing import Protocol

from omnibase_infra.runtime.first_effect_ledger.model_first_effect_authorization_request import (
    ModelFirstEffectAuthorizationRequest,
    Sha256Digest,
)
from omnibase_infra.runtime.first_effect_ledger.model_first_effect_ledger_record import (
    ModelFirstEffectLedgerRecord,
)


class ProtocolFirstEffectLedger(Protocol):
    """Durable observation/recording scaffold; it confers no effect permission."""

    async def issue(
        self, request: ModelFirstEffectAuthorizationRequest
    ) -> ModelFirstEffectLedgerRecord: ...

    async def consume_preflight(
        self, *, authorization_digest: Sha256Digest, expected_version: int
    ) -> ModelFirstEffectLedgerRecord | None: ...

    async def get(
        self, *, authorization_digest: Sha256Digest
    ) -> ModelFirstEffectLedgerRecord | None: ...

    async def record_publishing_started(
        self, *, authorization_digest: Sha256Digest, expected_version: int
    ) -> ModelFirstEffectLedgerRecord | None: ...

    async def record_published_unknown(
        self,
        *,
        authorization_digest: Sha256Digest,
        expected_version: int,
        publish_evidence_hash: Sha256Digest,
    ) -> ModelFirstEffectLedgerRecord | None: ...

    async def observe_terminal(
        self,
        *,
        authorization_digest: Sha256Digest,
        expected_version: int,
        publish_evidence_hash: Sha256Digest,
        terminal_receipt_hash: Sha256Digest,
    ) -> ModelFirstEffectLedgerRecord | None: ...

    async def block(
        self, *, authorization_digest: Sha256Digest, expected_version: int
    ) -> ModelFirstEffectLedgerRecord | None: ...


__all__ = ["ProtocolFirstEffectLedger"]
