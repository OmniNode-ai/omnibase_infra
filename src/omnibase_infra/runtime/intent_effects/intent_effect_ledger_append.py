# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Intent effect adapter for audit-ledger event persistence."""

from __future__ import annotations

import logging
from uuid import UUID

from omnibase_infra.nodes.node_ledger_write_effect.handlers import HandlerLedgerAppend
from omnibase_infra.nodes.node_registration_reducer.models import (
    ModelPayloadLedgerAppend,
)

logger = logging.getLogger(__name__)


class IntentEffectLedgerAppend:
    """Bridge `ledger.append` intents to HandlerLedgerAppend.

    ``node_ledger_projection_compute`` projects each consumed platform event
    into a ``ledger.append`` intent. Without this bridge the kernel cannot
    register a result applier for that contract, and auto-wiring then SKIPS the
    contract entirely (raw audit/projection consumers are only wired when an
    explicit effect path exists) — which is why ``event_ledger`` never held a
    row despite the topics carrying traffic (OMN-14516).
    """

    def __init__(self, handler: HandlerLedgerAppend) -> None:
        self._handler = handler

    async def execute(
        self,
        payload: object,
        *,
        correlation_id: UUID | None = None,
    ) -> None:
        typed_payload = (
            payload
            if isinstance(payload, ModelPayloadLedgerAppend)
            else ModelPayloadLedgerAppend.model_validate(payload)
        )
        result = await self._handler.append(typed_payload)
        logger.info(
            "ledger append intent executed: topic=%s partition=%s offset=%s "
            "duplicate=%s correlation_id=%s",
            typed_payload.topic,
            typed_payload.partition,
            typed_payload.kafka_offset,
            result.duplicate,
            str(correlation_id) if correlation_id is not None else None,
        )


__all__ = ["IntentEffectLedgerAppend"]
