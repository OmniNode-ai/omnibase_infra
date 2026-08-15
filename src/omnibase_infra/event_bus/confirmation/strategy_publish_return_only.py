# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The explicitly-named weak strategy: trust the publish return (OMN-15861).

This class exists so that "we acked because the publish call did not raise" is a
**named, greppable, attributable choice** instead of the unexamined default it
used to be. Every confirmation it emits carries ``strategy="publish_return_only"``,
so an audit of durable claims can separate them from readback-backed ones.

It is correct ONLY for genuinely lossy-tolerant traffic (telemetry, metrics,
best-effort notifications). Binding it to duty-critical traffic reintroduces
exactly the invariant-7 violation OMN-15861 closes; consumers that carry a
durability policy MUST fail fast at configuration time rather than at runtime.
"""

from __future__ import annotations

from datetime import UTC, datetime

from omnibase_infra.enums.enum_confirmation_state import EnumConfirmationState
from omnibase_infra.event_bus.models.model_durability_confirmation import (
    ModelDurabilityConfirmation,
)
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt

STRATEGY_NAME_PUBLISH_RETURN_ONLY = "publish_return_only"


class PublishReturnOnlyStrategy:
    """Confirms on the presence of a receipt alone -- no readback.

    Example:
        >>> import asyncio
        >>> asyncio.run(PublishReturnOnlyStrategy().confirm(None)).state
        <EnumConfirmationState.UNKNOWN: 'unknown'>
    """

    @property
    def name(self) -> str:
        """Stable identifier recorded on every confirmation."""
        return STRATEGY_NAME_PUBLISH_RETURN_ONLY

    async def confirm(
        self, receipt: ModelPublishReceipt | None
    ) -> ModelDurabilityConfirmation:
        """Confirm iff a coordinate exists; no authoritative surface is consulted.

        Even this weakest strategy refuses to confirm a ``None`` receipt. A
        transport that cannot report a coordinate has told us nothing at all,
        and "nothing at all" is ``UNKNOWN``, which fails closed.
        """
        now = datetime.now(UTC)
        if receipt is None:
            return ModelDurabilityConfirmation(
                state=EnumConfirmationState.UNKNOWN,
                strategy=self.name,
                receipt=None,
                checked_at=now,
                detail=(
                    "publish returned no durability coordinate; the transport "
                    "cannot support any durable claim"
                ),
            )
        return ModelDurabilityConfirmation(
            state=EnumConfirmationState.CONFIRMED,
            strategy=self.name,
            receipt=receipt,
            checked_at=now,
            detail="",
        )


__all__: list[str] = [
    "STRATEGY_NAME_PUBLISH_RETURN_ONLY",
    "PublishReturnOnlyStrategy",
]
