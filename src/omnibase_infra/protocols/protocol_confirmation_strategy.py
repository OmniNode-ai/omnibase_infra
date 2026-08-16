# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Decides whether a publish receipt authorises a durable claim (OMN-15861).

Canonical invariant 7: a publish return is not durability. This protocol is the
verdict layer the old code did not have at all -- without it, "the produce call
returned" *was* the verdict, and a durable outbox truncated its only copy of an
event on that basis.

See ``ProtocolReadbackSource`` for the fact-reporting half of the seam.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from omnibase_infra.event_bus.models.model_durability_confirmation import (
    ModelDurabilityConfirmation,
)
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt


@runtime_checkable
class ProtocolConfirmationStrategy(Protocol):
    """Decides whether a publish receipt authorises a durable claim."""

    @property
    def name(self) -> str:
        """Stable identifier recorded on every confirmation for attribution."""
        ...

    async def confirm(
        self, receipt: ModelPublishReceipt | None
    ) -> ModelDurabilityConfirmation:
        """Resolve ``receipt`` into a durability verdict.

        MUST NOT raise for an unreachable surface: an unresolvable check is a
        first-class ``UNKNOWN`` outcome, and swallowing it into an exception
        pushes the fail-closed decision onto every call site instead of keeping
        it here.

        Args:
            receipt: The coordinate to confirm. ``None`` means the produce path
                could not supply one at all, which MUST resolve to ``UNKNOWN``.
        """
        ...


__all__: list[str] = ["ProtocolConfirmationStrategy"]
