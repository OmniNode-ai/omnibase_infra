# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol for dispatch engines that support exact contract-owned scopes."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from omnibase_core.models.dispatch.model_message_delivery_context import (
        ModelMessageDeliveryContext,
    )
    from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
    from omnibase_infra.models.dispatch.model_dispatch_result import (
        ModelDispatchResult,
    )

__all__ = ["ProtocolContractScopedDispatchEngine"]


@runtime_checkable
class ProtocolContractScopedDispatchEngine(Protocol):
    """Dispatch through an explicit set of contract-owned dispatcher IDs."""

    def validate_contract_dispatcher_scope(
        self,
        contract_name: str,
        dispatcher_ids: Collection[str],
    ) -> frozenset[str]:
        """Return a validated scope or raise before consumer side effects."""
        ...

    async def dispatch_scoped(
        self,
        topic: str,
        envelope: ModelEventEnvelope[object],
        *,
        allowed_dispatcher_ids: Collection[str],
        delivery: ModelMessageDeliveryContext | None = None,
    ) -> ModelDispatchResult:
        """Dispatch through a contract-owned scope, with optional coordinates.

        OMN-18918. ``delivery`` carries the source message's own partition and
        offset from the consume boundary that still holds the record. It is
        optional with a ``None`` default for the same reason it is on
        ``ProtocolDispatchEngine``: every implementor written before it stays
        structurally valid, and a caller probes before passing it.

        This entry needs it as much as the process-global one does, and for a
        sharper reason -- the in-process projection writers, the surfaces the
        OMN-18905 defect is actually about, arrive HERE. A typed path that
        reached only the other protocol would leave them injecting nothing
        while looking complete.
        """
        ...
