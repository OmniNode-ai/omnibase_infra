# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol for dispatch engines that support exact contract-owned scopes."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
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
    ) -> ModelDispatchResult: ...
