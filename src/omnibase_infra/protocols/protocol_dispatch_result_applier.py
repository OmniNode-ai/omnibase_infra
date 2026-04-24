# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Protocol for applying dispatch results emitted by auto-wired handlers."""

from __future__ import annotations

__all__ = ["ProtocolDispatchResultApplier"]

from typing import TYPE_CHECKING, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_infra.models.dispatch.model_dispatch_result import (
        ModelDispatchResult,
    )


class ProtocolDispatchResultApplier(Protocol):
    """Applies a handler dispatch result to its output/effect path."""

    async def apply(
        self,
        result: ModelDispatchResult,
        correlation_id: UUID | None = None,
    ) -> None:
        raise NotImplementedError
