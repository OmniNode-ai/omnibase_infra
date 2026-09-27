# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Current chain annotation read after graph ownership admission."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from omnibase_core.models.execution_graph_replay.model_execution_graph_stored_chain_annotation import (
        ModelExecutionGraphStoredChainAnnotation,
    )
    from omnibase_infra.runtime.db.execution_graph_read_adapters import (
        DelegationOwnerProof,
    )
    from omnibase_infra.runtime.execution_graph_read_authority import (
        VerifiedExecutionGraphReadAuthority,
    )


class ProtocolExecutionGraphStoredChainReader(Protocol):
    async def read_current(
        self,
        authority: VerifiedExecutionGraphReadAuthority,
        owner: DelegationOwnerProof,
        owned_envelope_ids: tuple[UUID, ...],
    ) -> tuple[ModelExecutionGraphStoredChainAnnotation, ...]:
        """Read only annotations naming admitted envelope IDs."""
