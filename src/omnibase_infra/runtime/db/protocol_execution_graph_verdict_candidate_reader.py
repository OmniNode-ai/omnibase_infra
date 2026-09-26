# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verification-index and raw-terminal read boundary for graph requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from omnibase_infra.runtime.db.execution_graph_read_adapters import (
        DelegationOwnerProof,
        ExecutionGraphVerdictCandidates,
        PinnedExecutionGraphReadSet,
    )
    from omnibase_infra.runtime.execution_graph_read_authority import (
        VerifiedExecutionGraphReadAuthority,
    )


class ProtocolExecutionGraphVerdictCandidateReader(Protocol):
    async def read_full_current(
        self,
        authority: VerifiedExecutionGraphReadAuthority,
        owner: DelegationOwnerProof,
        read_set: PinnedExecutionGraphReadSet,
    ) -> ExecutionGraphVerdictCandidates:
        """Discover IDs via projection index, return raw verdicts only."""
