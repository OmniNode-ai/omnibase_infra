# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Raw ledger-read boundary for authorized execution graph requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from omnibase_infra.runtime.db.execution_graph_read_adapters import (
        DelegationOwnerProof,
        ExecutionGraphLedgerRecord,
        PinnedExecutionGraphReadSet,
    )
    from omnibase_infra.runtime.execution_graph_read_authority import (
        VerifiedExecutionGraphReadAuthority,
    )


class ProtocolExecutionGraphLedgerReader(Protocol):
    async def read_full_current(
        self,
        authority: VerifiedExecutionGraphReadAuthority,
        owner: DelegationOwnerProof,
        read_set: PinnedExecutionGraphReadSet,
    ) -> tuple[ExecutionGraphLedgerRecord, ...]:
        """Read every pinned-topic row, independent of replay bounds."""
