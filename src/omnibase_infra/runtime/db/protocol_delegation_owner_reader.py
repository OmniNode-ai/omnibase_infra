# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Owner-read boundary for authorized execution graph requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from omnibase_infra.runtime.db.execution_graph_read_adapters import (
        DelegationOwnerProof,
    )
    from omnibase_infra.runtime.execution_graph_read_authority import (
        VerifiedExecutionGraphReadAuthority,
    )


class ProtocolDelegationOwnerReader(Protocol):
    async def require_owner(
        self, authority: VerifiedExecutionGraphReadAuthority
    ) -> DelegationOwnerProof:
        """Require exactly one current owner row under tenant-local RLS."""
