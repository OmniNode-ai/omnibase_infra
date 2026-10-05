# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Central project tracker DI authority.

Single authoritative surface for selecting a `ProtocolProjectTracker`
implementation. The caller selects the backend:

* ``EnumProjectTrackerBackend.LINEAR`` (the default) returns
  `AdapterLinearGraphQLProjectTracker`, authenticated by ``LINEAR_API_KEY`` or
  ``LINEAR_TOKEN``. A missing or blank credential, or any construction
  failure, RAISES. It never falls back to the stub: a caller that asked for
  Linear and silently got a local JSON file would read and write tickets that
  do not exist (OMN-20595, Operating Rule 8).
* ``EnumProjectTrackerBackend.LOCAL_STUB`` returns `LocalStubProjectTracker`,
  and only when the caller names it.

Safe to call from any Python context (no MCP-runtime dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

from omnibase_infra.enums.enum_project_tracker_backend import (
    EnumProjectTrackerBackend,
)

if TYPE_CHECKING:
    from omnibase_spi.protocols.services.protocol_project_tracker import (
        ProtocolProjectTracker,
    )


def resolve_project_tracker(
    state_root: Path | None = None,
    *,
    backend: EnumProjectTrackerBackend = EnumProjectTrackerBackend.LINEAR,
) -> ProtocolProjectTracker:
    """Resolve the `ProtocolProjectTracker` implementation the caller selected.

    Args:
        state_root: State directory for the local JSON tracker's backing file.
            Ignored for Linear.
        backend: Linear (the default) or the local JSON tracker.

    Returns:
        A `ProtocolProjectTracker`-shaped instance.

    Raises:
        InfraAuthenticationError: Linear is selected and no credential resolves.
    """
    if backend is EnumProjectTrackerBackend.LOCAL_STUB:
        from omnibase_infra.adapters.project_tracker.local_stub_project_tracker import (
            LocalStubProjectTracker,
        )

        return cast(
            "ProtocolProjectTracker", LocalStubProjectTracker(state_root=state_root)
        )

    from omnibase_infra.adapters.project_tracker.linear_graphql_project_tracker_adapter import (
        AdapterLinearGraphQLProjectTracker,
    )

    # The adapter resolves LINEAR_API_KEY, then LINEAR_TOKEN, and raises
    # InfraAuthenticationError when neither carries a value.
    # cast: AdapterLinearGraphQLProjectTracker returns ModelStub* typed
    # variants that are structural-compatible subclasses of the canonical
    # ModelIssue/ModelComment/ModelProject declared by ProtocolProjectTracker.
    # Stub type consolidation is tracked separately (OMN-9210).
    return cast("ProtocolProjectTracker", AdapterLinearGraphQLProjectTracker())
