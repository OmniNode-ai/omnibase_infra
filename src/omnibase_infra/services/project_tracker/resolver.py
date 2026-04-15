# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Central project tracker DI authority.

Single authoritative surface for selecting a `ProtocolProjectTracker`
implementation. Token present (LINEAR_API_KEY or LINEAR_TOKEN) →
`LinearProjectTrackerAdapter`. Absent → `LocalStubProjectTracker`.
Fail-soft: NEVER raises; on any construction error (missing adapter
module, constructor failure, bad token) falls back to `LocalStubProjectTracker`
with a warning log.

Only `node_session_compose` and the canary skill's handler are permitted
callers. Skill prose MUST NOT re-encode token-detection or fallback logic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_spi.protocols.services.protocol_project_tracker import (
        ProtocolProjectTracker,
    )

log = logging.getLogger(__name__)


def resolve_project_tracker(
    state_root: Path | None = None,
    _force_construction_error: bool = False,
) -> ProtocolProjectTracker:
    """Resolve a `ProtocolProjectTracker` implementation.

    Args:
        state_root: Optional state directory for the local fallback's JSON backing file.
        _force_construction_error: Test-only hook; forces the Linear adapter path
            to fail so the fail-soft fallback is exercised.

    Returns:
        A `ProtocolProjectTracker`-shaped instance. Never raises.
    """
    token = os.environ.get("LINEAR_TOKEN") or os.environ.get("LINEAR_API_KEY")
    if token and not _force_construction_error:
        try:
            from omnibase_infra.adapters.project_tracker.linear_project_tracker_adapter import (
                LinearProjectTrackerAdapter,
            )

            return LinearProjectTrackerAdapter()
        except Exception as exc:  # noqa: BLE001 — fail-soft is the contract
            log.warning(
                "resolve_project_tracker: Linear adapter construction failed (%s); "
                "falling back to LocalStubProjectTracker",
                exc,
            )

    if _force_construction_error:
        log.warning(
            "resolve_project_tracker: forced construction error; "
            "falling back to LocalStubProjectTracker",
        )

    from omnibase_infra.adapters.project_tracker.local_stub_project_tracker import (
        LocalStubProjectTracker,
    )

    return LocalStubProjectTracker(state_root=state_root)
