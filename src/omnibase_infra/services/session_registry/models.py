# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session registry model for multi-session coordination.

Defines the materialized aggregate model for session registry entries.
Each entry represents the accumulated state of a single task_id across
all sessions that have worked on it.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 3).

Naming conventions:
    - Enums: Prefix with Enum* (per ONEX convention), one per file
    - Models: Prefix with Model* (per ONEX convention)
    - All models use Pydantic BaseModel with ConfigDict(frozen=True, extra="forbid")
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.services.session_registry.enum_session_phase import EnumSessionPhase
from omnibase_infra.services.session_registry.enum_session_registry_status import (
    EnumSessionRegistryStatus,
)

# Ordering for replay-safe phase advancement (D3)
_PHASE_ORDER: dict[EnumSessionPhase, int] = {
    EnumSessionPhase.PLANNING: 0,
    EnumSessionPhase.IMPLEMENTING: 1,
    EnumSessionPhase.REVIEWING: 2,
    EnumSessionPhase.MERGING: 3,
    EnumSessionPhase.DEPLOYING: 4,
    EnumSessionPhase.COMPLETED: 5,
    EnumSessionPhase.STALLED: 6,
}


def phase_is_forward(current: EnumSessionPhase, proposed: EnumSessionPhase) -> bool:
    """Check if proposed phase is forward from current (D3 compliance).

    Args:
        current: The current phase.
        proposed: The proposed new phase.

    Returns:
        True if the proposed phase is later in the lifecycle ordering.
    """
    return _PHASE_ORDER.get(proposed, -1) > _PHASE_ORDER.get(current, -1)


class ModelSessionRegistryEntry(BaseModel):
    """Materialized aggregate for a task's cross-session state.

    One row per task_id in the session_registry table. Accumulated from
    Kafka events by the session registry projector.

    Doctrine D3 compliance:
        - Arrays (files_touched, session_ids, correlation_ids, decisions)
          are deduplicated deterministically on upsert.
        - Scalar lifecycle fields (status, current_phase, last_activity)
          follow explicit ordering rules, not blind overwrite.

    Doctrine D4 compliance:
        - Resume queries return typed results (Found/NotFound/Unavailable),
          not this model directly. See session_registry_client.py.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # task_id is intentionally str (not UUID): it's a human-readable Linear ticket
    # ID like "OMN-1234", not a machine-generated UUID. See plan Doctrine D2.
    task_id: str = Field(  # onex:str-not-uuid — Linear ticket ID, not a UUID
        ...,
        min_length=1,
        max_length=64,
        description="Linear ticket ID (e.g., 'OMN-1234'). Primary key.",
    )
    status: EnumSessionRegistryStatus = Field(
        default=EnumSessionRegistryStatus.ACTIVE,
        description="Current task status.",
    )
    current_phase: EnumSessionPhase | None = Field(
        default=None,
        description="Current lifecycle phase. Only advances forward (D3).",
    )
    worktree_path: str | None = Field(
        default=None,
        description="Path to the git worktree for this task.",
    )
    files_touched: list[str] = Field(
        default_factory=list,
        description="Deduplicated list of files modified by this task.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Other task_ids this task depends on.",
    )
    session_ids: list[str] = Field(
        default_factory=list,
        description="All CLI session_ids that worked on this task.",
    )
    correlation_ids: list[str] = Field(
        default_factory=list,
        description="All correlation_ids across sessions for this task.",
    )
    decisions: list[str] = Field(
        default_factory=list,
        description="Key decisions made during the task (deduplicated by content).",
    )
    last_activity: datetime | None = Field(
        default=None,
        description="Timestamp of most recent activity. max(existing, incoming) per D3.",
    )
    created_at: datetime | None = Field(
        default=None,
        description="When this registry entry was first created.",
    )
