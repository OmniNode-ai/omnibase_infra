# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Authorization source classification for work authorization contracts.

Identifies how authorization was granted, which determines the default scope
and TTL of the authorization.

Ticket: OMN-2125
"""

from __future__ import annotations

from enum import Enum, unique


@unique
class EnumAuthSource(str, Enum):
    """Source of a work authorization grant.

    Each source carries different default scopes and TTLs:

    Attributes:
        PLAN_APPROVAL: Authorization via plan mode approval. All paths, 4hr TTL.
        TICKET_PIPELINE: Authorization via ticket pipeline. Ticket-declared files, run-bound.
        EXPLICIT: Authorization via /authorize command. All paths, 4hr TTL.
        SKILL_EXECUTION: Authorization via skill invocation. Inherits parent run, 2hr TTL.
        EMERGENCY_OVERRIDE: Emergency override via env vars. Run-scoped, 10min hard cap.
    """

    PLAN_APPROVAL = "plan_approval"
    """Granted by plan mode approval flow."""

    TICKET_PIPELINE = "ticket_pipeline"
    """Granted by ticket pipeline execution."""

    EXPLICIT = "explicit"
    """Granted by /authorize [reason] command."""

    SKILL_EXECUTION = "skill_execution"
    """Inherited from parent run during skill execution."""

    EMERGENCY_OVERRIDE = "emergency_override"
    """Emergency override via ONEX_UNSAFE_ALLOW_EDITS env var."""

    def __str__(self) -> str:
        """Return the string value for serialization."""
        return self.value


__all__: list[str] = ["EnumAuthSource"]
