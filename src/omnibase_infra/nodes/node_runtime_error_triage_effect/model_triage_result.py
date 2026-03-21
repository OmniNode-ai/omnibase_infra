# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Model for runtime error triage results emitted to Kafka.

Produced by HandlerRuntimeErrorTriage after processing a runtime error event.
Each result carries the triage action taken (auto-fix, ticket creation, dedup)
and is emitted to onex.evt.omnibase-infra.error-triaged.v1.

Related Tickets:
    - OMN-5650: Runtime error triage consumer
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.enums.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.enum_triage_action import (
    EnumTriageAction,
)
from omnibase_infra.nodes.node_runtime_error_triage_effect.enum_triage_action_status import (
    EnumTriageActionStatus,
)


class ModelRuntimeErrorTriageResult(BaseModel):
    """Triage result for a single runtime error event."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    event_id: UUID
    """Matches the source ModelRuntimeErrorEvent.event_id."""

    fingerprint: str
    """Error fingerprint from the source event."""

    action: EnumTriageAction
    """Triage action taken."""

    action_status: EnumTriageActionStatus
    """Whether the action succeeded."""

    triaged_at: datetime
    """Timezone-aware UTC timestamp of triage completion."""

    ticket_id: str | None = None
    """Linear ticket ID if created (e.g., 'OMN-5642')."""

    ticket_url: str | None = None
    """Linear ticket URL if created."""

    auto_fix_type: str | None = None
    """Type of auto-fix attempted (e.g., 'rpk_topic_create')."""

    auto_fix_command: str | None = None
    """Exact command attempted for auto-fix."""

    auto_fix_result: str | None = None
    """stdout/stderr of fix attempt."""

    auto_fix_verified: bool | None = None
    """Whether post-fix verification passed."""

    dedup_reason: str | None = None
    """Reason for dedup (e.g., 'fingerprint seen 3 times in last 24h')."""

    error_category: EnumRuntimeErrorCategory
    """Carried from source event."""

    container: str
    """Carried from source event."""

    severity: str
    """Carried from source event."""

    operator_attention_required: bool = False
    """True if auto-fix failed or unknown category."""

    notes: str | None = None
    """Additional context for the triage decision."""
