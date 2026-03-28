# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Session registry projector -- materializes session_registry rows from hook events.

Processes Kafka events from ``onex.evt.omniclaude.*`` topics and upserts
session_registry entries for events carrying a non-null ``task_id``.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 4).

Architecture:
    Kafka (onex.evt.omniclaude.*)
           |
           v
    SessionRegistryProjector.extract_registry_update()
           |
           v
    SessionRegistryStore.upsert_entry()

Design decisions:
    - Pure function extraction: ``extract_registry_update()`` is a pure function
      that extracts a ``ModelSessionRegistryEntry`` from a raw event dict. This
      enables unit testing without Kafka or Postgres dependencies.
    - Events without ``task_id`` are silently skipped (they predate the
      multi-session coordination feature).
    - Phase inference from event type: ``session.started`` -> PLANNING,
      ``tool.executed`` -> IMPLEMENTING, etc.
    - Replay-safe: all upserts use the store's replay-safe upsert logic (D3).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from omnibase_infra.services.session_registry.enum_session_phase import EnumSessionPhase
from omnibase_infra.services.session_registry.enum_session_registry_status import (
    EnumSessionRegistryStatus,
)
from omnibase_infra.services.session_registry.models import ModelSessionRegistryEntry

logger = logging.getLogger(__name__)

# Consumer group for the session registry projector.
CONSUMER_GROUP = "omnibase_infra.session_registry.project.v1"

# Map event types to inferred session phases.
_EVENT_TYPE_TO_PHASE: dict[str, EnumSessionPhase] = {
    "session.started": EnumSessionPhase.PLANNING,
    "session.ended": EnumSessionPhase.COMPLETED,
    "prompt.submitted": EnumSessionPhase.IMPLEMENTING,
    "tool.executed": EnumSessionPhase.IMPLEMENTING,
    "context.injected": EnumSessionPhase.IMPLEMENTING,
    "manifest.injected": EnumSessionPhase.IMPLEMENTING,
    "agent.status": EnumSessionPhase.IMPLEMENTING,
    "task.delegated": EnumSessionPhase.IMPLEMENTING,
    "decision.recorded": EnumSessionPhase.REVIEWING,
}

# Tool names that indicate file touches.
_FILE_TOUCH_TOOLS = frozenset({"Edit", "Write", "Read", "Bash"})


def extract_registry_update(
    event: dict[str, object],
) -> ModelSessionRegistryEntry | None:
    """Extract a session registry entry from a raw Kafka event.

    Returns None if the event has no task_id (pre-coordination events)
    or is malformed. Never raises -- all errors are logged and swallowed
    to preserve projector liveness.

    Args:
        event: Raw deserialized Kafka event dict.

    Returns:
        A ModelSessionRegistryEntry ready for upsert, or None if the event
        should be skipped.
    """
    try:
        task_id = event.get("task_id")
        if not task_id or not isinstance(task_id, str):
            return None

        raw_session_id = event.get("session_id", "")
        session_id = str(raw_session_id) if raw_session_id else ""
        raw_correlation_id = event.get("correlation_id", "")
        correlation_id = str(raw_correlation_id) if raw_correlation_id else ""
        raw_event_type = event.get("event_type", "")
        event_type = str(raw_event_type) if raw_event_type else ""

        # Parse emitted_at timestamp.
        emitted_at_raw = event.get("emitted_at")
        last_activity: datetime | None = None
        if isinstance(emitted_at_raw, str):
            try:
                last_activity = datetime.fromisoformat(emitted_at_raw)
            except ValueError:
                last_activity = datetime.now(UTC)
        elif isinstance(emitted_at_raw, datetime):
            last_activity = emitted_at_raw
        else:
            last_activity = datetime.now(UTC)

        # Extract files touched from tool.executed events.
        files_touched: list[str] = []
        if event_type == "tool.executed":
            raw_tool_name = event.get("tool_name", "")
            tool_name = str(raw_tool_name) if raw_tool_name else ""
            if tool_name in _FILE_TOUCH_TOOLS:
                file_path = event.get("file_path") or event.get("path")
                if isinstance(file_path, str) and file_path:
                    files_touched = [file_path]

        # Infer phase from event type.
        current_phase = _EVENT_TYPE_TO_PHASE.get(event_type)

        # Build session IDs and correlation IDs lists.
        session_ids = [session_id] if session_id else []
        correlation_ids = [str(correlation_id)] if correlation_id else []

        return ModelSessionRegistryEntry(
            task_id=task_id,
            status=EnumSessionRegistryStatus.ACTIVE,
            current_phase=current_phase,
            files_touched=files_touched,
            session_ids=session_ids,
            correlation_ids=correlation_ids,
            last_activity=last_activity,
        )

    except Exception:
        logger.exception("Failed to extract registry update from event")
        return None
