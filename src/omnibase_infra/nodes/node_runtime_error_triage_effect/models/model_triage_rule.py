# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Triage rule model for first-match-wins runtime error triage.

Each rule matches on a combination of logger_family prefix, error_category,
and optional message pattern. Rules are evaluated in order; first match wins.

Related Tickets:
    - OMN-5522: Create NodeRuntimeErrorTriageEffect
    - OMN-5529: Runtime Health Event Pipeline (epic)
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)


class ModelTriageRule(BaseModel):
    """A single triage rule for the first-match-wins rule engine.

    Rules are evaluated in priority order. First match determines the
    triage action for the runtime error event.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., description="Human-readable rule name.")
    priority: int = Field(
        default=100,
        description="Lower number = higher priority. Rules are sorted by priority.",
    )

    # Match conditions (all must match for rule to fire)
    logger_prefix: str = Field(
        default="",
        description="Logger family prefix to match (e.g., 'aiokafka'). Empty = match all.",
    )
    error_category: EnumRuntimeErrorCategory | None = Field(
        default=None,
        description="Error category to match. None = match all.",
    )
    message_pattern: str = Field(
        default="",
        description="Regex pattern to match against raw_message. Empty = match all.",
    )

    # Action
    action: Literal["suppress", "alert", "ticket"] = Field(
        default="alert",
        description="Action to take: suppress (ignore), alert (Slack), ticket (Linear).",
    )
    suppress_duration_minutes: int = Field(
        default=60,
        description="For 'suppress' action: suppress further alerts for this many minutes.",
    )

    def matches(
        self,
        logger_family: str,
        error_category: EnumRuntimeErrorCategory,
        raw_message: str,
    ) -> bool:
        """Check if this rule matches the given error event attributes.

        Args:
            logger_family: Logger name from the error event.
            error_category: Error category from the event.
            raw_message: Raw error message.

        Returns:
            True if all conditions match.
        """
        # Check logger prefix
        if self.logger_prefix and not logger_family.startswith(self.logger_prefix):
            return False

        # Check error category
        if self.error_category is not None and self.error_category != error_category:
            return False

        # Check message pattern
        if self.message_pattern:
            try:
                if not re.search(self.message_pattern, raw_message, re.IGNORECASE):
                    return False
            except re.error:
                # Invalid regex — treat as non-match
                return False

        return True


# Default triage rules for common runtime errors
DEFAULT_TRIAGE_RULES: list[ModelTriageRule] = [
    ModelTriageRule(
        name="aiokafka_heartbeat_failure",
        priority=10,
        logger_prefix="aiokafka",
        error_category=EnumRuntimeErrorCategory.KAFKA_CONSUMER,
        message_pattern=r"heartbeat|session.*timeout|coordinator.*dead",
        action="alert",
    ),
    ModelTriageRule(
        name="aiokafka_rebalance",
        priority=20,
        logger_prefix="aiokafka",
        error_category=EnumRuntimeErrorCategory.KAFKA_CONSUMER,
        message_pattern=r"rebalance|partition.*revoked",
        action="suppress",
        suppress_duration_minutes=5,
    ),
    ModelTriageRule(
        name="asyncpg_connection_error",
        priority=10,
        logger_prefix="asyncpg",
        error_category=EnumRuntimeErrorCategory.DATABASE,
        message_pattern=r"connection.*refused|connection.*reset|cannot connect",
        action="alert",
    ),
    ModelTriageRule(
        name="asyncpg_query_error",
        priority=30,
        logger_prefix="asyncpg",
        error_category=EnumRuntimeErrorCategory.DATABASE,
        message_pattern=r"relation.*does not exist|column.*does not exist",
        action="ticket",
    ),
    ModelTriageRule(
        name="aiohttp_connection_error",
        priority=20,
        logger_prefix="aiohttp",
        error_category=EnumRuntimeErrorCategory.HTTP_CLIENT,
        message_pattern=r"connection.*refused|timeout|server disconnected",
        action="alert",
    ),
    ModelTriageRule(
        name="catch_all",
        priority=999,
        action="alert",
    ),
]


__all__ = ["DEFAULT_TRIAGE_RULES", "ModelTriageRule"]
