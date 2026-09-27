# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One terminal publish attempt, reported rather than raised (OMN-19152)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict

from omnibase_infra.cli.enum_skill_terminal_publish_outcome import (
    EnumSkillTerminalPublishOutcome,
)


class ModelSkillTerminalPublishReport(BaseModel):
    """One publish attempt, reported rather than raised."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    outcome: EnumSkillTerminalPublishOutcome
    lane: str
    topic: str = ""
    correlation_id: UUID | None = None
    detail: str = ""

    def render(self) -> str:
        """One stderr line naming the outcome."""
        if self.outcome is EnumSkillTerminalPublishOutcome.PUBLISHED:
            return (
                f"onex skill: terminal published to lane {self.lane} topic "
                f"{self.topic} correlation_id={self.correlation_id}"
            )
        if self.outcome is EnumSkillTerminalPublishOutcome.FAILED:
            return (
                f"onex skill: terminal publish failed for lane {self.lane} "
                f"(verdict, receipt and exit code unchanged): {self.detail}"
            )
        return f"onex skill: terminal not published ({self.outcome.value})"
