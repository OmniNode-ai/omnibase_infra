# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runner health alert model with Slack message formatter."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.observability.runner_health.enum_runner_health_state import (
    EnumRunnerHealthState,
)
from omnibase_infra.observability.runner_health.model_runner_status import (
    ModelRunnerStatus,
)


class ModelRunnerHealthAlert(BaseModel):
    """Alert payload for degraded runner health.

    Includes a ``to_slack_message()`` method that formats runner-specific
    degradation details for Slack Block Kit consumption.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Trace correlation ID")
    alert_type: Literal["runner_health_degraded"] = Field(
        default="runner_health_degraded"
    )
    degraded_runners: tuple[ModelRunnerStatus, ...] = Field(
        default_factory=tuple, description="Runners in degraded state"
    )
    total_runners: int = Field(..., description="Expected runner count")
    healthy_count: int = Field(..., description="Healthy runner count")
    host: str = Field(..., description="CI host address")

    def to_slack_message(self) -> str:
        """Format alert as a Slack-compatible message string."""
        lines = [
            f":warning: *Runner Health Degraded* — {self.healthy_count}/{self.total_runners} healthy on `{self.host}`",
            "",
        ]
        for r in self.degraded_runners:
            state_emoji = {
                EnumRunnerHealthState.GITHUB_OFFLINE: ":red_circle:",
                EnumRunnerHealthState.CRASH_LOOPING: ":rotating_light:",
                EnumRunnerHealthState.DOCKER_UNHEALTHY: ":warning:",
                EnumRunnerHealthState.STALE_REGISTRATION: ":ghost:",
                EnumRunnerHealthState.MISSING: ":question:",
            }.get(r.state, ":x:")
            detail = f" — {r.error}" if r.error else ""
            lines.append(
                f"{state_emoji} `{r.name}`: {r.state.value} "
                f"(GitHub: {r.github_status}, Docker: {r.docker_status}){detail}"
            )
        return "\n".join(lines)
