# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The receipt's account of deploys around the probe window (OMN-19811)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_deploy_window_status import (
    EnumDeployWindowStatus,
)


class ModelDeployWindowEvidence(BaseModel):
    """What the deploy agent said before and after the probe fired.

    When ``status`` is ``DEPLOY_IN_WINDOW_RETRIED`` the receipt's top-level
    fields describe the RETRY, and the first attempt survives here
    (``first_attempt_*``) so a reader can see both runs and the deploy job
    that separated them.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    status: EnumDeployWindowStatus = Field(
        default=EnumDeployWindowStatus.NOT_CONFIGURED,
        description="Why the run did or did not retry.",
    )
    agent_url: str = Field(default="", description="Deploy agent surface read.")
    detail: str = Field(default="", description="One-line account of the decision.")

    preflight_deploy_correlation_id: UUID | None = Field(
        default=None,
        description="The deploy job in flight when the probe was about to fire, if any.",
    )
    preflight_waited_seconds: float = Field(
        default=0.0, ge=0.0, description="How long the probe waited before firing."
    )
    preflight_converged: bool = Field(
        default=True,
        description=(
            "False when a deploy was in flight and the lane had not converged "
            "when the pre-fire wait ran out; the probe fired anyway."
        ),
    )
    preflight_queued_commands: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Deploy commands the agent's /queue reported waiting at pre-fire; "
            "None when the queue was unread or its lag sample was stale."
        ),
    )
    preflight_agent_error: str = Field(
        default="", description="Sanitized agent read error at pre-fire, if any."
    )

    window_started_at: str = Field(
        default="", description="UTC start of the first attempt's window (ISO 8601)."
    )
    window_ended_at: str = Field(
        default="", description="UTC end of the first attempt's window (ISO 8601)."
    )
    deploy_correlation_ids: tuple[UUID, ...] = Field(
        default=(),
        description=(
            "Deploy-agent job correlation ids accepted, running or completed "
            "inside the first attempt's window."
        ),
    )

    first_attempt_probe_correlation_id: UUID | None = Field(
        default=None, description="The first attempt's probe correlation id."
    )
    first_attempt_verdict: EnumChainCanaryVerdict | None = Field(
        default=None, description="The first attempt's verdict."
    )
    first_attempt_detail: str = Field(
        default="", description="The first attempt's detail line."
    )
    convergence_waited_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="How long the run waited for the lane to converge before retrying.",
    )
    retried: bool = Field(
        default=False, description="True when the probe was fired a second time."
    )


__all__ = ["ModelDeployWindowEvidence"]
