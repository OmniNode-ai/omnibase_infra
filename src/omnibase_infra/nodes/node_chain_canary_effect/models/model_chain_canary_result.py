# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Result model for one event-chain canary run (OMN-16773)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_canary_verdict import (
    EnumChainCanaryVerdict,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_quarantine_check_status import (
    EnumQuarantineCheckStatus,
)


class ModelChainCanaryResult(BaseModel):
    """The receipt for one canary run.

    This IS the append-only evidence record: the workflow prints it into
    the run's job summary verbatim, so a run's verdict, the correlation id
    it used, and the reason are all recoverable from GitHub's own
    retention without any separate store.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(..., description="Canary run correlation ID.")
    probe_correlation_id: UUID = Field(
        ...,
        description=(
            "The correlation ID minted for THIS run's delegation and put on "
            "the wire. Fresh per run; the join key for tracing the probe "
            "through the bus by hand afterwards."
        ),
    )
    verdict: EnumChainCanaryVerdict = Field(
        ..., description="Terminal verdict for this run."
    )
    success: bool = Field(
        ...,
        description=(
            "True only for GREEN and SKIPPED_DISABLED. The workflow exits "
            "non-zero when this is False — red must be visible."
        ),
    )
    detail: str = Field(
        default="",
        description="One-line human-readable reason for the verdict.",
    )

    probe_url: str = Field(default="", description="Ingress base URL probed.")
    runtime_command: str = Field(default="", description="Runtime command dispatched.")
    task_type: str = Field(default="", description="Delegation task class probed.")
    budget_ms: int = Field(default=0, ge=0, description="Runtime ingress budget used.")
    elapsed_ms: int = Field(
        default=0, ge=0, description="Observed wall-clock time of the ingress call."
    )

    ingress_ok: bool = Field(
        default=False, description="The /skill response's own ok flag."
    )
    ingress_error_code: str = Field(
        default="",
        description="Error code from the ingress response, e.g. 'dispatch_timeout'.",
    )
    ingress_error_message: str = Field(
        default="", description="Sanitized error text from the ingress or transport."
    )
    terminal_event: str = Field(
        default="",
        description=(
            "Terminal event type reported by the ingress. Empty means no "
            "terminal — which is RED even when ok=true (OMN-16027)."
        ),
    )

    quarantine_status: EnumQuarantineCheckStatus = Field(
        default=EnumQuarantineCheckStatus.SKIPPED_NOT_CONFIGURED,
        description="Whether the correlation-scoped quarantine check ran, and what it saw.",
    )
    quarantine_topic: str = Field(default="", description="Quarantine topic scanned.")
    quarantine_records_scanned: int = Field(
        default=0, ge=0, description="Records read from the quarantine tail."
    )
    quarantine_error: str = Field(
        default="", description="Sanitized error from the quarantine scan, if any."
    )

    kill_switch_engaged: bool = Field(
        default=False,
        description=(
            "True when ONEX_CHAIN_CANARY_DISABLED was set — zero I/O was "
            "performed and no claim is made about the chain."
        ),
    )


__all__ = ["ModelChainCanaryResult"]
