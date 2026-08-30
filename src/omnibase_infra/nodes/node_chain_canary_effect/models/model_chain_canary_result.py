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
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_terminal_readback_status import (
    EnumTerminalReadbackStatus,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.model_chain_link_verdict import (
    ModelChainLinkVerdict,
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
    ingress_terminal_event: str = Field(
        default="",
        description=(
            "Terminal event type the ingress CLAIMED, verbatim. Recorded, "
            "never trusted: OMN-15468 is the live proof that this lane can "
            "answer ok=true with a terminal name while nothing durable "
            "landed. It is here so a receipt can show the discrepancy "
            "between the claim and terminal_event below."
        ),
    )
    terminal_event: str = Field(
        default="",
        description=(
            "The terminal actually READ BACK off the bus for this run's "
            "correlation id, named by the topic it was found on. Empty means "
            "the readback did not find one. This — not the ingress response "
            "— is what discharges OMN-16025 link 4 (OMN-16931)."
        ),
    )

    terminal_readback_status: EnumTerminalReadbackStatus = Field(
        default=EnumTerminalReadbackStatus.SKIPPED_NOT_CONFIGURED,
        description="Whether the correlation-scoped terminal readback ran, and what it saw.",
    )
    terminal_topic: str = Field(
        default="",
        description="Topic the terminal was read back from; empty when none was found.",
    )
    terminal_topics_scanned: tuple[str, ...] = Field(
        default=(),
        description="Declared terminal topics the readback consumed from.",
    )
    terminal_readback_records_scanned: int = Field(
        default=0, ge=0, description="Records read during the terminal readback."
    )
    terminal_readback_window_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Wall-clock window the readback was allowed — the remainder of "
            "budget_ms after the ingress answered, floored at "
            "terminal_readback_timeout_seconds."
        ),
    )
    terminal_readback_error: str = Field(
        default="", description="Sanitized error from the terminal readback, if any."
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

    link_verdicts: tuple[ModelChainLinkVerdict, ...] = Field(
        default=(),
        description=(
            "One verdict per OMN-16025 chain link. The scalar `verdict` "
            "above answers 'did this probe's own checks pass'; THIS answers "
            "'which links of the five-link gate are proven'. They are not "
            "the same question, and conflating them let a three-link probe "
            "report a five-link gate as green (OMN-16931)."
        ),
    )
    links_proven: int = Field(
        default=0,
        ge=0,
        description="Count of links with status PASS. Nothing else counts.",
    )
    links_total: int = Field(
        default=0, ge=0, description="Total links in the OMN-16025 gate (five)."
    )
    chain_proof_complete: bool = Field(
        default=False,
        description=(
            "True only when EVERY link is PASS. This is the field that "
            "answers 'is the delegation chain proven'. A GREEN verdict with "
            "chain_proof_complete=False means the probe's own checks passed "
            "and the gate is still open."
        ),
    )

    kill_switch_engaged: bool = Field(
        default=False,
        description=(
            "True when ONEX_CHAIN_CANARY_DISABLED was set — zero I/O was "
            "performed and no claim is made about the chain."
        ),
    )


__all__ = ["ModelChainCanaryResult"]
