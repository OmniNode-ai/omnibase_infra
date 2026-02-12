# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Contract governance fields for RRH validation tightening."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRRHContractGovernance(BaseModel):
    """Contract-level governance fields that tighten profile rules.

    These fields come from the ticket's ONEX contract and can only
    **tighten** the active profile — never loosen.

    Attributes:
        ticket_id: Ticket identifier (e.g. ``"OMN-2136"``).
        evidence_requirements: Evidence types required (e.g. ``["tests"]``).
            Activates RRH-1403 when ``"tests"`` is present.
        interfaces_touched: Interface categories modified (e.g. ``["topics"]``).
            Activates RRH-1201 when ``"topics"`` is present.
        deployment_targets: Deployment targets (e.g. ``["k8s"]``).
            Activates RRH-1301 when ``"k8s"`` is present.
        is_seam_ticket: Whether this is a cross-repo seam ticket.
            Switches to seam-ticket profile (all rules active).
        disallowed_fields: Contract fields that should not be present.
            Triggers RRH-1601 when non-empty.
        expected_branch_pattern: Regex pattern for expected branch name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    ticket_id: str = Field(default="", description="Ticket identifier.")
    evidence_requirements: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Required evidence types (e.g. 'tests').",
    )
    interfaces_touched: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Interface categories modified (e.g. 'topics').",
    )
    deployment_targets: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Deployment targets (e.g. 'k8s').",
    )
    is_seam_ticket: bool = Field(
        default=False,
        description="Cross-repo seam ticket flag.",
    )
    disallowed_fields: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Contract fields that should not be present.",
    )
    expected_branch_pattern: str = Field(
        default="",
        description="Expected branch name regex pattern (matched via re.fullmatch).",
    )


__all__: list[str] = ["ModelRRHContractGovernance"]
