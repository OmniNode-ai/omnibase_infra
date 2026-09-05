# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One In-Progress ticket's probe-hygiene outcome (OMN-17942)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_in_progress_probe_hygiene_effect.models.enum_probe_hygiene_decision import (
    EnumProbeHygieneDecision,
)


class ModelProbeHygieneOutcome(BaseModel):
    """What the sweep found, and said, about one ticket."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticket: str = Field(..., description="Linear ticket identifier, e.g. OMN-17926.")
    decision: EnumProbeHygieneDecision = Field(
        ..., description="Terminal verdict for this ticket."
    )
    reason: str = Field(
        default="",
        description="Why this verdict, in words a person can act on.",
    )
    occ_contract_checks: int = Field(
        default=0,
        ge=0,
        description=(
            "Executable checks declared across the ticket's OCC contract "
            "dod_evidence items. 0 means the closer's verifier has nothing to "
            "run for this ticket."
        ),
    )
    description_probe_lines: int = Field(
        default=0,
        ge=0,
        description=(
            "Well-formed probe lines in the Linear description — the "
            "OMN-17942 creation-gate grammar. A ticket filed before that gate "
            "landed has none, which is why the OCC contract is checked first "
            "and this is the second chance rather than the only one."
        ),
    )
    comment_posted: bool = Field(
        default=False,
        description="Whether THIS run wrote the hygiene comment.",
    )


__all__ = ["ModelProbeHygieneOutcome"]
