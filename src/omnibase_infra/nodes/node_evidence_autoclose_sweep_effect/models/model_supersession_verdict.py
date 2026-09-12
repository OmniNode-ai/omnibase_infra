# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The OMN-18233 verified-supersession predicate's answer, with its evidence.

A bare boolean would leave an auditor unable to tell a resolved predicate from
an unasked one, so the verdict carries which pull request replaced the closed
bump, which version was required, and which was delivered. ``superseded=False``
is deliberately BOTH "refuted" and "could not be resolved": the closer holds on
either, so they are one decision, and ``detail`` is where they differ.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSupersessionVerdict(BaseModel):
    """Whether a closed-unmerged cascade bump is PROVEN superseded, and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    superseded: bool = Field(
        default=False,
        description=(
            "True only when all four clauses resolved on evidence. False is "
            "both 'refuted' and 'could not be resolved' — the closer holds on "
            "either, so they are the same decision, and `detail` distinguishes "
            "them for whoever reads the receipt."
        ),
    )
    detail: str = Field(
        default="",
        min_length=1,
        description=(
            "One sentence naming the clause that resolved or failed, and the "
            "facts it read. Carried into the hold reason verbatim."
        ),
    )
    replacement_pr: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of the merged pull request that moved the pin. Zero when "
            "no clause-2 replacement was proven."
        ),
    )
    required_version: str = Field(
        default="",
        description="Clause 1's version, or empty when clause 1 did not resolve.",
    )
    delivered_version: str = Field(
        default="",
        description=(
            "The version readable from the repository's own default branch, "
            "or empty when clause 4 did not resolve."
        ),
    )


__all__ = ["ModelSupersessionVerdict"]
