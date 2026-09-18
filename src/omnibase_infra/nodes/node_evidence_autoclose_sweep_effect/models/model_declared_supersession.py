# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One supersession a TICKET declares for a closed-unmerged citation (OMN-18749).

The OMN-18233 predicate proves supersession for dependency-cascade bumps by
reading a pin. Most closed citations are not bumps — a proof pull request
reopened on a clean branch, a pull request closed by an accidental branch
rename — and for those the only durable statement of where the work went is the
one the ticket itself makes. This model is that statement, parsed; it proves
nothing on its own, because the successor's merge state is a separate probe.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDeclaredSupersession(BaseModel):
    """A `<closed> superseded by <successor>` line, parsed, not yet verified."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    closed_repo: str = Field(min_length=1, description="`owner/repo` of the closed PR.")
    closed_number: int = Field(gt=0, description="The closed pull request's number.")
    successor_repo: str = Field(
        default="",
        description=(
            "`owner/repo` of the declared successor, or empty when the right "
            "side of the line named nothing that resolves to a repository. "
            "Empty is NOT 'no declaration' — it is a declaration that cannot "
            "be checked, which holds, and says so."
        ),
    )
    successor_number: int = Field(
        default=0, ge=0, description="The successor's number, or zero when unresolved."
    )
    line: str = Field(
        min_length=1,
        description="The declaration line verbatim, carried into the hold reason.",
    )


__all__ = ["ModelDeclaredSupersession"]
