# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One OMN-16025 link's verdict inside a canary receipt (OMN-16931)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link import (
    EnumChainLink,
)
from omnibase_infra.nodes.node_chain_canary_effect.models.enum_chain_link_status import (
    EnumChainLinkStatus,
)


class ModelChainLinkVerdict(BaseModel):
    """What this run proved, failed to prove, or could not look at.

    ``owning_ticket`` is set for a ``NO_LEG`` link so the receipt itself
    routes the reader to the work that would close it. A gap that names its
    own ticket is a gap somebody can pick up; an unnamed gap is one that
    gets rediscovered.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    link: EnumChainLink = Field(..., description="Which OMN-16025 link this is.")
    status: EnumChainLinkStatus = Field(
        ..., description="Status of this link for this run. Only PASS counts."
    )
    detail: str = Field(
        default="",
        description="One line naming the evidence, or naming what is missing.",
    )
    owning_ticket: str = Field(
        default="",
        description=(
            "Ticket that owes this link a leg, for NO_LEG links. Empty for "
            "links the canary actually exercises."
        ),
    )


__all__ = ["ModelChainLinkVerdict"]
