# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""What one projection readback attempt found, or why it made no claim.

OMN-18060. The readback used to return ``(fsm_state | None, error)`` and the
handler classified the two-state tuple. That shape has no room for the third
thing this leg can now say: *the readback could have run and I declined to run
it* — a DSN that arrived on the command line, or a DSN whose role carries
``SUPERUSER`` / ``BYPASSRLS``.

Encoding a refusal as ``(None, "some message")`` would have classified it as
``ERROR``, which reads as "the store did not answer". That distinction is the
whole diagnostic: an error sends you to look at the database, a refusal sends
you to look at how the canary was wired. So the transport returns a typed
outcome and the classification lives in one place.

There is deliberately no field on this model that could carry a DSN.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_chain_canary_effect.models.enum_projection_readback_status import (
    EnumProjectionReadbackStatus,
)


class ModelProjectionReadbackOutcome(BaseModel):
    """One projection readback attempt, as a fact rather than a verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EnumProjectionReadbackStatus = Field(
        description="What the readback established, or why it established nothing"
    )
    state: str = Field(
        default="",
        description=(
            "The FSM state the projection holds for this correlation id. "
            "Non-empty only for TERMINAL and STRANDED."
        ),
    )
    error: str = Field(
        default="",
        description=(
            "Why no claim is made, for the non-passing members. Sanitized: "
            "never carries the DSN, and never carries a credential."
        ),
    )


__all__ = ["ModelProjectionReadbackOutcome"]
