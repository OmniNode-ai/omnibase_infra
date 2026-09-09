# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The outcome of attributing one consumer-flow delta (OMN-16753).

A ``(consumer_group, topic)`` flow delta observed on a topic some projection
declares has exactly THREE possible readings, and collapsing any two of them is
what produced two consecutive false verdicts on the ``.201`` stability lane:

* **ATTRIBUTED** — the delta's consumer group is provably one in-scope
  projection's subscription, so its ``messages_in``/``messages_dlq`` belong in
  that projection's ratio.
* **EXCLUDED** — the delta's consumer group is provably NOT a projection's at
  all: its ``{package}.{node}.consume.`` infix belongs to a contract that
  declares no ``db_io.db_tables`` projection target, so the group is a reducer,
  an effect or a forwarder that happens to share the topic. Flow a
  non-projection consumer took was never a projection's to take, and says
  nothing about projection liveness either way.
* **UNATTRIBUTABLE** — the delta's group IS (or may be) a projection's, but
  which one cannot be shown: several in-scope projections declare the topic and
  the group matched none of their infixes, or matched two. That is a question
  the runtime could not answer, and it degrades the dimension.

The third is the OMN-16753 round-2 rule and stays exactly as it was. The second
did not exist, so a reducer's flow was folded into the third: on 2026-09-08 at
23:52Z the stability lane's ``node_session_phase_reducer`` group carried the
two ``onex.evt.omniclaude.session-{started,ended}.v1`` topics that three
projections also declare, was matched by no projection infix, and — the topics
having three declarers, so the sole-declarer fallback was correctly guarded —
was recorded as unattributable. ``projection_dlq_saturation`` went DEGRADED
over it, the refresh gate refused, and the lane was rolled back onto the old
image with ``18085`` returning 503.

Why a validated three-state model rather than an optional name: the fold has to
tell "no projection" from "not a projection", and a bare ``str | None`` cannot.
The validator makes the impossible combinations unconstructable rather than
merely undocumented.

Related Tickets:
    - OMN-16753: this model, and the misattribution it closes
    - OMN-16994: the projection-liveness dimensions it feeds
    - OMN-15837: the stability refresh gate that rolled back over the verdict
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

#: The three readings of one flow delta. Declared as a Literal so a fourth
#: outcome is a type error where it is written rather than a value some
#: downstream ``if`` silently drops on the floor.
EnumFlowAttributionOutcome = Literal["ATTRIBUTED", "EXCLUDED", "UNATTRIBUTABLE"]


class ModelFlowAttribution(BaseModel):
    """How one ``(consumer_group, topic)`` flow delta was read this cycle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    consumer_group: str = Field(
        ...,
        min_length=1,
        description=(
            "The delta's consumer group, verbatim. Carried so an EXCLUDED "
            "reading can be rendered on the health detail with the group an "
            "operator can look up on the broker, rather than only the contract "
            "name it was resolved to."
        ),
    )
    outcome: EnumFlowAttributionOutcome = Field(
        ..., description="Which of the three readings this delta got"
    )
    projection: str | None = Field(
        default=None,
        description=(
            "The in-scope projection whose subscription produced this delta. "
            "Set on ATTRIBUTED and only on ATTRIBUTED."
        ),
    )
    owner_contract: str | None = Field(
        default=None,
        description=(
            "The non-projection contract whose consumer-group infix the delta's "
            "group carries. Set on EXCLUDED and only on EXCLUDED."
        ),
    )
    reason: str | None = Field(
        default=None,
        description=(
            "Why the delta was excluded, in the words the health detail "
            "renders. Set on EXCLUDED and only on EXCLUDED, so an exclusion "
            "can never reach the surface unexplained."
        ),
    )

    @model_validator(mode="after")
    def _fields_match_the_outcome(self) -> Self:
        """Reject the combinations that would make a reading unreadable."""
        if self.outcome == "ATTRIBUTED":
            if self.projection is None:
                raise ValueError("ATTRIBUTED requires a projection name")
            if self.owner_contract is not None or self.reason is not None:
                raise ValueError("ATTRIBUTED carries no exclusion owner or reason")
        elif self.outcome == "EXCLUDED":
            if self.owner_contract is None or self.reason is None:
                raise ValueError("EXCLUDED requires an owner contract and a reason")
            if self.projection is not None:
                raise ValueError("EXCLUDED carries no projection name")
        elif self.projection is not None or self.owner_contract is not None:
            raise ValueError("UNATTRIBUTABLE names neither a projection nor an owner")
        return self


__all__: list[str] = ["EnumFlowAttributionOutcome", "ModelFlowAttribution"]
