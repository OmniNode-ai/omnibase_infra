# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A contract-declared projection and the topics it must consume (OMN-16994).

The unit the projection-liveness dimensions are evaluated over. Selected from
the discovery manifest by
:func:`omnibase_infra.runtime.health.projection_liveness.select_projection_contracts`,
which admits a contract on exactly one signal — it declares ``db_io.db_tables``
and subscribes to at least one topic. That is the same discriminator the wiring
seam uses to choose the projection dispatch arm, so the health surface and the
wiring seam cannot disagree about what a projection is.

Related Tickets:
    - OMN-16994: projection liveness on the runtime health surface
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionContractRef(BaseModel):
    """A contract-declared projection and the topics it must consume."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(..., min_length=1, description="Contract/node name")
    subscribe_topics: tuple[str, ...] = Field(
        ..., min_length=1, description="Topics the projection must consume"
    )
    attributable_subscribe_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description=(
            "OMN-17562. The subset of ``subscribe_topics`` whose presence in "
            "the live, TOPIC-KEYED bus registry can only be this contract's "
            "own subscription -- every other contract in the manifest that "
            "declares the topic is itself wired with no live dispatcher here, "
            "so none of them could have put it there. Empty is the correct "
            "value whenever a contract with a live in-process dispatcher "
            "shares the topic: the registry cannot attribute a subscription "
            "to a contract, and reading a co-owner's subscription as this "
            "projection's is what reported three healthy projections as "
            "silent-loss sites on both .201 lanes on 2026-09-04. Populated "
            "only by "
            ":func:`omnibase_infra.runtime.health.projection_liveness."
            "select_kernel_nonwriting_projections`, which is the only caller "
            "that needs to attribute an attachment rather than expect one."
        ),
    )


__all__: list[str] = ["ModelProjectionContractRef"]
