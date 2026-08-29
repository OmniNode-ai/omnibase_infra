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


__all__: list[str] = ["ModelProjectionContractRef"]
