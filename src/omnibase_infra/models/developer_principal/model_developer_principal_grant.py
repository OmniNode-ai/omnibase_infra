# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One broker grant a planned developer principal receives (OMN-20821)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelDeveloperPrincipalGrant(BaseModel):
    """A resource, its pattern and the operations granted on it.

    ``name`` is already resolved for the planned principal: a templated
    ``{principal}`` topic carries the principal's name here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    resource: Literal["topic", "group"] = Field(
        ..., description="Broker resource type."
    )
    name: str = Field(..., min_length=1, description="Resource name or prefix.")
    pattern: Literal["literal", "prefixed"] = Field(..., description="Name match.")
    operations: tuple[Literal["describe", "read", "write"], ...] = Field(
        ..., min_length=1, description="Granted operations, sorted."
    )
    create_topic: bool = Field(
        default=False,
        description="The issuer creates this principal-owned topic before granting it.",
    )


__all__: list[str] = ["ModelDeveloperPrincipalGrant"]
