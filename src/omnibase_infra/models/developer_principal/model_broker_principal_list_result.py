# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of ``broker.principal.list`` (OMN-20822)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBrokerPrincipalListResult(BaseModel):
    """The principals the broker holds under the prefix, sorted."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    principals: tuple[str, ...] = Field(
        default_factory=tuple, description="Sorted names."
    )
