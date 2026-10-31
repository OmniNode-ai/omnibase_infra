# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""One counterfactual model rule for savings estimation, declared in the node contract.

Related Tickets:
    - OMN-21623: model names leave the handler; the node contract declares them
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelSavingsCounterfactualRule(BaseModel):
    """One ordered substring rule mapping an actual model to its counterfactual."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    match: str = Field(
        ...,
        min_length=1,
        description="Lower-case substring of the actual model identifier",
    )
    counterfactual_model: str = Field(
        ..., min_length=1, description="Model the session is priced against"
    )


__all__: list[str] = ["ModelSavingsCounterfactualRule"]
