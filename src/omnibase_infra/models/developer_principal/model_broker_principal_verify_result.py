# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Result of ``broker.principal.verify`` (OMN-20822)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBrokerPrincipalVerifyResult(BaseModel):
    """Whether the principal's ACLs match the plan and it can authenticate."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    principal: str = Field(..., min_length=1, description="The principal.")
    verified: bool = Field(..., description="ACLs match and the login listed topics.")
    visible_topics: tuple[str, ...] = Field(
        default_factory=tuple, description="Topics the principal can see, sorted."
    )
    reason: str = Field(default="", description="Why verification did not pass.")
