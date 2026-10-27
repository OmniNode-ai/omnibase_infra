# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Names-only consumer group membership record for secret gating (OMN-21423)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelSecretGatedMembership(BaseModel):
    """Record whether a runtime joined a contract requiring a named secret."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    contract: str = Field(..., description="Contract (node) name the record is for")
    consumer_group: str = Field(
        default="",
        description="Contract-declared consumer group; empty when runtime-derived",
    )
    secret_ref: str = Field(
        ..., description="Name of the required secret, never its value"
    )
    token: Literal["present", "absent"]
    joined: bool

    def render(self) -> str:
        """Render membership using names and presence only, never secret values."""
        return (
            f"contract={self.contract} secret={self.secret_ref} "
            f"token={self.token} joined={'true' if self.joined else 'false'}"
        )
