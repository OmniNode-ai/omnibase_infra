# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One ALLOW ACL of a developer principal: a resource, a pattern and one operation (OMN-20822)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelBrokerAclBinding(BaseModel):
    """One ALLOW binding, the unit the effect creates, reads back and compares as a set."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    resource: Literal["topic", "group"] = Field(..., description="Resource type.")
    name: str = Field(..., min_length=1, description="Resource name or prefix.")
    pattern: Literal["literal", "prefixed"] = Field(..., description="Name match.")
    operation: Literal["describe", "read", "write"] = Field(
        ..., description="The one operation this binding allows."
    )

    def label(self) -> str:
        """A stable human-readable form, used in refusals and errors."""
        return f"{self.resource}:{self.name}({self.pattern}):{self.operation}"
