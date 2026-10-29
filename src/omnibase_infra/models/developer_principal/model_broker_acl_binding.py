# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One ACL binding of a developer principal, as created or as read back (OMN-20822)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

_NAME = r"^[a-z_]+$"


class ModelBrokerAclBinding(BaseModel):
    """One binding: the unit the effect creates, reads back and compares as a set.

    The effect only ever creates ALLOW bindings for any host, on a topic or group,
    for describe, read or write. A binding read back from the broker keeps every
    field that sets it apart (its resource type, pattern, operation, permission and
    host, as the broker names them, lower-cased), so a DENY, a host-scoped ALLOW or
    an operation the plan cannot express is a distinct binding the drift check
    names, never one that compares equal to a planned binding or is dropped.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    resource: str = Field(..., pattern=_NAME, description="Resource type.")
    name: str = Field(..., min_length=1, description="Resource name or prefix.")
    pattern: str = Field(..., pattern=_NAME, description="Name match.")
    operation: str = Field(
        ..., pattern=_NAME, description="The one operation this binding names."
    )
    permission: str = Field(
        default="allow", pattern=_NAME, description="allow or deny."
    )
    host: str = Field(default="*", min_length=1, description="The host it applies to.")

    def label(self) -> str:
        """A stable human-readable form, used in refusals and errors."""
        base = f"{self.resource}:{self.name}({self.pattern}):{self.operation}"
        if self.permission == "allow" and self.host == "*":
            return base
        return f"{base}[{self.permission} host={self.host}]"
