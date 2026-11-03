# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for ``broker.principal.describe`` (OMN-20823)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBrokerPrincipalDescribeRequest(BaseModel):
    """Read one principal's ACLs. It names no password and changes nothing."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    principal: str = Field(..., min_length=1, description="The principal to read.")


__all__: list[str] = ["ModelBrokerPrincipalDescribeRequest"]
