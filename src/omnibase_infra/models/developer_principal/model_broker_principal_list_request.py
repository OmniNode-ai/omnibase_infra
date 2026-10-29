# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request for ``broker.principal.list`` (OMN-20822)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBrokerPrincipalListRequest(BaseModel):
    """List the broker's principals that carry a declaration's prefix."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    admin_url: str = Field(..., min_length=1, description="The Redpanda Admin API URL.")
    prefix: str = Field(
        ..., min_length=1, description="The declaration's principal prefix."
    )
