# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed delegation-route declaration resolved from a bounded lane."""

from pydantic import BaseModel, ConfigDict, Field


class ModelBoundedDelegationRoute(BaseModel):
    """Resolved lane route; broker is inherited from its lane declaration."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    lane: str = Field(min_length=1)
    broker: str = Field(min_length=1)
    consumer: str = Field(min_length=1)
    terminal_route: str = Field(min_length=1)
    repository_owner: str = Field(min_length=1)
