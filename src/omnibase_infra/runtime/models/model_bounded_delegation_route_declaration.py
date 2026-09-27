# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One declared ``delegation_routes`` row of a bounded lane (OMN-18933).

Shape carried from the Codex draft omnibase_infra#3951. The omnimarket lane
overlay (``config/ci_bus_lanes.yaml``) owns the rows.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelBoundedDelegationRouteDeclaration(BaseModel):
    """One ``delegation_routes`` row of a bounded lane, as declared."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    consumer: str = Field(min_length=1)
    terminal_route: str = Field(min_length=1)
    repository_owner: str = Field(min_length=1)


__all__ = ["ModelBoundedDelegationRouteDeclaration"]
