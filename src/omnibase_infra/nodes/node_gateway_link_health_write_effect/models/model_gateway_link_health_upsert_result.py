# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""ModelGatewayLinkHealthUpsertResult - outcome of a single gateway_link_health upsert.

gateway_link_health is a latest-known-state projection (ON CONFLICT
(tenant_id) DO UPDATE), so `was_insert` distinguishes a first-seen tenant
edge from a refresh of an existing row -- mirrors ModelPrStateUpsertResult.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayLinkHealthUpsertResult(BaseModel):
    """Outcome of one gateway_link_health upsert."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool = Field(
        ...,
        description="Whether the upsert completed without error.",
    )
    tenant_id: str = Field(
        ...,
        min_length=1,
        description="Tenant slug persisted to the row.",
    )
    was_insert: bool = Field(
        ...,
        description="True if this tenant edge had no prior row (first-seen); "
        "False if an existing row was refreshed.",
    )


__all__ = ["ModelGatewayLinkHealthUpsertResult"]
