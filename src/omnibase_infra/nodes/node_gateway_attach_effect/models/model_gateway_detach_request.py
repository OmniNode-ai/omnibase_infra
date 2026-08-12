# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Detach request -- explicit, edge-initiated session teardown."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelGatewayDetachRequest(BaseModel):
    """Input to ``gateway.detach``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: UUID
    reason: str = Field(min_length=1, max_length=500)


__all__ = ["ModelGatewayDetachRequest"]
