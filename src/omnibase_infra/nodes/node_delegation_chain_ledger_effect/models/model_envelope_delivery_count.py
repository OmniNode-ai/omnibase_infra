# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exact redelivery count for one replay envelope (OMN-19729)."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelEnvelopeDeliveryCount(BaseModel):
    """The number of exact recorded deliveries collapsed into an envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    envelope_id: UUID
    delivery_count: int = Field(ge=1)


__all__ = ["ModelEnvelopeDeliveryCount"]
