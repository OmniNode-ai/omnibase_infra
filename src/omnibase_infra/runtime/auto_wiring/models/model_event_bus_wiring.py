# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Event bus topic wiring declarations from a contract (OMN-7653)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEventBusWiring(BaseModel):
    """Event bus topic declarations extracted from a contract."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    subscribe_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topics this node subscribes to",
    )
    publish_topics: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Topics this node publishes to",
    )
    consumer_purpose: str | None = Field(
        default=None,
        description="Optional contract-declared consumer purpose",
    )
