# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul operation result model for registry operations."""

from pydantic import BaseModel, ConfigDict


class ModelConsulOperationResult(BaseModel):
    """Result of a Consul operation."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    success: bool
    service_id: str | None = None
    error: str | None = None
