# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Operation binding model for contract.yaml entries."""

from __future__ import annotations

from pydantic import BaseModel, Field

from omnibase_core.types import JsonType


class ModelOperationBinding(BaseModel):
    """Single binding entry from contract.yaml.

    Maps a handler parameter to an expression that extracts data from
    envelope, payload, or context.

    Example YAML:
        - parameter_name: "correlation_id"
          expression: "${envelope.correlation_id}"
          required: true
    """

    parameter_name: str = Field(
        ...,
        description="Target handler input field name",
    )
    expression: str = Field(
        ...,
        description="Source expression in ${source.path} format",
    )
    required: bool = Field(
        default=True,
        description="If True, fail fast when field is missing",
    )
    default: JsonType | None = Field(
        default=None,
        description="Default value if not required and missing",
    )

    model_config = {"frozen": True, "extra": "forbid"}


__all__ = ["ModelOperationBinding"]
