# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""MCP tool parameter model for representing tool input parameters."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelMCPToolParameter(BaseModel):
    """Parameter definition for an MCP tool.

    Represents a single parameter that can be passed to an MCP tool,
    including its type, validation constraints, and documentation.
    """

    name: str = Field(description="Parameter name")
    parameter_type: str = Field(
        default="string",
        description="JSON Schema type: string, number, boolean, array, object",
    )
    description: str = Field(default="", description="Human-readable description")
    required: bool = Field(
        default=True, description="Whether this parameter is required"
    )
    default_value: object | None = Field(
        default=None, description="Default value if not provided"
    )
    json_schema: dict[str, object] | None = Field(
        default=None,
        description="Additional JSON Schema constraints (enum, format, etc.)",
    )


__all__ = ["ModelMCPToolParameter"]
