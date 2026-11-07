# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Request model for architecture validation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.nodes.node_architecture_validator.models.model_stack_shape_snapshot import (
    ModelStackShapeSnapshot,
)


class ModelArchitectureValidationRequest(BaseModel):
    """Request to validate architecture compliance.

    Contains nodes and handlers to check against architecture rules.
    Used as input to NodeArchitectureValidatorCompute.

    Attributes:
        nodes: Nodes to validate against architecture rules.
        handlers: Handlers to validate against architecture rules.
        rule_ids: Specific rule IDs to check. None means check all rules.
        fail_fast: If True, stop on first violation found.
        correlation_id: Correlation ID for tracing.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    stack_shape: ModelStackShapeSnapshot | None = Field(
        default=None,
        description="T1 stack snapshots for command and console-script validation",
    )

    nodes: tuple[object, ...] = Field(
        default_factory=tuple,
        description="Nodes to validate against architecture rules",
    )

    handlers: tuple[object, ...] = Field(
        default_factory=tuple,
        description="Handlers to validate against architecture rules",
    )

    rule_ids: tuple[str, ...] | None = Field(
        default=None,
        description="Specific rule IDs to check. None means check all rules.",
    )

    fail_fast: bool = Field(
        default=False,
        description="If True, stop on first violation found",
    )

    correlation_id: str | None = Field(
        default=None,
        description="Correlation ID for tracing",
    )

    @field_validator("nodes", "handlers", "rule_ids", mode="before")
    @classmethod
    def _wire_sequences(cls, value: object) -> object:
        """Materialized JSON arrays retain the model's immutable tuple representation."""
        return tuple(value) if isinstance(value, list) else value


__all__ = ["ModelArchitectureValidationRequest"]
