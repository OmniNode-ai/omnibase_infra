# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Operation bindings subcontract model for contract.yaml section.

This model represents the full operation_bindings section from a contract.yaml file,
containing pre-parsed bindings for all operations plus optional global bindings.

.. versionadded:: 0.2.6
    Created as part of OMN-1518 - Declarative operation bindings.

.. versionchanged:: 0.2.7
    Added max_expression_length and max_path_segments for per-contract guardrail overrides.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.bindings.constants import (
    DEFAULT_JSON_RECURSION_DEPTH,
    MAX_EXPRESSION_LENGTH,
    MAX_JSON_RECURSION_DEPTH,
    MAX_PATH_SEGMENTS,
    MIN_JSON_RECURSION_DEPTH,
)
from omnibase_infra.models.bindings.model_parsed_binding import ModelParsedBinding

# =============================================================================
# Guardrail Bounds Constants
# =============================================================================

# Expression length bounds
MIN_EXPRESSION_LENGTH: int = 32
"""Minimum allowed value for max_expression_length override.

Prevents trivially small expressions that would be unusable for real bindings.
A minimal valid expression like ${payload.x} is 12 characters.
"""

MAX_EXPRESSION_LENGTH_LIMIT: int = 1024
"""Maximum allowed value for max_expression_length override.

Prevents DoS via excessively long expressions that could exhaust memory
or CPU during regex matching.
"""

# Path segment bounds
MIN_PATH_SEGMENTS: int = 3
"""Minimum allowed value for max_path_segments override.

Need at least source.path.field (3 segments) for useful bindings.
"""

MAX_PATH_SEGMENTS_LIMIT: int = 50
"""Maximum allowed value for max_path_segments override.

Prevents absurdly deep paths that could cause stack issues or DoS.
"""


class ModelOperationBindingsSubcontract(BaseModel):
    """Full operation_bindings section from contract.yaml.

    Contains pre-parsed bindings for all operations, plus optional
    global bindings applied to every operation.

    Example YAML:
        operation_bindings:
          version: { major: 1, minor: 0, patch: 0 }
          max_expression_length: 512  # Override default (256)
          max_path_segments: 30       # Override default (20)
          max_json_recursion_depth: 50  # Override default (100)
          additional_context_paths:
            - "request_id"      # Handler needs request ID
            - "tenant_id"       # Multi-tenant context
          global_bindings:
            - parameter_name: "correlation_id"
              expression: "${envelope.correlation_id}"
          bindings:
            "db.query":
              - parameter_name: "sql"
                expression: "${payload.sql}"
              - parameter_name: "tenant"
                expression: "${context.tenant_id}"
                required: true

    Attributes:
        version: Schema version for evolution tracking.
        additional_context_paths: Additional context paths this handler can resolve
            beyond the base set (now_iso, dispatcher_id, correlation_id).
            When declared, the dispatch engine is CONTRACTED to provide these
            values in the context dict. Pattern: ^[a-z][a-z0-9_]*$
        bindings: Mapping of operation name to list of parsed bindings.
        global_bindings: Optional bindings applied to all operations (can be overridden).
        max_expression_length: Override default expression length limit (32-1024).
        max_path_segments: Override default path segment limit (3-50).
        max_json_recursion_depth: Maximum depth for JSON compatibility validation (10-1000).
            Limits how deeply nested structures are validated to prevent stack overflow.

    .. versionadded:: 0.2.6
    .. versionchanged:: 0.2.7
        Added max_expression_length, max_path_segments, and max_json_recursion_depth
        for per-contract guardrail overrides.
        Added additional_context_paths for extensible context resolution.
    """

    version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Schema version for evolution tracking",
    )
    additional_context_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Additional context paths this handler can resolve beyond the base set "
            "(now_iso, dispatcher_id, correlation_id). When declared, the dispatch "
            "engine is CONTRACTED to provide these values. "
            "Pattern: ^[a-z][a-z0-9_]*$ (lowercase letters, numbers, underscores)"
        ),
    )
    bindings: dict[str, list[ModelParsedBinding]] = Field(
        default_factory=dict,
        description="Operation name -> list of parsed bindings",
    )
    global_bindings: list[ModelParsedBinding] | None = Field(
        default=None,
        description="Bindings applied to all operations (can be overridden)",
    )
    max_expression_length: int = Field(
        default=MAX_EXPRESSION_LENGTH,
        ge=MIN_EXPRESSION_LENGTH,
        le=MAX_EXPRESSION_LENGTH_LIMIT,
        description="Override default expression length limit (32-1024)",
    )
    max_path_segments: int = Field(
        default=MAX_PATH_SEGMENTS,
        ge=MIN_PATH_SEGMENTS,
        le=MAX_PATH_SEGMENTS_LIMIT,
        description="Override default path segment limit (3-50)",
    )
    max_json_recursion_depth: int = Field(
        default=DEFAULT_JSON_RECURSION_DEPTH,
        ge=MIN_JSON_RECURSION_DEPTH,
        le=MAX_JSON_RECURSION_DEPTH,
        description=(
            f"Maximum depth for JSON compatibility validation ({MIN_JSON_RECURSION_DEPTH}-"
            f"{MAX_JSON_RECURSION_DEPTH}). Limits how deeply nested structures are "
            "validated to prevent stack overflow on pathological inputs."
        ),
    )

    model_config = {"frozen": True, "extra": "forbid"}
