# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Error Models Module.

This module provides Pydantic models for error context and metadata
in the ONEX infrastructure layer.

Models:
    - ModelInfraErrorContext: Context for infrastructure errors including
      transport type, operation, and correlation ID

Related:
    - errors.error_infra: Error classes that use these models
    - OMN-927: Infrastructure error patterns

.. versionadded:: 0.5.0
"""

from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)

__all__: list[str] = [
    "ModelInfraErrorContext",
]
