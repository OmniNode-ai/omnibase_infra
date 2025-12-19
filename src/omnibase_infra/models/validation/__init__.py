# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""
ONEX Infrastructure Validation Models.

Provides validation models for infrastructure-specific concerns including
execution shape validation for message routing.

Exports:
    ModelExecutionShapeValidation: Validates message category to node kind routing
"""

from omnibase_infra.models.validation.model_execution_shape_validation import (
    ModelExecutionShapeValidation,
)

__all__ = ["ModelExecutionShapeValidation"]
