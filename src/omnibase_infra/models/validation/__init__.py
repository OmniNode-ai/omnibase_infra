# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Validation Models.

Provides models for ONEX execution shape validation, including
rules defining handler constraints, violation results for CI gate
integration, and message category to node kind routing validation.

Exports:
    ModelExecutionShapeRule: Rule defining handler type constraints
    ModelExecutionShapeValidation: Validates message category to node kind routing
    ModelExecutionShapeViolationResult: Result of violation detection
"""

from omnibase_infra.models.validation.model_execution_shape_rule import (
    ModelExecutionShapeRule,
)
from omnibase_infra.models.validation.model_execution_shape_validation import (
    ModelExecutionShapeValidation,
)
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)

__all__ = [
    "ModelExecutionShapeRule",
    "ModelExecutionShapeValidation",
    "ModelExecutionShapeViolationResult",
]
