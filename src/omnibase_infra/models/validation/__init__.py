# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Execution Shape Validation Models.

Provides models for ONEX execution shape validation, including
rules defining handler constraints and violation results for
CI gate integration.

Exports:
    ModelExecutionShapeRule: Rule defining handler type constraints
    ModelExecutionShapeViolationResult: Result of violation detection
"""

from omnibase_infra.models.validation.model_execution_shape_rule import (
    ModelExecutionShapeRule,
)
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)

__all__ = [
    "ModelExecutionShapeRule",
    "ModelExecutionShapeViolationResult",
]
