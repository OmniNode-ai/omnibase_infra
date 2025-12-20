# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Validation Models.

Provides models for ONEX execution shape validation and chain validation,
including rules defining handler constraints, violation results for CI gate
integration, message category to node kind routing validation, and
correlation/causation chain violation tracking.

Exports:
    ModelChainViolation: Result of chain violation detection
    ModelExecutionShapeRule: Rule defining handler type constraints
    ModelExecutionShapeValidation: Validates message category to node kind routing
    ModelExecutionShapeViolationResult: Result of violation detection
"""

from omnibase_infra.models.validation.model_chain_violation import ModelChainViolation
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
    "ModelChainViolation",
    "ModelExecutionShapeRule",
    "ModelExecutionShapeValidation",
    "ModelExecutionShapeViolationResult",
]
