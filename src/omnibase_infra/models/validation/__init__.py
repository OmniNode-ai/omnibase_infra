# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Validation Models.

Provides models for ONEX execution shape validation and chain validation,
including rules defining handler constraints, violation results for CI gate
integration, message category to node kind routing validation, coverage metrics
for routing validation, correlation/causation chain violation tracking,
and Any type violation detection.

Exports:
    ModelAnyTypeValidationResult: Aggregate result of Any type validation for CI
    ModelAnyTypeViolation: Result of Any type violation detection
    ModelCategoryMatchResult: Result of category matching operation
    ModelChainViolation: Result of chain violation detection
    ModelCoverageMetrics: Coverage metrics for routing validation
    ModelExecutionShapeRule: Rule defining handler type constraints
    ModelExecutionShapeValidation: Validates message category to node kind routing
    ModelExecutionShapeValidationResult: Aggregate result of execution shape validation
    ModelExecutionShapeViolationResult: Result of violation detection
    ModelValidationOutcome: Generic validation result model
"""

from omnibase_infra.models.validation.model_any_type_validation_result import (
    ModelAnyTypeValidationResult,
)
from omnibase_infra.models.validation.model_any_type_violation import (
    ModelAnyTypeViolation,
)
from omnibase_infra.models.validation.model_category_match_result import (
    ModelCategoryMatchResult,
)
from omnibase_infra.models.validation.model_chain_violation import ModelChainViolation
from omnibase_infra.models.validation.model_coverage_metrics import (
    ModelCoverageMetrics,
)
from omnibase_infra.models.validation.model_execution_shape_rule import (
    ModelExecutionShapeRule,
)
from omnibase_infra.models.validation.model_execution_shape_validation import (
    ModelExecutionShapeValidation,
)
from omnibase_infra.models.validation.model_execution_shape_validation_result import (
    ModelExecutionShapeValidationResult,
)
from omnibase_infra.models.validation.model_execution_shape_violation import (
    ModelExecutionShapeViolationResult,
)
from omnibase_infra.models.validation.model_validation_outcome import (
    ModelValidationOutcome,
)

__all__ = [
    "ModelAnyTypeValidationResult",
    "ModelAnyTypeViolation",
    "ModelCategoryMatchResult",
    "ModelChainViolation",
    "ModelCoverageMetrics",
    "ModelExecutionShapeRule",
    "ModelExecutionShapeValidation",
    "ModelExecutionShapeValidationResult",
    "ModelExecutionShapeViolationResult",
    "ModelValidationOutcome",
]
