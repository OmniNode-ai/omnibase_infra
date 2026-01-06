# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Validation Models.

Provides models for ONEX execution shape validation and chain validation,
including rules defining handler constraints, violation results for CI gate
integration, message category to node kind routing validation, coverage metrics
for routing validation, and correlation/causation chain violation tracking.

Exports:
    ModelCategoryMatchResult: Result of category matching operation
    ModelChainViolation: Result of chain violation detection
    ModelCoverageMetrics: Coverage metrics for routing validation
    ModelExecutionShapeRule: Rule defining handler type constraints
    ModelExecutionShapeValidation: Validates message category to node kind routing
    ModelExecutionShapeValidationResult: Aggregate result of execution shape validation
    ModelExecutionShapeViolationResult: Result of violation detection
    ModelValidationOutcome: Generic validation result model
"""

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
    "ModelCategoryMatchResult",
    "ModelChainViolation",
    "ModelCoverageMetrics",
    "ModelExecutionShapeRule",
    "ModelExecutionShapeValidation",
    "ModelExecutionShapeValidationResult",
    "ModelExecutionShapeViolationResult",
    "ModelValidationOutcome",
]
