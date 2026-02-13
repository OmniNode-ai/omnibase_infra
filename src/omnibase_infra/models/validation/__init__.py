# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Infrastructure Validation Models.

Provides models for ONEX execution shape validation and chain validation,
including rules defining handler constraints, violation results for CI gate
integration, message category to node kind routing validation, coverage metrics
for routing validation, correlation/causation chain violation tracking,
Any type violation detection, and shared validation pipeline models.

Exports:
    ModelAnyTypeValidationResult: Aggregate result of Any type validation for CI
    ModelAnyTypeViolation: Result of Any type violation detection
    ModelCategoryMatchResult: Result of category matching operation
    ModelCheckResult: Individual validation check result
    ModelChainViolation: Result of chain violation detection
    ModelCoverageMetrics: Coverage metrics for routing validation
    ModelExecutionShapeRule: Rule defining handler type constraints
    ModelExecutionShapeValidation: Validates message category to node kind routing
    ModelExecutionShapeValidationResult: Aggregate result of execution shape validation
    ModelExecutionShapeViolationResult: Result of violation detection
    ModelExecutorResult: Aggregated executor result for validation pipeline
    ModelLocalHandlerValidationResult: Aggregate result of LocalHandler validation for CI
    ModelLocalHandlerViolation: Result of LocalHandler import violation detection
    ModelOutputValidationParams: Parameters for output validation
    ModelPlannedCheck: Single check in a validation plan
    ModelValidateAndRaiseParams: Parameters for validate and raise operations
    ModelValidationErrorParams: Parameters for security validation error creation
    ModelValidationOutcome: Generic validation result model
    ModelValidationPlan: Validation plan produced by orchestrator
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
from omnibase_infra.models.validation.model_check_result import ModelCheckResult
from omnibase_infra.models.validation.model_coverage_metrics import (
    ModelCoverageMetrics,
)
from omnibase_infra.models.validation.model_declarative_node_validation_result import (
    ModelDeclarativeNodeValidationResult,
)
from omnibase_infra.models.validation.model_declarative_node_violation import (
    ModelDeclarativeNodeViolation,
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
from omnibase_infra.models.validation.model_executor_result import ModelExecutorResult
from omnibase_infra.models.validation.model_flake_detection_result import (
    ModelFlakeDetectionResult,
)
from omnibase_infra.models.validation.model_flake_record import ModelFlakeRecord
from omnibase_infra.models.validation.model_localhandler_validation_result import (
    ModelLocalHandlerValidationResult,
)
from omnibase_infra.models.validation.model_localhandler_violation import (
    ModelLocalHandlerViolation,
)
from omnibase_infra.models.validation.model_output_validation_params import (
    ModelOutputValidationParams,
)
from omnibase_infra.models.validation.model_planned_check import ModelPlannedCheck
from omnibase_infra.models.validation.model_validate_and_raise_params import (
    ModelValidateAndRaiseParams,
)
from omnibase_infra.models.validation.model_validation_error_params import (
    ModelValidationErrorParams,
)
from omnibase_infra.models.validation.model_validation_outcome import (
    ModelValidationOutcome,
)
from omnibase_infra.models.validation.model_validation_plan import ModelValidationPlan

__all__ = [
    "ModelAnyTypeValidationResult",
    "ModelAnyTypeViolation",
    "ModelCategoryMatchResult",
    "ModelCheckResult",
    "ModelDeclarativeNodeValidationResult",
    "ModelDeclarativeNodeViolation",
    "ModelChainViolation",
    "ModelCoverageMetrics",
    "ModelExecutionShapeRule",
    "ModelExecutionShapeValidation",
    "ModelExecutionShapeValidationResult",
    "ModelExecutionShapeViolationResult",
    "ModelExecutorResult",
    "ModelFlakeDetectionResult",
    "ModelFlakeRecord",
    "ModelLocalHandlerValidationResult",
    "ModelLocalHandlerViolation",
    "ModelOutputValidationParams",
    "ModelPlannedCheck",
    "ModelValidateAndRaiseParams",
    "ModelValidationErrorParams",
    "ModelValidationOutcome",
    "ModelValidationPlan",
]
