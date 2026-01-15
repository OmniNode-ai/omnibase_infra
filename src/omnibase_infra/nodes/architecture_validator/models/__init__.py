# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for the architecture validator nodes.

This module exports all models used by both architecture validators:
- OMN-1138: NodeArchitectureValidatorCompute (pluggable rules)
- OMN-1099: NodeArchitectureValidator (core pattern validation)
"""

# Canonical severity enum
from omnibase_infra.enums import EnumValidationSeverity

# OMN-1138: Models for NodeArchitectureValidatorCompute
# OMN-1099: Models for NodeArchitectureValidator
from omnibase_infra.nodes.architecture_validator.models.model_architecture_validation_request import (
    ModelArchitectureValidationRequest,
)
from omnibase_infra.nodes.architecture_validator.models.model_architecture_validation_result import (
    ModelArchitectureValidationResult,
)
from omnibase_infra.nodes.architecture_validator.models.model_architecture_violation import (
    ModelArchitectureViolation,
)
from omnibase_infra.nodes.architecture_validator.models.model_rule_check_result import (
    ModelRuleCheckResult,
)

__all__ = [
    # OMN-1138 models
    "ModelArchitectureValidationRequest",
    "ModelArchitectureValidationResult",
    "ModelArchitectureViolation",
    "ModelRuleCheckResult",
    # Canonical severity enum (from enums/ directory)
    "EnumValidationSeverity",
]
