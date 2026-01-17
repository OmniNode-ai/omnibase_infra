# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture Validator Node Package.

This package provides architecture validation COMPUTE_GENERIC nodes for validating
ONEX architecture patterns and compliance rules.

Available Validators:
    NodeArchitectureValidator (OMN-1099):
        Validates three core architecture rules:
        - ARCH-001: No Direct Handler Dispatch
        - ARCH-002: No Handler Publishing Events
        - ARCH-003: No Workflow FSM in Orchestrators

    NodeArchitectureValidatorCompute (OMN-1138):
        Generic architecture validator supporting pluggable rules via
        ProtocolArchitectureRule implementations.

Example:
    >>> from omnibase_core.models.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.architecture_validator import (
    ...     NodeArchitectureValidatorCompute,
    ...     ModelArchitectureValidationRequest,
    ... )
    >>>
    >>> # Create validator with rules
    >>> container = ModelONEXContainer()
    >>> validator = NodeArchitectureValidatorCompute(container, rules=my_rules)
    >>>
    >>> # Validate architecture
    >>> result = validator.compute(ModelArchitectureValidationRequest(
    ...     nodes=my_nodes,
    ...     handlers=my_handlers,
    ... ))
    >>> if result.valid:
    ...     print("Validation passed")

.. versionadded:: 0.8.0
    Added NodeArchitectureValidatorCompute as part of OMN-1138.
    Added NodeArchitectureValidator as part of OMN-1099.
"""

# Canonical severity enum
from omnibase_infra.enums import EnumValidationSeverity

# Models
from omnibase_infra.nodes.architecture_validator.models import (
    ModelArchitectureValidationRequest,
    ModelArchitectureValidationResult,
    ModelArchitectureViolation,
    ModelRuleCheckResult,
)
from omnibase_infra.nodes.architecture_validator.node import NodeArchitectureValidator
from omnibase_infra.nodes.architecture_validator.node_architecture_validator import (
    SUPPORTED_RULE_IDS,
    NodeArchitectureValidatorCompute,
)
from omnibase_infra.nodes.architecture_validator.protocols import (
    ProtocolArchitectureRule,
)
from omnibase_infra.nodes.architecture_validator.registry import (
    RegistryInfraArchitectureValidator,
)

__all__ = [
    # OMN-1138: NodeArchitectureValidatorCompute
    "NodeArchitectureValidatorCompute",
    "SUPPORTED_RULE_IDS",
    "EnumValidationSeverity",
    "ModelRuleCheckResult",
    "ProtocolArchitectureRule",
    # OMN-1099: NodeArchitectureValidator
    "NodeArchitectureValidator",
    "RegistryInfraArchitectureValidator",
    # Shared models
    "ModelArchitectureValidationRequest",
    "ModelArchitectureValidationResult",
    "ModelArchitectureViolation",
]
