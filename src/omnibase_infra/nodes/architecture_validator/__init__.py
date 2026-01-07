# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture validator node for ONEX compliance checking.

This module provides the NodeArchitectureValidatorCompute node for validating
architecture compliance against ONEX rules. The validator can check nodes,
handlers, and other architectural components for compliance with patterns
defined in ProtocolArchitectureRule implementations.

Example:
    >>> from omnibase_core.models.container import ModelONEXContainer
    >>> from omnibase_infra.nodes.architecture_validator import (
    ...     NodeArchitectureValidatorCompute,
    ...     ModelArchitectureValidationRequest,
    ... )
    >>>
    >>> # Create validator with rules
    >>> container = ModelONEXContainer.minimal()
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
"""

from omnibase_infra.nodes.architecture_validator.enums import EnumValidationSeverity
from omnibase_infra.nodes.architecture_validator.models import (
    ModelArchitectureValidationRequest,
    ModelArchitectureValidationResult,
    ModelArchitectureViolation,
    ModelRuleCheckResult,
)
from omnibase_infra.nodes.architecture_validator.node_architecture_validator import (
    NodeArchitectureValidatorCompute,
)
from omnibase_infra.nodes.architecture_validator.protocols import (
    ProtocolArchitectureRule,
)

__all__ = [
    # Node
    "NodeArchitectureValidatorCompute",
    # Enums
    "EnumValidationSeverity",
    # Models
    "ModelArchitectureValidationRequest",
    "ModelArchitectureValidationResult",
    "ModelArchitectureViolation",
    "ModelRuleCheckResult",
    # Protocols
    "ProtocolArchitectureRule",
]
