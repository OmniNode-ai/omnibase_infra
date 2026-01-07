# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Architecture Validator Node Package.

This package provides the Architecture Validator COMPUTE node for validating
ONEX architecture patterns. It detects violations of three core rules:

    ARCH-001: No Direct Handler Dispatch
        Handlers MUST NOT be invoked directly bypassing the RuntimeHost.

    ARCH-002: No Handler Publishing Events
        Handlers MUST NOT have direct event bus access.

    ARCH-003: No Workflow FSM in Orchestrators
        Orchestrators MUST NOT duplicate reducer FSM transitions.

Available Classes:
    - NodeArchitectureValidator: COMPUTE node for architecture validation
    - RegistryInfraArchitectureValidator: DI registration for the node

Available Models:
    - ModelArchitectureValidationRequest: Input request
    - ModelArchitectureValidationResult: Output result
    - ModelArchitectureViolation: Single violation
    - EnumViolationSeverity: Severity levels (ERROR, WARNING)

Ticket: OMN-1099
"""

from omnibase_infra.nodes.architecture_validator.models import (
    EnumViolationSeverity,
    ModelArchitectureValidationRequest,
    ModelArchitectureValidationResult,
    ModelArchitectureViolation,
)
from omnibase_infra.nodes.architecture_validator.node import NodeArchitectureValidator
from omnibase_infra.nodes.architecture_validator.registry import (
    RegistryInfraArchitectureValidator,
)

__all__ = [
    "EnumViolationSeverity",
    "ModelArchitectureValidationRequest",
    "ModelArchitectureValidationResult",
    "ModelArchitectureViolation",
    "NodeArchitectureValidator",
    "RegistryInfraArchitectureValidator",
]
