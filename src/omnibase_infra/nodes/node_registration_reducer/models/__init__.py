# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for NodeRegistrationReducer.

This module exports models used by the NodeRegistrationReducer (FSM-driven pattern).

Available Models:
    - ModelValidationResult: Validation result with error details
    - ModelRegistrationState: Immutable state for reducer FSM
    - ModelRegistrationConfirmation: Confirmation event from Effect layer
    - ModelPayloadConsulRegister: Payload for Consul registration intents
    - ModelPayloadPostgresUpsertRegistration: Payload for PostgreSQL upsert intents
"""

# Node-specific model
from omnibase_infra.nodes.node_registration_reducer.models.model_validation_result import (
    ModelValidationResult,
    ValidationErrorCode,
    ValidationResult,
)

# Re-export shared models from the reducers module for convenience
from omnibase_infra.nodes.reducers.models import (
    ModelPayloadConsulRegister,
    ModelPayloadPostgresUpsertRegistration,
    ModelRegistrationConfirmation,
    ModelRegistrationState,
)

__all__ = [
    "ModelPayloadConsulRegister",
    "ModelPayloadPostgresUpsertRegistration",
    "ModelRegistrationConfirmation",
    "ModelRegistrationState",
    "ModelValidationResult",
    "ValidationErrorCode",
    "ValidationResult",
]
