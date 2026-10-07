# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node-local state and validation models for NodeRegistrationReducer.

Shared intent payloads and Effect-layer updates live in
``omnibase_infra.models.ledger`` and ``omnibase_infra.models.registration``.

Available Models:
    - ModelValidationResult: Validation result with error details
    - ModelRegistrationState: Immutable state for reducer FSM
    - ModelRegistrationConfirmation: Confirmation event from Effect layer
"""

from __future__ import annotations

# Registration state models (migrated from nodes.reducers.models)
from omnibase_infra.nodes.node_registration_reducer.models.model_registration_confirmation import (
    ModelRegistrationConfirmation,
)
from omnibase_infra.nodes.node_registration_reducer.models.model_registration_state import (
    ModelRegistrationState,
)

# Node-specific model
from omnibase_infra.nodes.node_registration_reducer.models.model_validation_result import (
    ModelValidationResult,
    ValidationErrorCode,
    ValidationResult,
)

__all__ = [
    "ModelRegistrationConfirmation",
    "ModelRegistrationState",
    "ModelValidationResult",
    "ValidationErrorCode",
    "ValidationResult",
]
