# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Models for Registration Reducers.

This module exports models used by the RegistrationReducer (pure function pattern).

Available Models:
    - ModelRegistrationState: Immutable state for pure reducer pattern
    - ModelRegistrationConfirmation: Confirmation event from Effect layer (Phase 2)
"""

from omnibase_infra.nodes.reducers.models.model_registration_confirmation import (
    ConfirmationEventType,
    ModelRegistrationConfirmation,
)
from omnibase_infra.nodes.reducers.models.model_registration_state import (
    ModelRegistrationState,
)

__all__ = [
    "ConfirmationEventType",
    "ModelRegistrationConfirmation",
    "ModelRegistrationState",
]
