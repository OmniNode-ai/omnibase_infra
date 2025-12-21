# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Projection Models Module.

Provides Pydantic models for projection storage and ordering.
Used by projectors to persist materialized state and by orchestrators
to query current entity state.

Exports:
    ModelRegistrationProjection: Registration projection for orchestrator state queries
    ModelSequenceInfo: Sequence information for projection ordering and idempotency

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from omnibase_infra.models.projection.model_registration_projection import (
    ModelRegistrationProjection,
)
from omnibase_infra.models.projection.model_sequence_info import ModelSequenceInfo

__all__ = [
    "ModelRegistrationProjection",
    "ModelSequenceInfo",
]
