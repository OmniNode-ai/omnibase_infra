# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Projector Implementations Module.

Provides projector implementations for persisting and reading projections.
Projectors are used by the runtime to materialize handler outputs to storage
(PostgreSQL) and by orchestrators to query current entity state.

Exports:
    ProjectionReaderRegistration: Registration projection reader implementation
    ProjectorRegistration: Registration projector implementation

Related Tickets:
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.projectors.projector_registration import ProjectorRegistration

__all__ = [
    "ProjectionReaderRegistration",
    "ProjectorRegistration",
]
