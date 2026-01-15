# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""ONEX Projector Implementations Module.

Provides projector implementations for persisting, reading, and publishing
projections. Projectors are used by the runtime to materialize handler
outputs to storage (PostgreSQL) and by orchestrators to query current
entity state.

Exports:
    ProjectionReaderRegistration: Registration projection reader implementation
    SnapshotPublisherRegistration: Registration snapshot publisher for Kafka

Related Tickets:
    - OMN-947 (F2): Snapshot Publishing
    - OMN-944 (F1): Implement Registration Projection Schema
    - OMN-940 (F0): Define Projector Execution Model
"""

from omnibase_infra.projectors.projection_reader_registration import (
    ProjectionReaderRegistration,
)
from omnibase_infra.projectors.snapshot_publisher_registration import (
    SnapshotPublisherRegistration,
)

__all__ = [
    "ProjectionReaderRegistration",
    "SnapshotPublisherRegistration",
]
