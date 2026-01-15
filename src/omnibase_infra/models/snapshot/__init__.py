# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Generic Snapshot Models for Point-in-Time State Capture.

This module provides infrastructure primitives for managing point-in-time
snapshots of entity state. Snapshots support:

- Subject-based grouping (ModelSubjectRef)
- Structural diffing (ModelSnapshotDiff, ModelFieldChange)
- Immutable state capture with content hashing (ModelSnapshot)
- Lineage tracking for fork scenarios

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive
"""

from omnibase_infra.models.snapshot.model_field_change import ModelFieldChange
from omnibase_infra.models.snapshot.model_snapshot import ModelSnapshot
from omnibase_infra.models.snapshot.model_snapshot_diff import ModelSnapshotDiff
from omnibase_infra.models.snapshot.model_subject_ref import ModelSubjectRef

__all__: list[str] = [
    "ModelFieldChange",
    "ModelSnapshot",
    "ModelSnapshotDiff",
    "ModelSubjectRef",
]
