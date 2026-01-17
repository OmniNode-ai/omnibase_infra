# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Snapshot Diff Model for Structural Comparison.

Provides a model for representing structural differences between two snapshots.
This is a structural diff (not semantic) - it identifies which keys were added,
removed, or changed between two data dictionaries.

Thread Safety:
    This model is frozen (immutable) for safe sharing across threads.

Related Tickets:
    - OMN-1246: ServiceSnapshot Infrastructure Primitive
"""

from __future__ import annotations

from typing import Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.models.snapshot.model_field_change import ModelFieldChange


class ModelSnapshotDiff(BaseModel):
    """Structural diff between two snapshots.

    Represents the structural differences between a base snapshot and a target
    snapshot. This is a shallow diff that operates on top-level keys only -
    nested changes within a key are captured as a single ModelFieldChange.

    Design Notes:
        - Shallow comparison: Only top-level keys are compared
        - Structural, not semantic: No interpretation of what changes mean
        - Immutable: Diffs are point-in-time calculations

    Attributes:
        base_id: UUID of the base (original) snapshot.
        target_id: UUID of the target (new) snapshot.
        added: List of keys present in target but not in base.
        removed: List of keys present in base but not in target.
        changed: Dictionary mapping changed keys to their ModelFieldChange.

    Example:
        >>> from uuid import uuid4
        >>> base_data = {"name": "alice", "status": "pending"}
        >>> target_data = {"name": "alice", "status": "active", "score": 100}
        >>> diff = ModelSnapshotDiff.compute(
        ...     base_data=base_data,
        ...     target_data=target_data,
        ...     base_id=uuid4(),
        ...     target_id=uuid4(),
        ... )
        >>> diff.added
        ['score']
        >>> diff.removed
        []
        >>> diff.changed["status"].from_value
        'pending'
        >>> diff.changed["status"].to_value
        'active'
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    base_id: UUID = Field(
        ...,
        description="UUID of the base (original) snapshot",
    )
    target_id: UUID = Field(
        ...,
        description="UUID of the target (new) snapshot",
    )
    added: list[str] = Field(
        default_factory=list,
        description="Keys present in target but not in base",
    )
    removed: list[str] = Field(
        default_factory=list,
        description="Keys present in base but not in target",
    )
    changed: dict[str, ModelFieldChange] = Field(
        default_factory=dict,
        description="Keys with different values, mapped to their changes",
    )

    @classmethod
    def compute(
        cls,
        base_data: dict[str, JsonType],
        target_data: dict[str, JsonType],
        base_id: UUID,
        target_id: UUID,
    ) -> Self:
        """Compute structural diff between two data dictionaries.

        Performs a shallow comparison of top-level keys between base and target
        data dictionaries, identifying which keys were added, removed, or changed.

        Args:
            base_data: The base (original) data dictionary.
            target_data: The target (new) data dictionary.
            base_id: UUID of the base snapshot.
            target_id: UUID of the target snapshot.

        Returns:
            A new ModelSnapshotDiff capturing the structural differences.

        Example:
            >>> from uuid import uuid4
            >>> base = {"a": 1, "b": 2, "c": 3}
            >>> target = {"a": 1, "b": 99, "d": 4}
            >>> diff = ModelSnapshotDiff.compute(
            ...     base_data=base,
            ...     target_data=target,
            ...     base_id=uuid4(),
            ...     target_id=uuid4(),
            ... )
            >>> sorted(diff.added)
            ['d']
            >>> sorted(diff.removed)
            ['c']
            >>> diff.changed["b"].from_value
            2
            >>> diff.changed["b"].to_value
            99
        """
        base_keys = set(base_data.keys())
        target_keys = set(target_data.keys())

        # Keys only in target (added)
        added = sorted(target_keys - base_keys)

        # Keys only in base (removed)
        removed = sorted(base_keys - target_keys)

        # Keys in both - check for value changes
        changed: dict[str, ModelFieldChange] = {}
        common_keys = base_keys & target_keys
        for key in sorted(common_keys):
            base_value = base_data[key]
            target_value = target_data[key]
            if base_value != target_value:
                changed[key] = ModelFieldChange(
                    from_value=base_value,
                    to_value=target_value,
                )

        return cls(
            base_id=base_id,
            target_id=target_id,
            added=added,
            removed=removed,
            changed=changed,
        )

    def is_empty(self) -> bool:
        """Check if the diff contains no changes.

        Returns:
            True if there are no additions, removals, or changes.

        Example:
            >>> from uuid import uuid4
            >>> diff = ModelSnapshotDiff.compute(
            ...     base_data={"a": 1},
            ...     target_data={"a": 1},
            ...     base_id=uuid4(),
            ...     target_id=uuid4(),
            ... )
            >>> diff.is_empty()
            True
        """
        return not self.added and not self.removed and not self.changed

    @property
    def total_changes(self) -> int:
        """Count total number of changes (added + removed + changed).

        Returns:
            Total count of all changes.

        Example:
            >>> from uuid import uuid4
            >>> diff = ModelSnapshotDiff.compute(
            ...     base_data={"a": 1, "b": 2},
            ...     target_data={"a": 99, "c": 3},
            ...     base_id=uuid4(),
            ...     target_id=uuid4(),
            ... )
            >>> diff.total_changes  # 1 added (c), 1 removed (b), 1 changed (a)
            3
        """
        return len(self.added) + len(self.removed) + len(self.changed)


__all__: list[str] = ["ModelSnapshotDiff"]
