# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Explicit inputs collected by the migration-sequence runtime."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelMigrationSequenceCheckInput"]


class ModelMigrationSequenceCheckInput(BaseModel):
    """Staged paths and the nonrecursive on-disk migration scan, relative to root."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    staged_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Paths from git diff --cached --name-only, including deletions.",
    )
    migration_paths: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Paths found by glob('*.sql') directly in each migration directory.",
    )
    migration_dirs: tuple[str, ...] = Field(
        default=(
            "docker/migrations/forward",
            "src/omnibase_infra/migrations/forward",
        ),
        description="Directory prefixes forming the shared sequence namespace.",
    )
    excluded_subtree_prefixes: tuple[str, ...] = Field(
        default=("docker/migrations/forward/nodes/",),
        description="Subtrees whose migrations have separate sequence namespaces.",
    )
