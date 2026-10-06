# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Explicit, immutable inputs collected by the append-only runtime."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelMigrationAppendOnlyCheckInput"]


class ModelMigrationAppendOnlyCheckInput(BaseModel):
    """The git diff and ledger snapshots needed by the pure decision.

    Committed mode retains the script's working-tree ledger/existence reads;
    staged mode supplies index data. Only added paths can be successors, so
    existing_paths need only include the added paths under inspection.
    Base supersessions are carried as a snapshot, but do not authorise changes:
    a successor must be added by this diff even when its row existed at the base.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    base_ref: str = Field(description="Resolved merge-base used in error messages.")
    changed_paths: tuple[tuple[str, str], ...] = Field(
        default_factory=tuple,
        description="Repository-relative path/status pairs parsed from git name-status.",
    )
    base_manifest_text: str | None = Field(
        description="Application manifest at the base ref; None when absent."
    )
    base_supersessions_text: str | None = Field(
        default=None, description="Supersessions at the base ref; None when absent."
    )
    manifest_text: str | None = Field(
        description="Current working-tree or index application manifest."
    )
    supersessions_text: str | None = Field(
        default=None, description="Current working-tree or index supersessions."
    )
    existing_paths: frozenset[str] = Field(
        default_factory=frozenset,
        description="Repository-relative added paths that are files in the working tree or exist in the index.",
    )
