# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Immutable source record for a generated application-database ACL matrix."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelApplicationDatabaseAclSource(BaseModel):
    """One immutable contract or evidence blob used to generate the matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_key: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$")
    repository: str = Field(
        ...,
        pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$",
    )
    revision: str = Field(..., pattern=r"^[0-9a-f]{40}$")
    path: str = Field(..., pattern=r"^[A-Za-z0-9_./-]+$")
    sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    purpose: Literal[
        "topology",
        "relation_inventory",
        "service_ownership",
        "rendered_topology",
        "typed_loader",
        "legacy_fixture",
        "principal_inventory",
        "acl_policy",
        "catalog_query_evidence",
        "catalog_result_evidence",
        "activity_query_evidence",
        "activity_result_evidence",
    ]

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        """Reject absolute, parent-traversing, or noncanonical source paths."""
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise ValueError("source path must be canonical and repository-relative")
        return value


__all__ = ["ModelApplicationDatabaseAclSource"]
