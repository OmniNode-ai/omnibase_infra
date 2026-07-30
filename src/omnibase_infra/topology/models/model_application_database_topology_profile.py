# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed application-database topology profile binding."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["ModelApplicationDatabaseTopologyProfile"]


class ModelApplicationDatabaseTopologyProfile(BaseModel):
    """One explicit deployment-profile to topology-instance binding."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instance: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    deployment_repository: Literal[
        "OmniNode-ai/omnibase_infra",
        "OmniNode-ai/omninode_infra",
    ]
    injection_path: str = Field(min_length=1)
    runtime_policy_profile: str | None = Field(
        default=None,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
    )

    @field_validator("injection_path")
    @classmethod
    def injection_path_is_repository_relative(cls, value: str) -> str:
        """Reject absolute and parent-traversing evidence paths."""
        parts = value.split("/")
        if value.startswith("/") or ".." in parts or any(not part for part in parts):
            raise ValueError("injection_path must be a repository-relative path")
        return value
