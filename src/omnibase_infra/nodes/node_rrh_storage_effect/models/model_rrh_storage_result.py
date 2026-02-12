# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Output model for the RRH storage effect node."""

from __future__ import annotations

from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelRRHStorageResult(BaseModel):
    """Outcome of RRH artifact storage.

    Attributes:
        artifact_path: Absolute path to the written JSON artifact.
        ticket_symlink: Path to latest_by_ticket symlink (empty if N/A).
        repo_symlink: Path to latest_by_repo symlink (empty if N/A).
        success: Whether the write succeeded.
        error: Error message on failure (empty on success).
        correlation_id: Distributed tracing correlation ID.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    artifact_path: str = Field(..., description="Path to written JSON artifact.")
    ticket_symlink: str = Field(default="", description="latest_by_ticket symlink.")
    repo_symlink: str = Field(default="", description="latest_by_repo symlink.")
    success: bool = Field(..., description="Whether the write succeeded.")
    error: str = Field(default="", description="Error message on failure.")
    correlation_id: UUID = Field(
        default_factory=uuid4, description="Correlation ID for tracing."
    )


__all__: list[str] = ["ModelRRHStorageResult"]
