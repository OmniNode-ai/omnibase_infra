#!/usr/bin/env python3
"""
Models for Template Engine Loader.

Defines data structures for template discovery, loading, and artifact management.

ONEX v2.0 Compliance:
- Pydantic v2 models for type safety
- Field validation and descriptions
- Structured template metadata
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ModelStubInfo(BaseModel):
    """Information about a detected code stub."""

    method_name: str = Field(..., description="Method name (e.g., execute_effect)")
    stub_code: str = Field(..., description="Current stub implementation")
    line_start: int = Field(..., ge=1, description="Starting line number")
    line_end: int = Field(..., ge=1, description="Ending line number")
    signature: str = Field(..., description="Full method signature with type hints")
    docstring: Optional[str] = Field(None, description="Method docstring if present")


class ModelTemplateMetadata(BaseModel):
    """Metadata extracted from template file."""

    node_type: str = Field(
        ..., description="Node type (effect/compute/reducer/orchestrator)"
    )
    version: str = Field(..., description="Template version (e.g., v1_0_0)")
    description: Optional[str] = Field(None, description="Template description")
    author: Optional[str] = Field(None, description="Template author")
    created_at: Optional[datetime] = Field(None, description="Template creation date")
    tags: list[str] = Field(default_factory=list, description="Template tags")


class ModelTemplateInfo(BaseModel):
    """Information about a discovered template."""

    template_path: Path = Field(..., description="Full path to template file")
    node_type: str = Field(..., description="Node type")
    version: str = Field(..., description="Version string (v1_0_0)")
    template_name: str = Field(..., description="Template name (e.g., node_effect)")
    metadata: ModelTemplateMetadata = Field(..., description="Template metadata")
    discovered_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When template was discovered",
    )


class ModelTemplateArtifacts(BaseModel):
    """
    Template artifacts loaded from file system.

    Contains raw template code, detected stubs, and metadata.
    """

    # Template content
    template_code: str = Field(..., description="Raw Python template code")
    template_path: Path = Field(..., description="Path to template file")

    # Detected stubs
    stubs: list[ModelStubInfo] = Field(
        default_factory=list,
        description="Detected stubs for replacement",
    )

    # Metadata
    node_type: str = Field(
        ..., description="Node type (effect/compute/reducer/orchestrator)"
    )
    version: str = Field(..., description="Template version")
    metadata: ModelTemplateMetadata = Field(..., description="Template metadata")

    # Discovery info
    loaded_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When template was loaded",
    )

    def get_stub_count(self) -> int:
        """Get count of detected stubs."""
        return len(self.stubs)

    def has_stubs(self) -> bool:
        """Check if template has any stubs."""
        return len(self.stubs) > 0


__all__ = [
    "ModelStubInfo",
    "ModelTemplateMetadata",
    "ModelTemplateInfo",
    "ModelTemplateArtifacts",
]
