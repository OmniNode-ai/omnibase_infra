"""
Pydantic v2 models for .onextree format.

Defines the schema for project structure representation with semantic metadata
and performance statistics.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class OnexTreeNode(BaseModel):
    """
    Single node in the onextree hierarchy.

    Represents a file or directory with metadata and optional semantic information.
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    path: str = Field(..., description="Relative path from project root")
    name: str = Field(..., description="File or directory name")
    type: str = Field(..., description="file | directory")
    size: Optional[int] = Field(None, description="File size in bytes")
    extension: Optional[str] = Field(None, description="File extension without dot")
    last_modified: Optional[datetime] = Field(
        None, description="Last modification time"
    )
    children: Optional[list["OnexTreeNode"]] = Field(
        default_factory=list, description="Child nodes for directories"
    )

    # Semantic metadata (populated by future analysis)
    inferred_purpose: Optional[str] = Field(
        None, description="Auto-detected purpose (e.g., 'service', 'model', 'test')"
    )
    architectural_pattern: Optional[str] = Field(
        None,
        description="Detected pattern (e.g., 'service', 'repository', 'controller')",
    )
    related_files: Optional[list[str]] = Field(
        default_factory=list, description="Import/dependency paths"
    )


class ProjectStatistics(BaseModel):
    """
    Project-level statistics for quick insights.

    Tracks file counts, type distribution, and size metrics.
    """

    total_files: int = Field(default=0, description="Total number of files")
    total_directories: int = Field(default=0, description="Total number of directories")
    file_type_distribution: dict[str, int] = Field(
        default_factory=dict, description="Count of files by extension"
    )
    total_size_bytes: int = Field(default=0, description="Total size of all files")
    last_updated: datetime = Field(
        default_factory=datetime.now, description="Last statistics update time"
    )


class ModelOnextreeRoot(BaseModel):
    """
    Root model for .onextree YAML format.

    Contains complete project tree with metadata and statistics.
    """

    model_config = ConfigDict(frozen=False, validate_assignment=True)

    version: str = Field(default="1.0.0", description="OnexTree format version")
    project_root: str = Field(..., description="Absolute path to project root")
    generated_at: datetime = Field(
        default_factory=datetime.now, description="Tree generation timestamp"
    )
    tree: OnexTreeNode = Field(..., description="Root node of tree hierarchy")
    statistics: ProjectStatistics = Field(
        default_factory=ProjectStatistics, description="Project-level statistics"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata"
    )
