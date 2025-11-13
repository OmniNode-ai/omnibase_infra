"""Model for artifact storage request."""

from typing import Any, ClassVar, Optional

from pydantic import BaseModel, Field


class ModelArtifactStorageRequest(BaseModel):
    """
    Request to store a generated code artifact.

    Represents a request to persist generated code artifacts to the file system,
    along with optional metadata storage in PostgreSQL.
    """

    file_path: str = Field(
        ..., description="Destination file path for the artifact (absolute or relative)"
    )

    content: str = Field(
        ..., description="The artifact content to store (generated code)"
    )

    artifact_type: str = Field(
        default="node_file",
        description="Type of artifact (node_file, test_file, model_file, etc.)",
    )

    create_directories: bool = Field(
        default=True,
        description="Whether to create parent directories if they don't exist",
    )

    overwrite: bool = Field(
        default=True, description="Whether to overwrite existing file"
    )

    file_permissions: Optional[str] = Field(
        default="0644",
        description="File permissions in octal notation (e.g., '0644', '0755')",
    )

    store_metrics: bool = Field(
        default=False, description="Whether to store artifact metadata in PostgreSQL"
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata to store with the artifact",
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra: ClassVar[dict[str, Any]] = {
            "example": {
                "file_path": "/path/to/node.py",
                "content": "def foo(): return 42",
                "artifact_type": "node_file",
                "create_directories": True,
                "overwrite": True,
                "file_permissions": "0644",
                "store_metrics": True,
                "metadata": {"node_name": "NodeMyEffect", "lines_of_code": 150},
            }
        }
