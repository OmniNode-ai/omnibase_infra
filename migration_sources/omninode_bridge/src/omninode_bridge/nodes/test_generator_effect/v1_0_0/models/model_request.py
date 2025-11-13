"""
Test Generation Request Model - ONEX v2.0 Compliant.

Request model for test generation operations.
"""

from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class ModelTestGeneratorRequest(BaseModel):
    """
    Request model for test generation operations.

    Contains the test contract YAML and configuration for generating test files.
    """

    # === CORE REQUEST DATA ===

    test_contract_yaml: str = Field(
        ...,
        description="YAML string of ModelContractTest contract",
        min_length=1,
    )

    output_directory: Path = Field(
        ...,
        description="Directory where test files should be written",
    )

    node_name: str = Field(
        ...,
        description="Name of the node being tested (e.g., 'postgres_crud_effect')",
        min_length=1,
    )

    # === OPTIONAL CONFIGURATION ===

    template_directory: Path | None = Field(
        default=None,
        description="Custom template directory (uses default if None)",
    )

    enable_fixtures: bool = Field(
        default=True,
        description="Whether to generate pytest fixtures in conftest.py",
    )

    overwrite_existing: bool = Field(
        default=False,
        description="Whether to overwrite existing test files",
    )

    # === CORRELATION TRACKING ===

    correlation_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for correlation tracking",
    )

    execution_id: UUID = Field(
        default_factory=uuid4,
        description="UUID for tracking this execution instance",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "test_contract_yaml": "name: test_postgres_crud\nversion:\n  major: 1\n  minor: 0\n  patch: 0\n...",
                "output_directory": "/path/to/tests",
                "node_name": "postgres_crud_effect",
                "enable_fixtures": True,
                "overwrite_existing": False,
            }
        }
    )


__all__ = ["ModelTestGeneratorRequest"]
