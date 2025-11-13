#!/usr/bin/env python3
"""
Stub for ModelIOOperationConfig from omnibase_core.

Used for Effect node I/O operation configuration.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class ModelIOOperationConfig(BaseModel):
    """
    Configuration for I/O operations in Effect nodes.

    Stub implementation for testing.
    """

    operation_type: str = Field(
        ..., description="Type of I/O operation (e.g., 'database_query', 'api_call')"
    )
    atomic: bool = Field(default=True, description="Whether operation should be atomic")
    timeout_ms: Optional[int] = Field(
        None, description="Operation timeout in milliseconds"
    )
    retry_config: Optional[dict[str, Any]] = Field(
        None, description="Retry configuration"
    )
    metadata: Optional[dict[str, Any]] = Field(
        None, description="Additional operation metadata"
    )

    class Config:
        """Pydantic model configuration."""

        extra = "allow"
