#!/usr/bin/env python3
"""Response models for mysql."""

from pydantic import BaseModel, Field


class ModelMysqlResponse(BaseModel):
    """Response model for mysql operations."""

    # SCHEMA REQUIRED: Add response-specific fields below
    status: str = Field(..., description="Operation status")
    message: str = Field(..., description="Status message")


__all__ = ["ModelMysqlResponse"]
