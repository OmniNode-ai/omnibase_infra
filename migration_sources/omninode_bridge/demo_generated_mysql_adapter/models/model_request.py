#!/usr/bin/env python3
"""Request models for mysql."""

from pydantic import BaseModel, Field


class ModelMysqlRequest(BaseModel):
    """Request model for mysql operations."""

    # SCHEMA REQUIRED: Add request-specific fields below
    operation: str = Field(..., description="Operation to perform")


__all__ = ["ModelMysqlRequest"]
