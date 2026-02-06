# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL connection pool configuration.

Part of OMN-1976: Contract dependency materialization.
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field


class ModelPostgresPoolConfig(BaseModel):
    """PostgreSQL connection pool configuration.

    Sources configuration from POSTGRES_* environment variables.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, ge=1, le=65535, description="PostgreSQL port")
    user: str = Field(default="postgres", description="PostgreSQL user")
    password: str = Field(
        default="",
        repr=False,
        description="PostgreSQL password (never logged or included in repr)",
    )
    database: str = Field(default="omninode_bridge", description="PostgreSQL database")
    min_size: int = Field(default=2, ge=1, le=100, description="Minimum pool size")
    max_size: int = Field(default=10, ge=1, le=100, description="Maximum pool size")

    @classmethod
    def from_env(cls) -> ModelPostgresPoolConfig:
        """Create config from POSTGRES_* environment variables.

        Raises:
            ValueError: If numeric env vars contain non-numeric values.
        """
        try:
            return cls(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", ""),
                database=os.getenv("POSTGRES_DATABASE", "omninode_bridge"),
                min_size=int(os.getenv("POSTGRES_POOL_MIN_SIZE", "2")),
                max_size=int(os.getenv("POSTGRES_POOL_MAX_SIZE", "10")),
            )
        except (ValueError, TypeError) as e:
            msg = f"Invalid POSTGRES_* environment variable: {e}"
            raise ValueError(msg) from e


__all__ = ["ModelPostgresPoolConfig"]
