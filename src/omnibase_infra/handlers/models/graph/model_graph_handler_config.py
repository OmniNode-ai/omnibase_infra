# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Configuration model for HandlerGraph."""

from pydantic import BaseModel, ConfigDict, Field


class ModelGraphHandlerConfig(BaseModel):
    """Configuration for Graph handler initialization.

    Supports both Memgraph and Neo4j via Bolt protocol.

    Attributes:
        uri: Bolt URI (e.g., bolt://localhost:7687)
        username: Database username
        password: Database password
        database: Database name (default: neo4j for Neo4j, memgraph for Memgraph)
        timeout_seconds: Request timeout in seconds
        max_connection_pool_size: Maximum number of connections in pool
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    uri: str = Field(
        default="bolt://localhost:7687",
        description="Bolt URI for graph database",
    )
    username: str = Field(
        default="",
        description="Database username",
    )
    password: str = Field(
        default="",
        description="Database password",
    )
    database: str = Field(
        default="memgraph",
        description="Database name (memgraph or neo4j)",
    )
    timeout_seconds: float = Field(
        default=30.0,
        ge=0.1,
        le=3600.0,
        description="Request timeout in seconds",
    )
    max_connection_pool_size: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum number of connections in pool",
    )


__all__: list[str] = ["ModelGraphHandlerConfig"]
