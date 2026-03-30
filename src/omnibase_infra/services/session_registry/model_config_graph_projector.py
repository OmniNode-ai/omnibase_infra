# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Configuration model for the session graph projector.

Part of the Multi-Session Coordination Layer (OMN-6850, Task 9).
"""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelConfigGraphProjector"]

_DEFAULT_CONSUMER_GROUP = "omnibase_infra.session_registry.graph_project.v1"
_DEFAULT_TOPIC_PATTERN = r"onex\.evt\.omniclaude\..*"


class ModelConfigGraphProjector(BaseModel):
    """Configuration for the session graph projector.

    Connection defaults are sourced from OMNIMEMORY_MEMGRAPH_HOST/PORT
    environment variables, falling back to localhost:7687.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bootstrap_servers: str = Field(
        default="localhost:19092",
        description="Kafka bootstrap servers.",
    )
    consumer_group: str = Field(
        default=_DEFAULT_CONSUMER_GROUP,
        description="Kafka consumer group ID.",
    )
    topic_pattern: str = Field(
        default=_DEFAULT_TOPIC_PATTERN,
        description="Regex pattern for Kafka topic subscription.",
    )
    memgraph_host: str = Field(
        default_factory=lambda: os.environ.get("OMNIMEMORY_MEMGRAPH_HOST", "localhost"),
        description="Memgraph host.",
    )
    memgraph_port: int = Field(
        default_factory=lambda: int(os.environ.get("OMNIMEMORY_MEMGRAPH_PORT", "7687")),
        description="Memgraph bolt port.",
    )

    @property
    def bolt_uri(self) -> str:
        """Memgraph bolt connection URI."""
        return f"bolt://{self.memgraph_host}:{self.memgraph_port}"
