# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Explicit opt-in resources for the signed execution graph read handler."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
    ModelExecutionGraphTopologyVersion,
)
from omnibase_infra.runtime.models.model_execution_graph_read_databases import (
    ModelExecutionGraphReadDatabases,
)
from omnibase_infra.runtime.models.model_execution_graph_terminal_publisher_config import (
    ModelExecutionGraphTerminalPublisherConfig,
)


class ModelExecutionGraphReadRuntimeConfig(BaseModel):
    """All resources required to activate the graph read contract at boot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    databases: ModelExecutionGraphReadDatabases
    topology_version: ModelExecutionGraphTopologyVersion
    terminal_publisher: ModelExecutionGraphTerminalPublisherConfig
    private_key_path: Path = Field(
        description="Existing PEM key for the runtime terminal signer."
    )

    @field_validator("private_key_path", mode="before")  # type: ignore[untyped-decorator]
    @classmethod
    def require_absolute_key_path(cls, value: object) -> Path:
        """Do not resolve key files relative to an arbitrary runtime cwd."""
        if not isinstance(value, (str, Path)):
            raise ValueError("graph terminal private_key_path must be a path")
        path = Path(value)
        if not path.is_absolute() or not path.is_file():
            raise ValueError(
                "graph terminal private_key_path must be an existing absolute file"
            )
        return path


__all__ = ["ModelExecutionGraphReadRuntimeConfig"]
