# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Query payload model for Graph handler."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.handlers.models.graph.model_graph_record import ModelGraphRecord
from omnibase_infra.models.types import JsonDict


class ModelGraphQueryPayload(BaseModel):
    """Payload for graph.query operation result.

    Attributes:
        operation_type: Discriminator for payload type
        cypher: The Cypher query that was executed
        records: List of result records
        summary: Query execution summary (counters, timing, etc.)
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    operation_type: Literal["graph.query"] = Field(
        default="graph.query",
        description="Operation type discriminator",
    )
    cypher: str = Field(description="The Cypher query that was executed")
    records: list[ModelGraphRecord] = Field(
        default_factory=list,
        description="List of result records",
    )
    summary: JsonDict = Field(
        default_factory=dict,
        description="Query execution summary",
    )


__all__: list[str] = ["ModelGraphQueryPayload"]
