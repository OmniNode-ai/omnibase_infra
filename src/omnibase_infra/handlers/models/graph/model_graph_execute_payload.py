# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Execute payload model for Graph handler."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.types import JsonDict


class ModelGraphExecutePayload(BaseModel):
    """Payload for graph.execute operation result.

    Attributes:
        operation_type: Discriminator for payload type
        cypher: The Cypher statement that was executed
        counters: Operation counters (nodes_created, relationships_created, etc.)
        success: Whether the execution was successful
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    operation_type: Literal["graph.execute"] = Field(
        default="graph.execute",
        description="Operation type discriminator",
    )
    cypher: str = Field(description="The Cypher statement that was executed")
    counters: JsonDict = Field(
        default_factory=dict,
        description="Operation counters (nodes_created, etc.)",
    )
    success: bool = Field(
        default=True, description="Whether the execution was successful"
    )


__all__: list[str] = ["ModelGraphExecutePayload"]
