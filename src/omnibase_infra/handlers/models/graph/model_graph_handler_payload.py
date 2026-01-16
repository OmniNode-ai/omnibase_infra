# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler payload model for Graph operations."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag

from omnibase_infra.handlers.models.graph.model_graph_execute_payload import (
    ModelGraphExecutePayload,
)
from omnibase_infra.handlers.models.graph.model_graph_query_payload import (
    ModelGraphQueryPayload,
)


def _get_graph_operation_type(value: object) -> str:
    """Extract operation_type from Graph payload for discriminated union."""
    if isinstance(value, dict):
        return str(value.get("operation_type", ""))
    return getattr(value, "operation_type", "")


GraphPayload = Annotated[
    Annotated[ModelGraphQueryPayload, Tag("graph.query")]
    | Annotated[ModelGraphExecutePayload, Tag("graph.execute")],
    Discriminator(_get_graph_operation_type),
]
"""Discriminated union of all Graph payload types."""


class ModelGraphHandlerPayload(BaseModel):
    """Wrapper for Graph handler payloads using discriminated union.

    Attributes:
        data: The typed payload from the discriminated union
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    data: GraphPayload = Field(description="The typed payload")


__all__: list[str] = ["ModelGraphHandlerPayload", "GraphPayload"]
