# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed response model for local runtime ingress."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_core.models.dispatch.model_dispatch_bus_terminal_result import (
    ModelDispatchBusTerminalResult,
)
from omnibase_core.types import JsonType
from omnibase_infra.runtime.models.model_local_runtime_ingress_error import (
    ModelLocalRuntimeIngressError,
)


class ModelLocalRuntimeIngressResponse(BaseModel):
    """Structured response returned to local runtime ingress callers."""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    ok: bool = Field(..., description="Whether the request completed successfully.")
    command_name: str = Field(..., min_length=1, description="Resolved command name.")
    node_alias: str | None = Field(
        default=None,
        min_length=1,
        description="Deprecated compatibility alias when supplied by the caller.",
    )
    resolved_node_name: str | None = Field(
        default=None,
        description="Resolved node directory name when the request is routable.",
    )
    contract_name: str | None = Field(
        default=None,
        description="Resolved contract name when the request is routable.",
    )
    command_topic: str | None = Field(
        default=None,
        description="Resolved command topic used for dispatch.",
    )
    terminal_event: str | None = Field(
        default=None,
        description="Declared terminal event for the resolved node contract.",
    )
    correlation_id: UUID | None = Field(
        default=None,
        description="Resolved correlation identifier for the request.",
    )
    dispatch_result: ModelDispatchBusTerminalResult | None = Field(
        default=None,
        description="Terminal broker result for runtime execution.",
    )
    output_payloads: list[dict[str, JsonType]] | None = Field(
        default=None,
        description="Typed dict payloads extracted from successful terminal results.",
    )
    error: ModelLocalRuntimeIngressError | None = Field(
        default=None,
        description="Structured error for rejected or failed requests.",
    )

    @model_validator(mode="after")
    def _validate_success_shape(self) -> ModelLocalRuntimeIngressResponse:
        if self.ok and self.dispatch_result is None:
            raise ValueError("ok responses must include dispatch_result")
        if not self.ok and self.error is None:
            raise ValueError("non-ok responses must include error")
        return self


__all__ = ["ModelLocalRuntimeIngressResponse"]
