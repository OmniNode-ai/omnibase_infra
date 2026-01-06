# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Input model for registration orchestrator."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.models.registration import ModelNodeIntrospectionEvent


class ModelOrchestratorInput(BaseModel):
    """Input for the registration orchestrator workflow.

    Contains the introspection event to process and correlation ID for
    distributed tracing across the workflow execution.

    Attributes:
        introspection_event: The introspection event containing node registration data.
        correlation_id: Correlation ID for distributed tracing.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    introspection_event: ModelNodeIntrospectionEvent = Field(
        ...,
        description="The introspection event to process",
    )
    correlation_id: UUID = Field(
        ...,
        description="Correlation ID for distributed tracing",
    )


__all__ = ["ModelOrchestratorInput"]
