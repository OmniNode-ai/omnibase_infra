# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Routing subcontract model for orchestrator configuration.

This model represents the complete routing configuration
for an orchestrator, including all routing entries and strategy.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.models.routing.model_routing_entry import (
    ModelRoutingEntry,
)


class ModelRoutingSubcontract(BaseModel):
    """Complete routing configuration for an orchestrator.

    This subcontract defines how incoming events are routed to handlers.
    It is loaded from the handler_routing section of contract.yaml.

    Attributes:
        version: Semantic version of this routing configuration.
        routing_strategy: Strategy for matching events to handlers.
            Currently only "payload_type_match" is supported.
        handlers: List of routing entries mapping event models to handlers.
        default_handler: Optional fallback handler key for unmatched events.

    Example:
        ```python
        subcontract = ModelRoutingSubcontract(
            version=ModelSemVer(major=1, minor=0, patch=0),
            routing_strategy="payload_type_match",
            handlers=[
                ModelRoutingEntry(
                    routing_key="ModelNodeIntrospectionEvent",
                    handler_key="handler-node-introspected",
                ),
            ],
            default_handler=None,
        )
        ```
    """

    version: ModelSemVer = Field(
        default_factory=lambda: ModelSemVer(major=1, minor=0, patch=0),
        description="Semantic version of this routing configuration",
    )
    routing_strategy: Literal["payload_type_match"] = Field(
        default="payload_type_match",
        description="Strategy for matching events to handlers",
    )
    handlers: list[ModelRoutingEntry] = Field(
        default_factory=list,
        description="List of routing entries mapping event models to handlers",
    )
    default_handler: str | None = Field(
        default=None,
        description="Optional fallback handler key for unmatched events",
    )

    model_config = {"frozen": True}


__all__ = ["ModelRoutingSubcontract"]
