# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
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
            "payload_type_match" and "topic_match" are both contract-legal
            values (OMN-15215) — this model's ``handlers`` shape only carries
            ``routing_key``/``handler_key`` and has no per-entry ``topic``
            field, so it cannot represent topic_match's per-topic
            disambiguation. That is expected: this model backs
            ``handler_routing_loader``'s informational
            ``RuntimeContractConfigLoader`` boot-summary pass, not the live
            consumer-attach/dispatch decision — the real wiring path uses
            ``omnibase_infra.runtime.auto_wiring.models.ModelHandlerRouting``
            (an untyped ``routing_strategy: str`` with a per-entry ``topic``
            field, OMN-14580/OMN-13825).
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
    routing_strategy: Literal["payload_type_match", "topic_match"] = Field(
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
