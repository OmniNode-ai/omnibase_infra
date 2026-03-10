# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Output model for the setup orchestrator node.

Invariant I5 — Orchestrators do not return results.
This model intentionally omits a ``result`` field (ONEX architectural invariant).

Ticket: OMN-3491
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.nodes.node_setup_orchestrator.models.model_setup_event import (
    ModelSetupEvent,
)


class ModelSetupOrchestratorOutput(BaseModel):
    """Output from the setup orchestrator node.

    Carries the ordered sequence of setup lifecycle events emitted during
    orchestration. No ``result`` field — orchestrators emit events only
    (Invariant I5).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(
        ...,
        description="Correlation ID echoed from the input.",
    )
    events: tuple[ModelSetupEvent, ...] = Field(
        default=(),
        description="Ordered tuple of setup lifecycle events emitted during orchestration.",
    )
    # NOTE: No 'result' field — orchestrators do not return results (ONEX invariant I5).


__all__: list[str] = ["ModelSetupOrchestratorOutput"]
