# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Reducer state model for the registration orchestrator."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelReducerState(BaseModel):
    """State model for the registration reducer.

    This model captures the reducer's internal state between reductions.
    The orchestrator treats this as an opaque container that it passes
    to the reducer but does not inspect.

    Attributes:
        last_event_timestamp: ISO timestamp of the last processed event.
        processed_node_ids: Set of node IDs that have been processed.
        pending_registrations: Count of registrations awaiting confirmation.

    Note:
        This is a minimal placeholder. The actual reducer (OMN-889) may
        extend this with additional state fields for deduplication,
        rate limiting, or batching logic.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
    )

    last_event_timestamp: str | None = Field(
        default=None,
        description="ISO timestamp of the last processed event",
    )
    processed_node_ids: frozenset[UUID] = Field(
        default_factory=frozenset,
        description="Set of node IDs that have been processed",
    )
    pending_registrations: int = Field(
        default=0,
        ge=0,
        description="Count of registrations awaiting confirmation",
    )

    @classmethod
    def initial(cls) -> ModelReducerState:
        """Create an initial empty state.

        Returns:
            A fresh ModelReducerState with default values.
        """
        return cls()


__all__ = ["ModelReducerState"]
