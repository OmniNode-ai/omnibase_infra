# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Input of the event-provenance stamp handler (OMN-23981)."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, SkipValidation

from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.event_bus.model_event_provenance_overlay import (
    ModelEventProvenanceOverlay,
)


class ModelEventProvenanceStampRequest(BaseModel):
    """An envelope about to be published and the overlay of the process publishing it."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    # SkipValidation: the envelope is already a validated instance of some
    # parametrization of the generic; revalidating it would round-trip it.
    envelope: Annotated[ModelEventEnvelope[Any], SkipValidation]
    overlay: ModelEventProvenanceOverlay


__all__: list[str] = ["ModelEventProvenanceStampRequest"]
