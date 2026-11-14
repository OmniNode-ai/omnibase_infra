# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Output of the event-provenance stamp handler (OMN-23981)."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, SkipValidation


class ModelEventProvenanceStampResult(BaseModel):
    """The envelope to publish, whether it was stamped, and why not when it was not."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    # SkipValidation: the envelope is already a validated instance of some
    # parametrization of the generic; revalidating it would round-trip it.
    envelope: Annotated[BaseModel, SkipValidation]
    stamped: bool = Field(description="True when this call put the provenance on.")
    reason: str = Field(
        description="stamped, already_stamped, or why the overlay stamped nothing."
    )


__all__: list[str] = ["ModelEventProvenanceStampResult"]
