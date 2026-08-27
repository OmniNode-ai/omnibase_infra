# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""A per-topic DLQ arrival allowance, with its mandatory justification (OMN-16769)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDlqTopicThresholdOverride(BaseModel):
    """A per-topic allowance, which MUST carry its own justification.

    There are no silent allowances. A topic only escapes the default
    bound by naming, in-band, why it is allowed to carry standing
    traffic and what would end the allowance. An override without a
    reason is rejected at parse time — that is the whole point of
    modelling this as a typed row instead of a ``dict[str, int]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    topic: str = Field(..., min_length=1, description="Exact DLQ topic name.")
    max_arrivals_per_window: int = Field(
        ...,
        ge=0,
        description="Arrivals allowed in one window before this topic alerts.",
    )
    reason: str = Field(
        ...,
        min_length=20,
        description=(
            "Why this topic is allowed to exceed the default bound. Must be a "
            "real explanation naming the traffic source — not 'noisy' or 'TODO'."
        ),
    )
    ratify_by: str = Field(
        ...,
        min_length=10,
        description=(
            "The STATE-BASED condition that ends this allowance (never a date). "
            "E.g. 'when OMN-16767 lands and the sink stops receiving'."
        ),
    )


__all__ = ["ModelDlqTopicThresholdOverride"]
