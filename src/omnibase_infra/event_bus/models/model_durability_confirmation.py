# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Outcome of a durability confirmation attempt (OMN-15861).

A ``ModelPublishReceipt`` says where a produce path *claimed* a record landed.
A ``ModelDurabilityConfirmation`` says whether an authoritative surface agreed.
Only ``EnumConfirmationState.CONFIRMED`` authorises a durable claim; every other
state -- including ``UNKNOWN`` -- fails closed.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums.enum_confirmation_state import EnumConfirmationState
from omnibase_infra.event_bus.models.model_publish_receipt import ModelPublishReceipt
from omnibase_infra.utils import validate_timezone_aware_datetime


class ModelDurabilityConfirmation(BaseModel):
    """Whether a produced record has been proven durable, and by what.

    Attributes:
        state: Tri-state outcome. ``CONFIRMED`` is the only durable-authorising
            value; ``UNKNOWN`` fails closed.
        strategy: Name of the ``ProtocolConfirmationStrategy`` implementation
            that produced this outcome. Recorded so a durable claim is always
            attributable -- "confirmed by publish-return-only" is a materially
            weaker fact than "confirmed by broker readback", and the audit trail
            must be able to tell them apart after the fact.
        receipt: The coordinate that was checked, when one existed.
        checked_at: Timezone-aware instant the outcome was decided.
        detail: Human-readable reason. REQUIRED for every non-``CONFIRMED``
            state so a stuck outbox record always carries its own explanation.

    Example:
        >>> from datetime import UTC, datetime
        >>> outcome = ModelDurabilityConfirmation(
        ...     state=EnumConfirmationState.UNKNOWN,
        ...     strategy="broker_readback",
        ...     receipt=None,
        ...     checked_at=datetime(2026, 8, 13, tzinfo=UTC),
        ...     detail="readback source raised ConnectionRefusedError",
        ... )
        >>> outcome.is_durable
        False
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    state: EnumConfirmationState = Field(..., description="Tri-state outcome")
    strategy: str = Field(
        ..., min_length=1, description="Strategy implementation that decided"
    )
    receipt: ModelPublishReceipt | None = Field(
        default=None, description="Coordinate checked, when one existed"
    )
    checked_at: datetime = Field(
        ..., description="Timezone-aware instant the outcome was decided"
    )
    detail: str = Field(
        default="", description="Reason; required unless state is CONFIRMED"
    )

    @field_validator("checked_at")
    @classmethod
    def _require_tz_aware(cls, value: datetime) -> datetime:
        """Reject naive datetimes -- an ambiguous instant is not evidence."""
        return validate_timezone_aware_datetime(value)

    @field_validator("detail")
    @classmethod
    def _require_detail_when_not_confirmed(cls, value: str, info: object) -> str:
        """Force every non-confirmed outcome to carry its own explanation.

        A bare ``UNCONFIRMED`` with no reason is the shape that produces
        un-triageable stuck outbox records; making the reason structurally
        mandatory is cheaper than chasing it in logs later.
        """
        data = getattr(info, "data", {})
        state = data.get("state") if isinstance(data, dict) else None
        if state is not None and state is not EnumConfirmationState.CONFIRMED:
            if not value.strip():
                raise ValueError(
                    f"detail is required when state is {state}; a non-confirmed "
                    "durability outcome must explain itself"
                )
        return value

    @property
    def is_durable(self) -> bool:
        """True only for ``CONFIRMED``. ``UNKNOWN`` fails closed to False."""
        return self.state is EnumConfirmationState.CONFIRMED


__all__: list[str] = ["ModelDurabilityConfirmation"]
