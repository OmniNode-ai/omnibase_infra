# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One ask to recreate the dev runtime (OMN-25912)."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.models.model_dev_recreate_policy import ModelDevRecreatePolicy


def _aware(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("a recreate window time must carry a UTC offset")
    return value.astimezone(UTC)


class ModelDevRecreateRequest(BaseModel):
    """One ask to recreate the dev runtime at ``now``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    policy: ModelDevRecreatePolicy
    now: datetime
    requested_by: str = Field(min_length=1)
    last_recreate_at: datetime | None = None

    @field_validator("now", "last_recreate_at")
    @classmethod
    def _times_aware(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _aware(value)
