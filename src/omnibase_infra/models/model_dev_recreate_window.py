# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Policy, request, decision and announcement of the dev recreate window (OMN-25912)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from omnibase_infra.enums.enum_dev_recreate_verdict import (
    EnumDevRecreateReason,
    EnumDevRecreateVerdict,
)

ENV_WINDOW_SECONDS = "DEV_RECREATE_WINDOW_SECONDS"
ENV_OFFSET_SECONDS = "DEV_RECREATE_WINDOW_OFFSET_SECONDS"
ENV_GRACE_SECONDS = "DEV_RECREATE_START_GRACE_SECONDS"
ENV_PINNED_PREFIXES = "DEV_RECREATE_PINNED_FIX_REQUESTER_PREFIXES"


def _aware(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("a recreate window time must carry a UTC offset")
    return value.astimezone(UTC)


class ModelDevRecreatePolicy(BaseModel):
    """The declared window. Every value is read from runtime-policy.env, none is a literal."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    window_seconds: int = Field(ge=1)
    offset_seconds: int = Field(ge=0)
    start_grace_seconds: int = Field(ge=0)
    pinned_fix_requester_prefixes: tuple[str, ...] = ()

    def render_env(self) -> dict[str, str]:
        """The policy as the env names the deploy agent reads."""
        return {
            ENV_WINDOW_SECONDS: str(self.window_seconds),
            ENV_OFFSET_SECONDS: str(self.offset_seconds),
            ENV_GRACE_SECONDS: str(self.start_grace_seconds),
            ENV_PINNED_PREFIXES: ",".join(self.pinned_fix_requester_prefixes),
        }

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> ModelDevRecreatePolicy:
        """Read the policy; an absent or malformed value refuses, there is no fallback."""
        missing = [
            name
            for name in (ENV_WINDOW_SECONDS, ENV_OFFSET_SECONDS, ENV_GRACE_SECONDS)
            if not env.get(name, "").strip()
        ]
        if missing:
            raise ValueError(
                "the dev recreate window policy is not declared: " + ", ".join(missing)
            )
        prefixes = tuple(
            part.strip()
            for part in env.get(ENV_PINNED_PREFIXES, "").split(",")
            if part.strip()
        )
        return cls(
            window_seconds=int(env[ENV_WINDOW_SECONDS]),
            offset_seconds=int(env[ENV_OFFSET_SECONDS]),
            start_grace_seconds=int(env[ENV_GRACE_SECONDS]),
            pinned_fix_requester_prefixes=prefixes,
        )


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


class ModelDevRecreateDecision(BaseModel):
    """Recreate now, or hold until ``next_recreate_at``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: EnumDevRecreateVerdict
    reason: EnumDevRecreateReason
    next_recreate_at: datetime


class ModelDevRecreateAnnouncement(BaseModel):
    """The event published before a recreate: the next recreate time and why."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: UUID
    runtime_lane: str = Field(min_length=1)
    requested_by: str = Field(min_length=1)
    announced_at: datetime
    next_recreate_at: datetime
    reason: EnumDevRecreateReason
    immediate: bool
