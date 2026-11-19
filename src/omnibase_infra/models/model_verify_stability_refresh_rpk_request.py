# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One Kafka API read the stability refresh health gate takes (OMN-25927)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelVerifyStabilityRefreshRpkRequest(BaseModel):
    """The broker container and the rpk arguments of one read, e.g. ``("topic", "list")``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker_container: str = Field(min_length=1)
    rpk_args: tuple[str, ...] = Field(min_length=1)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    environ: dict[str, str] = Field(default_factory=dict, repr=False, exclude=True)
    """Injected environment the gate's password is read from; empty keeps the read anonymous."""
