# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Which auto-wired contracts the runtime joins at boot (OMN-18087)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelBootSubscriptionsConfig(BaseModel):
    """Boot subscription policy, declared in the runtime config contract.

    Attributes:
        policy: ``all`` joins every auto-wired contract's consumer groups at
            boot. ``in_use`` joins at boot only the contracts with a subscribe
            topic holding a retained record, as the broker reports it, and
            joins each of the rest when its first request arrives.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    policy: Literal["all", "in_use"] = Field(
        default="all",
        description="Which auto-wired contracts join their consumer groups at boot",
    )


__all__: list[str] = ["ModelBootSubscriptionsConfig"]
