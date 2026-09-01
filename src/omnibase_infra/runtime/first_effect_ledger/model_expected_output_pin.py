# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Immutable deployment-owned output pin for verified first-effect grants."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from omnibase_infra.runtime.first_effect_ledger.verified_first_effect_grant_types import (
    CanonicalModelEventClass,
    VerifiedGrantExpectedOutputTopic,
)


class ModelExpectedOutputPin(BaseModel):
    """Deployment configuration, not a per-request input or grant rehydration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: VerifiedGrantExpectedOutputTopic
    event_class: CanonicalModelEventClass
    index: Literal[0]


__all__ = ["ModelExpectedOutputPin"]
