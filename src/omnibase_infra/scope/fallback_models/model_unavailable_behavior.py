# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Compatibility subset of the core unavailable-behavior model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

UnavailableTier = Literal["hidden", "explain", "warn", "block"]


class ModelUnavailableBehavior(BaseModel):
    """Compatibility subset of the core unavailable-behavior model."""

    model_config = ConfigDict(frozen=True, extra="allow")

    default: UnavailableTier = "hidden"
    diagnostics: UnavailableTier = "explain"


__all__ = ["ModelUnavailableBehavior"]
