# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Validator catch signal model for savings estimation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelValidatorCatchSignal(BaseModel):
    """Signal from a validator catching an error before LLM regen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    validator_type: str
    severity: str


__all__: list[str] = ["ModelValidatorCatchSignal"]
