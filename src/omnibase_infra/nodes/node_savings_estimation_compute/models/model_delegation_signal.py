# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Delegation signal model for savings estimation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelDelegationSignal(BaseModel):
    """Signal from avoided subagent delegation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subagent_calls_avoided: int


__all__: list[str] = ["ModelDelegationSignal"]
