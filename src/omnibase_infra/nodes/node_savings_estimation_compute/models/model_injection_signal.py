# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Injection signal model for savings estimation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelInjectionSignal(BaseModel):
    """Signal from pattern injection into a prompt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens_injected: int
    patterns_count: int


__all__: list[str] = ["ModelInjectionSignal"]
