# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""LLM call record model for savings estimation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelLlmCallRecord(BaseModel):
    """Record of a single LLM call with token counts and model info."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # ONEX_EXCLUDE: pattern_validator - model_id is a human-readable model name, not a UUID
    model_id: str
    prompt_tokens: int
    completion_tokens: int


__all__: list[str] = ["ModelLlmCallRecord"]
