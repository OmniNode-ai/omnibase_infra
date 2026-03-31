# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Inference intent model emitted by the delegation orchestrator."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelInferenceIntent(BaseModel):
    """Intent emitted when the orchestrator requests LLM inference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: str = Field(default="llm_inference")
    base_url: str
    model: str
    system_prompt: str
    prompt: str
    max_tokens: int
    correlation_id: UUID


__all__: list[str] = ["ModelInferenceIntent"]
