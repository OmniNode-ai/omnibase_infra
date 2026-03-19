# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""RAG signal model for savings estimation."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelRagSignal(BaseModel):
    """Signal from memory/RAG retrieval avoiding regeneration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tokens_retrieved: int
    regen_tokens_estimate: int


__all__: list[str] = ["ModelRagSignal"]
