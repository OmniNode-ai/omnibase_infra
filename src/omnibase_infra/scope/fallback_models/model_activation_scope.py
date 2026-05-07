# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Compatibility subset of the core activation scope model."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelActivationScope(BaseModel):
    """Compatibility subset of the core activation scope model."""

    model_config = ConfigDict(frozen=True, extra="allow")

    requires_tokens: list[str] = Field(default_factory=list)


__all__ = ["ModelActivationScope"]
