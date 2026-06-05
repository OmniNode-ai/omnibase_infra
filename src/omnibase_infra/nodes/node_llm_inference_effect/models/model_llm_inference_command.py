# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bus command model for LLM inference requests."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from omnibase_core.types import JsonType
from omnibase_infra.enums import EnumLlmOperationType


class ModelLlmInferenceCommand(BaseModel):
    """Typed command consumed from the node's contract-declared bus topic."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    correlation_id: UUID = Field(default_factory=uuid4)
    model: str = Field(..., min_length=1)
    messages: tuple[dict[str, JsonType], ...] = Field(default_factory=tuple)
    prompt: str | None = None
    operation_type: EnumLlmOperationType = EnumLlmOperationType.CHAT_COMPLETION
    endpoint_url: str | None = None
    base_url: str | None = None
    provider_config: dict[str, JsonType] = Field(default_factory=dict)
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    stop: tuple[str, ...] = Field(default_factory=tuple)
    api_key: str | None = Field(default=None, repr=False)
    extra_headers: dict[str, str] = Field(default_factory=dict, repr=False)
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=600.0)
    gpu_type: str | None = Field(default=None, min_length=1, max_length=64)
    gpu_count: int | None = Field(default=None, ge=1, le=32767)
    compute_usage_source: str | None = None

    def provider_value(self, key: str) -> Any:
        """Return a provider_config value using a validated command key."""
        return self.provider_config.get(key)


__all__: list[str] = ["ModelLlmInferenceCommand"]
