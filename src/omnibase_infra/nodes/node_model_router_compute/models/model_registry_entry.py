# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Model registry entry — parsed from model_registry.yaml."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelRegistryEntry(BaseModel):
    """A single model from the registry, used as scoring input."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    model_key: str = Field(..., description="Unique model identifier.")
    provider: str = Field(..., description="Provider: local, anthropic, qwen, zhipu.")
    transport: str = Field(..., description="Transport: http, sdk, or oauth.")
    base_url_env: str = Field(
        default="",
        description="Env var name for base URL (http transport).",
    )
    api_key_env: str = Field(
        default="",
        description="Env var name for API key (sdk transport).",
    )
    capabilities: tuple[str, ...] = Field(
        default_factory=tuple, description="Declared capabilities."
    )
    context_window: int = Field(default=4096, description="Max context window tokens.")
    seed_cost_per_1k_tokens: float = Field(
        default=0.0, description="Bootstrap cost estimate per 1K tokens."
    )
    seed_tokens_per_sec: float = Field(
        default=0.0, description="Bootstrap throughput estimate."
    )
    tier: str = Field(default="local", description="Tier: local or frontier_api.")
    concurrency_limit: int | None = Field(
        default=None,
        description="Max concurrent requests to this model. None means unlimited.",
    )
