# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The downstream delegation chain declared by installed node contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelDelegateDownstreamChain(BaseModel):
    """Topics and consumer contract a deployed orchestrator hands work to."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    command_topic: str = Field(
        ..., min_length=1, description="Downstream command topic."
    )
    completed_topic: str = Field(
        ..., min_length=1, description="Downstream completion topic."
    )
    consumer_contract: str = Field(
        ..., description="Downstream consumer contract name."
    )
    consumer_contract_path: str = Field(
        ..., description="Consumer contract filesystem path."
    )
    subscribe_topics: tuple[str, ...] = Field(
        ..., description="Full subscription footprint in consumer contract order."
    )
