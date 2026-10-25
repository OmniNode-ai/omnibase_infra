# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The installed read node's wire contract, resolved without a market dependency."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelWorkLedgerQueryContract(BaseModel):
    """Topics and typed models owned by the query node, never copied into infra."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    command_topic: str = Field(min_length=1)
    completed_topic: str = Field(min_length=1)
    request_model: type[BaseModel]
    result_model: type[BaseModel]
