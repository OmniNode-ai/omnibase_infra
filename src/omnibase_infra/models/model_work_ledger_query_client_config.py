# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The client half of the work-ledger read node's wire contract."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelWorkLedgerQueryClientConfig(BaseModel):
    """A dev-only caller, with its topics and exact granted reply-group prefix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lane: Literal["dev"]
    command_topic: str = Field(min_length=1)
    completed_topic: str = Field(min_length=1)
    consumer_group_prefix: str = Field(min_length=16, pattern=r"\.$")
    queries: tuple[str, ...] = Field(min_length=1)
