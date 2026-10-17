# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A consumer group prefix a principal reads and describes under."""

from pydantic import BaseModel, ConfigDict, Field


class ModelBrokerGroupPrefix(BaseModel):
    """Groups named ``prefix`` plus a per-call suffix the derivation cannot know."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker: str = Field(min_length=1, description="Logical broker reference.")
    prefix: str = Field(min_length=1, description="Leading characters of the group.")
