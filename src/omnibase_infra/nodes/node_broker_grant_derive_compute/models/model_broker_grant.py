# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One exact broker resource permission."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelBrokerGrant(BaseModel):
    """An exact resource grant; wildcard permissions are never inferred."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker: str = Field(min_length=1)
    resource_type: Literal["TOPIC", "GROUP"]
    resource: str = Field(min_length=1)
    operation: Literal["READ", "WRITE", "DESCRIBE"]
