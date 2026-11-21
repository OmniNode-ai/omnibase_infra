# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A request held at the front door for a dormant contract (OMN-21592)."""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__: list[str] = ["ModelLazyStartRequest"]


class ModelLazyStartRequest(BaseModel):
    """The target contract, the topic the request was addressed to, and the envelope.

    The envelope is delivered to the started node's own dispatcher in this
    process; it never rides the node's command topic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    operation: Literal["node.start.requested"] = "node.start.requested"
    contract_name: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    correlation_id: UUID
    envelope: object
