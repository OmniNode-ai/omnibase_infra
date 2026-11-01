# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One topic a contract is about to subscribe (OMN-21753)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelContractSubscribePreflightRequest(BaseModel):
    """The topic whose consumer ``HandlerContractSubscribePreflight`` prepares.

    Attributes:
        topic: Canonical topic, exactly as it is about to be passed to
            ``subscribe``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str = Field(..., min_length=1)
