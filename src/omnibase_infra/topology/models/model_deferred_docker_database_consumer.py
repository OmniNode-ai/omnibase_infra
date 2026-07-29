# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed fail-closed hold for an unclassified Docker database consumer."""

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelDeferredDockerDatabaseConsumer"]


class ModelDeferredDockerDatabaseConsumer(BaseModel):
    """Explicit hold for a consumer awaiting semantic classification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tracking_ticket: str = Field(pattern=r"^OMN-[0-9]+$")
    reason: str = Field(min_length=1)
