# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed input for the delegation identity check."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ModelDelegationIdentityRequest(BaseModel):
    """Typed input of the delegation identity check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    onex_home: Path
    local_identity: str | None = Field(default=None)
    identity_unreadable: bool = False


__all__ = ["ModelDelegationIdentityRequest"]
