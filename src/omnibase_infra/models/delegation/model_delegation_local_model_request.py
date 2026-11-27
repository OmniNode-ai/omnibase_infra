# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed input for the delegation local-model check."""

from pydantic import BaseModel, ConfigDict


class ModelDelegationLocalModelRequest(BaseModel):
    """Typed input of the delegation local-model check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    hosted_providers: tuple[str, ...] | None = None


__all__ = ["ModelDelegationLocalModelRequest"]
