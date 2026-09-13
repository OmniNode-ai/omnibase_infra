# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boundary model for the terminal event that calls the chain writer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelDelegationTerminalPayload(BaseModel):
    """Accept the producer-owned terminal body without making it evidence."""

    model_config = ConfigDict(extra="allow", frozen=True)


__all__ = ["ModelDelegationTerminalPayload"]
