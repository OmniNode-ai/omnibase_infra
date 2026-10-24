# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""One correlated receipt event the receipt-mode client returns (OMN-21341)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, JsonValue


class ModelReceiptModeEvent(BaseModel):
    """A correlated receipt event exposed to the hook adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    topic: str
    payload: dict[str, JsonValue]


__all__ = ["ModelReceiptModeEvent"]
