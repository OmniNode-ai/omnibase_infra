# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``ModelWorkflowReceiptReplayDiagnostic`` (OMN-15095)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelWorkflowReceiptReplayDiagnostic(BaseModel):
    """One field-level mismatch between the recorded receipt and the replayed
    reconstruction -- never a silent pass, never a bare exception."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str
    receipt_value: str
    replayed_value: str


__all__ = ["ModelWorkflowReceiptReplayDiagnostic"]
