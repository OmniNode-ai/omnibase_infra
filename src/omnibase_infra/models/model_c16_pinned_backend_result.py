# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verdict of the C16 pinned-backend probe (OMN-23176)."""

from pydantic import BaseModel, ConfigDict


class ModelC16PinnedBackendResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    outcome: str
    reasons: tuple[str, ...] = ()
    evidence: dict[str, object] = {}
    error_message: str | None = None
