# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verdict of the C11 delegation legs (OMN-23177)."""

from pydantic import BaseModel, ConfigDict


class ModelC11DelegationLegsResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    outcome: str
    reasons: tuple[str, ...] = ()
    evidence: dict[str, object] = {}
    error_message: str | None = None
