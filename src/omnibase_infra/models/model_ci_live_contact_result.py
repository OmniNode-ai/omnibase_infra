# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Admission decision, including actionable evidence (OMN-18648)."""

from pydantic import BaseModel, ConfigDict


class ModelCILiveContactResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    success: bool
    reason: str
    changed_tooling: tuple[str, ...] = ()
    live_contact_tests: tuple[str, ...] = ()
    error_message: str | None = None
