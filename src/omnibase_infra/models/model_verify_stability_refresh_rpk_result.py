# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The outcome of one stability health gate Kafka API read (OMN-25927)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelVerifyStabilityRefreshRpkResult(BaseModel):
    """rpk's exit code and output, and whether the read went as the gate's principal.

    ``detail`` is the tail of rpk's standard error, or why the read did not finish.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    exit_code: int
    authenticated: bool
    stdout: str = ""
    detail: str = ""
