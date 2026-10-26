# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Local host operation requested from the boot effect."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelDevBootRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    operation: Literal["prepare", "wait", "preflight", "report"]
