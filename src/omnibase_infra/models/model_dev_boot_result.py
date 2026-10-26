# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Successful local host boot operation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelDevBootResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    operation: Literal["prepare", "wait", "preflight", "report"]
    status: Literal["pass"]
