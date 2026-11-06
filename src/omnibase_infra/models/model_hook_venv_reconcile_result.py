# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Readback result of an exact hook environment reconciliation."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelHookVenvReconcileResult(BaseModel):
    """A refusal or a revision-verified sync, with the package tool's output."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    status: Literal["already_in_sync", "reconciled", "refused"]
    detail: str
    output: str = ""
