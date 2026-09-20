# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The execution bound declared for one delegate task class."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["ModelTaskClassExecutionBudget"]


class ModelTaskClassExecutionBudget(BaseModel):
    """A task class's execution and terminal-delivery windows.

    The ceiling is deliberately below the deployed dispatch port's 300-second
    wait.  The delivery margin belongs to the terminal waiter, not the model
    execution deadline.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_class_timeout_ceiling_seconds: int = Field(gt=0, le=240)
    terminal_delivery_margin_seconds: int = Field(gt=0)
