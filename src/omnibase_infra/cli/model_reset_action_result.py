# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Model for a single demo reset action result.

.. versionadded:: 0.9.1
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from omnibase_infra.cli.enum_reset_action import EnumResetAction

__all__: list[str] = [
    "ModelResetActionResult",
]


class ModelResetActionResult(BaseModel):
    """Result of a single reset action.

    Attributes:
        resource: Name of the resource affected.
        action: What was done (reset, preserved, skipped, error).
        detail: Human-readable description of what happened.
    """

    model_config = ConfigDict(frozen=True)

    resource: str
    action: EnumResetAction
    detail: str
