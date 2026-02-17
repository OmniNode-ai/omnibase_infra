# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Model for a single demo reset action result.

.. versionadded:: 0.9.1
"""

from __future__ import annotations

from dataclasses import dataclass

from omnibase_infra.cli.enum_reset_action import EnumResetAction

__all__: list[str] = [
    "ModelResetActionResult",
]


@dataclass(frozen=True)
class ModelResetActionResult:
    """Result of a single reset action.

    Attributes:
        resource: Name of the resource affected.
        action: What was done (reset, preserved, skipped, error).
        detail: Human-readable description of what happened.
    """

    resource: str
    action: EnumResetAction
    detail: str
