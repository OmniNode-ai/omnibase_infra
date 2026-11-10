# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker's boot health verdict."""

from typing import Any

from typing_extensions import TypedDict


class ModelDevBootHealthStatus(TypedDict, total=False):
    Status: str
    FailingStreak: int
    Log: list[dict[str, Any]]
