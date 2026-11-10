# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Docker's boot health verdict."""

from typing_extensions import TypedDict


class ModelDevBootHealthProbe(TypedDict, total=False):
    Start: str
    End: str
    ExitCode: int
    Output: str


class ModelDevBootHealthStatus(TypedDict, total=False):
    Status: str
    FailingStreak: int
    Log: list[ModelDevBootHealthProbe]
