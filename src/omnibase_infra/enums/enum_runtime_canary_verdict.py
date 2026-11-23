# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The verdict of one runtime canary run (OMN-26463)."""

from __future__ import annotations

from enum import StrEnum

__all__: list[str] = ["EnumRuntimeCanaryVerdict"]


class EnumRuntimeCanaryVerdict(StrEnum):
    """PASS only when the canary booted, woke every dormant contract and held its heartbeat."""

    PASS = "PASS"
    FAIL = "FAIL"
