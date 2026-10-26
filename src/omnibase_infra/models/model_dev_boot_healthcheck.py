# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Declared Docker health timings in nanoseconds."""

from typing_extensions import TypedDict


class ModelDevBootHealthcheck(TypedDict, total=False):
    StartPeriod: int
    Retries: int
    Interval: int
    Timeout: int
