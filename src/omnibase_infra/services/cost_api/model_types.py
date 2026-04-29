# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared type aliases for LLM cost API models."""

from __future__ import annotations

from typing import Literal

AggregationWindow = Literal["24h", "7d", "30d"]
TrendBucket = Literal["hour", "day"]
