# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Enum for LLM endpoint slot status values defined in llm_endpoints.yaml."""

from __future__ import annotations

from enum import Enum


class EnumLlmEndpointStatus(str, Enum):
    """Closed set of slot status values from contracts/llm_endpoints.yaml."""

    RUNNING = "running"
    DISABLED = "disabled"
    ON_DEMAND = "on_demand"
    PLANNED = "planned"


__all__ = ["EnumLlmEndpointStatus"]
