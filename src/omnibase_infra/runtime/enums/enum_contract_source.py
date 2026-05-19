# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract source enum for dynamic materialization (OMN-11244)."""

from __future__ import annotations

from enum import Enum


class EnumContractSource(str, Enum):
    """Origin of a contract that was materialized."""

    FILESYSTEM = "filesystem"
    DYNAMIC_KAFKA = "dynamic_kafka"
    REGISTRY = "registry"


__all__ = ["EnumContractSource"]
