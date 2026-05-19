# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Materialization status enum for dynamic contract wiring (OMN-11244)."""

from __future__ import annotations

from enum import Enum


class EnumMaterializationStatus(str, Enum):
    """Outcome of a materialize_cached_contract() call."""

    MATERIALIZED = "materialized"
    ALREADY_MATERIALIZED = "already_materialized"
    REJECTED = "rejected"


__all__ = ["EnumMaterializationStatus"]
