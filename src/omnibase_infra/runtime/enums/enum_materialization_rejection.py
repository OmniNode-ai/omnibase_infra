# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Materialization rejection reason enum for dynamic contract wiring (OMN-11244)."""

from __future__ import annotations

from enum import Enum


class EnumMaterializationRejection(str, Enum):
    """Reason a materialize_cached_contract() call was rejected."""

    VERSION_CONFLICT = "version_conflict"
    HASH_MISMATCH = "hash_mismatch"
    PROFILE_MISMATCH = "profile_mismatch"
    HANDLER_ALLOWLIST = "handler_allowlist"
    PARSE_FAILURE = "parse_failure"


__all__ = ["EnumMaterializationRejection"]
