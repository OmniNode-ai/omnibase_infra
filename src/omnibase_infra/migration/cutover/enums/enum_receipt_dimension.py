# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Required dimensions of a transformation-aware receipt."""

from enum import StrEnum


class EnumReceiptDimension(StrEnum):
    """Every family receipt evaluates every dimension explicitly."""

    EVIDENCE_CONTRACTS = "evidence_contracts"
    KEY_SET = "key_set"
    ROW_COUNT = "row_count"
    TRANSFORMATION_HASH = "transformation_hash"
    FOREIGN_KEYS = "foreign_keys"
    SEQUENCES = "sequences"
    OWNERS = "owners"
    GRANTS = "grants"
    POLICIES = "policies"
    VIEWS_FUNCTIONS = "views_functions"
    EVENT_OFFSETS = "event_offsets"
    CONTROL_PLANE_DELTA = "control_plane_delta"
    COLLISIONS = "collisions"
    DEPENDENCIES = "dependencies"


__all__ = ["EnumReceiptDimension"]
