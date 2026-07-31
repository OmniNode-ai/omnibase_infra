# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed application-relation ownership violation codes."""

from enum import StrEnum, unique


@unique
class EnumApplicationRelationViolation(StrEnum):
    """Deterministic global ownership violation codes."""

    MISSING_OWNER = "missing_owner"
    DUPLICATE_OWNER = "duplicate_owner"
    CONFLICTING_LOCATION = "conflicting_location"
    UNKNOWN_RELATION = "unknown_relation"
    UNKNOWN_DATABASE = "unknown_database"
    UNKNOWN_SCHEMA = "unknown_schema"
    MISSING_TYPED_LOCATION = "missing_typed_location"
    DOMAIN_MISMATCH = "domain_mismatch"
    OWNER_MISMATCH = "owner_mismatch"
    PURPOSE_MISMATCH = "purpose_mismatch"
    BLOCKED_RELATION = "blocked_relation"
    INCOMPLETE_CENSUS = "incomplete_census"


__all__ = ["EnumApplicationRelationViolation"]
