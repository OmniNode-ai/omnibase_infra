# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Semantic purposes for application relations."""

from enum import StrEnum, unique


@unique
class EnumApplicationRelationPurpose(StrEnum):
    """Semantic purpose kept separate from a PostgreSQL object kind."""

    DATA = "data"
    MIGRATION_LEDGER = "migration_ledger"


__all__ = ["EnumApplicationRelationPurpose"]
