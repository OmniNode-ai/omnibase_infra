# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Evidence source kinds accepted by application-database principal census."""

from enum import StrEnum, unique


@unique
class EnumApplicationDatabasePrincipalInventorySourceKind(StrEnum):
    """Distinguish authorized catalog evidence from synthetic proof input."""

    AUTHORIZED_CATALOG = "authorized_catalog"
    SYNTHETIC_FIXTURE = "synthetic_fixture"


__all__ = ["EnumApplicationDatabasePrincipalInventorySourceKind"]
