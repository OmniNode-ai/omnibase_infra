# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Audited control-plane operations for tenant identity roots."""

from enum import StrEnum, unique


@unique
class EnumApplicationDatabaseIdentityRootOperation(StrEnum):
    """Operations that require an explicit non-runtime RLS bypass identity."""

    TENANT_CREATION = "tenant_creation"
    CROSS_TENANT_ENUMERATION = "cross_tenant_enumeration"


__all__ = ["EnumApplicationDatabaseIdentityRootOperation"]
