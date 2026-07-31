# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Narrow tenant identity-root exceptions declared by the approved plan."""

from enum import StrEnum, unique


@unique
class EnumApplicationDatabaseIdentityRoot(StrEnum):
    """Closed identity-root contracts; ordinary tenant tables are excluded."""

    CANONICAL_TENANT = "canonical_tenant_identity_root"
    PRE_TENANT_BOOTSTRAP = "pre_tenant_bootstrap_identity_root"


__all__ = ["EnumApplicationDatabaseIdentityRoot"]
