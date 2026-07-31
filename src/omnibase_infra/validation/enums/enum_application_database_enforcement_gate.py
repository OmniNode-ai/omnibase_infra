# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Application database enforcement gate families."""

from enum import StrEnum, unique


@unique
class EnumApplicationDatabaseEnforcementGate(StrEnum):
    """Closed set of source and deployment assertions owned by OMN-15361."""

    CLASSIFICATION = "classification"
    SCHEMA_QUALIFICATION = "schema_qualification"
    TENANT_RLS = "tenant_rls"
    INTERNAL_CATALOG = "internal_catalog"
    ROLE_ACL = "role_acl"
    ONE_DATABASE = "one_database"
    ADAPTER = "adapter"
    TOPOLOGY_PARITY = "topology_parity"


__all__ = ["EnumApplicationDatabaseEnforcementGate"]
