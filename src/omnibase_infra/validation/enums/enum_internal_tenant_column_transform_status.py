# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Internal tenant-column transformation evidence states."""

from enum import StrEnum, unique


@unique
class EnumInternalTenantColumnTransformStatus(StrEnum):
    """Finite states emitted by the OMN-15423 classification inventory."""

    NOT_APPLICABLE_NO_SOURCE_TENANT_ID = "not_applicable_no_source_tenant_id"
    SOURCE_DEPENDENCY_INVENTORY_COMPLETE_RUNTIME_COLLISION_SCAN_BLOCKED = (
        "source_dependency_inventory_complete_runtime_collision_scan_blocked"
    )


__all__ = ["EnumInternalTenantColumnTransformStatus"]
