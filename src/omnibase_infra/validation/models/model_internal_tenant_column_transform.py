# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed internal-domain tenant-column transformation evidence."""

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.validation.enums.enum_internal_tenant_column_transform_status import (
    EnumInternalTenantColumnTransformStatus,
)


class ModelInternalTenantColumnTransform(BaseModel):
    """Evidence required before an internal relation drops tenant stamping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EnumInternalTenantColumnTransformStatus
    source_occurrences: tuple[str, ...]
    key_fk_index_partition_dependencies: tuple[str, ...]
    runtime_collision_scan: str = Field(..., min_length=1)


__all__ = ["ModelInternalTenantColumnTransform"]
