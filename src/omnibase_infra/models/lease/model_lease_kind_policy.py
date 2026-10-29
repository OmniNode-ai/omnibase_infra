# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""How many holders one resource of a lease kind admits, and its TTL (OMN-21424)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelLeaseKindPolicy(BaseModel):
    """One lease kind: ``capacity`` concurrent grants per resource for ``ttl_s``."""

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    kind: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    capacity: int = Field(default=1, ge=1, le=64)
    resource_capacity: dict[str, int] = Field(default_factory=dict)
    ttl_s: int = Field(default=30, ge=2)

    @field_validator("resource_capacity")
    @classmethod
    def validate_resource_capacity(cls, value: dict[str, int]) -> dict[str, int]:
        """Apply the same capacity bounds to each resource override."""
        if any(capacity < 1 or capacity > 64 for capacity in value.values()):
            raise ValueError("resource capacities must be between 1 and 64")
        return value

    def capacity_of(self, resource: str) -> int:
        """Return the resource override or the kind's default capacity."""
        return self.resource_capacity.get(resource, self.capacity)


__all__ = ["ModelLeaseKindPolicy"]
