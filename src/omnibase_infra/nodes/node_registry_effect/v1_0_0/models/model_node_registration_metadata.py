# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node registration runtime metadata model.

NOTE: This is a temporary local model for MVP. When omnibase_core >= 0.5.0
is released with ModelNodeRegistrationMetadata, migrate to use that instead.
See Linear tickets:
- OMN-900: Add ModelNodeRegistrationMetadata to omnibase_core
- OMN-901: Migrate to core ModelNodeRegistrationMetadata when available
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .enum_environment import EnumEnvironment

# Pre-compiled regex for label key validation (k8s-style)
_LABEL_KEY_PATTERN = re.compile(
    r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?(\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*$"
)


class ModelNodeRegistrationMetadata(BaseModel):
    """Runtime/deployment metadata for node registration.

    Separate from ModelNodeCapabilitiesMetadata (authorship/docs).
    This is environment-specific, deployment-specific, mutable data
    captured during node registration.

    Attributes:
        environment: Deployment environment (dev, staging, prod, etc.)
        tags: Categorization tags (bounded list, normalized lowercase)
        labels: Kubernetes-style labels (str -> str, validated keys)
        release_channel: Optional release channel (stable, canary, beta)
        region: Optional deployment region (us-east-1, eu-west-1, etc.)
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    environment: EnumEnvironment = Field(..., description="Deployment environment")
    tags: list[str] = Field(
        default_factory=list,
        description="Categorization tags (max 20)",
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Kubernetes-style labels (validated keys, max 50)",
    )
    release_channel: str | None = Field(
        default=None,
        description="Release channel (e.g., stable, canary, beta)",
    )
    region: str | None = Field(
        default=None,
        description="Deployment region (e.g., us-east-1)",
    )

    @field_validator("tags", mode="before")
    @classmethod
    def normalize_tags(cls, v: list[str]) -> list[str]:
        """Normalize tags to lowercase and deduplicate."""
        if not v:
            return []
        normalized = [tag.lower().strip() for tag in v if tag and tag.strip()]
        # Deduplicate while preserving order
        return list(dict.fromkeys(normalized))[:20]  # Max 20 tags

    @field_validator("labels", mode="before")
    @classmethod
    def validate_labels(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate label keys follow k8s naming conventions."""
        if not v:
            return {}
        if len(v) > 50:
            raise ValueError("Maximum 50 labels allowed")
        result = {}
        for key, val in v.items():
            key_lower = key.lower()
            if not _LABEL_KEY_PATTERN.match(key_lower):
                raise ValueError(f"Invalid label key format: {key}")
            result[key_lower] = str(val)
        return result
