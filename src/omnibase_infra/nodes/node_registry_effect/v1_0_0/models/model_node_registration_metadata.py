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

# Maximum limits for bounded collections
MAX_TAGS: int = 20
MAX_LABELS: int = 50

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
        description=f"Categorization tags (max {MAX_TAGS})",
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description=f"Kubernetes-style labels (validated keys, max {MAX_LABELS})",
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
        """Normalize tags to lowercase, deduplicate, and enforce maximum limit.

        SECURITY NOTE: This validator enforces a maximum of MAX_TAGS to prevent
        DoS via oversized payloads. When the limit is exceeded, an explicit error
        is raised to prevent silent data loss.

        Args:
            v: List of tags to normalize

        Returns:
            Normalized, deduplicated, and bounded list of tags

        Raises:
            ValueError: If too many tags after normalization and deduplication
        """
        if not v:
            return []
        normalized = [tag.lower().strip() for tag in v if tag and tag.strip()]
        # Deduplicate while preserving order
        unique_tags = list(dict.fromkeys(normalized))

        # EXPLICIT ERROR: Tags exceeding limit is an error (not silent truncation)
        # This is consistent with label validation behavior (see validate_labels)
        if len(unique_tags) > MAX_TAGS:
            raise ValueError(
                f"Maximum {MAX_TAGS} tags allowed, received {len(unique_tags)} "
                f"(after normalization and deduplication). "
                f"Reduce the number of tags or split across multiple registrations."
            )

        return unique_tags

    @field_validator("labels", mode="before")
    @classmethod
    def validate_labels(cls, v: dict[str, str]) -> dict[str, str]:
        """Validate label keys follow k8s naming conventions.

        SECURITY NOTE: This validator enforces a maximum of MAX_LABELS to prevent
        DoS via oversized payloads. Unlike tags, labels raise an error when the
        limit is exceeded because label semantics are stricter (k8s-style).

        Args:
            v: Dictionary of labels to validate

        Returns:
            Validated and normalized labels dictionary

        Raises:
            ValueError: If too many labels or invalid key format
        """
        if not v:
            return {}
        if len(v) > MAX_LABELS:
            # EXPLICIT ERROR: Labels exceeding limit is an error (not silent truncation)
            # This is intentional - labels have stricter semantics than tags
            raise ValueError(
                f"Maximum {MAX_LABELS} labels allowed, received {len(v)}. "
                f"Reduce the number of labels or split across multiple registrations."
            )
        result = {}
        for key, val in v.items():
            key_lower = key.lower()
            if not _LABEL_KEY_PATTERN.match(key_lower):
                # Sanitize key for error message (truncate and remove special chars)
                sanitized_key = key[:30].replace("\n", "").replace("\r", "")
                raise ValueError(
                    f"Invalid label key format: '{sanitized_key}'. "
                    f"Keys must follow k8s naming: lowercase alphanumeric with hyphens/dots."
                )
            result[key_lower] = str(val)
        return result
