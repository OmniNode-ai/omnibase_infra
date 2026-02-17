# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Topic catalog query request model.

Defines the request payload for querying the topic catalog. Clients publish
this on the ``topic-catalog-query`` topic to request a filtered view of the
current catalog. The response arrives on ``topic-catalog-response`` with a
matching ``correlation_id``.

Related Tickets:
    - OMN-2310: Topic Catalog model + suffix foundation

.. versionadded:: 0.9.0
"""

from __future__ import annotations

import re
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Pattern validation: allow alphanumeric, dots, asterisks, question marks,
# underscores, and hyphens. This covers fnmatch glob characters.
_TOPIC_PATTERN_REGEX = re.compile(r"^[a-zA-Z0-9.*?_\-]+$")

# Maximum length for topic_pattern field.
_MAX_TOPIC_PATTERN_LENGTH = 256


class ModelTopicCatalogQuery(BaseModel):
    """Query request for the topic catalog.

    Published by clients to request a snapshot of the current topic catalog.
    Supports optional filtering by topic pattern (fnmatch semantics) and
    active/inactive status.

    Attributes:
        correlation_id: Unique identifier for request-response correlation.
        client_id: Identifier of the requesting client (node ID or service name).
        include_inactive: Whether to include topics with zero publishers and
            zero subscribers. Defaults to False.
        topic_pattern: Optional fnmatch-style glob pattern to filter topics.
            Must contain only ``[a-zA-Z0-9.*?_-]`` characters, max 256 chars.
        schema_version: Schema version for forward compatibility. Currently ``1``.

    Example:
        >>> from uuid import uuid4
        >>> query = ModelTopicCatalogQuery(
        ...     correlation_id=uuid4(),
        ...     client_id="node-registration-orchestrator",
        ...     topic_pattern="onex.evt.platform.*",
        ... )
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        from_attributes=True,
    )

    correlation_id: UUID = Field(
        ...,
        description="Unique identifier for request-response correlation.",
    )
    client_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Identifier of the requesting client (node ID or service name).",
    )
    include_inactive: bool = Field(
        default=False,
        description=(
            "Whether to include topics with zero publishers and zero subscribers."
        ),
    )
    topic_pattern: str | None = Field(
        default=None,
        max_length=_MAX_TOPIC_PATTERN_LENGTH,
        description=(
            "Optional fnmatch-style glob pattern to filter topics. "
            "Must contain only [a-zA-Z0-9.*?_-] characters."
        ),
    )
    schema_version: int = Field(
        default=1,
        ge=1,
        description="Schema version for forward compatibility.",
    )

    @field_validator("topic_pattern", mode="after")
    @classmethod
    def validate_topic_pattern_chars(cls, v: str | None) -> str | None:
        """Validate topic_pattern contains only allowed characters.

        Args:
            v: The topic_pattern value after basic validation.

        Returns:
            The validated pattern, or None if no pattern specified.

        Raises:
            ValueError: If pattern contains disallowed characters.
        """
        if v is not None and not _TOPIC_PATTERN_REGEX.match(v):
            raise ValueError(
                f"topic_pattern contains invalid characters: '{v}'. "
                "Only [a-zA-Z0-9.*?_-] are allowed."
            )
        return v


__all__: list[str] = ["ModelTopicCatalogQuery"]
