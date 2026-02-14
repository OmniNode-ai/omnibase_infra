# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Event Registry Fingerprint Element Model.

Frozen Pydantic model representing the canonical fingerprint of a single
event registration.  Extracted to its own file per the one-model-per-file
architectural rule.

Related:
    - OMN-2088: Handshake hardening -- Event registry fingerprint + startup assertion
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ModelEventRegistryFingerprintElement(BaseModel):
    """Canonical fingerprint for a single event registration.

    Each element captures the full identity of one ``ModelEventRegistration``
    in a deterministic, hashable form.

    Attributes:
        event_type: Semantic event type identifier.
        topic_template: Canonical ONEX topic suffix (no env prefix).
        schema_version: Semantic version of the event schema.
        partition_key_field: Partition key field name, or empty string if None.
        required_fields: Sorted tuple of required payload field names.
        element_sha256: SHA-256 hex digest of the canonical tuple for this element.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    event_type: str = Field(..., description="Semantic event type identifier")
    topic_template: str = Field(
        ..., description="Canonical ONEX topic suffix (no env prefix)"
    )
    schema_version: str = Field(..., description="Semantic version of the event schema")
    partition_key_field: str = Field(
        ..., description="Partition key field name, or empty string if None"
    )
    required_fields: tuple[str, ...] = Field(
        ..., description="Sorted tuple of required payload field names"
    )
    element_sha256: str = Field(
        ..., description="SHA-256 hex digest of the canonical element tuple"
    )


__all__: list[str] = [
    "ModelEventRegistryFingerprintElement",
]
