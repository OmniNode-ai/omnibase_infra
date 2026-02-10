# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Minimal projection record wrapper model.

This model wraps projection record data into a Pydantic BaseModel for use
with ModelPayloadPostgresUpsertRegistration. Critical columns are declared
as explicit fields for validation. Dynamic projection columns (which vary
by projector schema) are stored in the ``data`` dict field.

Related:
    - HandlerNodeIntrospected: Primary consumer of this model
    - ModelPayloadPostgresUpsertRegistration: Uses this as record field type
    - IntentEffectPostgresUpsert: Merges ``data`` into top-level record dict
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionRecord(BaseModel):
    """Minimal model for wrapping projection record dicts.

    Critical columns (``entity_id``, ``current_state``, ``domain``,
    ``node_type``) are declared explicitly so that typos in these required
    fields fail validation instead of silently being ignored.

    Dynamic projection columns that vary by projector schema are stored in
    the ``data`` dict field. Consumers (e.g., IntentEffectPostgresUpsert)
    merge ``data`` into the top-level record dict before passing to the
    projector for database upsert.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)

    entity_id: UUID = Field(
        ...,
        description=(
            "Entity UUID. Required for upsert conflict resolution "
            "on the registration_projections table."
        ),
    )
    # Note: entity_id accepts both UUID and string inputs via model_validate().
    # Pydantic coerces strings to UUID automatically. model_dump() returns
    # a native UUID object, which IntentEffectPostgresUpsert._normalize_for_asyncpg()
    # passes through directly to asyncpg.
    current_state: str = Field(
        ...,
        description=(
            "FSM state value (e.g., 'pending_registration'). Required for "
            "registration projection state tracking."
        ),
    )
    domain: str = Field(
        default="registration",
        description=(
            "Projection domain discriminator. Identifies the projector schema "
            "this record belongs to, enabling consumers to distinguish between "
            "different projection record types."
        ),
    )
    node_type: str = Field(
        ...,
        description=(
            "Node type value (e.g., 'effect', 'compute'). Required NOT NULL "
            "in schema_registration_projection.sql."
        ),
    )
    # ONEX_EXCLUDE: any_type - dict[str, Any] required for dynamic projection columns
    # that vary by projector schema. Cannot use a typed model because columns differ
    # per projector (e.g., ack_deadline, capabilities, node_version).
    data: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Dynamic projection columns that vary by projector schema. "
            "Consumers merge this dict into the top-level record dict "
            "before database upsert. Replaces the previous extra='allow' "
            "approach with an explicit, schema-compliant field."
        ),
    )


__all__: list[str] = ["ModelProjectionRecord"]
