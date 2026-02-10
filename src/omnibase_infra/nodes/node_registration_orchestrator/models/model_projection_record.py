# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Minimal projection record wrapper model.

This model wraps arbitrary projection record dicts into a Pydantic BaseModel
for use with ModelPayloadPostgresUpsertRegistration. Uses extra='allow' so
all dict keys are preserved as extra fields, and SerializeAsAny ensures
model_dump() serializes all extra fields.

Related:
    - HandlerNodeIntrospected: Primary consumer of this model
    - ModelPayloadPostgresUpsertRegistration: Uses this as record field type
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ModelProjectionRecord(BaseModel):
    """Minimal model for wrapping projection record dicts.

    Uses extra='allow' so all dict keys are preserved as extra fields.
    This allows the record to be stored as SerializeAsAny[BaseModel] in
    ModelPayloadPostgresUpsertRegistration while retaining all data.
    SerializeAsAny ensures model_dump() serializes all extra fields.

    Critical columns (``entity_id``, ``current_state``, ``domain``) are
    declared explicitly so that typos in these required fields fail
    validation instead of silently passing through as extra fields.
    Non-critical columns remain as extra fields.

    Warning:
        **Non-standard ``extra`` config**: Uses ``extra="allow"`` instead of the
        project convention ``extra="forbid"`` (see CLAUDE.md Pydantic Model
        Standards). This is intentional — the model acts as a pass-through
        wrapper for arbitrary projection column dicts whose keys vary by
        projector schema. ``extra="forbid"`` would reject unknown columns.
    """

    model_config = ConfigDict(extra="allow", frozen=True, from_attributes=True)

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


__all__: list[str] = ["ModelProjectionRecord"]
