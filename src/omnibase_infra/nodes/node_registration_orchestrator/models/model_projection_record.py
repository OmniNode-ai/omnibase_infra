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

from pydantic import BaseModel, ConfigDict


class ModelProjectionRecord(BaseModel):
    """Minimal model for wrapping projection record dicts.

    Uses extra='allow' so all dict keys are preserved as extra fields.
    This allows the record to be stored as SerializeAsAny[BaseModel] in
    ModelPayloadPostgresUpsertRegistration while retaining all data.
    SerializeAsAny ensures model_dump() serializes all extra fields.
    """

    model_config = ConfigDict(extra="allow", frozen=True, from_attributes=True)


__all__: list[str] = ["ModelProjectionRecord"]
