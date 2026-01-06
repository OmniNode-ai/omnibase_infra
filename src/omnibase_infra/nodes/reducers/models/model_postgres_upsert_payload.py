# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""PostgreSQL upsert intent payload wrapper.

This wrapper class implements ProtocolIntentPayload to satisfy ModelIntent's
payload type requirement while preserving the "postgres.upsert_registration"
intent type. This is necessary because omnibase_core 0.6.2 changed
ModelIntent.payload to require ProtocolIntentPayload instances instead of
accepting dicts.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelPostgresUpsertPayload(BaseModel):
    """Payload wrapper for PostgreSQL upsert intents.

    Wraps PostgreSQL upsert data to implement ProtocolIntentPayload
    while preserving the "postgres.upsert_registration" intent type.

    Attributes:
        data: The serialized PostgreSQL upsert intent data.
    """

    data: dict[str, object]

    @property
    def intent_type(self) -> str:
        """Return the intent type for routing."""
        return "postgres.upsert_registration"
