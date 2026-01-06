# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Consul registration intent payload wrapper.

This wrapper class implements ProtocolIntentPayload to satisfy ModelIntent's
payload type requirement while preserving the "consul.register" intent type.
This is necessary because omnibase_core 0.6.2 changed ModelIntent.payload to
require ProtocolIntentPayload instances instead of accepting dicts.
"""

from __future__ import annotations

from pydantic import BaseModel


class ModelConsulRegisterPayload(BaseModel):
    """Payload wrapper for Consul registration intents.

    Wraps Consul registration data to implement ProtocolIntentPayload
    while preserving the "consul.register" intent type.

    Attributes:
        data: The serialized Consul registration intent data.
    """

    data: dict[str, object]

    @property
    def intent_type(self) -> str:
        """Return the intent type for routing."""
        return "consul.register"
