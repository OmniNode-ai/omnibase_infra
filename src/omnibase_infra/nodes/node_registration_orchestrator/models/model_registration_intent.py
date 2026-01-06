# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration intent type aliases for the registration orchestrator.

This module provides type aliases that aggregate the individual intent and
payload models into discriminated unions. These aliases enable type narrowing
based on the `kind` field in intent models.

Design Note:
    Rather than using a loose dict[str, JsonType] for payloads, we use
    typed payload models that match the exact structure expected by each
    infrastructure adapter. This follows the ONEX principle of "no Any types"
    and provides compile-time validation of intent payloads.

    The pattern uses Literal discriminators for the `kind` field, enabling
    type narrowing in effect node handlers.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field

from omnibase_infra.nodes.node_registration_orchestrator.models.model_consul_intent_payload import (
    ModelConsulIntentPayload,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_consul_registration_intent import (
    ModelConsulRegistrationIntent,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_intent_payload import (
    ModelPostgresIntentPayload,
)
from omnibase_infra.nodes.node_registration_orchestrator.models.model_postgres_upsert_intent import (
    ModelPostgresUpsertIntent,
)

# Type alias for intent payloads
IntentPayload = ModelConsulIntentPayload | ModelPostgresIntentPayload

# Discriminated union of all intent types using Annotated pattern
# This enables type narrowing based on the `kind` field
# TODO(OMN-912): Migrate to discriminated intent unions once available in omnibase_core
ModelRegistrationIntent = Annotated[
    ModelConsulRegistrationIntent | ModelPostgresUpsertIntent,
    Field(discriminator="kind"),
]


__all__ = [
    "IntentPayload",
    "ModelConsulIntentPayload",
    "ModelConsulRegistrationIntent",
    "ModelPostgresIntentPayload",
    "ModelPostgresUpsertIntent",
    "ModelRegistrationIntent",
]
