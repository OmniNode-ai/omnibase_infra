# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Registration intent type aliases for the registration orchestrator.

This module provides type aliases that aggregate the individual intent and
payload models into discriminated unions. These aliases enable type narrowing
based on the `kind` field in intent models.

IMPORTANT - Union Sync Requirements (OMN-1007):
    This module defines explicit union types that MUST stay in sync with the
    RegistryIntent in model_registry_intent.py. There are two parallel systems:

    1. **RegistryIntent** (dynamic): Decorator-based registration for runtime
       type resolution. Used by ModelReducerExecutionResult for deserialization.

    2. **ModelRegistrationIntent** (static): Annotated union for Pydantic field
       validation. Used by ProtocolEffect.execute_intent() and similar APIs.

    When adding a new intent type, you MUST update BOTH:
    - Register with @RegistryIntent.register("kind") on the model class
    - Add the model to ModelRegistrationIntent union below

    Use validate_union_registry_sync() in tests to verify consistency.

    For function signatures, prefer ProtocolRegistrationIntent (duck-typed)
    over ModelRegistrationIntent (union-typed) when possible.

Design Note:
    Rather than using a loose dict[str, JsonType] for payloads, we use
    typed payload models that match the exact structure expected by each
    infrastructure adapter. This follows the ONEX principle of "no Any types"
    and provides compile-time validation of intent payloads.

    The pattern uses Literal discriminators for the `kind` field, enabling
    type narrowing in effect node handlers.

Related:
    - model_registry_intent.py: RegistryIntent and ModelRegistryIntent base
    - protocols.py: ProtocolRegistrationIntent for duck-typed signatures
    - OMN-1007: Union reduction refactoring
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
from omnibase_infra.nodes.node_registration_orchestrator.models.model_registry_intent import (
    RegistryIntent,
)

# Type alias for intent payloads
IntentPayload = ModelConsulIntentPayload | ModelPostgresIntentPayload

# =============================================================================
# Discriminated Union Definitions
#
# SYNC REQUIREMENT: These unions MUST stay in sync with RegistryIntent.
# When adding a new intent type:
#   1. Create model inheriting from ModelRegistryIntent
#   2. Register with @RegistryIntent.register("kind")
#   3. Add model to the union below
#   4. Run validate_union_registry_sync() to verify
# =============================================================================

# Discriminated union of all intent types using Annotated pattern
# This enables type narrowing based on the `kind` field
# SYNC: Must include all types registered in RegistryIntent
ModelRegistrationIntent = Annotated[
    ModelConsulRegistrationIntent | ModelPostgresUpsertIntent,
    Field(discriminator="kind"),
]

# Explicit list of intent model classes in the union
# Used by validate_union_registry_sync() for verification
_UNION_INTENT_TYPES: tuple[type, ...] = (
    ModelConsulRegistrationIntent,
    ModelPostgresUpsertIntent,
)


def validate_union_registry_sync() -> tuple[bool, list[str]]:
    """Validate that ModelRegistrationIntent union matches RegistryIntent.

    This function checks that:
    1. All types in the union are registered in RegistryIntent
    2. All types in RegistryIntent are included in the union

    Use this in tests to catch sync issues early.

    Returns:
        Tuple of (is_valid, list of error messages).
        If is_valid is True, error list is empty.

    Example:
        >>> is_valid, errors = validate_union_registry_sync()
        >>> if not is_valid:
        ...     for error in errors:
        ...         print(f"ERROR: {error}")

    .. versionadded:: 0.7.1
        Created as part of OMN-1007 union sync fix.
    """
    errors: list[str] = []
    registry_types = RegistryIntent.get_all_types()

    # Check 1: All union types are registered
    for union_type in _UNION_INTENT_TYPES:
        # Get the kind from the class default
        kind_default = getattr(union_type, "model_fields", {}).get("kind")
        if kind_default is None:
            errors.append(
                f"Invalid union type '{union_type.__name__}': expected 'kind' field, got None"
            )
            continue

        # Get the default value from the field
        kind_value = kind_default.default
        if kind_value not in registry_types:
            errors.append(
                f"Union type {union_type.__name__} (kind='{kind_value}') "
                f"is not registered in RegistryIntent"
            )
        elif registry_types[kind_value] is not union_type:
            errors.append(
                f"Kind '{kind_value}' is registered to "
                f"{registry_types[kind_value].__name__}, "
                f"but union contains {union_type.__name__}"
            )

    # Check 2: All registered types are in union
    union_type_set = set(_UNION_INTENT_TYPES)
    for kind, registered_type in registry_types.items():
        if registered_type not in union_type_set:
            errors.append(
                f"Registered type {registered_type.__name__} (kind='{kind}') "
                f"is missing from ModelRegistrationIntent union. "
                f"Add it to the union in model_registration_intent.py"
            )

    return len(errors) == 0, errors


def get_union_intent_types() -> tuple[type, ...]:
    """Get the tuple of intent model types in the ModelRegistrationIntent union.

    Returns:
        Tuple of intent model classes that are part of the union.

    .. versionadded:: 0.7.1
    """
    return _UNION_INTENT_TYPES


__all__ = [
    "IntentPayload",
    "ModelConsulIntentPayload",
    "ModelConsulRegistrationIntent",
    "ModelPostgresIntentPayload",
    "ModelPostgresUpsertIntent",
    "ModelRegistrationIntent",
    "get_union_intent_types",
    "validate_union_registry_sync",
]
