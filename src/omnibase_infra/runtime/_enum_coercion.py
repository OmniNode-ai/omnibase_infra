# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Shared enum coercion utilities for the runtime layer.

This module exists at a layer below both ``registry_dispatcher`` and
``service_message_dispatch_engine`` so that both can import ``coerce_message_category``
without introducing a circular dependency.

Circular import chain that motivated this extraction (OMN-4087):

    registry_dispatcher
      → service_message_dispatch_engine
        → dispatch_context_enforcer
          → registry_dispatcher   ← cycle

By placing ``coerce_message_category`` here — with zero imports from either
``registry_dispatcher`` or ``service_message_dispatch_engine`` — both modules can safely
import from this shared location.

.. versionadded:: 0.8.1
    Extracted from ``service_message_dispatch_engine`` (OMN-4087) to break the circular
    import cycle and eliminate inlined coercion copies in ``registry_dispatcher``.
"""

from __future__ import annotations

from omnibase_core.enums import EnumMessageCategory


def coerce_message_category(value: object) -> EnumMessageCategory:
    """Normalize any category input to the canonical ``EnumMessageCategory``.

    Accepts:
    - A canonical ``EnumMessageCategory`` instance (pass-through).
    - A string matching a valid enum value (e.g. ``"EVENT"``).
    - A foreign enum instance whose ``.value`` matches a valid enum value.

    Raises:
        ValueError: When ``value`` cannot be resolved to a valid enum member.
            The error message lists all valid values.

    .. versionadded:: 0.8.0
        Added for boundary coercion at dispatcher registration entry points (OMN-4034).

    .. versionchanged:: 0.8.1
        Moved to ``_enum_coercion`` to break circular import cycle (OMN-4087).
    """
    if isinstance(value, EnumMessageCategory):
        return value
    raw: object = value.value if hasattr(value, "value") else value
    try:
        return EnumMessageCategory(raw)
    except ValueError:
        raise ValueError(
            f"Invalid message category: {value!r}. "
            f"Expected one of: {[e.value for e in EnumMessageCategory]}"
        )
