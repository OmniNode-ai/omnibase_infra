# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Unit tests for the shared ``_enum_coercion`` module (OMN-4087).

These tests verify that ``coerce_message_category`` behaves correctly when imported
from the new ``_enum_coercion`` module, and that the re-export from
``service_message_dispatch_engine`` remains backward-compatible.

The module was extracted to break the circular import chain:
    registry_dispatcher → service_message_dispatch_engine → dispatch_context_enforcer
    → registry_dispatcher
"""

from __future__ import annotations

from enum import Enum

import pytest

from omnibase_core.enums import EnumMessageCategory
from omnibase_infra.runtime._enum_coercion import coerce_message_category

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ForeignCategory(Enum):
    """Foreign enum that mirrors EnumMessageCategory values but is a different class."""

    EVENT = "event"
    COMMAND = "command"
    INTENT = "intent"


class UnrelatedCategory(Enum):
    """Foreign enum with values that do not match EnumMessageCategory."""

    UNKNOWN = "unknown_garbage_xyzzy"


# ---------------------------------------------------------------------------
# Tests — canonical pass-through
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_canonical_passthrough() -> None:
    """A canonical EnumMessageCategory instance is returned as-is."""
    for member in EnumMessageCategory:
        result = coerce_message_category(member)
        assert result is member, (
            f"Expected pass-through for canonical member {member!r}, got {result!r}"
        )


# ---------------------------------------------------------------------------
# Tests — string coercion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_string_coercion() -> None:
    """String values matching valid enum members coerce to the canonical member."""
    for member in EnumMessageCategory:
        result = coerce_message_category(member.value)
        assert result == member, (
            f"String coercion of {member.value!r} returned {result!r}, expected {member!r}"
        )
        assert type(result) is EnumMessageCategory, (
            f"type(result) is {type(result)!r} after string coercion of {member.value!r}"
        )


# ---------------------------------------------------------------------------
# Tests — foreign enum coercion
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_foreign_enum_coercion() -> None:
    """Foreign enum instances with matching values coerce to the canonical class."""
    assert coerce_message_category(ForeignCategory.EVENT) is EnumMessageCategory.EVENT
    assert (
        coerce_message_category(ForeignCategory.COMMAND) is EnumMessageCategory.COMMAND
    )
    assert coerce_message_category(ForeignCategory.INTENT) is EnumMessageCategory.INTENT


# ---------------------------------------------------------------------------
# Tests — invalid inputs raise ValueError
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_invalid_string_raises_value_error() -> None:
    """An unrecognised string raises ValueError with a descriptive message."""
    with pytest.raises(ValueError, match="Invalid message category"):
        coerce_message_category("not_a_real_category_xyzzy")


@pytest.mark.unit
def test_invalid_int_raises_value_error() -> None:
    """An integer with no matching enum value raises ValueError."""
    with pytest.raises(ValueError, match="Expected one of"):
        coerce_message_category(42)


@pytest.mark.unit
def test_unrelated_foreign_enum_raises_value_error() -> None:
    """A foreign enum with non-matching value raises ValueError."""
    with pytest.raises(ValueError, match="Invalid message category"):
        coerce_message_category(UnrelatedCategory.UNKNOWN)


# ---------------------------------------------------------------------------
# Tests — backward-compatible re-export from service_message_dispatch_engine
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reexport_from_dispatch_engine_is_same_function() -> None:
    """coerce_message_category re-exported from service_message_dispatch_engine is the same object."""
    from omnibase_infra.runtime.service_message_dispatch_engine import (
        coerce_message_category as engine_coerce,
    )

    assert engine_coerce is coerce_message_category, (
        "service_message_dispatch_engine.coerce_message_category must be the same function "
        "object as _enum_coercion.coerce_message_category (OMN-4087 backward compat)"
    )


# ---------------------------------------------------------------------------
# Tests — no circular import: registry_dispatcher imports cleanly
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_registry_dispatcher_imports_without_circular_error() -> None:
    """Importing registry_dispatcher must not raise ImportError.

    This is a canary test: if the circular import chain reappears, this test
    will fail with an ImportError before any assertion is reached.
    """
    import importlib

    module = importlib.import_module("omnibase_infra.runtime.registry_dispatcher")
    assert hasattr(module, "RegistryDispatcher"), (
        "RegistryDispatcher not found in registry_dispatcher module"
    )
