# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
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


# ---------------------------------------------------------------------------
# Tests — unregister ghost entry fix (OMN-4087)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_unregister_dispatcher_with_foreign_enum_category() -> None:
    """Unregister with a foreign-enum category must not leave a ghost entry.

    Before the OMN-4087 fix, unregister_dispatcher() looked up the category
    key using the *raw* dispatcher.category value. When the dispatcher was
    registered with a foreign enum (e.g. ForeignCategory.EVENT), the canonical
    key EnumMessageCategory.EVENT was stored at registration time — but the
    unregister path used ForeignCategory.EVENT as the key, which was never a
    key, so the entry was silently left behind (ghost entry).

    After the fix, unregister coerces the category the same way registration
    does, so the lookup always uses the canonical key.
    """
    from datetime import UTC, datetime

    from omnibase_core.enums import EnumNodeKind
    from omnibase_infra.enums import EnumMessageCategory as InfraEnumMessageCategory
    from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
    from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
    from omnibase_infra.runtime.registry_dispatcher import RegistryDispatcher

    class ForeignEnumDispatcher:
        """Dispatcher whose .category is a foreign enum with a matching value."""

        def __init__(self) -> None:
            self._dispatcher_id = "foreign-enum-dispatcher"
            # ForeignCategory is defined at module level in this test file
            self._category = ForeignCategory.EVENT
            self._node_kind = EnumNodeKind.REDUCER
            self._message_types: set[str] = set()

        @property
        def dispatcher_id(self) -> str:
            return self._dispatcher_id

        @property
        def category(self) -> ForeignCategory:  # type: ignore[override]
            return self._category

        @property
        def message_types(self) -> set[str]:
            return self._message_types

        @property
        def node_kind(self) -> EnumNodeKind:
            return self._node_kind

        async def handle(self, envelope: object) -> ModelDispatchResult:
            return ModelDispatchResult(
                status=EnumDispatchStatus.SUCCESS,
                topic="test.events",
                dispatcher_id=self._dispatcher_id,
                started_at=datetime(2025, 1, 1, tzinfo=UTC),
            )

    registry = RegistryDispatcher()
    dispatcher = ForeignEnumDispatcher()

    # Register — canonical key EnumMessageCategory.EVENT is stored
    registry.register_dispatcher(dispatcher)
    assert registry.dispatcher_count == 1

    # Unregister — must coerce category to canonical key before lookup
    removed = registry.unregister_dispatcher("foreign-enum-dispatcher")
    assert removed is True, (
        "unregister_dispatcher should return True for a registered dispatcher"
    )
    assert registry.dispatcher_count == 0, (
        "dispatcher count should be 0 after unregister"
    )

    # Freeze and verify no ghost entry remains in the category index
    registry.freeze()
    result = registry.get_dispatchers(InfraEnumMessageCategory.EVENT)
    assert result == [], (
        f"Ghost entry detected: get_dispatchers(EVENT) returned {result!r} "
        "after unregister — unregister did not coerce the foreign enum category key"
    )
