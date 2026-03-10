# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2025 OmniNode Team
"""Regression tests for EnumMessageCategory class-identity split (OMN-4031).

These tests reproduce the original cross-package class-identity failure where
EnumMessageCategory imported from omnibase_infra was a distinct class object
from the one imported from omnibase_core, causing isinstance checks to fail
silently at dispatcher registration boundaries.

All tests must remain passing after any future refactor of the enum re-export
chain. Removing ``coerce_message_category`` from the dispatcher must cause
``test_foreign_enum_coercion_regression`` to fail again (acts as a canary).
"""

from __future__ import annotations

from enum import Enum

import pytest

from omnibase_core.enums import EnumMessageCategory
from omnibase_infra.runtime.service_message_dispatch_engine import (
    coerce_message_category,
)


class FakeEnumMessageCategory(Enum):
    """Simulates a foreign / plugin-side EnumMessageCategory.

    This is the minimal reproduction of the class-identity split: a plugin
    loaded in a different import context defines its own copy of the enum.
    Values are identical to the real enum but the class object is different.

    Values must match the canonical EnumMessageCategory values exactly so that
    coerce_message_category() can normalise via ``.value`` lookup.
    """

    EVENT = "event"
    COMMAND = "command"
    INTENT = "intent"


@pytest.mark.unit
def test_foreign_enum_coercion_regression() -> None:
    """Reproduce the original plugin/runtime class-identity split.

    A foreign enum instance (same values, different class) must be coerced
    to the canonical EnumMessageCategory by coerce_message_category().

    Before OMN-4034: this test would FAIL because the isinstance check in the
    dispatch engine accepted the foreign instance without coercion, leading to
    downstream type errors. After OMN-4034: the coercer normalises the foreign
    value to the canonical class, so all three members convert correctly.
    """
    for fake_member, expected in (
        (FakeEnumMessageCategory.EVENT, EnumMessageCategory.EVENT),
        (FakeEnumMessageCategory.COMMAND, EnumMessageCategory.COMMAND),
        (FakeEnumMessageCategory.INTENT, EnumMessageCategory.INTENT),
    ):
        result = coerce_message_category(fake_member)
        assert type(result) is EnumMessageCategory, (
            f"type(result) is {type(result)!r}, expected EnumMessageCategory. "
            f"Class-identity split still present for {fake_member!r}."
        )
        assert result == expected


@pytest.mark.unit
def test_infra_export_resolves_to_core() -> None:
    """Infra re-export and core definition must be the same class object.

    OMN-4033 introduced enum_message_category.py in omnibase_infra as a thin
    re-export shim. This test asserts that ``omnibase_infra.enums.EnumMessageCategory``
    and ``omnibase_core.enums.EnumMessageCategory`` are the identical class object —
    not merely equal, but ``is`` the same.
    """
    from omnibase_infra.enums import EnumMessageCategory as InfraEnum

    assert InfraEnum is EnumMessageCategory, (
        f"Import identity mismatch: "
        f"infra={InfraEnum.__module__}.{InfraEnum.__qualname__}, "
        f"core={EnumMessageCategory.__module__}.{EnumMessageCategory.__qualname__}"
    )


@pytest.mark.unit
def test_string_coercion() -> None:
    """String values matching valid enum members must coerce to the canonical member.

    Covers the boundary case where event-bus messages arrive as raw strings
    (e.g. from JSON deserialization) and must be normalised before reaching
    the dispatcher.
    """
    for member in EnumMessageCategory:
        result = coerce_message_category(member.value)
        assert result == member, (
            f"String coercion of {member.value!r} returned {result!r}, expected {member!r}"
        )
        assert type(result) is EnumMessageCategory, (
            f"type(result) is {type(result)!r} after string coercion of {member.value!r}"
        )


@pytest.mark.unit
def test_invalid_value_raises_clear_error() -> None:
    """Invalid inputs must raise ValueError with a descriptive message.

    Ensures the coercer fails loudly rather than silently returning None
    or an unexpected fallback when given unrecognised strings or non-string
    types that have no valid ``.value``.
    """
    with pytest.raises(ValueError, match="Invalid message category"):
        coerce_message_category("not_a_real_category_xyzzy")

    with pytest.raises(ValueError, match="Expected one of"):
        coerce_message_category(42)
