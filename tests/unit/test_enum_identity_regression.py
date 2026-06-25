# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
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
from omnibase_infra.runtime.message_dispatch_engine import (
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
def test_dispatch_status_infra_export_resolves_to_core() -> None:
    """Infra EnumDispatchStatus must be the identical core class object (OMN-12545 S-1a).

    S-1a consolidates EnumDispatchStatus to a single core copy and converts the
    infra module to a thin re-export shim (mirroring enum_message_category.py).
    Both ``omnibase_infra.enums.EnumDispatchStatus`` and
    ``omnibase_core.enums.enum_dispatch_status.EnumDispatchStatus`` must be the
    same class object — not merely equal, but ``is`` identical — so isinstance
    checks at dispatch boundaries cannot silently split.
    """
    from omnibase_core.enums.enum_dispatch_status import (
        EnumDispatchStatus as CoreEnum,
    )
    from omnibase_infra.enums import EnumDispatchStatus as InfraTopLevel
    from omnibase_infra.enums.enum_dispatch_status import (
        EnumDispatchStatus as InfraModule,
    )

    assert InfraTopLevel is CoreEnum, (
        f"Import identity mismatch (top-level): "
        f"infra={InfraTopLevel.__module__}.{InfraTopLevel.__qualname__}, "
        f"core={CoreEnum.__module__}.{CoreEnum.__qualname__}"
    )
    assert InfraModule is CoreEnum, (
        f"Import identity mismatch (module): "
        f"infra={InfraModule.__module__}.{InfraModule.__qualname__}, "
        f"core={CoreEnum.__module__}.{CoreEnum.__qualname__}"
    )


@pytest.mark.unit
def test_dispatch_status_carries_core_superset_members() -> None:
    """The single-source EnumDispatchStatus must carry the core superset (OMN-12545 S-1a).

    Core is the canonical superset: it preserves the live infra members
    NO_DISPATCHER and INTERNAL_ERROR while also exposing NO_HANDLER. After the
    infra re-export, the infra-imported enum must expose every core member.
    """
    from omnibase_core.enums.enum_dispatch_status import (
        EnumDispatchStatus as CoreEnum,
    )
    from omnibase_infra.enums import EnumDispatchStatus as InfraEnum

    core_values = {m.value for m in CoreEnum}
    infra_values = {m.value for m in InfraEnum}
    assert infra_values == core_values
    # Live infra members must be preserved by the canonical core copy.
    assert "no_dispatcher" in infra_values
    assert "internal_error" in infra_values


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
