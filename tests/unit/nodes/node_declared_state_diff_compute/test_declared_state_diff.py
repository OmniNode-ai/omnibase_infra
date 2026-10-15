# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The diff plans exactly the operations that move observed state to declared."""

import pytest

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind,
)
from omnibase_core.models.config_overlay.model_declared_state_observation import (
    ModelDeclaredStateObservation,
)
from omnibase_infra.handlers.handler_declared_state_diff import (
    HandlerDeclaredStateDiff,
)
from omnibase_infra.models.declared_state.enum_declared_state_operation_kind import (
    EnumDeclaredStateOperationKind as Kind,
)
from omnibase_infra.models.declared_state.model_declared_state_diff_request import (
    ModelDeclaredStateDiffRequest,
)
from omnibase_infra.models.declared_state.model_declared_state_item import (
    ModelDeclaredStateItem,
)

pytestmark = pytest.mark.unit
BROKER = EnumDeclaredStateSurfaceKind.BROKER_ACL


def declared(**items: str) -> tuple[ModelDeclaredStateItem, ...]:
    return tuple(
        ModelDeclaredStateItem(item_key=k, digest=d) for k, d in sorted(items.items())
    )


def seen(key: str, digest: str | None) -> ModelDeclaredStateObservation:
    return ModelDeclaredStateObservation(
        surface_kind=BROKER,
        surface_id="scratch",
        item_key=key,
        observed_digest=digest,
        read_ok=True,
        detail="read",
    )


def unreadable(key: str) -> ModelDeclaredStateObservation:
    return ModelDeclaredStateObservation(
        surface_kind=BROKER,
        surface_id="scratch",
        item_key=key,
        observed_digest=None,
        read_ok=False,
        detail="broker admin API timed out",
    )


def diff(items, observations, generation="g1"):
    return HandlerDeclaredStateDiff().handle(
        ModelDeclaredStateDiffRequest(
            surface_kind=BROKER,
            surface_key="scratch",
            generation=generation,
            declared=items,
            observations=tuple(observations),
        )
    )


def plan(result):
    return [(o.kind, o.item_key, o.digest) for o in result.operations]


def test_missing_drifted_and_undeclared_items_each_get_one_operation() -> None:
    result = diff(
        declared(a="1", b="2", c="3"),
        [seen("a", None), seen("b", "old"), seen("c", "3"), seen("z", "9")],
    )
    assert result.unobservable_reason is None
    assert plan(result) == [
        (Kind.CREATE, "a", "1"),
        (Kind.UPDATE, "b", "2"),
        (Kind.DELETE, "z", None),
    ]
    assert result.generation == "g1"


def test_matching_state_plans_zero_operations() -> None:
    result = diff(declared(a="1"), [seen("a", "1")])
    assert result.operations == ()
    assert result.unobservable_reason is None


def test_creates_and_updates_precede_deletes() -> None:
    result = diff(declared(m="1"), [seen("a", "x"), seen("m", None)])
    assert [o.kind for o in result.operations] == [Kind.CREATE, Kind.DELETE]


def test_unreadable_item_makes_the_surface_unobservable() -> None:
    result = diff(declared(a="1", b="2"), [seen("a", None), unreadable("b")])
    assert result.operations == ()
    assert result.unobservable_reason is not None
    assert "b" in result.unobservable_reason


def test_declared_item_the_round_never_reached_is_unobservable() -> None:
    result = diff(declared(a="1", b="2"), [seen("a", "1")])
    assert result.operations == ()
    assert result.unobservable_reason is not None
    assert "b" in result.unobservable_reason


def test_zero_observations_for_declared_items_is_never_clean() -> None:
    result = diff(declared(a="1"), [])
    assert result.operations == ()
    assert result.unobservable_reason is not None


def test_nothing_declared_and_nothing_observed_is_in_sync() -> None:
    result = diff((), [])
    assert result.operations == ()
    assert result.unobservable_reason is None


def test_observation_for_another_surface_is_refused() -> None:
    other = ModelDeclaredStateObservation(
        surface_kind=BROKER,
        surface_id="elsewhere",
        item_key="a",
        observed_digest="1",
        read_ok=True,
        detail="read",
    )
    with pytest.raises(ValueError, match="surface"):
        diff(declared(a="1"), [other])


def test_duplicate_observation_rows_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        diff(declared(a="1"), [seen("a", "1"), seen("a", "2")])
