# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rounds observe first, so apply converges and a later hand edit is repaired."""

import pytest

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind as SurfaceKind,
)
from omnibase_core.models.config_overlay.model_declared_state_observation import (
    ModelDeclaredStateObservation,
)
from omnibase_infra.handlers.handler_declared_state_apply import (
    HandlerDeclaredStateApply,
)
from omnibase_infra.handlers.handler_declared_state_diff import (
    HandlerDeclaredStateDiff,
)
from omnibase_infra.handlers.handler_declared_state_lease_local import (
    HandlerDeclaredStateLeaseLocal,
)
from omnibase_infra.handlers.handler_declared_state_reconcile import (
    HandlerDeclaredStateReconcile,
)
from omnibase_infra.models.declared_state.enum_declared_state_apply_status import (
    EnumDeclaredStateApplyStatus as Status,
)
from omnibase_infra.models.declared_state.enum_declared_state_operation_kind import (
    EnumDeclaredStateOperationKind as Kind,
)
from omnibase_infra.models.declared_state.enum_declared_state_reconcile_mode import (
    EnumDeclaredStateReconcileMode as Mode,
)
from omnibase_infra.models.declared_state.model_declared_state_apply_policy import (
    ModelDeclaredStateApplyPolicy,
)
from omnibase_infra.models.declared_state.model_declared_state_item import (
    ModelDeclaredStateItem,
)
from omnibase_infra.models.declared_state.model_declared_state_operation import (
    ModelDeclaredStateOperation,
)
from omnibase_infra.models.declared_state.model_declared_state_reconcile_request import (
    ModelDeclaredStateReconcileRequest,
)
from omnibase_infra.models.declared_state.model_declared_state_snapshot import (
    ModelDeclaredStateSnapshot,
)

pytestmark = pytest.mark.unit
POLICY = ModelDeclaredStateApplyPolicy(
    allowed_kinds=frozenset({SurfaceKind.BROKER_ACL, SurfaceKind.DEPLOYED_BUNDLE})
)
DECLARED = (
    ModelDeclaredStateItem(item_key="a", digest="1"),
    ModelDeclaredStateItem(item_key="b", digest="2"),
)


class FakeSurface:
    """One in-memory surface that is also its own observer."""

    def __init__(self, kind: SurfaceKind, items: dict[str, str]) -> None:
        self.kind = kind
        self.items = dict(items)
        self.generation = 0
        self.before_apply = None

    async def observe(self, surface_kind, surface_key, round_key, declared):
        rows = {
            item.item_key: ModelDeclaredStateObservation(
                surface_kind=surface_kind,
                surface_id=surface_key,
                item_key=item.item_key,
                observed_digest=self.items.get(item.item_key),
                read_ok=True,
                detail="read",
            )
            for item in declared
        }
        for key, digest in self.items.items():
            rows.setdefault(
                key,
                ModelDeclaredStateObservation(
                    surface_kind=surface_kind,
                    surface_id=surface_key,
                    item_key=key,
                    observed_digest=digest,
                    read_ok=True,
                    detail="read",
                ),
            )
        snapshot = ModelDeclaredStateSnapshot(
            generation=f"g{self.generation}", observations=tuple(rows.values())
        )
        if self.before_apply is not None:
            self.before_apply(self)
        return snapshot

    async def read_generation(self, surface_kind, surface_key) -> str:
        return f"g{self.generation}"

    async def apply_operation(
        self, operation: ModelDeclaredStateOperation, holder
    ) -> str:
        if operation.kind is Kind.DELETE:
            del self.items[operation.item_key]
        else:
            assert operation.digest is not None
            self.items[operation.item_key] = operation.digest
        self.generation += 1
        return f"g{self.generation}"


def orchestrator(surface: FakeSurface) -> HandlerDeclaredStateReconcile:
    return HandlerDeclaredStateReconcile(
        observer=surface,
        diff=HandlerDeclaredStateDiff(),
        apply=HandlerDeclaredStateApply(
            surfaces={surface.kind: surface}, lease=HandlerDeclaredStateLeaseLocal()
        ),
    )


def round_request(surface: FakeSurface, round_key: str, mode: Mode = Mode.APPLY):
    return ModelDeclaredStateReconcileRequest(
        surface_kind=surface.kind,
        surface_key="scratch",
        declared=DECLARED,
        mode=mode,
        policy=POLICY,
        round_key=round_key,
    )


async def test_four_rounds_wrong_fixed_hand_edited_fixed_again() -> None:
    surface = FakeSurface(SurfaceKind.BROKER_ACL, {"a": "wrong", "extra": "9"})
    node = orchestrator(surface)

    first = await node.handle(round_request(surface, "r1"))
    assert first.operation_count == 3  # create b, update a, delete extra
    assert first.apply is not None and first.apply.status is Status.APPLIED
    assert surface.items == {"a": "1", "b": "2"}

    second = await node.handle(round_request(surface, "r2"))
    assert second.operation_count == 0
    assert second.apply is None

    surface.items["b"] = "hand-edited"
    surface.generation += 1

    third = await node.handle(round_request(surface, "r3"))
    assert third.operation_count == 1
    assert third.apply is not None and third.apply.status is Status.APPLIED
    assert surface.items == {"a": "1", "b": "2"}

    fourth = await node.handle(round_request(surface, "r4"))
    assert fourth.operation_count == 0


async def test_item_whose_surface_moved_after_observation_is_refused() -> None:
    surface = FakeSurface(SurfaceKind.BROKER_ACL, {})
    surface.before_apply = lambda s: setattr(s, "generation", s.generation + 1)
    result = await orchestrator(surface).handle(round_request(surface, "r1"))
    assert result.operation_count == 2
    assert result.apply is not None
    assert result.apply.status is Status.REFUSED_STALE
    assert surface.items == {}


async def test_observe_mode_plans_and_changes_nothing() -> None:
    surface = FakeSurface(SurfaceKind.BROKER_ACL, {})
    result = await orchestrator(surface).handle(
        round_request(surface, "r1", Mode.OBSERVE)
    )
    assert result.operation_count == 2
    assert result.apply is None
    assert surface.items == {}


async def test_kind_not_allowed_by_policy_is_refused_by_name() -> None:
    surface = FakeSurface(SurfaceKind.HOST_SETTING, {})
    result = await orchestrator(surface).handle(round_request(surface, "r1"))
    assert result.operation_count == 2
    assert result.apply is not None
    assert result.apply.status is Status.REFUSED_POLICY
    assert result.apply.refusal is not None and "host_setting" in result.apply.refusal
    assert surface.items == {}


async def test_unobservable_surface_reports_its_reason_and_changes_nothing() -> None:
    surface = FakeSurface(SurfaceKind.BROKER_ACL, {})

    async def silent(surface_kind, surface_key, round_key, declared):
        return ModelDeclaredStateSnapshot(generation="g0", observations=())

    surface.observe = silent
    result = await orchestrator(surface).handle(round_request(surface, "r1"))
    assert result.operation_count == 0
    assert result.unobservable_reason is not None
    assert (
        result.apply is not None and result.apply.status is Status.REFUSED_UNOBSERVABLE
    )
    assert surface.items == {}
