# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Apply changes only allowed kinds, only from a fresh plan, one holder at a time."""

import asyncio

import pytest

from omnibase_core.enums.enum_declared_state_surface_kind import (
    EnumDeclaredStateSurfaceKind as SurfaceKind,
)
from omnibase_infra.handlers.handler_declared_state_apply import (
    HandlerDeclaredStateApply,
)
from omnibase_infra.handlers.handler_declared_state_lease_local import (
    HandlerDeclaredStateLeaseLocal,
)
from omnibase_infra.models.declared_state.enum_declared_state_apply_status import (
    EnumDeclaredStateApplyStatus as Status,
)
from omnibase_infra.models.declared_state.enum_declared_state_operation_kind import (
    EnumDeclaredStateOperationKind as Kind,
)
from omnibase_infra.models.declared_state.model_declared_state_apply_policy import (
    ModelDeclaredStateApplyPolicy,
)
from omnibase_infra.models.declared_state.model_declared_state_apply_request import (
    ModelDeclaredStateApplyRequest,
)
from omnibase_infra.models.declared_state.model_declared_state_diff import (
    ModelDeclaredStateDiff,
)
from omnibase_infra.models.declared_state.model_declared_state_operation import (
    ModelDeclaredStateOperation,
)

pytestmark = pytest.mark.unit
POLICY = ModelDeclaredStateApplyPolicy(
    allowed_kinds=frozenset(
        {
            SurfaceKind.BROKER_ACL,
            SurfaceKind.COMPOSE_SERVICE,
            SurfaceKind.DEPLOYED_BUNDLE,
        }
    )
)


class FakeSurface:
    """An in-memory surface whose generation moves on every change."""

    def __init__(self, items: dict[str, str] | None = None) -> None:
        self.items = dict(items or {})
        self.generation = 0
        self.reads = 0
        self.log: list[tuple[str, str]] = []
        self.running = 0
        self.max_running = 0
        self.after_apply = None

    async def read_generation(self, surface_kind, surface_key) -> str:
        self.reads += 1
        return f"g{self.generation}"

    async def apply_operation(self, operation, holder) -> str:
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        await asyncio.sleep(0)
        self.log.append((holder, operation.item_key))
        if operation.kind is Kind.DELETE:
            del self.items[operation.item_key]
        else:
            self.items[operation.item_key] = operation.digest
        self.generation += 1
        produced = f"g{self.generation}"
        self.running -= 1
        if self.after_apply is not None:
            self.after_apply(self)
        return produced

    def hand_edit(self, key: str, digest: str) -> None:
        self.items[key] = digest
        self.generation += 1


def op(kind, key, digest, surface_kind=SurfaceKind.BROKER_ACL):
    return ModelDeclaredStateOperation(
        kind=kind,
        surface_kind=surface_kind,
        surface_key="scratch",
        item_key=key,
        digest=digest,
    )


def request(
    operations,
    generation="g0",
    *,
    surface_kind=SurfaceKind.BROKER_ACL,
    round_key="r1",
    reason=None,
):
    return ModelDeclaredStateApplyRequest(
        diff=ModelDeclaredStateDiff(
            surface_kind=surface_kind,
            surface_key="scratch",
            generation=generation,
            operations=tuple(operations),
            unobservable_reason=reason,
        ),
        policy=POLICY,
        round_key=round_key,
    )


def handler(surface: FakeSurface, kind=SurfaceKind.BROKER_ACL):
    return HandlerDeclaredStateApply(
        surfaces={kind: surface}, lease=HandlerDeclaredStateLeaseLocal()
    )


async def test_fresh_plan_is_applied_and_every_operation_reported() -> None:
    surface = FakeSurface({"old": "x"})
    result = await handler(surface).handle(
        request([op(Kind.CREATE, "a", "1"), op(Kind.DELETE, "old", None)])
    )
    assert result.status is Status.APPLIED
    assert result.applied_count == 2
    assert surface.items == {"a": "1"}


async def test_kind_the_policy_does_not_allow_is_refused_by_name_and_untouched() -> (
    None
):
    surface = FakeSurface()
    host = SurfaceKind.HOST_SETTING
    result = await handler(surface, host).handle(
        request([op(Kind.CREATE, "sysctl", "1", host)], surface_kind=host)
    )
    assert result.status is Status.REFUSED_POLICY
    assert result.applied_count == 0
    assert result.refusal is not None and "host_setting" in result.refusal
    assert surface.items == {}
    assert surface.reads == 0


async def test_unobservable_surface_is_never_applied_to() -> None:
    surface = FakeSurface()
    result = await handler(surface).handle(
        request([], reason="adapter returned nothing")
    )
    assert result.status is Status.REFUSED_UNOBSERVABLE
    assert result.refusal is not None and "adapter returned nothing" in result.refusal
    assert surface.reads == 0


async def test_item_whose_surface_moved_since_observation_is_refused() -> None:
    surface = FakeSurface()
    surface.hand_edit("x", "1")
    result = await handler(surface).handle(request([op(Kind.CREATE, "a", "1")]))
    assert result.status is Status.REFUSED_STALE
    assert result.applied_count == 0
    assert "a" not in surface.items


async def test_surface_moving_between_operations_stops_the_apply() -> None:
    surface = FakeSurface()
    surface.after_apply = (
        lambda s: s.hand_edit("intruder", "1") if len(s.log) == 1 else None
    )
    result = await handler(surface).handle(
        request([op(Kind.CREATE, "a", "1"), op(Kind.CREATE, "b", "2")])
    )
    assert result.status is Status.REFUSED_STALE
    assert result.applied_count == 1
    assert "b" not in surface.items


async def test_lease_is_released_after_a_refusal() -> None:
    surface = FakeSurface()
    surface.hand_edit("x", "1")
    h = handler(surface)
    assert (
        await h.handle(request([op(Kind.CREATE, "a", "1")]))
    ).status is Status.REFUSED_STALE
    again = await asyncio.wait_for(
        h.handle(
            request([op(Kind.CREATE, "a", "1")], generation=f"g{surface.generation}")
        ),
        timeout=1,
    )
    assert again.status is Status.APPLIED


async def test_lease_is_released_when_the_adapter_raises() -> None:
    surface = FakeSurface()

    async def boom(operation, holder) -> str:
        raise RuntimeError("broker admin API down")

    surface.apply_operation = boom
    lease = HandlerDeclaredStateLeaseLocal()
    h = HandlerDeclaredStateApply(
        surfaces={SurfaceKind.BROKER_ACL: surface}, lease=lease
    )
    with pytest.raises(RuntimeError, match="admin API"):
        await h.handle(request([op(Kind.CREATE, "a", "1")]))
    surface2 = FakeSurface()
    h2 = HandlerDeclaredStateApply(
        surfaces={SurfaceKind.BROKER_ACL: surface2}, lease=lease
    )
    result = await asyncio.wait_for(
        h2.handle(request([op(Kind.CREATE, "a", "1")])), timeout=1
    )
    assert result.status is Status.APPLIED


async def test_two_concurrent_rounds_on_one_surface_serialize_on_the_lease() -> None:
    surface = FakeSurface()
    h = handler(surface)
    first = request(
        [op(Kind.CREATE, "a", "1"), op(Kind.CREATE, "b", "2")], round_key="r1"
    )
    second = request(
        [op(Kind.CREATE, "a", "1"), op(Kind.CREATE, "b", "2")], round_key="r2"
    )
    results = await asyncio.gather(h.handle(first), h.handle(second))
    assert surface.max_running == 1
    holders = [holder for holder, _ in surface.log]
    assert holders == sorted(holders, key=holders.index)
    assert holders in (["r1", "r1"], ["r2", "r2"])
    statuses = sorted(r.status for r in results)
    assert statuses == sorted([Status.APPLIED, Status.REFUSED_STALE])


async def test_rounds_on_different_surfaces_do_not_wait_on_each_other() -> None:
    lease = HandlerDeclaredStateLeaseLocal()
    await lease.acquire(SurfaceKind.BROKER_ACL, "scratch", "r1")
    await asyncio.wait_for(
        lease.acquire(SurfaceKind.BROKER_ACL, "other", "r2"), timeout=1
    )
    await asyncio.wait_for(
        lease.acquire(SurfaceKind.COMPOSE_SERVICE, "scratch", "r3"), timeout=1
    )
