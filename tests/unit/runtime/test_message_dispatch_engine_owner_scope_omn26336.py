# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Proving a contract's dispatcher scope must not rescan every registered dispatcher.

OMN-26336: ``subscribe_wired_contract_topics`` proves each contract's scope against the live
engine on the bootstrap coroutine. ``validate_contract_dispatcher_scope`` rebuilt the owner's
complete scope by iterating every registered dispatcher, once per contract, so the boot held
the event loop in proportion to contracts x dispatchers (1.391 s with the full dev contract
set pinned to one core; composed-runtime-boot job 137964427003 read 1.505 s). The engine now
keeps its owner scopes at registration.
"""

from __future__ import annotations

import gc
import time
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from typing import Any

import pytest

from omnibase_core.enums import EnumMessageCategory
from omnibase_core.models.errors.model_onex_error import ModelOnexError
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_CONTRACTS = 1500
_DISPATCHERS_PER_CONTRACT = 4


class _CountingDispatchers(dict[str, Any]):
    """Counts every walk of the dispatcher table: items(), iteration, keys and values."""

    walks = 0

    def __iter__(self) -> Iterator[str]:
        type(self).walks += 1
        return super().__iter__()

    def items(self) -> ItemsView[str, Any]:
        type(self).walks += 1
        return super().items()

    def keys(self) -> KeysView[str]:
        type(self).walks += 1
        return super().keys()

    def values(self) -> ValuesView[Any]:
        type(self).walks += 1
        return super().values()


async def _noop(envelope: object) -> None:
    return None


def _engine_with_owned_dispatchers() -> tuple[
    MessageDispatchEngine, dict[str, tuple[str, ...]]
]:
    engine = MessageDispatchEngine()
    scopes: dict[str, tuple[str, ...]] = {}
    for contract_index in range(_CONTRACTS):
        contract_name = f"node_contract_{contract_index}"
        dispatcher_ids = tuple(
            f"dispatcher.{contract_name}.{slot}"
            for slot in range(_DISPATCHERS_PER_CONTRACT)
        )
        for dispatcher_id in dispatcher_ids:
            engine.register_dispatcher(
                dispatcher_id=dispatcher_id,
                dispatcher=_noop,
                category=EnumMessageCategory.EVENT,
                message_types={f"Model{dispatcher_id}"},
                owner_contract_name=contract_name,
            )
        scopes[contract_name] = dispatcher_ids
    return engine, scopes


@pytest.mark.unit
def test_scope_proof_never_walks_the_dispatcher_table() -> None:
    engine, scopes = _engine_with_owned_dispatchers()
    counting = _CountingDispatchers(engine._dispatchers)
    _CountingDispatchers.walks = 0
    engine._dispatchers = counting

    for contract_name, dispatcher_ids in scopes.items():
        assert engine.validate_contract_dispatcher_scope(
            contract_name, dispatcher_ids
        ) == frozenset(dispatcher_ids)

    assert _CountingDispatchers.walks == 0


@pytest.mark.unit
def test_scope_proof_at_the_dev_contract_count_holds_the_loop_under_the_bound() -> None:
    engine, scopes = _engine_with_owned_dispatchers()

    started = time.perf_counter()
    for contract_name, dispatcher_ids in scopes.items():
        engine.validate_contract_dispatcher_scope(contract_name, dispatcher_ids)
    held = time.perf_counter() - started

    assert held < 0.25, (
        f"proving {_CONTRACTS} contract scopes held the caller {held:.3f} s"
    )


@pytest.mark.unit
def test_scope_proof_still_rejects_a_foreign_dispatcher_a_gap_and_an_unknown_id() -> (
    None
):
    engine, scopes = _engine_with_owned_dispatchers()
    first, second = list(scopes)[:2]

    with pytest.raises(ModelOnexError, match="not owned by contract"):
        engine.validate_contract_dispatcher_scope(
            first, (*scopes[first][:-1], scopes[second][0])
        )
    with pytest.raises(ModelOnexError, match="complete live owner set"):
        engine.validate_contract_dispatcher_scope(first, scopes[first][:-1])
    with pytest.raises(ModelOnexError, match="not registered"):
        engine.validate_contract_dispatcher_scope(first, (*scopes[first], "missing"))
    assert (
        engine.validate_contract_dispatcher_scope("node_without_dispatchers", ())
        == frozenset()
    )


@pytest.mark.unit
def test_freeze_moves_the_gc_heap_to_the_permanent_generation_only_when_asked() -> None:
    gc.unfreeze()
    try:
        MessageDispatchEngine().freeze()
        assert gc.get_freeze_count() == 0

        MessageDispatchEngine().freeze(freeze_gc_heap=True)
        assert gc.get_freeze_count() > 0
    finally:
        gc.unfreeze()
