# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The owned-database gate admits adapter owners serially and others in parallel (OMN-26430).

Drives the real ``owned_database_gate`` with concurrent callers on the
runtime's dispatch loop.
"""

from __future__ import annotations

import asyncio

import pytest

from omnibase_infra.handlers.projection_database_ownership import (
    owned_database_gate,
    serves_its_own_database,
)

CALLS = 6
HOLD_SECONDS = 0.05


class _Adapter:
    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None


class _Owner:
    def __init__(self) -> None:
        self.db = _Adapter()


class _Injected:
    pass


def _peak_inside(handler: object) -> int:
    state = {"inside": 0, "peak": 0}

    async def _one() -> None:
        async with owned_database_gate(handler):
            state["inside"] += 1
            state["peak"] = max(state["peak"], state["inside"])
            await asyncio.sleep(HOLD_SECONDS)
            state["inside"] -= 1

    async def _all() -> None:
        await asyncio.gather(*(_one() for _ in range(CALLS)))

    asyncio.run(_all())
    return state["peak"]


@pytest.mark.integration
def test_injected_handler_passes_the_gate_in_parallel() -> None:
    assert serves_its_own_database(_Injected()) is False
    assert _peak_inside(_Injected()) > 1


@pytest.mark.integration
def test_adapter_owner_is_admitted_one_at_a_time_on_one_loop() -> None:
    owner = _Owner()
    assert serves_its_own_database(owner) is True
    assert _peak_inside(owner) == 1
