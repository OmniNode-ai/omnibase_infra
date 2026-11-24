# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel freezes the engine with the GC heap before any consumer attaches.

OMN-26336: with the full dev contract set wired (1.56M tracked objects) one full collection
held the event loop 0.955 s pinned to one core. The wired graph is a full pass for every
later collection, so a boot step that allocates after wiring can be the call a stack sampler
names for a hold over one second. ``MessageDispatchEngine.freeze(freeze_gc_heap=True)`` moves
the graph to the permanent generation. This reads the bootstrap coroutine's own source: its
one engine freeze passes the flag and comes before the first consumer attach.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from omnibase_infra.runtime import service_kernel


def _bootstrap_calls() -> list[ast.Call]:
    source = Path(service_kernel.__file__).read_text(encoding="utf-8")
    bootstrap = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "bootstrap"
    )
    return [node for node in ast.walk(bootstrap) if isinstance(node, ast.Call)]


def _called_as(call: ast.Call, dotted: str) -> bool:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return f"{func.value.id}.{func.attr}" == dotted
    return isinstance(func, ast.Name) and func.id == dotted


@pytest.mark.unit
def test_bootstrap_freezes_the_engine_with_the_gc_heap_before_consumers_attach() -> (
    None
):
    calls = _bootstrap_calls()
    freezes = [call for call in calls if _called_as(call, "dispatch_engine.freeze")]
    attaches = [
        call for call in calls if _called_as(call, "subscribe_wired_contract_topics")
    ]

    assert len(freezes) == 1
    assert [
        (keyword.arg, ast.literal_eval(keyword.value))
        for keyword in freezes[0].keywords
    ] == [("freeze_gc_heap", True)]
    assert attaches
    assert freezes[0].lineno < min(call.lineno for call in attaches)
