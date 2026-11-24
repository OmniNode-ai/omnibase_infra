# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The kernel's boot never awaits the savings signal joins (OMN-26418).

On the 2026-11-23T10:55Z dev boot the kernel's savings step awaited three
``event_bus.subscribe`` calls in a row: 25.1 s from "Baselines batch compute
loop started" to "Savings correlation started", with nothing later in the boot
reading those consumers. The step must hand the joins to the savings correlation
task, which the kernel already cancels at shutdown, and must not await a
subscribe itself.

Read from the source tree, so it needs no broker and no clock.

Ticket: OMN-26418
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_KERNEL = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "service_kernel.py"
)
_LOOP = "_savings_correlation_loop"
_JOIN = "subscribe_topics_concurrently"


def _kernel_tree() -> ast.Module:
    return ast.parse(_KERNEL.read_text(encoding="utf-8"))


def _called_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                names.append(func.id)
            elif isinstance(func, ast.Attribute):
                names.append(func.attr)
    return names


def _savings_loop() -> ast.AsyncFunctionDef:
    loops = [
        n
        for n in ast.walk(_kernel_tree())
        if isinstance(n, ast.AsyncFunctionDef) and n.name == _LOOP
    ]
    assert len(loops) == 1, f"expected one {_LOOP} in the kernel, found {len(loops)}"
    return loops[0]


def test_the_savings_task_owns_the_joins_before_its_periodic_loop() -> None:
    body = _savings_loop().body
    first_while = next(i for i, s in enumerate(body) if isinstance(s, ast.While))
    before_loop = ast.Module(body=body[:first_while], type_ignores=[])
    assert _JOIN in _called_names(before_loop), (
        f"{_LOOP} must join the signal topics through {_JOIN} before its loop"
    )


def test_the_kernel_boot_never_awaits_a_savings_subscribe_itself() -> None:
    source = _KERNEL.read_text(encoding="utf-8").splitlines()
    start = 1 + next(
        i for i, line in enumerate(source) if "3.9. Wire savings estimation" in line
    )
    end = 1 + next(
        i
        for i, line in enumerate(source)
        if "3.10. Start runtime health monitor" in line
    )
    loop = _savings_loop()
    loop_lines = range(loop.lineno, (loop.end_lineno or loop.lineno) + 1)
    awaited_subscribes = [
        n.lineno
        for n in ast.walk(_kernel_tree())
        if isinstance(n, ast.Await)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "subscribe"
        and start <= n.lineno < end
        and n.lineno not in loop_lines
    ]
    assert awaited_subscribes == [], (
        f"the savings step awaits event_bus.subscribe on the boot path at {awaited_subscribes}"
    )
