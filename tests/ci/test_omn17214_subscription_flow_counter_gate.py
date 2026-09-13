# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17214 AC6 — every subscription path must register a flow counter.

Why this is a gate and not a sweep
----------------------------------
Operating Rule 5: detection that is not enforcement gets ignored. This defect is
itself an instance of an un-gated seam drifting. ``_make_event_bus_callback`` was
given the flow-counter seam by OMN-16777; ``_make_raw_event_projection_callback``
was not, and nothing anywhere noticed for the three months that followed. 57 live
Stable subscriptions emitted no row at all, and because a MISSING row is read as
``UNKNOWN`` rather than as zero traffic, the flow projection could not even say
that it did not know.

What this gate asserts
----------------------
The subscription-wiring loop in ``handler_wiring`` selects ONE callback factory
per subscription. The set of factories it can select is read from the source
itself — every ``callback = _make_*(...)`` assignment in the module — so a THIRD
branch added tomorrow is enumerated automatically rather than needing this file
to be updated. For each selected factory:

  1. the call site passes ``consumer_group=``, and
  2. the factory declares a ``consumer_group`` parameter, and
  3. the factory body calls ``.register(`` on the flow counters.

Honest limit, stated rather than implied
----------------------------------------
This is a SOURCE gate over one module. It proves the seam is present on every
branch the wiring can select; it does not prove the runtime reaches that branch,
and it cannot see a subscription wired from a different module entirely (the 16
service-owned consumers in this ticket's AC2 residual are off the auto-wiring
path and are outside what any gate on this file can answer). The behavioural
proof that both branches actually count is
``tests/unit/runtime/auto_wiring/test_omn17214_raw_projection_flow_seam.py``;
this gate exists so that proof cannot be quietly outgrown by a new branch.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_HANDLER_WIRING = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "omnibase_infra"
    / "runtime"
    / "auto_wiring"
    / "handler_wiring.py"
)

# The register call the seam is made of, and the accessor that produces the
# object it is called on. Named here so the failure message can quote them.
_REGISTER_METHOD = "register"
_COUNTERS_ACCESSOR = "get_consumer_flow_counters"
_GROUP_PARAM = "consumer_group"
# The list every bus subscription is driven from. Used to locate the wiring
# function structurally, so the gate does not depend on that function's name.
_CALLBACK_LIST = "topic_callbacks"


def _module() -> ast.Module:
    return ast.parse(_HANDLER_WIRING.read_text(encoding="utf-8"))


def _subscription_wiring_function(
    tree: ast.Module,
) -> ast.AsyncFunctionDef | ast.FunctionDef:
    """Return the ONE function that builds the per-topic bus subscriptions.

    Identified by what it does, not by its name: it is the function that fills
    ``topic_callbacks``, the list every ``typed_bus.subscribe`` call is driven
    from. Renaming it does not evade this gate; moving the loop somewhere that
    no longer fills that list fails the positive control below instead of
    silently passing.
    """
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(
            isinstance(inner, ast.Name) and inner.id == _CALLBACK_LIST
            for inner in ast.walk(node)
        )
    ]
    assert len(matches) == 1, (
        f"expected exactly one function filling {_CALLBACK_LIST!r}; found "
        f"{[m.name for m in matches]}. The gate cannot locate the subscription "
        "wiring, so it must fail rather than report a vacuous pass"
    )
    return matches[0]


def _selected_factories(tree: ast.Module) -> dict[str, ast.Call]:
    """Return ``{factory_name: the Call that selects it}`` for the wiring loop.

    Read from the source, not from a hand-maintained list: the subscription loop
    binds its chosen factory to the local name ``callback`` before appending it
    to ``topic_callbacks``, so every branch — including one added after this file
    was written — appears here. Scoped to that one function, because
    ``_make_*_callback`` is also the naming convention for the per-handler
    DISPATCHER callbacks registered on the dispatch engine, which join no
    consumer group and correctly count nothing.
    """
    selected: dict[str, ast.Call] = {}
    for node in ast.walk(_subscription_wiring_function(tree)):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(t, ast.Name) and t.id == "callback" for t in node.targets
        ):
            continue
        call = node.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        if not call.func.id.startswith("_make_"):
            continue
        selected[call.func.id] = call
    return selected


def _function_defs(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and isinstance(node, ast.FunctionDef)
    }


def _declares_group_param(fn: ast.FunctionDef) -> bool:
    args = fn.args
    names = {a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    return _GROUP_PARAM in names


def _registers_a_flow_counter(fn: ast.FunctionDef) -> bool:
    """True when the factory body both obtains the counters and registers."""
    obtains = False
    registers = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == _COUNTERS_ACCESSOR:
                obtains = True
            if isinstance(func, ast.Attribute) and func.attr == _REGISTER_METHOD:
                registers = True
    return obtains and registers


@pytest.mark.unit
def test_the_gate_enumerates_more_than_one_subscription_branch() -> None:
    """Positive control: an empty enumeration must not read as a clean pass.

    A zero-row sweep is not evidence of absence. If a refactor renames the loop
    local, ``_selected_factories`` would return nothing and every assertion below
    would vacuously pass — so the count is asserted first, against the branches
    known to exist (``_make_event_bus_callback`` and
    ``_make_raw_event_projection_callback``).
    """
    selected = _selected_factories(_module())
    assert len(selected) >= 2, (
        "the subscription-wiring enumeration found "
        f"{sorted(selected)} — fewer than the two branches known to exist. The "
        "gate is reading the wrong thing and every assertion it makes is vacuous"
    )
    assert "_make_event_bus_callback" in selected
    assert "_make_raw_event_projection_callback" in selected


@pytest.mark.unit
def test_every_selected_subscription_factory_registers_a_flow_counter() -> None:
    """AC6: a new subscription branch without the flow seam fails this check."""
    tree = _module()
    selected = _selected_factories(tree)
    defs = _function_defs(tree)

    offenders: list[str] = []
    for name in sorted(selected):
        fn = defs.get(name)
        if fn is None:
            offenders.append(
                f"{name}: selected by the wiring loop but not defined here"
            )
            continue
        if not _declares_group_param(fn):
            offenders.append(f"{name}: declares no {_GROUP_PARAM!r} parameter")
            continue
        if not _registers_a_flow_counter(fn):
            offenders.append(
                f"{name}: never calls {_COUNTERS_ACCESSOR}() + .{_REGISTER_METHOD}()"
            )

    assert not offenders, (
        "a subscription path registers no flow counter, so every subscription it "
        "wires emits NO row — which the consumer-flow projection materializes as "
        "UNKNOWN, not as observed-idle (OMN-17214):\n  " + "\n  ".join(offenders)
    )


@pytest.mark.unit
def test_every_selected_subscription_factory_is_passed_the_consumer_group() -> None:
    """A factory that accepts the group but is never handed one counts nothing.

    This is the half that actually broke: the raw-projection branch was selected
    with four arguments and the group was simply not among them, so the seam
    inside the factory — had it existed — would still have been dead.
    """
    selected = _selected_factories(_module())

    offenders = [
        name
        for name, call in sorted(selected.items())
        if not any(kw.arg == _GROUP_PARAM for kw in call.keywords)
    ]

    assert not offenders, (
        "the subscription-wiring loop selects these factories without passing "
        f"{_GROUP_PARAM!r}, so they cannot count anything: {offenders}"
    )
