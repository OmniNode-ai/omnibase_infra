# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17214 AC6 — a publish seam that attributes no flow output fails here.

Sibling of ``test_omn17214_subscription_flow_counter_gate.py``. That one guards
the SUBSCRIBE side: every wiring branch must register a flow counter. This one
guards the PUBLISH side: every seam in ``handler_wiring`` that publishes a
handler's own declared output on ``event_bus`` must attribute it through the
single ``record_flow_output`` entry point.

Why a gate and not just the behaviour tests. Per Operating Rule 5, detection
that is not enforcement gets ignored — and this defect is itself an instance of
an un-gated seam drifting. The external result applier recorded its publishes
from the first day of OMN-16777; the two seams inside ``handler_wiring`` never
did; nothing said so, and the consequence was the Phase 1 deliverable reporting
its own writer as STALLED while that writer was demonstrably producing
(measured live on the ``.201`` dev lane 2026-09-17, in=76 out=0 across 19
consecutive windows while the terminal topic advanced).

The seams are located from the module's own AST, not from a list of names, so a
THIRD publish seam added later is covered without editing this file.

Deliberately out of scope, stated rather than implied: the DLQ route
(``await publish(dlq_topic, ...)``) and the boundary FAILURE terminal
(``await publish_fn(terminal_topic, ...)``). Those outcomes are already counted
as ``messages_dlq`` / ``handler_errors``, and counting a failure terminal as a
successful output would make a consumer that is failing every message read
``FLOWING``. Both are calls to injected callables rather than to ``event_bus``,
which is the distinction this gate keys on.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from omnibase_infra.runtime.auto_wiring import handler_wiring

# The two ``event_bus`` methods that publish a handler's own declared output.
_OUTPUT_PUBLISH_METHODS = frozenset({"publish", "publish_envelope"})

# The single entry point every such seam must route its attribution through.
_ATTRIBUTION_CALL = "record_flow_output"


def _module_tree() -> ast.Module:
    return ast.parse(inspect.getsource(handler_wiring))


def _publishing_functions() -> dict[str, list[str]]:
    """Map every handler_wiring function publishing on ``event_bus`` to its methods."""
    found: dict[str, list[str]] = {}

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

        def _walk_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            self.stack.append(node)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._walk_function(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._walk_function(node)

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr in _OUTPUT_PUBLISH_METHODS
                and isinstance(func.value, ast.Name)
                and func.value.id == "event_bus"
                and self.stack
            ):
                found.setdefault(self.stack[-1].name, []).append(func.attr)
            self.generic_visit(node)

    _Visitor().visit(_module_tree())
    return found


def _attributing_functions() -> set[str]:
    """Functions containing an actual CALL to the attribution entry point.

    A call, never a mention: matching the bare name would let an unused
    ``from ... import record_flow_output`` satisfy the gate. That is not
    hypothetical — deleting the call while leaving the import in place kept this
    test green until the check was narrowed to ``ast.Call``.
    """
    attributing: set[str] = set()
    for node in ast.walk(_module_tree()):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Name)
                and inner.func.id == _ATTRIBUTION_CALL
            ):
                attributing.add(node.name)
    return attributing


@pytest.mark.unit
def test_the_gate_finds_the_publish_seams_it_is_meant_to_guard() -> None:
    """Positive control against a vacuous pass.

    If the AST query silently matched nothing, the gate below would pass on an
    empty set and report green for a module with no attribution anywhere.
    """
    publishing = _publishing_functions()
    assert publishing, (
        "the publish-seam query matched no function in handler_wiring — the gate "
        "below would pass vacuously"
    )
    assert "_emit_projection_terminal_event" in publishing, (
        "the known projection terminal seam was not located; the AST query has "
        f"drifted from the module. found={sorted(publishing)}"
    )


@pytest.mark.unit
def test_every_event_bus_publish_seam_records_flow_output() -> None:
    """Every output-publishing seam attributes its output, or this names it."""
    attributing = _attributing_functions()
    unattributed = [name for name in _publishing_functions() if name not in attributing]

    assert not unattributed, (
        "these handler_wiring functions publish a handler's declared output on "
        "event_bus without calling record_flow_output, so every node whose "
        "output leaves on that seam reports messages_out=0 and derives STALLED "
        f"while producing: {sorted(unattributed)}"
    )
