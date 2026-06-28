# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-13713 proof: malformed-contract nodes now resolve a handler spec.

Before this ticket both contracts failed at dispatch over RuntimeLocal with::

    Workflow contract missing 'terminal_event' topic and no handler spec found
    (need handler_routing.default_handler or handler.module/class).

Root cause: neither contract declared ``handler_routing.default_handler`` (in
``module:Class`` form) nor a top-level ``handler.module``/``handler.class``
block, so ``RuntimeLocal._resolve_handler_spec()`` returned ``None`` and the
compute execution path could never start.

Fix: add ``handler_routing.default_handler`` to each contract. These tests
assert the resolver now returns the concrete (module, class) and that the
resolved handler class is importable — i.e. the named contract-resolution
defect is gone. (Full end-to-end execution of these effect/DB nodes additionally
requires a live Postgres pool / deployment topology that RuntimeLocal does not
inject; that runtime-dependency wiring is tracked separately and is out of scope
for this contract-resolution fix.)
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from omnibase_core.runtime.runtime_local import RuntimeLocal, load_workflow_contract

_NODES = Path(__file__).resolve().parents[3] / "src/omnibase_infra/nodes"

_CASES = [
    (
        _NODES / "node_decision_store_query_compute/contract.yaml",
        "omnibase_infra.nodes.node_decision_store_query_compute.handlers.handler_query_decisions",
        "HandlerQueryDecisions",
    ),
    (
        _NODES / "node_setup_preflight_effect/contract.yaml",
        "omnibase_infra.nodes.node_setup_preflight_effect.handlers.handler_preflight_check",
        "HandlerPreflightCheck",
    ),
]


@pytest.mark.parametrize(
    ("contract_path", "expected_module", "expected_class"),
    _CASES,
    ids=["decision_store_query_compute", "setup_preflight_effect"],
)
def test_runtime_local_resolves_handler_spec(
    tmp_path: Path,
    contract_path: Path,
    expected_module: str,
    expected_class: str,
) -> None:
    """RuntimeLocal resolves a concrete handler (no 'no handler spec found')."""
    runtime = RuntimeLocal(
        workflow_path=contract_path,
        state_root=tmp_path / "state",
        timeout=5,
    )
    runtime._contract = load_workflow_contract(contract_path)

    resolved = runtime._resolve_handler_spec()

    assert resolved is not None, (
        f"{contract_path.parent.name}: handler spec still unresolved — "
        "runtime would fail with 'missing terminal_event / no handler spec found'"
    )
    module_name, class_name = resolved
    assert module_name == expected_module
    assert class_name == expected_class

    # The resolved handler must be importable so the compute path can proceed
    # to instantiation rather than dying at resolution.
    module = importlib.import_module(module_name)
    handler_cls = getattr(module, class_name)
    assert callable(handler_cls)
