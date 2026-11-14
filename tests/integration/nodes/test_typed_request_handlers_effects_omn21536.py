# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21536: effect handlers of five nodes take one typed request.

Loads the real ``contract.yaml`` of each node from disk, resolves every routed
handler and requires ``handle`` to take exactly one argument next to ``self``.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest
import yaml

from omnibase_infra.nodes import (
    node_chain_retrieval_effect,
    node_decision_store_effect,
    node_delta_bundle_effect,
    node_delta_metrics_effect,
    node_merge_gate_effect,
)

pytestmark = pytest.mark.integration

_NODE_PACKAGES = (
    node_chain_retrieval_effect,
    node_decision_store_effect,
    node_delta_bundle_effect,
    node_delta_metrics_effect,
    node_merge_gate_effect,
)


def _routed_handlers() -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for package in _NODE_PACKAGES:
        contract_path = Path(str(package.__file__)).parent / "contract.yaml"
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        for entry in contract["handler_routing"]["handlers"]:
            handler = entry["handler"]
            rows.append((package.__name__, handler["module"], handler["name"]))
    return rows


@pytest.mark.parametrize(("node", "handler_module", "handler_name"), _routed_handlers())
def test_routed_handler_takes_one_typed_argument(
    node: str, handler_module: str, handler_name: str
) -> None:
    handler_cls = getattr(importlib.import_module(handler_module), handler_name)

    parameters = list(inspect.signature(handler_cls.handle).parameters)

    assert len(parameters) == 2, f"{node}: {handler_name}.handle takes {parameters}"
    assert parameters[0] == "self"
