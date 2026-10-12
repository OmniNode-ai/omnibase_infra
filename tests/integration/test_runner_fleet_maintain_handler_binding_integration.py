# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21041 -- the runner-fleet-maintain handlers bind the command the runtime hands them.

The runtime passes a handler the whole validated command. This test reads the
real orchestrator contract from disk and checks that every handler it routes
declares ``handle(request)`` annotated with the routed command model, and that
the contract names its terminal event for ``onex run-node``.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import yaml

import omnibase_infra.nodes.node_runner_fleet_maintain_orchestrator as _orchestrator_pkg

_CONTRACT = Path(_orchestrator_pkg.__file__).parent / "contract.yaml"


def _load_contract() -> dict[str, object]:
    loaded = yaml.safe_load(_CONTRACT.read_text())
    assert isinstance(loaded, dict)
    return loaded


def test_orchestrator_contract_declares_a_terminal_event() -> None:
    assert _load_contract().get("terminal_event"), (
        "onex run-node refuses a contract with no top-level terminal_event"
    )


def test_every_routed_handler_takes_one_request_parameter() -> None:
    routing = _load_contract()["handler_routing"]
    assert isinstance(routing, dict)
    handlers = routing["handlers"]
    assert handlers, "the orchestrator contract routes no handlers"
    for entry in handlers:
        handler = entry["handler"]
        module = importlib.import_module(handler["module"])
        cls = getattr(module, handler["name"])
        params = [
            name for name in inspect.signature(cls.handle).parameters if name != "self"
        ]
        assert params[:1] == ["request"], (
            f"{handler['name']}.handle must take the whole command as `request`, "
            f"found {params}"
        )
