# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The infra delegate path holds no fixed 240 s budget of its own (OMN-21604 AC3).

A dogfood fault route's timeout is whatever omnimarket's lane declaration says;
the consumer guard still admits only a request that asks for exactly that
figure. The dispatch-deadline refusal itself is covered by
test_delegation_budget_deadline_check_omn21604.py.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import omnibase_infra
from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.runtime.dogfood_delegation_fault_routes import (
    resolve_dogfood_delegation_fault_route,
    validate_dogfood_delegation_fault_request,
)

pytestmark = pytest.mark.unit

_SRC = Path(omnibase_infra.__file__).resolve().parent

# The modules a delegation passes through on its way from `onex delegate` to the
# consumer that runs it.
_DELEGATE_PATH = (
    "cli/cli_delegate.py",
    "cli/model_delegate_terminal.py",
    "models/delegation/model_dispatch_budget.py",
    "runtime/protocols/protocol_delegation_dispatch_port.py",
    "runtime/service_delegation_dispatch_port.py",
    "runtime/dogfood_delegation_fault_routes.py",
)

_DECLARED_SECONDS = 300
_BROKER = "dogfood-broker:9092"


def _write_lanes(path: Path) -> None:
    route = {
        "backend_id": "dogfood-fault-503",
        "endpoint_url": "http://dogfood-delegation-fault-503:8080/v1/chat/completions",
        "expected_http_status": 503,
        "requested_timeout_seconds": _DECLARED_SECONDS,
        "max_attempts": 1,
        "no_escalation": True,
    }
    document = {
        "lanes": {"dogfood": {"broker": _BROKER, "delegation_fault_routes": [route]}}
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _request(timeout: int) -> SimpleNamespace:
    return SimpleNamespace(
        backend_id="dogfood-fault-503",
        no_escalation=True,
        requested_timeout_seconds=timeout,
    )


def test_fault_route_timeout_comes_from_the_lane_declaration(tmp_path: Path) -> None:
    config = tmp_path / "ci_bus_lanes.yaml"
    _write_lanes(config)

    route = resolve_dogfood_delegation_fault_route(
        environment="dogfood",
        bootstrap_servers=_BROKER,
        backend_id="dogfood-fault-503",
        path_for_test=config,
    )

    assert route.requested_timeout_seconds == _DECLARED_SECONDS


def test_consumer_guard_admits_only_the_declared_timeout(tmp_path: Path) -> None:
    config = tmp_path / "ci_bus_lanes.yaml"
    _write_lanes(config)
    bus = SimpleNamespace(environment="dogfood", bootstrap_servers=_BROKER)

    validate_dogfood_delegation_fault_request(
        request=_request(_DECLARED_SECONDS), event_bus=bus, path_for_test=config
    )
    with pytest.raises(InfraUnavailableError, match="exact timeout policy"):
        validate_dogfood_delegation_fault_request(
            request=_request(240), event_bus=bus, path_for_test=config
        )


@pytest.mark.parametrize("relative", _DELEGATE_PATH)
def test_delegate_path_module_holds_no_240_literal(relative: str) -> None:
    source = (_SRC / relative).read_text(encoding="utf-8")
    hits = [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Constant)
        and type(node.value) in (int, float)
        and node.value == 240
    ]
    assert hits == [], f"{relative} still holds a 240 literal on lines {hits}"
