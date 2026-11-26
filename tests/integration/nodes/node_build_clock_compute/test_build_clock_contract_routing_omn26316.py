# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration: the build clock node's contract on disk routes to a handler that judges (OMN-26316).

Loads the shipped ``contract.yaml``, resolves the ``build.clock.measure`` handler and its
input and output models by the module paths the contract declares, and runs a fast build
and a 6 minute build through them. A contract that names a handler or model that does not
import fails here, not at runtime boot.
"""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

import omnibase_infra.nodes.node_build_clock_compute as node_pkg

pytestmark = pytest.mark.integration

START = datetime(2026, 11, 21, 9, 10, 40, tzinfo=UTC)
DOCKER_START = "2026-11-21T09:10:40.000000000Z"


def _resolve(module: str, name: str) -> type:
    return getattr(importlib.import_module(module), name)


def _contract() -> dict:
    path = Path(node_pkg.__file__).parent / "contract.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _route(contract: dict, operation: str) -> dict:
    matches = [
        h
        for h in contract["handler_routing"]["handlers"]
        if h["operation"] == operation
    ]
    assert len(matches) == 1
    return matches[0]


def test_contract_declared_handler_and_models_import() -> None:
    route = _route(_contract(), "build.clock.measure")
    handler_cls = _resolve(route["handler"]["module"], route["handler"]["name"])
    request_cls = _resolve(route["input_model"]["module"], route["input_model"]["name"])
    assert handler_cls.__name__ == "HandlerBuildClock"
    assert request_cls.__name__ == "ModelBuildClockRequest"


@pytest.mark.parametrize(
    ("after", "verdict"),
    [(timedelta(seconds=14), "pass"), (timedelta(minutes=6), "fail")],
)
def test_contract_routed_handler_judges_build(after: timedelta, verdict: str) -> None:
    contract = _contract()
    route = _route(contract, "build.clock.measure")
    handler = _resolve(route["handler"]["module"], route["handler"]["name"])()
    request_cls = _resolve(route["input_model"]["module"], route["input_model"]["name"])
    output_cls = _resolve(
        contract["output_model"]["module"], contract["output_model"]["name"]
    )
    result = handler.handle(
        request_cls(container_started_at=DOCKER_START, first_healthy_at=START + after)
    )
    assert isinstance(result, output_cls)
    assert result.verdict == verdict
    assert result.limit_minutes == 5.0
