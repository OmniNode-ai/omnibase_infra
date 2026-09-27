# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The dogfood Bifrost render materializes the declared fault backends (OMN-18931)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import omnibase_infra.runtime.render_bifrost_delegation_contract as renderer
from omnibase_infra.errors import ProtocolConfigurationError
from omnibase_infra.runtime.dogfood_delegation_fault_routes import (
    load_dogfood_delegation_fault_routes,
)
from omnibase_infra.runtime.models.enum_bifrost_lane_locale import (
    EnumBifrostLaneLocale,
)
from omnibase_infra.runtime.models.model_bifrost_lane_overlay import (
    ModelBifrostLaneOverlay,
)

pytestmark = pytest.mark.unit


def _fault_route(status: int) -> dict[str, object]:
    return {
        "backend_id": f"dogfood-fault-{status}",
        "endpoint_url": f"http://dogfood-delegation-fault-{status}:8080/v1/chat/completions",
        "expected_http_status": status,
        "requested_timeout_seconds": 240,
        "max_attempts": 1,
        "no_escalation": True,
    }


@pytest.fixture
def declared_fault_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lanes = tmp_path / "ci_bus_lanes.yaml"
    lanes.write_text(
        yaml.safe_dump(
            {
                "lanes": {
                    "dogfood": {
                        "broker": "dogfood-broker:9092",
                        "delegation_fault_routes": [
                            _fault_route(429),
                            _fault_route(503),
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        renderer,
        "load_dogfood_delegation_fault_routes",
        lambda: load_dogfood_delegation_fault_routes(path_for_test=lanes),
    )


def _overlay(lane: str) -> ModelBifrostLaneOverlay:
    return ModelBifrostLaneOverlay(
        schema_version="bifrost_lane_overlay.v3",
        lane=lane,
        locale=EnumBifrostLaneLocale.CLOUD,
        backends=(),
    )


@pytest.mark.usefixtures("declared_fault_routes")
def test_dogfood_render_appends_each_declared_fault_backend() -> None:
    base: dict[str, object] = {
        "backends": [{"backend_id": "local-coder", "provider": "openai_compatible"}]
    }

    renderer._append_dogfood_fault_backends(base, overlay=_overlay("dogfood"))

    backends = base["backends"]
    assert isinstance(backends, list)
    assert [b["backend_id"] for b in backends] == [
        "local-coder",
        "dogfood-fault-429",
        "dogfood-fault-503",
    ]
    fault = backends[1]
    assert fault["endpoint_url"] == (
        "http://dogfood-delegation-fault-429:8080/v1/chat/completions"
    )
    assert fault["timeout_ms"] == 240_000
    assert fault["tier"] == "dogfood_fault"


@pytest.mark.usefixtures("declared_fault_routes")
@pytest.mark.parametrize("lane", ["dev", "stability-test", "prod"])
def test_no_other_lane_renders_a_fault_backend(lane: str) -> None:
    base: dict[str, object] = {"backends": [{"backend_id": "local-coder"}]}

    renderer._append_dogfood_fault_backends(base, overlay=_overlay(lane))

    assert base == {"backends": [{"backend_id": "local-coder"}]}


@pytest.mark.usefixtures("declared_fault_routes")
def test_a_fault_backend_colliding_with_a_base_backend_fails_the_render() -> None:
    base: dict[str, object] = {"backends": [{"backend_id": "dogfood-fault-503"}]}

    with pytest.raises(ProtocolConfigurationError, match="collides"):
        renderer._append_dogfood_fault_backends(base, overlay=_overlay("dogfood"))
