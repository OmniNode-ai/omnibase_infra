# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The publisher must know the three OMN-18931 delegation keys before they land.

OMN-18931. omnibase_infra#3951 reads three new lane keys out of the packaged
omnimarket ``config/ci_bus_lanes.yaml``: ``broker_topology`` and
``delegation_fault_routes`` on the dogfood lane, and ``delegation_routes`` on
the dev and dogfood lanes. ``scripts/trigger_rebuild_on_merge.py`` validates
that same file ``extra="forbid"``. If omnimarket declared the keys before this
model learned them, the rebuild trigger and the required
``ci-bus-overlay-binding`` context would reject the whole overlay. That is how
the OMN-18012, OMN-18060 and OMN-16964 keys each broke. omnimarket's
``ci-bus-overlay-parity`` gate refuses the omnimarket PR until the keys are
learned here, so this lands first.

The publisher does not act on any of the three keys. It models their shape so
that ``extra="forbid"`` still means "unknown key". The policy (a 240-second,
single-attempt, no-escalation route for HTTP 429 and 503 only) belongs to
``omnibase_infra.runtime.dogfood_delegation_fault_routes``, the one component
that acts on it, and is deliberately not duplicated here.

Hermetic: the overlay fixtures are written to tmp_path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "trigger_rebuild_on_merge.py"
)


def _import_trigger_module():
    """Import the publisher by file path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "trigger_rebuild_on_merge_delegation_route_keys", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_ROUTE: dict[str, object] = {
    "consumer": "omnimarket.nodes.node_delegation_orchestrator",
    "terminal_route": "terminal_events",
    "repository_owner": "omnimarket",
}


def _fault_route(status: int) -> dict[str, object]:
    return {
        "backend_id": f"dogfood-fault-{status}",
        "endpoint_url": (
            f"http://dogfood-delegation-fault-{status}:8080/v1/chat/completions"
        ),
        "expected_http_status": status,
        "requested_timeout_seconds": 240,
        "max_attempts": 1,
        "no_escalation": True,
    }


def _dogfood_lane() -> dict[str, object]:
    return {
        "broker": "192.0.2.10:47092",
        "security_protocol": "PLAINTEXT",
        "broker_topology": {
            "external_bootstrap_servers": "192.0.2.10:47092",
            "internal_bootstrap_servers": "redpanda:9092",
        },
        "delegation_fault_routes": [_fault_route(429), _fault_route(503)],
        "delegation_routes": [dict(_ROUTE)],
    }


def _dev_lane() -> dict[str, object]:
    return {
        "broker": "example.invalid:19092",
        "security_protocol": "SASL_PLAINTEXT",
        "sasl_mechanism": "SCRAM-SHA-256",
        "delegation_routes": [dict(_ROUTE)],
    }


def _overlay(tmp_path: Path, lanes: dict[str, object]) -> Path:
    path = tmp_path / "ci_bus_lanes.yaml"
    path.write_text(
        yaml.safe_dump({"default": "inmemory", "lanes": lanes}), encoding="utf-8"
    )
    return path


@pytest.mark.unit
def test_the_declared_shape_is_accepted(tmp_path: Path) -> None:
    mod = _import_trigger_module()

    model = mod.load_ci_bus_overlay(
        _overlay(tmp_path, {"dev": _dev_lane(), "dogfood": _dogfood_lane()})
    )

    dogfood = model.lanes["dogfood"]
    assert dogfood.broker_topology is not None
    assert dogfood.broker_topology.internal_bootstrap_servers == "redpanda:9092"
    statuses = [route.expected_http_status for route in dogfood.delegation_fault_routes]
    assert statuses == [429, 503]
    assert dogfood.delegation_routes[0].terminal_route == "terminal_events"
    assert model.lanes["dev"].delegation_routes[0].repository_owner == "omnimarket"


@pytest.mark.unit
def test_the_keys_stay_optional(tmp_path: Path) -> None:
    """Every other lane declares none of them, so none may be required."""
    lane = {
        key: value
        for key, value in _dogfood_lane().items()
        if key in {"broker", "security_protocol"}
    }
    mod = _import_trigger_module()

    model = mod.load_ci_bus_overlay(_overlay(tmp_path, {"dogfood": lane}))

    assert model.lanes["dogfood"].broker_topology is None
    assert model.lanes["dogfood"].delegation_fault_routes == ()
    assert model.lanes["dogfood"].delegation_routes == ()


@pytest.mark.unit
@pytest.mark.parametrize(
    ("block", "typo"),
    [
        ("broker_topology", "internal_bootstrap_server"),
        ("delegation_fault_routes", "no_escalaton"),
        ("delegation_routes", "terminal_rout"),
    ],
)
def test_an_unknown_key_inside_a_block_is_still_refused(
    tmp_path: Path, block: str, typo: str
) -> None:
    """Learning the key must not weaken the strictness that motivated it."""
    lane = _dogfood_lane()
    value = lane[block]
    if isinstance(value, list):
        value[0][typo] = "x"
    else:
        assert isinstance(value, dict)
        value[typo] = "x"
    mod = _import_trigger_module()

    with pytest.raises(Exception, match=typo):
        mod.load_ci_bus_overlay(_overlay(tmp_path, {"dogfood": lane}))


@pytest.mark.unit
def test_a_genuinely_unknown_lane_key_is_still_refused(tmp_path: Path) -> None:
    lane = _dogfood_lane()
    lane["delegation_fault_route"] = lane.pop("delegation_fault_routes")
    mod = _import_trigger_module()

    with pytest.raises(Exception, match=r"delegation_fault_route|Extra inputs"):
        mod.load_ci_bus_overlay(_overlay(tmp_path, {"dogfood": lane}))


@pytest.mark.unit
def test_topology_runtime_environment_is_learned_before_it_lands(
    tmp_path: Path,
) -> None:
    """OMN-18933: the dev lane's topology names ``runtime_environment: local``.

    Learned here first for the same reason as the three keys above: omnimarket
    declares it only after this model accepts it.
    """
    dev = _dev_lane()
    dev["broker_topology"] = {
        "external_bootstrap_servers": "example.invalid:19092",
        "internal_bootstrap_servers": "redpanda:9092",
        "runtime_environment": "local",
    }
    mod = _import_trigger_module()

    model = mod.load_ci_bus_overlay(_overlay(tmp_path, {"dev": dev}))

    topology = model.lanes["dev"].broker_topology
    assert topology is not None
    assert topology.runtime_environment == "local"
