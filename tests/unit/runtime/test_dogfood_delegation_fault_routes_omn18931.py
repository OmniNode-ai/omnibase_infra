# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed route selection for OMN-18931 fault-cohort pins."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.runtime.dogfood_delegation_fault_routes import (
    resolve_dogfood_delegation_fault_route,
    validate_dogfood_delegation_fault_request,
)

pytestmark = pytest.mark.unit


def _write_lanes(path: Path, routes: object) -> None:
    path.write_text(
        """lanes:\n  dogfood:\n    broker: dogfood-broker:9092\n    delegation_fault_routes:\n"""
        + "\n".join(f"      - {line}" for line in [])
        + "\n",
        encoding="utf-8",
    )
    path.write_text(
        yaml.safe_dump(
            {
                "lanes": {
                    "dogfood": {
                        "broker": "dogfood-broker:9092",
                        "delegation_fault_routes": routes,
                    }
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _route(status: int) -> dict[str, object]:
    return {
        "backend_id": f"dogfood-fault-{status}",
        "endpoint_url": f"http://dogfood-delegation-fault-{status}:8080/v1/chat/completions",
        "expected_http_status": status,
        "requested_timeout_seconds": 240,
        "max_attempts": 1,
        "no_escalation": True,
    }


def test_declared_dogfood_pin_resolves_exact_single_hop_policy(tmp_path: Path) -> None:
    config = tmp_path / "ci_bus_lanes.yaml"
    _write_lanes(config, [_route(429), _route(503)])

    route = resolve_dogfood_delegation_fault_route(
        environment="dogfood",
        bootstrap_servers="dogfood-broker:9092",
        backend_id="dogfood-fault-429",
        path_for_test=config,
    )

    assert route.expected_http_status == 429
    assert route.requested_timeout_seconds == 240
    assert route.max_attempts == 1
    assert route.no_escalation is True


@pytest.mark.parametrize(
    ("environment", "broker", "backend_id"),
    [
        ("dev", "dogfood-broker:9092", "dogfood-fault-429"),
        ("prod", "dogfood-broker:9092", "dogfood-fault-429"),
        ("dogfood", "wrong-broker:9092", "dogfood-fault-429"),
        ("dogfood", "dogfood-broker:9092", "not-declared"),
        ("dogfood", "", "dogfood-fault-429"),
    ],
)
def test_unbounded_identity_or_unknown_pin_refuses(
    tmp_path: Path, environment: str, broker: str, backend_id: str
) -> None:
    config = tmp_path / "ci_bus_lanes.yaml"
    _write_lanes(config, [_route(429), _route(503)])

    with pytest.raises(InfraUnavailableError):
        resolve_dogfood_delegation_fault_route(
            environment=environment,
            bootstrap_servers=broker,
            backend_id=backend_id,
            path_for_test=config,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("requested_timeout_seconds", 241), ("max_attempts", 2), ("no_escalation", False)],
)
def test_declared_route_cannot_weaken_single_hop_policy(
    tmp_path: Path, field: str, value: object
) -> None:
    config = tmp_path / "ci_bus_lanes.yaml"
    route = _route(503)
    route[field] = value
    _write_lanes(config, [route])

    with pytest.raises(InfraUnavailableError):
        resolve_dogfood_delegation_fault_route(
            environment="dogfood",
            bootstrap_servers="dogfood-broker:9092",
            backend_id="dogfood-fault-503",
            path_for_test=config,
        )


def test_fault_pin_accepts_only_explicit_internal_topology_identity(
    tmp_path: Path,
) -> None:
    config = tmp_path / "ci_bus_lanes.yaml"
    _write_lanes(config, [_route(429)])
    import yaml

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    raw["lanes"]["dogfood"]["broker_topology"] = {
        "external_bootstrap_servers": "dogfood-broker:9092",
        "internal_bootstrap_servers": "redpanda:9092",
    }
    config.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    route = resolve_dogfood_delegation_fault_route(
        environment="dogfood",
        bootstrap_servers="redpanda:9092",
        backend_id="dogfood-fault-429",
        path_for_test=config,
    )

    assert route.expected_http_status == 429


def test_consumer_guard_accepts_only_exact_declared_fault_policy(
    tmp_path: Path,
) -> None:
    config = tmp_path / "ci_bus_lanes.yaml"
    _write_lanes(config, [_route(429)])
    request = SimpleNamespace(
        backend_id="dogfood-fault-429",
        no_escalation=True,
        requested_timeout_seconds=240,
    )
    bus = SimpleNamespace(
        environment="dogfood", bootstrap_servers="dogfood-broker:9092"
    )

    validate_dogfood_delegation_fault_request(
        request=request,
        event_bus=bus,
        path_for_test=config,
    )


@pytest.mark.parametrize(
    ("backend_id", "no_escalation", "timeout", "environment", "broker"),
    [
        ("not-declared", True, 240, "dogfood", "dogfood-broker:9092"),
        ("dogfood-fault-429", False, 240, "dogfood", "dogfood-broker:9092"),
        ("dogfood-fault-429", True, 239, "dogfood", "dogfood-broker:9092"),
        ("dogfood-fault-429", True, 240, "dev", "dogfood-broker:9092"),
        ("dogfood-fault-429", True, 240, "dogfood", "wrong-broker:9092"),
    ],
)
def test_consumer_guard_refuses_untrusted_raw_pin_or_policy(
    tmp_path: Path,
    backend_id: str,
    no_escalation: bool,
    timeout: int,
    environment: str,
    broker: str,
) -> None:
    config = tmp_path / "ci_bus_lanes.yaml"
    _write_lanes(config, [_route(429)])
    request = SimpleNamespace(
        backend_id=backend_id,
        no_escalation=no_escalation,
        requested_timeout_seconds=timeout,
    )
    bus = SimpleNamespace(environment=environment, bootstrap_servers=broker)

    with pytest.raises(InfraUnavailableError):
        validate_dogfood_delegation_fault_request(
            request=request,
            event_bus=bus,
            path_for_test=config,
        )


def test_consumer_guard_refuses_unpinned_no_escalation_before_workflow() -> None:
    request = SimpleNamespace(
        backend_id=None,
        no_escalation=True,
        requested_timeout_seconds=None,
    )

    with pytest.raises(InfraUnavailableError, match="requires a declared backend pin"):
        validate_dogfood_delegation_fault_request(request=request, event_bus=None)
