# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Unit seam of the bounded delegation route resolver (OMN-18933, K6).

Carried from the Codex draft omnibase_infra#3951 (tests/unit/runtime/
test_bounded_delegation_routes.py there) and extended for the runtime
environment pair and the vendor/manifest binding. The K6 acceptance cases
(missing row, broker mismatch, consumer mismatch refused before dispatch, and
the offset-489 positive control) live in tests/ci/test_dispatcher_route_coverage_gate.py.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from importlib import metadata
from pathlib import Path

import pytest
import yaml

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.event_bus.event_bus_inmemory import EventBusInmemory
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.runtime import bounded_delegation_routes as routes_module
from omnibase_infra.runtime.bounded_delegation_routes import (
    resolve_bounded_delegation_route,
)
from omnibase_infra.runtime.protocols.protocol_addressed_broker_transport import (
    ProtocolAddressedBrokerTransport,
)
from omnibase_infra.runtime.runtime_local_ingress import ModelRuntimeLocalIngressRoute

pytestmark = pytest.mark.unit

_DOGFOOD_BROKER = "192.0.2.10:47092"
_INTERNAL = "redpanda:9092"


class _AddressedBus:
    def __init__(
        self, environment: str = "dogfood", bootstrap_servers: str = _DOGFOOD_BROKER
    ) -> None:
        self.environment = environment
        self.bootstrap_servers = bootstrap_servers


def _route(**changes: object) -> ModelRuntimeLocalIngressRoute:
    fields: dict[str, object] = {
        "node_name": "node_delegation_orchestrator",
        "contract_name": "node_delegation_orchestrator",
        "command_topic": "onex.cmd.omnibase-infra.delegation-request.v1",
        "event_type": "omnimarket.delegation-request",
        "terminal_event": "onex.evt.omnibase-infra.delegation-completed.v1",
        "terminal_events": (
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
        ),
        "contract_path": "/contracts/node_delegation_orchestrator/contract.yaml",
        "package_name": "omnimarket",
    }
    fields.update(changes)
    return ModelRuntimeLocalIngressRoute(**fields)  # type: ignore[arg-type]


def _lanes(row_change: tuple[str, str] | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "consumer": "omnimarket.nodes.node_delegation_orchestrator",
        "terminal_route": "terminal_events",
        "repository_owner": "omnimarket",
    }
    if row_change is not None:
        row[row_change[0]] = row_change[1]
    return {
        "dogfood": {
            "broker": _DOGFOOD_BROKER,
            "security_protocol": "PLAINTEXT",
            "delegation_routes": [row],
        }
    }


def _write(tmp_path: Path, lanes: dict[str, object]) -> Path:
    path = tmp_path / "ci_bus_lanes.yaml"
    path.write_text(
        yaml.safe_dump({"default": "inmemory", "lanes": lanes}, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_kafka_bus_exposes_public_addressed_transport_identity() -> None:
    bus = EventBusKafka(
        config=ModelKafkaEventBusConfig(
            bootstrap_servers=_DOGFOOD_BROKER, environment="dogfood"
        )
    )

    assert isinstance(bus, ProtocolAddressedBrokerTransport)
    assert bus.bootstrap_servers == _DOGFOOD_BROKER
    assert bus.environment == "dogfood"


def test_in_process_bus_has_no_broker_to_bound() -> None:
    assert not isinstance(EventBusInmemory(), ProtocolAddressedBrokerTransport)


def test_declared_row_resolves_with_every_identity_the_decision_read(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _lanes())
    resolved = resolve_bounded_delegation_route(
        transport=_AddressedBus(),
        selected_route=_route(),
        overlay_path_for_test=path,
    )

    assert resolved is not None
    assert resolved.lane == "dogfood"
    assert resolved.broker == _DOGFOOD_BROKER
    assert resolved.runtime_environment == "dogfood"
    assert resolved.consumer == "omnimarket.nodes.node_delegation_orchestrator"
    assert resolved.repository_owner == "omnimarket"
    assert resolved.terminal_route == "terminal_events"
    assert resolved.command_topic == "onex.cmd.omnibase-infra.delegation-request.v1"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert resolved.declaration_sha256 == digest
    assert resolved.manifest_sha256 == digest


@pytest.mark.parametrize("field", ["environment", "bootstrap_servers"])
def test_empty_transport_identity_is_refused(tmp_path: Path, field: str) -> None:
    bus = _AddressedBus()
    setattr(bus, field, "  ")

    with pytest.raises(InfraUnavailableError, match="non-empty runtime environment"):
        resolve_bounded_delegation_route(
            transport=bus,
            selected_route=_route(),
            overlay_path_for_test=_write(tmp_path, _lanes()),
        )


@pytest.mark.parametrize(
    ("row_change", "route_changes", "message"),
    [
        (("repository_owner", "omnibase_infra"), {}, "repository owner mismatch"),
        (("consumer", "omnimarket.nodes.other"), {}, "consumer mismatch"),
        (("terminal_route", "wrong"), {}, "terminal route does not match"),
        (None, {"terminal_events": ("only-one",)}, "terminal route does not match"),
        (None, {"command_topic": " "}, "no command topic"),
        (None, {"package_name": "other_pkg"}, "consumer mismatch"),
    ],
)
def test_route_contract_mutations_are_refused(
    tmp_path: Path,
    row_change: tuple[str, str] | None,
    route_changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(InfraUnavailableError, match=message):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(),
            selected_route=_route(**route_changes),
            overlay_path_for_test=_write(tmp_path, _lanes(row_change)),
        )


@pytest.mark.parametrize("missing", ["consumer", "terminal_route", "repository_owner"])
def test_each_row_field_is_required(tmp_path: Path, missing: str) -> None:
    lanes = _lanes()
    row = lanes["dogfood"]["delegation_routes"][0]  # type: ignore[index]
    del row[missing]

    with pytest.raises(InfraUnavailableError, match="incomplete or invalid"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(),
            selected_route=_route(),
            overlay_path_for_test=_write(tmp_path, lanes),
        )


def test_two_rows_on_one_lane_are_refused(tmp_path: Path) -> None:
    lanes = _lanes()
    rows = lanes["dogfood"]["delegation_routes"]  # type: ignore[index]
    rows.append(dict(rows[0]))

    with pytest.raises(InfraUnavailableError, match="exactly one delegation route"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(),
            selected_route=_route(),
            overlay_path_for_test=_write(tmp_path, lanes),
        )


def test_declared_external_broker_under_another_environment_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(InfraUnavailableError, match="unexpected runtime environment"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(environment="prod"),
            selected_route=_route(),
            overlay_path_for_test=_write(tmp_path, _lanes()),
        )


def _with_topology(runtime_environment: str | None = "dogfood") -> dict[str, object]:
    lanes = _lanes()
    topology: dict[str, object] = {
        "external_bootstrap_servers": _DOGFOOD_BROKER,
        "internal_bootstrap_servers": _INTERNAL,
    }
    if runtime_environment is not None:
        topology["runtime_environment"] = runtime_environment
    lanes["dogfood"]["broker_topology"] = topology  # type: ignore[index]
    return lanes


@pytest.mark.parametrize(
    "environment", ["stability-test", "judge", "lakshman", "local"]
)
def test_shared_internal_listener_never_claims_a_lane_alone(
    tmp_path: Path, environment: str
) -> None:
    """Every compose lane's runtime reports ``redpanda:9092`` (read 2026-09-24).

    Only the declared (runtime_environment, internal listener) pair claims a
    lane; the listener alone would sweep stability-test, judge and lakshman in.
    """
    assert (
        resolve_bounded_delegation_route(
            transport=_AddressedBus(
                environment=environment, bootstrap_servers=_INTERNAL
            ),
            selected_route=_route(),
            overlay_path_for_test=_write(tmp_path, _with_topology()),
        )
        is None
    )


def test_declared_internal_pair_is_accepted(tmp_path: Path) -> None:
    resolved = resolve_bounded_delegation_route(
        transport=_AddressedBus(bootstrap_servers=_INTERNAL),
        selected_route=_route(),
        overlay_path_for_test=_write(tmp_path, _with_topology()),
    )

    assert resolved is not None
    assert resolved.broker == _DOGFOOD_BROKER
    assert resolved.runtime_bootstrap_servers == _INTERNAL


def test_internal_listener_without_a_declared_topology_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(InfraUnavailableError, match="broker mismatch"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(bootstrap_servers=_INTERNAL),
            selected_route=_route(),
            overlay_path_for_test=_write(tmp_path, _lanes()),
        )


def test_topology_external_identity_must_equal_the_lane_broker(
    tmp_path: Path,
) -> None:
    lanes = _with_topology()
    lanes["dogfood"]["broker_topology"]["external_bootstrap_servers"] = (  # type: ignore[index]
        "198.51.100.7:47092"
    )

    with pytest.raises(InfraUnavailableError, match="topology external identity"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(),
            selected_route=_route(),
            overlay_path_for_test=_write(tmp_path, lanes),
        )


@pytest.mark.parametrize(
    ("lane", "key"),
    [
        ("ci-bus", "delegation_routes"),
        ("prod", "delegation_routes"),
        ("ci-bus", "broker_topology"),
    ],
)
def test_ci_bus_and_prod_may_not_declare_a_route(
    tmp_path: Path, lane: str, key: str
) -> None:
    lanes = _lanes()
    value: object = (
        lanes["dogfood"]["delegation_routes"]  # type: ignore[index]
        if key == "delegation_routes"
        else {
            "external_bootstrap_servers": "a:1",
            "internal_bootstrap_servers": "b:2",
        }
    )
    lanes[lane] = {"broker": "inmemory", key: value}

    with pytest.raises(InfraUnavailableError, match="ci-bus is transport-only"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(),
            selected_route=_route(),
            overlay_path_for_test=_write(tmp_path, lanes),
        )


def _patch_packaged(
    monkeypatch: pytest.MonkeyPatch, result: tuple[bytes, str | None, str] | None
) -> None:
    monkeypatch.setattr(routes_module, "_read_packaged_overlay", lambda: result)


def _overlay_bytes() -> bytes:
    return yaml.safe_dump({"lanes": _lanes()}, sort_keys=False).encode("utf-8")


def test_vendor_bytes_that_disagree_with_the_manifest_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_packaged(monkeypatch, (_overlay_bytes(), "0" * 64, "omnimarket==test"))

    with pytest.raises(InfraUnavailableError, match="vendor/manifest mismatch"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(), selected_route=_route()
        )


def test_vendor_bytes_without_a_manifest_entry_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_packaged(monkeypatch, (_overlay_bytes(), None, "omnimarket==test"))

    with pytest.raises(InfraUnavailableError, match="no sha256 RECORD entry"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(), selected_route=_route()
        )


def test_matching_vendor_and_manifest_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _overlay_bytes()
    digest = hashlib.sha256(data).hexdigest()
    _patch_packaged(monkeypatch, (data, digest, "omnimarket==test"))

    resolved = resolve_bounded_delegation_route(
        transport=_AddressedBus(), selected_route=_route()
    )

    assert resolved is not None
    assert resolved.manifest_sha256 == digest
    assert resolved.declaration_source == "omnimarket==test"


def test_absent_overlay_refuses_a_lane_that_names_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_packaged(monkeypatch, None)

    with pytest.raises(InfraUnavailableError, match="not installed"):
        resolve_bounded_delegation_route(
            transport=_AddressedBus(), selected_route=_route()
        )


def test_absent_overlay_leaves_an_unnamed_runtime_unbounded_and_says_so(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_packaged(monkeypatch, None)

    with caplog.at_level(logging.WARNING, logger=routes_module.__name__):
        resolved = resolve_bounded_delegation_route(
            transport=_AddressedBus(
                environment="stability-test", bootstrap_servers=_INTERNAL
            ),
            selected_route=_route(),
        )

    assert resolved is None
    assert "gate not armed" in caplog.text


def test_record_hash_is_read_from_the_installed_manifest(tmp_path: Path) -> None:
    data = b"lanes: {}\n"
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    # importlib.metadata lists only RECORD entries that exist on disk.
    installed = tmp_path / "omnimarket" / "config" / "ci_bus_lanes.yaml"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(data)
    dist_info = tmp_path / "omnimarket-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: omnimarket\nVersion: 0.0.0\n", encoding="utf-8"
    )
    (dist_info / "RECORD").write_text(
        f"omnimarket/config/ci_bus_lanes.yaml,sha256={encoded},{len(data)}\n"
        "omnimarket-0.0.0.dist-info/RECORD,,\n",
        encoding="utf-8",
    )

    dist = metadata.PathDistribution(dist_info)

    assert routes_module._record_sha256_hex(dist) == hashlib.sha256(data).hexdigest()
