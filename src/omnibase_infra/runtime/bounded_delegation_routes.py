# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fail-closed validation for the explicitly bounded delegation lanes."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.nodes.node_chain_canary_effect.lane_transport import (
    load_lane_transport,
)
from omnibase_infra.runtime.model_bounded_delegation_route import (
    ModelBoundedDelegationRoute,
)
from omnibase_infra.runtime.model_bounded_lane_broker_topology import (
    ModelBoundedLaneBrokerTopology,
)
from omnibase_infra.runtime.protocol_addressed_broker_transport import (
    ProtocolAddressedBrokerTransport,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.runtime_local_ingress import (
        ModelRuntimeLocalIngressRoute,
    )

_BOUNDED_LANES = frozenset({"dev", "dogfood"})
_LANE_OVERLAY_PACKAGE = "omnimarket"
_LANE_OVERLAY_RESOURCE = "config/ci_bus_lanes.yaml"


def validate_bounded_lane_broker_identity(
    *,
    lane: str,
    lane_data: dict[str, object],
    runtime_bootstrap_servers: str,
) -> str:
    """Validate a runtime broker against the lane's declared exact topology.

    A plain lane declaration accepts only its literal ``broker``. A lane with an
    explicit ``broker_topology`` additionally accepts its declared internal
    listener, but only if the external member exactly equals ``broker``. This
    prevents a hostname alias or arbitrary broker from becoming a lane match.
    """

    runtime_broker = runtime_bootstrap_servers.strip()
    declared_broker = str(lane_data.get("broker") or "").strip()
    if not runtime_broker or not declared_broker:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} requires non-empty broker identity"
        )
    if runtime_broker == declared_broker:
        return declared_broker

    raw_topology = lane_data.get("broker_topology")
    if not isinstance(raw_topology, dict):
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} broker mismatch: runtime bus "
            f"uses {runtime_broker!r}, declaration resolves to {declared_broker!r}"
        )
    try:
        topology = ModelBoundedLaneBrokerTopology.model_validate(raw_topology)
    except ValidationError as exc:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} broker topology is invalid"
        ) from exc
    if topology.external_bootstrap_servers != declared_broker:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} topology external identity does "
            "not equal its declared broker"
        )
    if runtime_broker != topology.internal_bootstrap_servers:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} runtime broker {runtime_broker!r} "
            "is not its declared internal topology identity"
        )
    return declared_broker


def _overlay_text(overlay_path_for_test: Path | None) -> str:
    if overlay_path_for_test is not None:
        try:
            return overlay_path_for_test.read_text(encoding="utf-8")
        except OSError as exc:
            raise InfraUnavailableError(
                "cannot read bounded delegation lane fixture "
                f"{overlay_path_for_test}: {exc}"
            ) from exc

    try:
        resource = resources.files(_LANE_OVERLAY_PACKAGE).joinpath(
            *_LANE_OVERLAY_RESOURCE.split("/")
        )
        return resource.read_text(encoding="utf-8")
    except (ModuleNotFoundError, OSError) as exc:
        raise InfraUnavailableError(
            "bounded delegation lane overlay is not installed as an "
            "omnimarket package resource"
        ) from exc


def resolve_bounded_delegation_route(
    *,
    transport: ProtocolAddressedBrokerTransport,
    selected_route: ModelRuntimeLocalIngressRoute,
    overlay_path_for_test: Path | None = None,
) -> ModelBoundedDelegationRoute | None:
    """Validate a selected contract against the declared dev/dogfood lane.

    Production dispatch always reads the installed Market package resource;
    ``overlay_path_for_test`` is only for isolated mutation tests.

    Other runtime environments are intentionally outside this bounded gate.
    On either named lane, missing or contradictory declarations are refusals.
    The lane overlay owns the broker; the installed consumer contract owns its
    command and terminal topics.
    """

    lane = transport.environment.strip().lower()
    broker = transport.bootstrap_servers.strip()
    if not lane or not broker:
        raise InfraUnavailableError(
            "bounded delegation route requires non-empty runtime environment "
            "and broker identity"
        )

    try:
        raw: object = yaml.safe_load(_overlay_text(overlay_path_for_test))
    except yaml.YAMLError as exc:
        raise InfraUnavailableError(
            f"bounded delegation lane overlay is invalid YAML: {exc}"
        ) from exc

    lanes = raw.get("lanes") if isinstance(raw, dict) else None
    # Only a lane's declared external broker classifies a foreign environment.
    # A topology's internal member is a compose-network name (``redpanda:9092``)
    # that every compose lane shares, so it can confirm a lane that names
    # itself, which validate_bounded_lane_broker_identity does below, but it
    # never identifies one.
    targeted_brokers: set[str] = set()
    if isinstance(lanes, dict):
        for name in _BOUNDED_LANES:
            lane_data = lanes.get(name)
            if not isinstance(lane_data, dict):
                continue
            declared_broker = str(lane_data.get("broker") or "").strip()
            if declared_broker:
                targeted_brokers.add(declared_broker)
    if lane not in _BOUNDED_LANES and broker not in targeted_brokers:
        return None
    if lane not in _BOUNDED_LANES:
        raise InfraUnavailableError(
            "a broker declared for a bounded delegation lane was selected "
            f"under unexpected runtime environment {lane!r}"
        )
    lane_data = lanes.get(lane) if isinstance(lanes, dict) else None
    if not isinstance(lane_data, dict):
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} has no lane declaration"
        )

    if overlay_path_for_test is None:
        # The packaged resource is read directly above; re-use the same typed
        # lane parser through a short-lived materialized resource path.
        resource = resources.files(_LANE_OVERLAY_PACKAGE).joinpath(
            *_LANE_OVERLAY_RESOURCE.split("/")
        )
        with resources.as_file(resource) as packaged_path:
            lane_transport = load_lane_transport(packaged_path, lane)
    else:
        lane_transport = load_lane_transport(overlay_path_for_test, lane)

    validate_bounded_lane_broker_identity(
        lane=lane,
        lane_data=lane_data,
        runtime_bootstrap_servers=transport.bootstrap_servers,
    )

    declarations = lane_data.get("delegation_routes")
    if not isinstance(declarations, list) or len(declarations) != 1:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} must declare exactly one "
            "delegation route"
        )
    declaration = declarations[0]
    if not isinstance(declaration, dict):
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} route declaration is not a mapping"
        )

    try:
        resolved = ModelBoundedDelegationRoute.model_validate(
            {
                "lane": lane,
                "broker": lane_transport.bootstrap_servers,
                **declaration,
            }
        )
    except ValueError as exc:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} route declaration is incomplete "
            f"or invalid: {exc}"
        ) from exc
    actual_consumer = (
        f"{selected_route.package_name}.nodes.{selected_route.contract_name}"
    )
    if resolved.consumer != actual_consumer:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} consumer mismatch: declaration "
            f"names {resolved.consumer!r}, selected contract is {actual_consumer!r}"
        )
    if resolved.repository_owner != selected_route.package_name:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} repository owner mismatch: "
            f"declaration names {resolved.repository_owner!r}, selected package "
            f"is {selected_route.package_name!r}"
        )
    if (
        resolved.terminal_route != "terminal_events"
        or len(selected_route.terminal_events) < 2
    ):
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} terminal route does not match "
            "the selected contract's terminal_events"
        )
    if not selected_route.command_topic.strip():
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} selected contract has no command topic"
        )
    return resolved


__all__ = [
    "resolve_bounded_delegation_route",
    "validate_bounded_lane_broker_identity",
]
