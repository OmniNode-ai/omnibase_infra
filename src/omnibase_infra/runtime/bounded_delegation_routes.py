# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pre-dispatch validation of the bounded delegation lanes (OMN-18933, K6).

Carried from the Codex draft omnibase_infra#3951 (OMN-18925 capture root) and
reworked for three facts read live on 2026-09-24:

* The .201 dev lane's runtime reports ``ONEX_ENVIRONMENT=local`` on the compose
  listener ``redpanda:9092``. Every compose lane on .201 and the .105 dogfood
  lane report the same internal listener, so neither the lane key ``dev`` nor the
  listener identifies a lane. A lane is claimed by its name, by its declared
  ``(runtime_environment, internal_bootstrap_servers)`` pair, or by its declared
  external broker, and by nothing looser.
* omnimarket is installed in the runtime image as a built wheel, so the lane
  overlay it packages has a RECORD entry. The installed bytes (the vendored copy
  this runtime acts on) must hash to that manifest entry, or the decision refuses.
* Only ``dev`` and ``dogfood`` (the authorized isolated lab) are bounded. A
  ``delegation_routes`` row or ``broker_topology`` on any other lane is refused:
  ``ci-bus`` is transport-only and ``prod`` is excluded.

On a bounded lane every missing or contradictory declaration refuses BEFORE the
broker is constructed, so a refusal publishes no command and can produce no
terminal and no projection row. A runtime that claims no bounded lane is outside
this gate and returns ``None``.
"""

from __future__ import annotations

import base64
import hashlib
import logging
from importlib import metadata, resources
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import ValidationError

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.runtime.models.model_bounded_delegation_route import (
    ModelBoundedDelegationRoute,
)
from omnibase_infra.runtime.models.model_bounded_delegation_route_declaration import (
    ModelBoundedDelegationRouteDeclaration,
)
from omnibase_infra.runtime.models.model_bounded_lane_broker_topology import (
    ModelBoundedLaneBrokerTopology,
)
from omnibase_infra.runtime.protocols.protocol_addressed_broker_transport import (
    ProtocolAddressedBrokerTransport,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.runtime_local_ingress import (
        ModelRuntimeLocalIngressRoute,
    )

logger = logging.getLogger(__name__)

BOUNDED_DELEGATION_LANES: tuple[str, ...] = ("dev", "dogfood")
_TERMINAL_ROUTE = "terminal_events"
_OVERLAY_DISTRIBUTION = "omnimarket"
_OVERLAY_PACKAGE_PATH = ("config", "ci_bus_lanes.yaml")
_OVERLAY_RECORD_PATH = "omnimarket/config/ci_bus_lanes.yaml"


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record_sha256_hex(dist: metadata.Distribution) -> str | None:
    """Return the RECORD sha256 of the packaged overlay, hex, or None if absent."""
    for packaged in dist.files or ():
        if str(packaged).replace("\\", "/") != _OVERLAY_RECORD_PATH:
            continue
        file_hash = packaged.hash
        if file_hash is None or file_hash.mode != "sha256":
            return None
        padded = file_hash.value + "=" * (-len(file_hash.value) % 4)
        return base64.urlsafe_b64decode(padded).hex()
    return None


def _read_packaged_overlay() -> tuple[bytes, str | None, str] | None:
    """Read the installed omnimarket lane overlay, or None when not packaged."""
    try:
        dist = metadata.distribution(_OVERLAY_DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None
    try:
        resource = resources.files(_OVERLAY_DISTRIBUTION).joinpath(
            *_OVERLAY_PACKAGE_PATH
        )
        if not resource.is_file():
            return None
        data = resource.read_bytes()
    except (ModuleNotFoundError, OSError):
        return None
    direct_url = dist.read_text("direct_url.json") or ""
    source = f"{_OVERLAY_DISTRIBUTION}=={dist.version} {_OVERLAY_RECORD_PATH}"
    if direct_url:
        source = f"{source} direct_url={direct_url.strip()}"
    return data, _record_sha256_hex(dist), source


def _read_overlay(
    overlay_path_for_test: Path | None,
) -> tuple[bytes, str | None, str] | None:
    if overlay_path_for_test is None:
        return _read_packaged_overlay()
    try:
        data = overlay_path_for_test.read_bytes()
    except OSError as exc:
        raise InfraUnavailableError(
            f"cannot read bounded delegation lane fixture {overlay_path_for_test}: "
            f"{exc}"
        ) from exc
    # A fixture has no installed RECORD; its own bytes stand in as the manifest
    # so the equality check below still runs on every path.
    return data, _sha256_hex(data), f"fixture {overlay_path_for_test}"


def _parse_topology(
    lane: str, lane_data: dict[str, object]
) -> ModelBoundedLaneBrokerTopology | None:
    raw = lane_data.get("broker_topology")
    if raw is None:
        return None
    try:
        return ModelBoundedLaneBrokerTopology.model_validate(raw)
    except ValidationError as exc:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} broker topology is invalid: {exc}"
        ) from exc


def _claimed_lane(
    *, environment: str, broker: str, lanes: dict[str, object]
) -> str | None:
    """Return the bounded lane this runtime identity claims, or None."""
    if environment in BOUNDED_DELEGATION_LANES:
        return environment
    for name in BOUNDED_DELEGATION_LANES:
        lane_data = lanes.get(name)
        if not isinstance(lane_data, dict):
            continue
        topology = _parse_topology(name, lane_data)
        if (
            topology is not None
            and topology.runtime_environment == environment
            and topology.internal_bootstrap_servers == broker
        ):
            return name
    for name in BOUNDED_DELEGATION_LANES:
        lane_data = lanes.get(name)
        if not isinstance(lane_data, dict):
            continue
        # Only a declared EXTERNAL broker classifies by address. The internal
        # listener (redpanda:9092) is shared by every compose lane, so it can
        # confirm a lane already claimed above but never claims one alone.
        if str(lane_data.get("broker") or "").strip() == broker:
            return name
    return None


def _refuse_unbounded_declarations(lanes: dict[str, object]) -> None:
    for name, lane_data in lanes.items():
        if name in BOUNDED_DELEGATION_LANES or not isinstance(lane_data, dict):
            continue
        for key in ("delegation_routes", "broker_topology"):
            if lane_data.get(key):
                raise InfraUnavailableError(
                    f"lane {name!r} declares {key}; only "
                    f"{', '.join(BOUNDED_DELEGATION_LANES)} are bounded delegation "
                    "lanes (ci-bus is transport-only and prod is excluded)"
                )


def resolve_bounded_delegation_route(
    *,
    transport: ProtocolAddressedBrokerTransport,
    selected_route: ModelRuntimeLocalIngressRoute,
    overlay_path_for_test: Path | None = None,
) -> ModelBoundedDelegationRoute | None:
    """Validate the selected delegation contract against its declared lane row.

    Returns the resolved route when the runtime identity claims a bounded lane
    and every declaration agrees; ``None`` when it claims none. Raises
    ``InfraUnavailableError`` on any refusal. ``overlay_path_for_test`` replaces
    the installed omnimarket resource for isolated tests only.
    """

    environment = transport.environment.strip()
    broker = transport.bootstrap_servers.strip()
    if not environment or not broker:
        raise InfraUnavailableError(
            "bounded delegation route requires non-empty runtime environment "
            "and broker identity"
        )

    overlay = _read_overlay(overlay_path_for_test)
    if overlay is None:
        if environment in BOUNDED_DELEGATION_LANES:
            raise InfraUnavailableError(
                f"bounded delegation lane {environment!r} cannot be validated: the "
                f"{_OVERLAY_DISTRIBUTION} lane overlay is not installed as a "
                "package resource"
            )
        logger.warning(
            "bounded delegation route gate not armed: the %s lane overlay is not "
            "installed as a package resource (runtime environment=%s broker=%s)",
            _OVERLAY_DISTRIBUTION,
            environment,
            broker,
        )
        return None
    data, manifest_sha256, source = overlay

    try:
        raw: object = yaml.safe_load(data.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise InfraUnavailableError(
            f"bounded delegation lane overlay is not valid YAML: {exc}"
        ) from exc
    lanes = raw.get("lanes") if isinstance(raw, dict) else None
    if not isinstance(lanes, dict):
        raise InfraUnavailableError(
            "bounded delegation lane overlay declares no lanes mapping"
        )
    _refuse_unbounded_declarations(lanes)

    lane = _claimed_lane(environment=environment, broker=broker, lanes=lanes)
    if lane is None:
        return None
    lane_data = lanes.get(lane)
    if not isinstance(lane_data, dict):
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} has no lane declaration"
        )

    declaration_sha256 = _sha256_hex(data)
    if manifest_sha256 is None:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r}: the installed overlay "
            f"{_OVERLAY_RECORD_PATH} has no sha256 RECORD entry, so its bytes "
            "cannot be bound to the build manifest"
        )
    if manifest_sha256 != declaration_sha256:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} vendor/manifest mismatch: installed "
            f"overlay sha256 {declaration_sha256} != manifest sha256 "
            f"{manifest_sha256}"
        )

    declared_broker = str(lane_data.get("broker") or "").strip()
    if not declared_broker:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} declares no broker"
        )
    topology = _parse_topology(lane, lane_data)
    if topology is not None and topology.external_bootstrap_servers != declared_broker:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} topology external identity "
            f"{topology.external_bootstrap_servers!r} is not its declared broker "
            f"{declared_broker!r}"
        )
    allowed_environments = {lane}
    if topology is not None and topology.runtime_environment is not None:
        allowed_environments.add(topology.runtime_environment)
    if environment not in allowed_environments:
        raise InfraUnavailableError(
            f"a broker declared for bounded delegation lane {lane!r} was selected "
            f"under unexpected runtime environment {environment!r}"
        )
    broker_matches = broker == declared_broker or (
        topology is not None and broker == topology.internal_bootstrap_servers
    )
    if not broker_matches:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} broker mismatch: runtime bus uses "
            f"{broker!r}, declaration names {declared_broker!r}"
            + (
                f" (internal {topology.internal_bootstrap_servers!r})"
                if topology is not None
                else ""
            )
        )

    rows = lane_data.get("delegation_routes")
    if not isinstance(rows, list) or len(rows) != 1:
        found = len(rows) if isinstance(rows, list) else 0
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} must declare exactly one delegation "
            f"route row; found {found}"
        )
    try:
        row = ModelBoundedDelegationRouteDeclaration.model_validate(rows[0])
    except ValidationError as exc:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} route row is incomplete or "
            f"invalid: {exc}"
        ) from exc

    actual_consumer = (
        f"{selected_route.package_name}.nodes.{selected_route.contract_name}"
    )
    if row.consumer != actual_consumer:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} consumer mismatch: row names "
            f"{row.consumer!r}, selected contract is {actual_consumer!r}"
        )
    if row.repository_owner != selected_route.package_name:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} repository owner mismatch: row names "
            f"{row.repository_owner!r}, selected package is "
            f"{selected_route.package_name!r}"
        )
    if row.terminal_route != _TERMINAL_ROUTE or len(selected_route.terminal_events) < 2:
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} terminal route does not match the "
            "selected contract's terminal_events"
        )
    if not selected_route.command_topic.strip():
        raise InfraUnavailableError(
            f"bounded delegation lane {lane!r} selected contract has no command topic"
        )

    return ModelBoundedDelegationRoute(
        lane=lane,
        broker=declared_broker,
        runtime_environment=environment,
        runtime_bootstrap_servers=broker,
        consumer=row.consumer,
        terminal_route=row.terminal_route,
        repository_owner=row.repository_owner,
        command_topic=selected_route.command_topic,
        terminal_events=tuple(selected_route.terminal_events),
        declaration_sha256=declaration_sha256,
        manifest_sha256=manifest_sha256,
        declaration_source=source,
    )


__all__ = [
    "BOUNDED_DELEGATION_LANES",
    "resolve_bounded_delegation_route",
]
