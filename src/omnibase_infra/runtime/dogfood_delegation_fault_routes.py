# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Typed, fail-closed dogfood-only provider fault route declarations."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from omnibase_infra.errors import InfraUnavailableError
from omnibase_infra.runtime.bounded_delegation_routes import (
    validate_bounded_lane_broker_identity,
)

_LANE_RESOURCE_PACKAGE: Final = "omnimarket"
_LANE_RESOURCE_PATH: Final = "config/ci_bus_lanes.yaml"
_DOGFOOD_LANE: Final = "dogfood"
_ALLOWED_STATUSES: Final = frozenset({429, 503})


class ModelDogfoodDelegationFaultRoute(BaseModel):
    """One no-secret, single-hop fault backend declared for dogfood only."""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    backend_id: str = Field(min_length=1)
    endpoint_url: str = Field(min_length=1)
    expected_http_status: int
    requested_timeout_seconds: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    no_escalation: bool

    @model_validator(mode="after")
    def _validate_single_hop_fault_route(self) -> ModelDogfoodDelegationFaultRoute:
        if self.expected_http_status not in _ALLOWED_STATUSES:
            raise ValueError("dogfood fault route must return exactly HTTP 429 or 503")
        if self.requested_timeout_seconds != 240:
            raise ValueError("dogfood fault route timeout must be exactly 240 seconds")
        if self.max_attempts != 1 or not self.no_escalation:
            raise ValueError(
                "dogfood fault route must be exactly one attempt with no escalation"
            )
        parsed = urlsplit(self.endpoint_url)
        expected_host = f"dogfood-delegation-fault-{self.expected_http_status}"
        if (
            parsed.scheme != "http"
            or parsed.hostname != expected_host
            or parsed.port != 8080
            or parsed.path != "/v1/chat/completions"
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "dogfood fault route endpoint must be its internal fixed-status service"
            )
        return self


def _load_lane_document(path_for_test: Path | None = None) -> object:
    try:
        if path_for_test is not None:
            content = path_for_test.read_text(encoding="utf-8")
        else:
            resource = resources.files(_LANE_RESOURCE_PACKAGE).joinpath(
                *_LANE_RESOURCE_PATH.split("/")
            )
            content = resource.read_text(encoding="utf-8")
        return yaml.safe_load(content)
    except (ModuleNotFoundError, OSError, yaml.YAMLError) as exc:
        raise InfraUnavailableError(
            "cannot load the packaged dogfood delegation fault route declaration"
        ) from exc


def load_dogfood_delegation_fault_routes(
    *, path_for_test: Path | None = None
) -> tuple[ModelDogfoodDelegationFaultRoute, ...]:
    """Load the dogfood-only fault rows from the packaged Market authority.

    This is the one endpoint/policy authority used by both the admission guard
    and the Bifrost renderer. It intentionally does not accept runtime identity:
    identity is a consumer-boundary concern, while render-time derivation only
    materializes the declared dogfood rows.
    """
    raw = _load_lane_document(path_for_test)
    lanes = raw.get("lanes") if isinstance(raw, dict) else None
    dogfood = lanes.get(_DOGFOOD_LANE) if isinstance(lanes, dict) else None
    if not isinstance(dogfood, dict):
        raise InfraUnavailableError("dogfood lane declaration is absent")
    raw_routes = dogfood.get("delegation_fault_routes")
    if not isinstance(raw_routes, list) or not raw_routes:
        raise InfraUnavailableError("dogfood fault route declaration is absent")

    routes: list[ModelDogfoodDelegationFaultRoute] = []
    backend_ids: set[str] = set()
    for raw_route in raw_routes:
        try:
            route = ModelDogfoodDelegationFaultRoute.model_validate(raw_route)
        except ValidationError as exc:
            raise InfraUnavailableError(
                "dogfood fault route declaration is invalid"
            ) from exc
        if route.backend_id in backend_ids:
            raise InfraUnavailableError("dogfood fault route backend_id is duplicated")
        backend_ids.add(route.backend_id)
        routes.append(route)
    return tuple(routes)


def validate_dogfood_delegation_fault_request(
    *,
    request: object,
    event_bus: object | None,
    path_for_test: Path | None = None,
) -> None:
    """Accept only a declared dogfood fault pin on the trusted consumer bus.

    Requests without a pin retain the ordinary delegation route. A pinned request
    is validated at the consumer boundary as well as the producer port, so a raw
    broker record cannot turn the caller-controlled pin or no-escalation flag
    into a retry-policy bypass.
    """

    backend_id = getattr(request, "backend_id", None)
    no_escalation = getattr(request, "no_escalation", False)
    if backend_id is None:
        if no_escalation:
            raise InfraUnavailableError(
                "no-escalation delegation request requires a declared backend pin"
            )
        return
    if not isinstance(backend_id, str) or not backend_id.strip():
        raise InfraUnavailableError("delegation backend pin must be a non-empty string")
    if event_bus is None:
        raise InfraUnavailableError(
            "dogfood fault backend pin requires trusted runtime bus identity"
        )
    environment = getattr(event_bus, "environment", None)
    bootstrap_servers = getattr(event_bus, "bootstrap_servers", None)
    if not isinstance(environment, str) or not isinstance(bootstrap_servers, str):
        raise InfraUnavailableError(
            "dogfood fault backend pin requires trusted runtime bus identity"
        )
    route = resolve_dogfood_delegation_fault_route(
        environment=environment,
        bootstrap_servers=bootstrap_servers,
        backend_id=backend_id,
        path_for_test=path_for_test,
    )
    if no_escalation is not True:
        raise InfraUnavailableError(
            "declared dogfood fault backend pin requires no-escalation policy"
        )
    requested_timeout_seconds = getattr(request, "requested_timeout_seconds", None)
    if requested_timeout_seconds != route.requested_timeout_seconds:
        raise InfraUnavailableError(
            "declared dogfood fault backend pin requires its exact timeout policy"
        )


def resolve_dogfood_delegation_fault_route(
    *,
    environment: str,
    bootstrap_servers: str,
    backend_id: str,
    path_for_test: Path | None = None,
) -> ModelDogfoodDelegationFaultRoute:
    """Resolve a pinned fault route only on its declared dogfood broker.

    The declaration lives with the packaged Market lane resource; callers do not
    carry an allowlist or endpoint literals. Any non-dogfood environment, broker
    mismatch, missing identity, or unknown backend is a refusal.
    """
    lane = environment.strip().lower()
    broker = bootstrap_servers.strip()
    if lane != _DOGFOOD_LANE or not broker:
        raise InfraUnavailableError(
            "dogfood fault backend pins require dogfood lane and broker identity"
        )
    raw = _load_lane_document(path_for_test)
    lanes = raw.get("lanes") if isinstance(raw, dict) else None
    dogfood = lanes.get(_DOGFOOD_LANE) if isinstance(lanes, dict) else None
    if not isinstance(dogfood, dict):
        raise InfraUnavailableError("dogfood lane declaration is absent")
    validate_bounded_lane_broker_identity(
        lane=_DOGFOOD_LANE,
        lane_data=dogfood,
        runtime_bootstrap_servers=broker,
    )
    for route in load_dogfood_delegation_fault_routes(path_for_test=path_for_test):
        if route.backend_id == backend_id:
            return route
    raise InfraUnavailableError("backend pin is not a declared dogfood fault route")


__all__ = [
    "ModelDogfoodDelegationFaultRoute",
    "load_dogfood_delegation_fault_routes",
    "resolve_dogfood_delegation_fault_route",
    "validate_dogfood_delegation_fault_request",
]
