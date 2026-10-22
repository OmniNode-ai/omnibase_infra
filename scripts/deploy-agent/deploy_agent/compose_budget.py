# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Derive the Phase.RUNTIME compose-up ceiling from the compose model (OMN-18057).

WHY THIS EXISTS
---------------

``PHASE_TIMEOUTS[Phase.RUNTIME]`` was the bare constant ``300``. The compose
model it was supposed to bound declares ``start_period: 1800s`` on
``omninode-runtime``, and ``runtime-effects``, ``runtime-worker`` and
``omninode-contract-resolver`` all gate on it with ``depends_on: {condition:
service_healthy}``. ``docker compose up`` honours that condition for every
selected service, so the command cannot finish before the gating dependency
reports healthy -- a contract SIX TIMES larger than the ceiling that killed it.

MEASURED, 2026-09-08 (``docs/tracking/ROLLING_WORK_LEDGER.md:5076``): the
runtime's own log recorded ``Bootstrap time: 249.600s`` and the lane first bound
:8085 at t+321s. The 300s ceiling fired 21 seconds early, leaving three services
in ``Created``, :8086 down and the OCC mint down fleet-wide.

A second bare constant would have the same defect one number later. The ceiling
is therefore READ from the same compose files the deploy actually invokes:

    ceiling = max(floor, max(start_period over gating services) + margin)

and the derivation names the service it came from, so a later compose change
moves the ceiling with it instead of silently re-opening this failure.

WHAT THE MARGIN COVERS
----------------------

``start_period`` is the window in which a failing healthcheck is not yet counted
against the container; compose stops waiting when the service reports healthy
OR unhealthy, and the unhealthy verdict lands at ``start_period + interval *
retries`` -- 1800s + 5*30s = 1950s for the runtime family, a number the compose
file states in its own comment. The margin must therefore cover that detection
tail plus the stop/create/start of the rest of the selected set. It is a single
declared constant, not a per-service guess.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Seconds per compose duration unit. Compose durations are unit-suffixed
# ("1800s", "30m", "1h30m"); a bare number is refused rather than guessed at,
# because docker reads an unsuffixed healthcheck duration as NANOSECONDS and a
# silent misread here is the whole class of defect this module closes.
_DURATION_UNITS: dict[str, float] = {
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}
_DURATION_TOKEN = re.compile(r"(\d+(?:\.\d+)?)(us|ms|s|m|h)")
_DURATION_FULL = re.compile(r"^(?:\d+(?:\.\d+)?(?:us|ms|s|m|h))+$")


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that understands Compose's merge tags.

    ``docker-compose.dev-lane.yml`` uses ``!override`` (and the spec also
    defines ``!reset``) on overlay keys. ``yaml.safe_load`` refuses an unknown
    tag outright, which would make the ceiling derivation unreadable on exactly
    the lane it matters most for. The tags are resolved the way compose
    resolves them: ``!override`` replaces the tagged value rather than merging
    its mapping, ``!reset`` yields ``None`` (the key is removed).
    """


class _ComposeOverrideMap(dict[str, Any]):
    """A mapping Compose explicitly replaces instead of merging."""


def _compose_tag_constructor(
    loader: _ComposeLoader, tag_suffix: str, node: yaml.Node
) -> Any:
    if tag_suffix == "reset":
        return None
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        mapping = loader.construct_mapping(node, deep=True)
        return (
            _ComposeOverrideMap({str(key): value for key, value in mapping.items()})
            if tag_suffix == "override"
            else mapping
        )
    return None


# OMN-18640: registered through the module-level function, which the yaml
# stubs annotate, rather than the classmethod, which they do not. Identical
# effect -- it forwards to the same registry on the Loader named here.
yaml.add_multi_constructor("!", _compose_tag_constructor, Loader=_ComposeLoader)


class ComposeDurationError(ValueError):
    """Raised when a compose duration cannot be read exactly."""


class ComposeBudgetError(RuntimeError):
    """Raised when the compose model cannot be read to derive a ceiling."""


class ModelPhaseBudget(BaseModel):
    """A derived phase ceiling and the compose facts it was derived from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_seconds: int
    floor_seconds: int
    margin_seconds: int
    source_service: str | None
    source_start_period_seconds: int
    gating_services: tuple[str, ...]
    compose_files: tuple[str, ...]

    def describe(self) -> str:
        """One-line, log-ready statement of the ceiling and its source."""
        if self.source_service is None:
            return (
                f"{self.timeout_seconds}s (floor {self.floor_seconds}s; no gated "
                f"service in the compose model declares a healthcheck start_period)"
            )
        return (
            f"{self.timeout_seconds}s = start_period "
            f"{self.source_start_period_seconds}s of {self.source_service!r} "
            f"(gated on via depends_on condition: service_healthy) "
            f"+ margin {self.margin_seconds}s, floor {self.floor_seconds}s"
        )


def parse_compose_duration(value: object) -> int:
    """Return whole seconds for a compose duration string such as ``1800s``.

    Rounds UP to the next whole second: a ceiling derived from a duration must
    never come out below the duration it is derived from.
    """
    if isinstance(value, bool) or not isinstance(value, str):
        raise ComposeDurationError(
            f"compose duration must be a unit-suffixed string (e.g. '1800s'); got {value!r}"
        )
    text = value.strip()
    if not _DURATION_FULL.match(text):
        raise ComposeDurationError(
            f"{text!r} is not a compose duration; expected unit-suffixed "
            f"components from {sorted(_DURATION_UNITS)} (e.g. '1800s', '1h30m')"
        )
    total = 0.0
    for amount, unit in _DURATION_TOKEN.findall(text):
        total += float(amount) * _DURATION_UNITS[unit]
    return int(-(-total // 1))  # ceil, without importing math for one call


def _merge_service_maps(documents: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Merge each compose file's ``services`` map, later files winning per key.

    Healthcheck mappings merge per key unless explicitly tagged !override.
    Other keys retain the phase-budget reader's replacement semantics.
    """
    merged: dict[str, dict[str, Any]] = {}
    for document in documents:
        services = document.get("services") or {}
        if not isinstance(services, dict):
            continue
        for name, spec in services.items():
            if not isinstance(spec, dict):
                continue
            previous = merged.setdefault(str(name), {})
            health = spec.get("healthcheck")
            inherited_health = previous.get("healthcheck")
            if (
                isinstance(health, dict)
                and not isinstance(health, _ComposeOverrideMap)
                and isinstance(inherited_health, dict)
            ):
                spec = {**spec, "healthcheck": {**inherited_health, **health}}
            previous.update(spec)
    return merged


def _services_gated_on_health(services: dict[str, dict[str, Any]]) -> set[str]:
    """Return every service some other service waits on via ``service_healthy``."""
    gated: set[str] = set()
    for spec in services.values():
        depends_on = spec.get("depends_on")
        if not isinstance(depends_on, dict):
            # The list form carries no condition, so it never blocks on health.
            continue
        for dependency, condition in depends_on.items():
            if (
                isinstance(condition, dict)
                and condition.get("condition") == "service_healthy"
            ):
                gated.add(str(dependency))
    return gated


def _read_compose_services(
    compose_files: tuple[str, ...] | list[str],
) -> dict[str, dict[str, Any]]:
    """Read the lane's compose model, refusing an unreadable declaration."""
    documents: list[dict[str, Any]] = []
    for compose_file in compose_files:
        path = Path(compose_file)
        try:
            document = yaml.load(
                path.read_text(encoding="utf-8"),
                # S506 is about arbitrary object construction. _ComposeLoader
                # subclasses SafeLoader and adds exactly one multi-constructor,
                # for the "!" tag family, which returns plain scalars/lists/dicts
                # and constructs nothing. Compose's own !override/!reset tags
                # cannot be read by safe_load, and refusing to read the dev-lane
                # overlay would make the ceiling underivable on the lane it
                # matters most for.
                Loader=_ComposeLoader,  # noqa: S506
            )
        except OSError as exc:
            raise ComposeBudgetError(
                f"cannot read compose file {path} to derive the runtime phase "
                f"ceiling: {exc}"
            ) from exc
        except yaml.YAMLError as exc:
            raise ComposeBudgetError(
                f"cannot parse compose file {path} to derive the runtime phase "
                f"ceiling: {exc}"
            ) from exc
        if isinstance(document, dict):
            documents.append(document)

    return _merge_service_maps(documents)


def derive_runtime_health_budget(
    compose_files: tuple[str, ...] | list[str],
    service: str,
    *,
    floor_seconds: int,
) -> int:
    """Bound readiness by the target's own healthcheck, including its failure tail.

    Unlike compose up, a readiness probe must wait for the selected service even
    when no dependent service gates on it. Include each check's timeout as well
    as its interval, matching runtime_health_wait.sh's conservative bound.
    """
    services = _read_compose_services(compose_files)
    if service not in services:
        raise ComposeBudgetError(
            f"health probe service {service!r} is absent from compose"
        )
    health = services[service].get("healthcheck") or {}
    if not isinstance(health, dict):
        raise ComposeBudgetError(f"healthcheck for {service!r} must be a mapping")
    if not health or health.get("disable") is True or health.get("test") == ["NONE"]:
        return floor_seconds
    retries = health.get("retries", 3)
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ComposeBudgetError(
            f"healthcheck retries for {service!r} must be nonnegative"
        )
    start = parse_compose_duration(health.get("start_period", "0s"))
    interval = parse_compose_duration(health.get("interval", "30s")) or 30
    timeout = parse_compose_duration(health.get("timeout", "30s")) or 30
    return max(floor_seconds, start + (retries or 3) * (interval + timeout))


def derive_runtime_phase_budget(
    compose_files: tuple[str, ...] | list[str],
    scope_services: tuple[str, ...] | list[str],
    *,
    margin_seconds: int,
    floor_seconds: int,
) -> ModelPhaseBudget:
    """Derive compose up's ceiling from health-gated services within its scope.

    Unreadable compose files raise ComposeBudgetError rather than silently
    reverting to the floor.
    """
    services = _read_compose_services(compose_files)
    gated = _services_gated_on_health(services)
    in_scope = [name for name in scope_services if name in gated]

    source_service: str | None = None
    source_start_period = 0
    for name in in_scope:
        healthcheck = services.get(name, {}).get("healthcheck")
        if not isinstance(healthcheck, dict):
            continue
        start_period = healthcheck.get("start_period")
        if start_period is None:
            continue
        seconds = parse_compose_duration(start_period)
        if seconds > source_start_period:
            source_start_period = seconds
            source_service = name

    derived = source_start_period + margin_seconds if source_service else 0
    return ModelPhaseBudget(
        timeout_seconds=max(floor_seconds, derived),
        floor_seconds=floor_seconds,
        margin_seconds=margin_seconds,
        source_service=source_service,
        source_start_period_seconds=source_start_period,
        gating_services=tuple(in_scope),
        compose_files=tuple(str(f) for f in compose_files),
    )
