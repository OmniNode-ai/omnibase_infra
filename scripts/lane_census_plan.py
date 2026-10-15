# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""lane_census_plan.py — Pure desired-vs-actual lane reconciler (OMN-13011).

THE CLASS FIX for the recurring lane-drift regression. Nothing reconciled the
declared desired state of a runtime lane against what is actually running, so the
same failures kept recurring with zero signal:

  - volume config drift (OMN-12945 family)
  - WORKER_REPLICAS silent zero — worker scaled to 0, no alert (OMN-12988/12990)
  - 2026-06-11: prod runtime containers + broker network silently absent for hours

This module is a *pure* planner: it takes the versioned lane manifest
(config/lab/lane-census/lane-manifest.yaml under the workspace config root,
private deployment config since OMN-20934) as the DESIRED state and a JSON envelope
of the live docker inventory as the ACTUAL state, and emits a deterministic list
of typed drift findings. It performs NO I/O — the shell driver
(scripts/lane-census-check.sh) gathers the docker inventory and publishes the
bus event / ticket. Keeping the decision logic here makes drift detection fully
unit-testable, including the 2026-06-11 prod red fixture.

Drift kinds (named exactly so the auto-ticket says precisely what is wrong):
  container_absent     — a required `service` container is not running
  replicas_zero        — a required service resolved to 0 running replicas
  network_detached     — the lane's declared network does not exist
  oneshot_failed       — a run-to-completion container Exited non-zero
  oneshot_stuck        — a run-to-completion container is still Running
                         (kind `keepalive` is exempt: a healthcheck sentinel
                         such as migration-gate is a long-running keepalive BY
                         DESIGN — `while true; do sleep 3600; done` + a
                         continuous healthcheck — so Running is its healthy
                         steady state, not drift; OMN-13772)
  image_tag_mismatch   — a running container's image tag fails the lane pattern
  unexpected_container — a container labeled for the lane that the manifest does
                         not declare
  profile_gated_present — a container the lane's compose file deliberately
                         disables (`profiles: !override [...-disabled]`, kind
                         `profile_gated`) is nevertheless running. The inverse
                         of container_absent: for this kind ABSENT is correct
                         and PRESENT is the drift (OMN-16803)
  lane_on_undeclared_host — a lane is running on a host its `hosts:` list does
                         not name (OMN-19088). Matched by declared container
                         name, compose project label or lane label, running
                         containers only

Every kind, with its one severity, is declared in ``CENSUS_FINDING_KINDS``; the
eleven lab-sync kinds that compare the inventory with the generated desired
state (``lab-desired-state.v1``) are declared in ``LAB_SYNC_FINDING_KINDS``
(OMN-19411). OMN-19416 evaluates unhealthy and declared restart-loop; OMN-19414
evaluates the other nine against the desired state the envelope carries as
``desired_state`` (one document or a list, one per surface):

  revision_mismatch, config_hash_mismatch, package_version_mismatch,
  broker_config_mismatch      — a declared container or broker against its row
  undeclared_container        — a container nothing declares, from a host
                                surface's ``allowed_undeclared``
  runner_count_mismatch, runner_workdir_mismatch, runner_offline
                              — a runner_fleet surface against the fleet rows
                                and the GitHub registration
  desired_state_unreadable    — ``desired_state`` is null, malformed or absent
                                from a run that was asked to read one. It FAILS
                                CLOSED: the census never reads clean without a
                                desired state to compare against

The planner stays pure. ``main`` reads ``$LANE_CENSUS_DESIRED_STATE`` (one
document, or a directory of ``*.json`` documents) through the stdlib validator
``lab_sync/desired_state.py`` and puts the result in the envelope; with the
variable unset and no ``desired_state`` in the envelope the desired-state
comparisons are not run and the plan says so (``lab_sync_evaluated: false``). A
declared facet the collector did not supply (``Packages`` on a row,
``broker_config``, ``github_runners``) raises, never reads clean.

Host scoping (OMN-19088): every lane declares the host(s) it runs on
(`hosts:`, naming entries of the top-level `hosts:` registry). The census
evaluates only the lanes declared for the host it runs on and reports the rest
in `lanes_not_applicable`, rather than filing every service of every other
host's lanes as absent. The host is resolved from the envelope's `host` (the
driver passes `LANE_CENSUS_HOST`, default `hostname`) through the registry's
aliases. A host the registry does not declare is REFUSED: evaluating every lane
there is the defect this closes, and evaluating none would be a silent clean.

Fail-fast policy: any drift on a non-optional lane is a hard signal. There is no
warn-only mode (gates-block policy). The driver exits non-zero on drift.

Input envelope (stdin JSON):
  {
    "lane": "<lane name>" | null,   # null => reconcile all non-optional lanes
    "host": "<hostname or host id>", # required; resolved via the hosts registry
    "containers": [                 # `docker ps -a --format '{{json .}}'` rows,
      {"Names": "...", "State": "running"|"exited"|...,
       "Image": "repo:tag", "Labels": "k=v,k2=v2",
       "Status": "Up 3 hours" | "Exited (0) ..."},
      ...
    ],
    "networks": ["omnibase-infra-prod-network", ...],  # `docker network ls`
    "runtime_tag": "0.37.0" | null, # resolved runtime tag (optional)
    "desired_state": {...} | [{...}] | null,  # lab-desired-state.v1 (optional)
    "desired_state_error": "why it could not be read"  # optional, with null
  }

Output (stdout JSON):
  {
    "schema_version": "1.1.0",
    "host": "lab-201",                 # the resolved host id
    "lanes_checked": ["prod", ...],    # lanes declared for this host
    "lanes_not_applicable": ["dogfood", ...],  # lanes declared elsewhere
    "lanes_skipped_optional_down": ["prepr-1", ...],  # optional + entirely down
    "lab_sync_evaluated": true|false,  # the desired-state kinds were graded
    "findings": [ {lane, kind, container, detail, severity}, ... ],
    "has_drift": true|false
  }

``lanes_checked`` and ``lanes_not_applicable`` partition the requested lanes.
A not-applicable lane produces no finding unless it is running here, which is
``lane_on_undeclared_host``.

``lanes_skipped_optional_down`` is a subset of ``lanes_checked``: those lanes
were looked at, found to be optional and entirely absent, and deliberately
produced no findings. It exists so a consumer can tell that silence apart from
"reconciled and clean" (OMN-18890).
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any, NamedTuple

import yaml

SCHEMA_VERSION = "1.1.0"
DEFAULT_UNHEALTHY_AFTER_SECONDS = 1800

#: OMN-19414: a container an ``allowed_undeclared`` entry admits has no manifest
#: service to declare a ``restart_bound``, and E7 was a restart loop (6051
#: restarts) in exactly such a project. The bound the lab alarm used for it.
DEFAULT_ALLOWED_UNDECLARED_RESTART_BOUND = 5

ENV_DESIRED_STATE = "LANE_CENSUS_DESIRED_STATE"
IMAGE_REVISION_LABEL = "org.opencontainers.image.revision"
COMPOSE_CONFIG_HASH_LABEL = "com.docker.compose.config-hash"
COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
COMPOSE_SERVICE_LABEL = "com.docker.compose.service"
COMPOSE_WORKDIR_LABEL = "com.docker.compose.project.working_dir"
#: Shortest image revision label read as a prefix of the desired commit. Images
#: are labelled with a 12-character prefix of the commit they were built from.
MIN_REVISION_PREFIX = 12

# Names that identify no single host and so may never be a host alias. Every
# Docker Desktop host reports the daemon name `docker-desktop` (measured on the
# .101, .105 and .200 hosts, 2026-09-23), so a census attributed by daemon name
# would read one host's inventory as another's.
_AMBIGUOUS_HOST_NAMES = frozenset({"docker-desktop", "localhost"})


#: Exit status of ``main`` when the host is undeclared or the host declarations
#: are malformed. Distinct from the driver's drift code 30 and its
#: unobservable-inventory code 4: an unattributable census is neither.
EXIT_HOST_UNDECLARED = 5

#: Exit status of ``main`` when no lane manifest can be located (OMN-20934).
#: The driver's "missing dependency" code: the manifest is private config the
#: host must supply, and without it there is no desired state to compare.
EXIT_NOT_CONFIGURED = 3

#: Exit status of ``main`` when the plan carries ``desired_state_unreadable``
#: (OMN-19414). The plan is still printed: the driver treats it as drift, so a
#: census without a readable desired state exits non-zero and publishes.
EXIT_DESIRED_STATE_UNREADABLE = 9


class HostDeclarationError(ValueError):
    """The manifest's host declarations are malformed, or this host is undeclared."""


# Drift severities. critical => a required service is down / network gone (the
# 2026-06-11 class of outage), or the lane's proof is about something other than
# what it was meant to run. warning => degraded but not lane-down. Severity only
# ranks the alert: every finding of every kind is drift, and drift blocks.
_SEVERITY_CRITICAL = "critical"
_SEVERITY_WARNING = "warning"


class FindingKindSpec(NamedTuple):
    """One finding kind the planner may emit, and what grading it needs.

    A ``NamedTuple`` rather than a dataclass: the census scripts are loaded by
    path, without a ``sys.modules`` entry, which a dataclass cannot survive.

    ``subject`` names what a finding of this kind is about, which is what its
    ``container`` field holds: a ``container`` name, a ``broker`` container name,
    a ``runner`` name, or the literal ``desired_state``.

    ``envelope_fields`` is the collector side of the seam (L0.2): the envelope
    keys, per container row or top level, that must be present for the kind to
    be graded. A kind whose fields are absent cannot be graded, and T1.2
    (OMN-19414) reports that as a finding, never as clean.
    """

    kind: str
    severity: str
    subject: str
    envelope_fields: tuple[str, ...]


#: The kinds the planner emits today (OMN-13011, OMN-16803, OMN-19088), with the
#: severity each has always carried. ``_finding`` resolves severity from this
#: table, so an undeclared kind raises instead of publishing.
CENSUS_FINDING_KINDS: tuple[FindingKindSpec, ...] = (
    FindingKindSpec("container_absent", _SEVERITY_CRITICAL, "container", ()),
    FindingKindSpec("replicas_zero", _SEVERITY_CRITICAL, "container", ()),
    FindingKindSpec("network_detached", _SEVERITY_CRITICAL, "network", ()),
    FindingKindSpec("oneshot_failed", _SEVERITY_CRITICAL, "container", ()),
    FindingKindSpec("oneshot_stuck", _SEVERITY_WARNING, "container", ()),
    FindingKindSpec("image_tag_mismatch", _SEVERITY_WARNING, "container", ()),
    FindingKindSpec("unexpected_container", _SEVERITY_WARNING, "container", ()),
    FindingKindSpec("profile_gated_present", _SEVERITY_WARNING, "container", ()),
    FindingKindSpec("lane_on_undeclared_host", _SEVERITY_CRITICAL, "container", ()),
)

#: OMN-19411 (lab release sync plan, T0.2, section 4 B): the eleven kinds that
#: compare the inventory with the generated desired state (``lab-desired-state.v1``,
#: OMN-19410). OMN-19416 evaluates health and declared restart bounds. The remaining
#: comparisons await evaluation, against the fixtures in
#: ``tests/fixtures/lab_sync/``, one per kind.
#:
#: Container-row fields beyond ``docker ps``: ``Health`` (``{"Status",
#: "FailingStreak"}`` from ``State.Health``), ``HealthcheckIntervalSeconds``
#: (``Config.Healthcheck.Interval``), ``RestartCount`` and ``Packages``
#: (``{distribution: version}`` installed in the container). Top-level fields:
#: ``broker_config`` (``{broker container: {key: value}}`` from
#: ``rpk cluster config get``), ``github_runners`` (``[{"name", "status",
#: "labels"}]`` from the org runner registration) and ``desired_state`` (the
#: ``lab-desired-state.v1`` document, or null when it could not be read).
LAB_SYNC_FINDING_KINDS: tuple[FindingKindSpec, ...] = (
    # A declared container runs an image built from another commit than the
    # desired ref (org.opencontainers.image.revision).
    FindingKindSpec("revision_mismatch", _SEVERITY_CRITICAL, "container", ("Labels",)),
    # com.docker.compose.config-hash differs from `docker compose config --hash`
    # at the desired ref: image, env, mounts or command drifted.
    FindingKindSpec(
        "config_hash_mismatch", _SEVERITY_CRITICAL, "container", ("Labels",)
    ),
    # An installed distribution's version differs from the desired ref's lock.
    FindingKindSpec(
        "package_version_mismatch", _SEVERITY_CRITICAL, "container", ("Packages",)
    ),
    # Docker-unhealthy for FailingStreak x interval >= the bound (default
    # 1800 s, plan section 4 D).
    FindingKindSpec(
        "container_unhealthy",
        _SEVERITY_CRITICAL,
        "container",
        ("Health", "HealthcheckIntervalSeconds"),
    ),
    # RestartCount above the container's restart bound.
    FindingKindSpec(
        "container_restart_loop", _SEVERITY_CRITICAL, "container", ("RestartCount",)
    ),
    # A container on the host that no lane or surface declares and that is not
    # listed in the desired state's allowed_undeclared with an owner.
    FindingKindSpec(
        "undeclared_container", _SEVERITY_WARNING, "container", ("Labels",)
    ),
    # A declared broker key reads another value than the lane's broker profile.
    FindingKindSpec(
        "broker_config_mismatch", _SEVERITY_CRITICAL, "broker", ("broker_config",)
    ),
    # Fewer or more runner containers of a class on a host than expected_count.
    FindingKindSpec("runner_count_mismatch", _SEVERITY_WARNING, "runner", ("Labels",)),
    # A runner's com.docker.compose.project.working_dir is not the fleet path.
    FindingKindSpec(
        "runner_workdir_mismatch", _SEVERITY_CRITICAL, "runner", ("Labels",)
    ),
    # A declared runner is not registered online with GitHub.
    FindingKindSpec("runner_offline", _SEVERITY_WARNING, "runner", ("github_runners",)),
    # The desired state is missing or does not validate. Fails closed: the
    # census never reads clean without a desired state to compare against.
    FindingKindSpec(
        "desired_state_unreadable",
        _SEVERITY_CRITICAL,
        "desired_state",
        ("desired_state",),
    ),
)

#: Every kind the census event may carry, with its one severity. The event
#: validator (``lane_census_event.validate_event``) reads this mapping.
FINDING_KIND_SEVERITY: dict[str, str] = {
    spec.kind: spec.severity
    for spec in (*CENSUS_FINDING_KINDS, *LAB_SYNC_FINDING_KINDS)
}

#: OMN-20934: the lane manifest and the committed census describe one lab's
#: hosts, lanes and addresses, so they are private deployment config (operator
#: RULING 2026-10-10T21:57:38Z) and this repository ships neither. They live in
#: the workspace config root: ``$ONEX_WORKSPACE_CONFIG_ROOT`` when set,
#: otherwise the owner the runtime resolves for ``$OMNI_HOME``
#: (``workspace_runtime_config_root``, the one resolver of that root).
ENV_WORKSPACE_CONFIG_ROOT = "ONEX_WORKSPACE_CONFIG_ROOT"
LANE_CENSUS_RELATIVE = Path("config") / "lab" / "lane-census"
MANIFEST_NAME = "lane-manifest.yaml"
SNAPSHOT_NAME = "census-snapshot.json"


class LaneCensusConfigError(RuntimeError):
    """The private lane census directory cannot be located."""


def _not_configured(reason: str) -> LaneCensusConfigError:
    return LaneCensusConfigError(
        f"the lane census is not configured: {reason}; set "
        f"{ENV_WORKSPACE_CONFIG_ROOT} to the directory that holds "
        f"{LANE_CENSUS_RELATIVE / MANIFEST_NAME}. This repository ships no "
        f"lane manifest"
    )


def lane_census_dir() -> Path:
    """The directory holding the lane manifest and the committed census.

    Fails loud rather than falling back to anything in this repository: a
    fork with no overlay is told how to supply one, never pointed at our lab.
    The census driver runs under a bare interpreter on a lab host, where the
    runtime package is not importable, so there the variable is required.
    """
    configured = os.environ.get(ENV_WORKSPACE_CONFIG_ROOT, "").strip()
    if configured:
        return Path(configured).resolve() / LANE_CENSUS_RELATIVE
    registry = os.environ.get("OMNI_HOME", "").strip()
    if not registry:
        raise _not_configured(
            f"neither {ENV_WORKSPACE_CONFIG_ROOT} nor OMNI_HOME is set"
        )
    try:
        from omnibase_infra.runtime.service_kernel import (
            workspace_runtime_config_root,
        )
    except ImportError as exc:
        raise _not_configured(
            f"{ENV_WORKSPACE_CONFIG_ROOT} is unset and this interpreter cannot "
            f"resolve the root from OMNI_HOME ({exc})"
        ) from exc
    return workspace_runtime_config_root(Path(registry)) / LANE_CENSUS_RELATIVE


def _required_file(name: str) -> Path:
    path = lane_census_dir() / name
    if not path.is_file():
        raise _not_configured(f"{path} does not exist")
    return path


def default_manifest_path() -> Path:
    """``LANE_MANIFEST`` when set, else the manifest in the lane census dir."""
    explicit = os.environ.get("LANE_MANIFEST", "").strip()
    if explicit:
        return Path(explicit)
    return _required_file(MANIFEST_NAME)


def default_snapshot_path() -> Path:
    """The committed census in the lane census dir."""
    return _required_file(SNAPSHOT_NAME)


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the versioned lane manifest (desired state)."""
    manifest_path = path or default_manifest_path()
    with open(manifest_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "lanes" not in data:
        raise ValueError(f"lane manifest missing 'lanes': {manifest_path}")
    validate_service_bounds(data)
    return data


def validate_service_bounds(manifest: dict[str, Any]) -> None:
    """OMN-19416: health may alarm sooner; a declaration cannot defer it."""
    for lane, spec in manifest["lanes"].items():
        for service in spec.get("services", []):
            threshold = service.get(
                "unhealthy_after_seconds", DEFAULT_UNHEALTHY_AFTER_SECONDS
            )
            if (
                type(threshold) is not int
                or not 0 < threshold <= DEFAULT_UNHEALTHY_AFTER_SECONDS
            ):
                raise ValueError(
                    f"{lane}/{service['name']}: unhealthy_after_seconds must be "
                    f"an integer in 1..{DEFAULT_UNHEALTHY_AFTER_SECONDS}"
                )
            if "restart_bound" in service:
                bound = service["restart_bound"]
                if type(bound) is not int or bound < 0:
                    raise ValueError(
                        f"{lane}/{service['name']}: restart_bound must be a non-negative integer"
                    )


def restart_bounds_for_lane(manifest: dict[str, Any], lane: str) -> dict[str, int]:
    """The alarm and detector share the per-service manifest declarations."""
    validate_service_bounds(manifest)
    if lane not in manifest["lanes"]:
        raise ValueError(f"unknown census lane {lane!r}")
    bounds = {
        svc["name"]: svc["restart_bound"]
        for svc in manifest["lanes"][lane].get("services", [])
        if "restart_bound" in svc
    }
    if not bounds:
        raise ValueError(f"census lane {lane!r} declares no restart_bound")
    return bounds


def _container_health_findings(
    lane: str, service: dict[str, Any], actual: dict[str, Any]
) -> list[dict[str, str]]:
    """Grade declared containers using inspect readings, never the uptime string."""
    findings: list[dict[str, str]] = []
    name = service["name"]
    bound = service.get("restart_bound")
    if bound is not None:
        count = actual.get("RestartCount")
        if type(count) is not int or count < 0:
            raise ValueError(f"{name}: RestartCount is missing or invalid")
        if count > bound:
            findings.append(
                _finding(
                    lane,
                    "container_restart_loop",
                    name,
                    f"container {name!r} has restarted {count} times, past its declared bound of {bound}",
                )
            )
    health = actual.get("Health")
    if not _running(actual.get("State", ""), actual.get("Status", "")):
        return findings
    if not isinstance(health, dict) or health.get("Status") != "unhealthy":
        return findings
    streak = health.get("FailingStreak")
    interval = actual.get("HealthcheckIntervalSeconds")
    if (
        type(streak) is not int
        or streak < 0
        or not isinstance(interval, (int, float))
        or isinstance(interval, bool)
        or not math.isfinite(interval)
        or interval <= 0
    ):
        raise ValueError(
            f"{name}: unhealthy Health.FailingStreak or HealthcheckIntervalSeconds is missing or invalid"
        )
    elapsed = streak * interval
    threshold = service.get("unhealthy_after_seconds", DEFAULT_UNHEALTHY_AFTER_SECONDS)
    if elapsed >= threshold:
        findings.append(
            _finding(
                lane,
                "container_unhealthy",
                name,
                f"container {name!r} is unhealthy for {elapsed:g}s "
                f"(FailingStreak={streak} x interval={interval:g}s), "
                f"at or above its {threshold}s bound",
            )
        )
    return findings


def _normalize_host(name: str) -> str:
    """Lowercase short name: ``Stickybeatz.local`` and ``x.tailnet.ts.net`` -> first label."""
    return name.strip().lower().rstrip(".").split(".", 1)[0]


def validate_hosts(manifest: dict[str, Any]) -> None:
    """Refuse a manifest whose host declarations cannot attribute a census.

    Every lane must name at least one registered host, every alias must be
    claimed by exactly one host, and no alias may be a name that identifies no
    single host. A lane with no host would be evaluated nowhere, which is a
    silent clean; an alias claimed twice would evaluate one host's lanes on
    another.
    """
    registry = manifest.get("hosts")
    if not isinstance(registry, dict) or not registry:
        raise HostDeclarationError("lane manifest declares no 'hosts' registry")

    claimed: dict[str, str] = {}
    for host_id, spec in registry.items():
        aliases = (spec or {}).get("aliases") or []
        if not isinstance(aliases, list):
            raise HostDeclarationError(f"host {host_id!r}: 'aliases' must be a list")
        for name in [host_id, *aliases]:
            key = _normalize_host(str(name))
            if not key:
                raise HostDeclarationError(f"host {host_id!r} declares an empty name")
            if key in _AMBIGUOUS_HOST_NAMES:
                raise HostDeclarationError(
                    f"host {host_id!r} declares {name!r}, which identifies no single "
                    "host (every Docker Desktop daemon reports 'docker-desktop')"
                )
            owner = claimed.get(key)
            if owner is not None and owner != host_id:
                raise HostDeclarationError(
                    f"name {name!r} is claimed by both host {owner!r} and host "
                    f"{host_id!r}; a census there cannot be attributed"
                )
            claimed[key] = host_id

    for lane_name, lane_spec in manifest["lanes"].items():
        hosts = lane_spec.get("hosts")
        if not isinstance(hosts, list) or not hosts:
            raise HostDeclarationError(
                f"lane {lane_name!r} declares no 'hosts'; every lane must name the "
                "host(s) it runs on"
            )
        for host_id in hosts:
            if host_id not in registry:
                raise HostDeclarationError(
                    f"lane {lane_name!r} names host {host_id!r}, which the 'hosts' "
                    "registry does not declare"
                )


def resolve_host(raw: str, manifest: dict[str, Any]) -> str:
    """Resolve a hostname, daemon name or host id to exactly one host id."""
    validate_hosts(manifest)
    key = _normalize_host(raw or "")
    if not key:
        raise HostDeclarationError(
            "no host given: the census must be told which host it runs on "
            "(LANE_CENSUS_HOST, default `hostname`)"
        )
    for host_id, spec in manifest["hosts"].items():
        names = [host_id, *((spec or {}).get("aliases") or [])]
        if key in {_normalize_host(str(n)) for n in names}:
            return str(host_id)
    raise HostDeclarationError(
        f"host {raw!r} is declared by no entry of the lane manifest's 'hosts' "
        "registry. Declare it (with the lanes it runs) before running the census "
        "there; evaluating every lane on an undeclared host is the OMN-19088 defect"
    )


def _undeclared_presence(
    lane_name: str,
    lane_spec: dict[str, Any],
    host_id: str,
    containers: list[tuple[str, dict[str, Any], dict[str, str]]],
) -> list[dict[str, str]]:
    """AC-2: a lane RUNNING on a host its declaration does not name is a finding.

    Matched by declared container name, compose project label or lane label,
    so a renamed container is still recognised by the project that started it.
    Exited leftovers do not count: a lane is on a host when it runs there.
    """
    declared_names = {svc["name"] for svc in lane_spec.get("services", [])}
    project = lane_spec.get("compose_project")
    running_here = sorted(
        name
        for name, row, labels in containers
        if _running(row.get("State", ""), row.get("Status", ""))
        and (
            name in declared_names
            or (project and labels.get("com.docker.compose.project") == project)
            or labels.get("com.omninode.lane") == lane_name
        )
    )
    if not running_here:
        return []
    return [
        _finding(
            lane_name,
            "lane_on_undeclared_host",
            running_here[0],
            f"lane {lane_name!r} is declared for host(s) "
            f"{', '.join(lane_spec.get('hosts', []))} but is running on "
            f"{host_id!r}: {', '.join(running_here)}. Stop it here, or declare "
            f"{host_id!r} in the lane's 'hosts' in the same change",
        )
    ]


def _running(state: str, status: str) -> bool:
    """A container counts as running only when docker reports it running."""
    return state.lower() == "running" or status.lower().startswith("up")


def _exit_code(status: str) -> int | None:
    """Parse the exit code out of a docker status string, e.g. 'Exited (0) ...'."""
    match = re.search(r"Exited \((\d+)\)", status)
    return int(match.group(1)) if match else None


def _tag_of(image: str) -> str:
    """Return the tag portion of a 'repo:tag' image reference (default 'latest')."""
    # Strip a registry host:port (the first colon may belong to the host).
    # docker image refs: [registry[:port]/]repo[:tag][@digest]
    ref = image.split("@", 1)[0]
    last_segment = ref.rsplit("/", 1)[-1]
    if ":" in last_segment:
        return last_segment.rsplit(":", 1)[1]
    return "latest"


def _finding(lane: str, kind: str, container: str, detail: str) -> dict[str, str]:
    """One finding. Its severity is the kind's, from ``FINDING_KIND_SEVERITY``.

    A kind the table does not declare raises ``KeyError`` here, so a typo or an
    undeclared kind can never reach the event (OMN-19411).
    """
    return {
        "lane": lane,
        "kind": kind,
        "container": container,
        "detail": detail,
        "severity": FINDING_KIND_SEVERITY[kind],
    }


# ---------------------------------------------------------------------------
# OMN-19414 — the desired-state comparisons (lab release sync plan, T1.2)
# ---------------------------------------------------------------------------

_LabRow = tuple[str, dict[str, Any], dict[str, str]]


def _read_desired_state(path: Path) -> tuple[list[dict[str, Any]] | None, str]:
    """Read ``lab-desired-state.v1`` documents; ``(None, reason)`` when unreadable.

    ``path`` is one document or a directory of ``*.json`` documents. Every
    document goes through the stdlib validator, so a refused, truncated or
    hash-mismatched document is unreadable here and never partly compared.
    """
    module_path = Path(__file__).resolve().parent / "lab_sync" / "desired_state.py"
    spec = importlib.util.spec_from_file_location("lab_sync_desired_state", module_path)
    if spec is None or spec.loader is None:
        return None, f"the desired-state validator {module_path} cannot be loaded"
    module = importlib.util.module_from_spec(spec)
    # A dataclass resolves its module through sys.modules while it is defined.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except OSError as exc:
        return None, f"the desired-state validator {module_path} cannot be read: {exc}"
    if path.is_dir():
        paths = sorted(path.glob("*.json"))
        if not paths:
            return None, f"{path} holds no desired-state document"
    else:
        paths = [path]
    documents: list[dict[str, Any]] = []
    for candidate in paths:
        try:
            documents.append(module.load_desired_state(candidate).document)
        except module.DesiredStateRefusalError as exc:
            return None, f"{candidate}: {exc}"
    return documents, ""


def _desired_documents(envelope: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """The envelope's desired-state documents, or why there are none."""
    desired = envelope.get("desired_state")
    if isinstance(desired, dict):
        return [desired], ""
    if (
        isinstance(desired, list)
        and desired
        and all(isinstance(doc, dict) for doc in desired)
    ):
        return list(desired), ""
    reason = str(envelope.get("desired_state_error") or "").strip()
    if reason:
        return [], reason
    if desired is None:
        return [], "no desired state was supplied"
    return [], f"the desired state is a {type(desired).__name__}, not a document"


def _ungradable(subject: str, field: str, kind: str) -> ValueError:
    return ValueError(
        f"{subject}: the desired state declares {kind} but the envelope carries no "
        f"{field}; the collector must supply it, and a census that cannot grade it "
        "does not read it as clean"
    )


def _scalar_text(value: Any) -> str:
    """A value as ``rpk cluster config get`` prints it, so int 1048576 == '1048576'."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name)).lower()


def _revision_matches(actual: str, expected: str) -> bool:
    if not actual:
        return False
    if actual == expected:
        return True
    return len(actual) >= MIN_REVISION_PREFIX and expected.startswith(actual)


def _container_sync_findings(
    lane: str, document: dict[str, Any], rows: dict[str, _LabRow]
) -> list[dict[str, str]]:
    """revision, config hash and package drift of each declared container."""
    findings: list[dict[str, str]] = []
    for want in document.get("containers") or []:
        name = want.get("name", "")
        if name not in rows:
            continue  # a declared container that is gone is container_absent's
        _, row, labels = rows[name]
        revision = want.get("revision")
        if revision is not None:
            actual = labels.get(IMAGE_REVISION_LABEL, "")
            if not _revision_matches(actual, revision):
                findings.append(
                    _finding(
                        lane,
                        "revision_mismatch",
                        name,
                        f"container {name!r} runs an image built from "
                        f"{actual or '(no revision label)'}, the desired state "
                        f"declares {revision}",
                    )
                )
        config_hash = want.get("config_hash")
        if config_hash is not None:
            actual = labels.get(COMPOSE_CONFIG_HASH_LABEL, "")
            if actual != config_hash:
                findings.append(
                    _finding(
                        lane,
                        "config_hash_mismatch",
                        name,
                        f"container {name!r} carries compose config-hash "
                        f"{actual or '(none)'}, the desired state declares "
                        f"{config_hash}: its image, env or mounts drifted",
                    )
                )
        packages = want.get("packages") or {}
        if packages:
            installed = row.get("Packages")
            if not isinstance(installed, dict):
                raise _ungradable(name, "Packages on its container row", "packages")
            have = {_distribution(k): str(v) for k, v in installed.items()}
            diffs = [
                f"{dist} {have.get(_distribution(dist), '(not installed)')} "
                f"!= {version}"
                for dist, version in sorted(packages.items())
                if have.get(_distribution(dist)) != version
            ]
            if diffs:
                findings.append(
                    _finding(
                        lane,
                        "package_version_mismatch",
                        name,
                        f"container {name!r} has installed distributions off the "
                        f"desired lock: {'; '.join(diffs)}",
                    )
                )
    return findings


def _broker_sync_findings(
    lane: str,
    document: dict[str, Any],
    rows: list[_LabRow],
    broker_config: Any,
) -> list[dict[str, str]]:
    """Each declared ``rpk cluster config`` key against what the broker reports."""
    broker = document.get("broker")
    if not isinstance(broker, dict):
        return []
    service = broker.get("service")
    project = (document.get("surface") or {}).get("compose_project")
    findings: list[dict[str, str]] = []
    for name, _, labels in rows:
        if labels.get(COMPOSE_SERVICE_LABEL) != service:
            continue
        if project and labels.get(COMPOSE_PROJECT_LABEL) != project:
            continue
        if not isinstance(broker_config, dict) or name not in broker_config:
            raise _ungradable(name, "broker_config for the broker", "broker keys")
        read = broker_config[name]
        diffs = [
            f"{key} reads {_scalar_text(read.get(key))}, declared {want}"
            for key, want in sorted((broker.get("cluster_config") or {}).items())
            if _scalar_text(read.get(key)) != _scalar_text(want)
        ]
        if diffs:
            findings.append(
                _finding(
                    lane,
                    "broker_config_mismatch",
                    name,
                    f"broker {name!r} departs from the lane's broker profile: "
                    f"{'; '.join(diffs)}",
                )
            )
    return findings


def _runner_sync_findings(
    document: dict[str, Any],
    rows: dict[str, _LabRow],
    github_runners: Any,
) -> list[dict[str, str]]:
    """A runner fleet against its container rows and its GitHub registration."""
    runners = document.get("runners")
    if not isinstance(runners, dict):
        return []
    surface = document.get("surface") or {}
    project = surface.get("compose_project")
    host = str(surface.get("host") or "")
    fleet = {
        name: entry
        for name, entry in rows.items()
        if project and entry[2].get(COMPOSE_PROJECT_LABEL) == project
    }
    pools = {pool["name"]: pool for pool in runners.get("pools") or []}
    declared = {c["name"]: c for c in runners.get("containers") or []}
    sole_pool = next(iter(pools.values())) if len(pools) == 1 else None
    findings: list[dict[str, str]] = []
    for name, want in sorted(declared.items()):
        expected = want.get("config_hash")
        if name in rows and expected is not None:
            actual = rows[name][2].get(COMPOSE_CONFIG_HASH_LABEL, "")
            if actual != expected:
                findings.append(
                    _finding(
                        "",
                        "config_hash_mismatch",
                        name,
                        f"runner {name!r} carries compose config-hash "
                        f"{actual or '(none)'}, the fleet render declares {expected}",
                    )
                )
    for name, (_, _, labels) in sorted(fleet.items()):
        pool = pools.get(declared.get(name, {}).get("pool", "")) or sole_pool
        workdir = (pool or {}).get("workdir")
        actual = labels.get(COMPOSE_WORKDIR_LABEL, "")
        if workdir and actual != workdir:
            findings.append(
                _finding(
                    "",
                    "runner_workdir_mismatch",
                    name,
                    f"runner {name!r} runs from {actual or '(no working_dir label)'}, "
                    f"the fleet path is {workdir}",
                )
            )
    counts = [pool.get("expected_count") for pool in pools.values()]
    if pools and all(type(count) is int for count in counts):
        expected_total = sum(counts)
        running = sorted(
            name
            for name, (_, row, _) in fleet.items()
            if _running(row.get("State", ""), row.get("Status", ""))
        )
        if len(running) != expected_total:
            findings.append(
                _finding(
                    "",
                    "runner_count_mismatch",
                    host,
                    f"host {host!r} runs {len(running)} runner container(s) in "
                    f"project {project!r} ({', '.join(running) or 'none'}), the "
                    f"fleet declares {expected_total}",
                )
            )
    present = sorted(name for name in declared if name in fleet)
    if present:
        if not isinstance(github_runners, list):
            raise _ungradable(host, "github_runners", "runners")
        online = {
            str(entry.get("name"))
            for entry in github_runners
            if isinstance(entry, dict) and entry.get("status") == "online"
        }
        for name in present:
            if name not in online:
                findings.append(
                    _finding(
                        "",
                        "runner_offline",
                        name,
                        f"runner {name!r} has a container but is not registered "
                        "online with GitHub",
                    )
                )
    return findings


def _allowed_by(entry: dict[str, Any], row: _LabRow) -> bool:
    """Whether one ``allowed_undeclared`` entry admits a container (all selectors)."""
    name, data, labels = row
    match = entry.get("match") or {}
    if not match:
        return False
    pattern = match.get("name_pattern")
    if pattern is not None:
        try:
            if not re.fullmatch(pattern, name):
                return False
        except re.error as exc:
            raise ValueError(
                f"allowed_undeclared {entry.get('name')!r}: name_pattern is not a "
                f"regular expression: {exc}"
            ) from exc
    project = match.get("compose_project")
    if project is not None and labels.get(COMPOSE_PROJECT_LABEL) != project:
        return False
    label = match.get("label")
    if label is not None:
        key, _, value = label.partition("=")
        if labels.get(key) != value:
            return False
    max_age = match.get("max_age_seconds")
    if max_age is not None:
        age = data.get("AgeSeconds")
        # An age the collector did not read cannot be shown to be young enough.
        if isinstance(age, bool) or not isinstance(age, (int, float)) or age > max_age:
            return False
    return True


def _undeclared_sync_findings(
    documents: list[dict[str, Any]], manifest: dict[str, Any], rows: list[_LabRow]
) -> list[dict[str, str]]:
    """Containers nothing declares and no ``allowed_undeclared`` entry admits.

    Graded only when a host surface is among the documents: that surface owns
    ``allowed_undeclared``. A container is declared by a manifest service, a
    desired-state container or runner, or the runner fleet's compose project,
    and a lane-labelled container is the lane census's (``unexpected_container``).
    An admitted container still answers to ``DEFAULT_ALLOWED_UNDECLARED_RESTART_BOUND``.
    """
    host_documents = [
        doc for doc in documents if (doc.get("surface") or {}).get("kind") == "host"
    ]
    if not host_documents:
        return []
    allowed = [
        entry for doc in host_documents for entry in doc.get("allowed_undeclared") or []
    ]
    declared = {
        svc["name"]
        for lane in manifest["lanes"].values()
        for svc in lane.get("services", [])
    }
    fleet_projects = set()
    for doc in documents:
        declared.update(c["name"] for c in doc.get("containers") or [])
        runners = doc.get("runners")
        if isinstance(runners, dict):
            declared.update(c["name"] for c in runners.get("containers") or [])
            project = (doc.get("surface") or {}).get("compose_project")
            if project:
                fleet_projects.add(project)
    findings: list[dict[str, str]] = []
    for row in rows:
        name, data, labels = row
        if (
            name in declared
            or labels.get("com.omninode.lane")
            or labels.get(COMPOSE_PROJECT_LABEL) in fleet_projects
        ):
            continue
        admitted = next((entry for entry in allowed if _allowed_by(entry, row)), None)
        if admitted is None:
            findings.append(
                _finding(
                    "",
                    "undeclared_container",
                    name,
                    f"container {name!r} (compose project "
                    f"{labels.get(COMPOSE_PROJECT_LABEL) or '(none)'}) is declared by "
                    "no lane or surface and no allowed_undeclared entry admits it",
                )
            )
            continue
        count = data.get("RestartCount")
        if type(count) is not int or count < 0:
            raise ValueError(f"{name}: RestartCount is missing or invalid")
        if count > DEFAULT_ALLOWED_UNDECLARED_RESTART_BOUND:
            findings.append(
                _finding(
                    "",
                    "container_restart_loop",
                    name,
                    f"container {name!r} (admitted by allowed_undeclared "
                    f"{admitted.get('name')!r}, owner {admitted.get('owner')}) has "
                    f"restarted {count} times, past the "
                    f"{DEFAULT_ALLOWED_UNDECLARED_RESTART_BOUND} an admitted "
                    "container may",
                )
            )
    return findings


def lab_sync_findings(
    envelope: dict[str, Any], manifest: dict[str, Any], rows: list[_LabRow]
) -> list[dict[str, str]]:
    """Compare the inventory with the generated desired state (OMN-19414).

    Pure. An unreadable desired state is ``desired_state_unreadable`` and nothing
    else is compared: there is nothing to compare, and the census must not read
    clean without it.
    """
    lane = str(envelope.get("lane") or "")
    documents, reason = _desired_documents(envelope)
    if not documents:
        return [
            _finding(
                lane,
                "desired_state_unreadable",
                "desired_state",
                f"the desired state cannot be read ({reason}); no desired-state "
                "comparison ran, so this census proves nothing about drift",
            )
        ]
    by_name = {row[0]: row for row in rows}
    findings: list[dict[str, str]] = []
    for document in documents:
        surface = document.get("surface") or {}
        is_lane = surface.get("kind") == "compose_lane"
        findings.extend(
            _container_sync_findings(lane if is_lane else "", document, by_name)
        )
        findings.extend(
            _broker_sync_findings(lane, document, rows, envelope.get("broker_config"))
        )
        findings.extend(
            _runner_sync_findings(document, by_name, envelope.get("github_runners"))
        )
    findings.extend(_undeclared_sync_findings(documents, manifest, rows))
    return findings


def _is_optional_and_entirely_down(
    lane_spec: dict[str, Any], actual_by_name: dict[str, dict[str, Any]]
) -> bool:
    """True when an optional lane has none of its declared services running.

    This is the condition :func:`reconcile_lane` returns no findings on. It is
    lifted into its own predicate and REPORTED (OMN-18890) because zero findings
    is otherwise two different facts wearing one face: "reconciled and clean"
    and "skipped without looking". A consumer that cannot tell them apart states
    the first when the truth is the second — which is how the generated lane
    table came to render an entirely-down optional lane as
    ``N running (census clean)``, the phantom-lane claim retro B-6 exists to make
    unwritable. The ephemeral pre-PR pool makes that the NORMAL case rather than
    a rare one, since an empty pool is its steady state.
    """
    if not lane_spec.get("optional", False):
        return False
    return not any(
        _running(
            actual_by_name.get(svc["name"], {}).get("State", ""),
            actual_by_name.get(svc["name"], {}).get("Status", ""),
        )
        for svc in lane_spec.get("services", [])
        if svc.get("kind", "service") == "service"
    )


def reconcile_lane(
    lane_name: str,
    lane_spec: dict[str, Any],
    *,
    actual_by_name: dict[str, dict[str, Any]],
    networks: set[str],
    lane_labeled_names: set[str],
    default_tag_pattern: str,
    runtime_tag: str | None,
) -> list[dict[str, str]]:
    """Diff one lane's desired state against the actual inventory."""
    findings: list[dict[str, str]] = []

    declared = lane_spec.get("services", [])
    declared_names = {svc["name"] for svc in declared}

    # An optional lane that is entirely down is NOT drift (developer lane, the
    # collaborator lane, and every ephemeral pre-PR slot, whose steady state is
    # absent). "Entirely down" means none of its declared service containers are
    # running. If ANY are running, it is partially up and we reconcile it.
    if _is_optional_and_entirely_down(lane_spec, actual_by_name):
        return findings  # lane legitimately down; no ticket

    # 1. Network presence — the broker/lane network must exist.
    declared_network = lane_spec.get("network")
    if declared_network and declared_network not in networks:
        findings.append(
            _finding(
                lane_name,
                "network_detached",
                declared_network,
                f"lane network {declared_network!r} is not present in "
                f"`docker network ls` — containers cannot reach the broker",
            )
        )

    tag_pattern = lane_spec.get("image_tag_pattern") or default_tag_pattern
    # ${RUNTIME_TAG} is resolved from the deploy-agent runtime version when
    # available; an unresolvable token relaxes to the default pattern (a missing
    # runtime tag is a separate signal, not a census drift).
    if "${RUNTIME_TAG}" in tag_pattern:
        tag_pattern = re.escape(runtime_tag) if runtime_tag else default_tag_pattern
    compiled_pattern = re.compile(tag_pattern)

    # 2. Per-declared-service diff.
    for svc in declared:
        name = svc["name"]
        kind = svc.get("kind", "service")
        actual = actual_by_name.get(name)

        if kind == "profile_gated":
            # OMN-16803: the lane's compose overlay disables this service via a
            # profile override, so it is NOT a member of the lane's active
            # profile and no sanctioned `up` can start it. Absent is the correct
            # steady state — the assertion runs the other way. A running
            # container here means something started it outside the profile
            # (e.g. a warm restart that named it explicitly on the CLI, which
            # bypasses profile filtering), which is exactly the surprise worth
            # surfacing.
            if actual is None or not _running(
                actual.get("State", ""), actual.get("Status", "")
            ):
                continue
            findings.append(
                _finding(
                    lane_name,
                    "profile_gated_present",
                    name,
                    f"container {name!r} is running, but lane {lane_name!r} "
                    f"disables it via a compose profile override "
                    f"(kind=profile_gated, expected absent) — something started "
                    f"it outside the lane's active profile",
                )
            )
            continue

        if actual is not None:
            findings.extend(_container_health_findings(lane_name, svc, actual))

        if kind in ("oneshot", "keepalive"):
            # Run-to-completion container. Absence (compose removed it) is fine.
            if actual is None:
                continue
            status = actual.get("Status", "")
            if _running(actual.get("State", ""), status):
                if kind == "keepalive":
                    # Healthcheck sentinel (e.g. migration-gate): a long-running
                    # keepalive BY DESIGN. Running is its healthy steady state,
                    # never oneshot_stuck (OMN-13772 census false-positive).
                    continue
                findings.append(
                    _finding(
                        lane_name,
                        "oneshot_stuck",
                        name,
                        f"migration/init container {name!r} is still Running "
                        f"(expected run-to-completion): {status}",
                    )
                )
                continue
            code = _exit_code(status)
            if code is not None and code != 0:
                findings.append(
                    _finding(
                        lane_name,
                        "oneshot_failed",
                        name,
                        f"migration/init container {name!r} Exited non-zero "
                        f"(code {code}): {status}",
                    )
                )
            continue

        # kind == service: MUST be running with the required replica count.
        required_replicas = int(svc.get("replicas", 1))
        if actual is None or not _running(
            actual.get("State", ""), actual.get("Status", "")
        ):
            # Distinguish replicas_zero (declared scaled-out but down) only for
            # clarity; for a single declared container, absent == container_absent.
            kind_name = (
                "replicas_zero" if required_replicas == 0 else "container_absent"
            )
            findings.append(
                _finding(
                    lane_name,
                    kind_name,
                    name,
                    f"required service container {name!r} is not running "
                    f"(desired replicas={required_replicas}); lane "
                    f"{lane_name!r} is degraded",
                )
            )
            continue

        # Running — verify image tag matches the lane pattern.
        tag = _tag_of(actual.get("Image", ""))
        if not compiled_pattern.fullmatch(tag):
            findings.append(
                _finding(
                    lane_name,
                    "image_tag_mismatch",
                    name,
                    f"container {name!r} runs image tag {tag!r} which does not "
                    f"match lane pattern {tag_pattern!r}",
                )
            )

    # 3. Unexpected containers — anything labeled for this lane but not declared.
    for actual_name in lane_labeled_names:
        if actual_name not in declared_names:
            findings.append(
                _finding(
                    lane_name,
                    "unexpected_container",
                    actual_name,
                    f"container {actual_name!r} carries com.omninode.lane="
                    f"{lane_name!r} but is not declared in the lane manifest",
                )
            )

    return findings


def _labels_to_dict(labels: Any) -> dict[str, str]:
    """Coerce a container's Labels to a mapping.

    The Docker Engine API returns Labels as a real mapping; the Docker CLI
    returns a comma-joined ``k=v`` string. The collector
    (``lane_census_inventory.py``) normalizes both to a mapping, so the mapping
    branch is the live path — the string branch is retained for envelopes
    produced by older callers and for direct CLI-shaped fixtures. Note the string
    form is lossy by construction: a label VALUE may itself contain commas
    (``com.docker.compose.project.config_files``), which a flat split cannot
    recover. Prefer the mapping form (OMN-15466).
    """
    if isinstance(labels, dict):
        return {str(k): str(v) for k, v in labels.items()}
    out: dict[str, str] = {}
    for pair in str(labels or "").split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def build_plan(envelope: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the deterministic drift plan from the desired manifest + actual state."""
    validate_service_bounds(manifest)
    requested_lane = envelope.get("lane")
    containers = envelope.get("containers", [])
    networks = set(envelope.get("networks", []))
    runtime_tag = envelope.get("runtime_tag")
    default_pattern = manifest.get("default_image_tag_pattern", ".+")
    host_id = resolve_host(str(envelope.get("host") or ""), manifest)

    actual_by_name: dict[str, dict[str, Any]] = {}
    lane_labels: dict[str, set[str]] = {}
    labelled_rows: list[tuple[str, dict[str, Any], dict[str, str]]] = []
    for row in containers:
        name = (row.get("Names") or row.get("Name") or "").lstrip("/").strip()
        if not name:
            continue
        actual_by_name[name] = row
        labels = _labels_to_dict(row.get("Labels", ""))
        labelled_rows.append((name, row, labels))
        lane_label = labels.get("com.omninode.lane")
        if lane_label:
            lane_labels.setdefault(lane_label, set()).add(name)

    lanes = manifest["lanes"]
    if requested_lane:
        if requested_lane not in lanes:
            raise ValueError(f"unknown lane {requested_lane!r}")
        target_lanes = {requested_lane: lanes[requested_lane]}
    else:
        target_lanes = lanes

    findings: list[dict[str, str]] = []
    checked: list[str] = []
    not_applicable: list[str] = []
    skipped_optional_down: list[str] = []
    for lane_name, lane_spec in target_lanes.items():
        if host_id not in lane_spec["hosts"]:
            not_applicable.append(lane_name)
            findings.extend(
                _undeclared_presence(lane_name, lane_spec, host_id, labelled_rows)
            )
            continue
        checked.append(lane_name)
        if _is_optional_and_entirely_down(lane_spec, actual_by_name):
            skipped_optional_down.append(lane_name)
        findings.extend(
            reconcile_lane(
                lane_name,
                lane_spec,
                actual_by_name=actual_by_name,
                networks=networks,
                lane_labeled_names=lane_labels.get(lane_name, set()),
                default_tag_pattern=default_pattern,
                runtime_tag=runtime_tag,
            )
        )

    # OMN-19414: the desired-state comparisons run when the envelope carries a
    # `desired_state` key, null included: null is the unreadable case.
    lab_sync_evaluated = "desired_state" in envelope
    if lab_sync_evaluated:
        findings.extend(lab_sync_findings(envelope, manifest, labelled_rows))

    return {
        "schema_version": SCHEMA_VERSION,
        "host": host_id,
        "lanes_checked": checked,
        "lanes_not_applicable": not_applicable,
        "lanes_skipped_optional_down": skipped_optional_down,
        "lab_sync_evaluated": lab_sync_evaluated,
        "findings": findings,
        "has_drift": len(findings) > 0,
    }


def main() -> int:
    try:
        manifest = load_manifest()
    except LaneCensusConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_NOT_CONFIGURED
    envelope = json.load(sys.stdin)
    # OMN-19088: the same host resolution as the driver — LANE_CENSUS_HOST, else
    # this machine's hostname. An envelope that already carries a host (a
    # replayed fixture) keeps its own.
    envelope.setdefault(
        "host", os.environ.get("LANE_CENSUS_HOST") or socket.gethostname()
    )
    # OMN-19414: the generated desired state, from the path the driver names. An
    # envelope that already carries one (a replayed fixture) keeps its own.
    desired_path = os.environ.get(ENV_DESIRED_STATE, "").strip()
    if desired_path and "desired_state" not in envelope:
        documents, reason = _read_desired_state(Path(desired_path))
        envelope["desired_state"] = documents
        if documents is None:
            envelope["desired_state_error"] = reason
    try:
        plan = build_plan(envelope, manifest)
    except HostDeclarationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_HOST_UNDECLARED
    json.dump(plan, sys.stdout)
    sys.stdout.write("\n")
    if any(f["kind"] == "desired_state_unreadable" for f in plan["findings"]):
        return EXIT_DESIRED_STATE_UNREADABLE
    # Exit 0 on any plan — the planner reports; the shell driver decides exit
    # policy so the JSON plan is always emittable for dry-run/inspection. The
    # non-zero exits are an unattributable host (EXIT_HOST_UNDECLARED) and an
    # absent manifest (EXIT_NOT_CONFIGURED), for which no plan exists to emit,
    # and an unreadable desired state (EXIT_DESIRED_STATE_UNREADABLE), whose plan
    # is emitted above.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
