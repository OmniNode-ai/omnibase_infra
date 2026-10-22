# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21288: projection writers, consumers and effects run inside the one runtime.

Mechanism for the operator ruling that no container runs a single projection and
that no separate container runs effects: every projection writer, event
consumer, effects process and worker runs as nodes inside the one runtime
container, so the dev stack holds the runtime plus its backing services.

AC1 -- the stack check refuses a service that runs a projection writer, an
event consumer or an effects/worker process outside the runtime container, shown
over fixture compose files (a positive and a negative control).

AC2 -- the same check over the real dev stack (the base infra compose file with
the dev-lane overlay merged on top). The services that still run outside the
runtime today are named in ``SEPARATE_SERVICES``: a shrink-only list. A service
absent from the list that the check flags fails the run; a listed service that
the check no longer flags fails the run too, so the list can only lose entries.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_DOCKER = _REPO / "docker"
_FIXTURES = _REPO / "tests" / "fixtures" / "omn21288"

# The runtime image every ONEX kernel and sidecar of this repo is built from.
_RUNTIME_DOCKERFILE = "docker/Dockerfile.runtime"

# The one runtime container runs under this profile; any other profile on a
# runtime-image service is a second kernel process (effects, workers, a
# profile carrier, a projection API).
_RUNTIME_PROFILE = "main"

# A runtime-image service whose command is ``python -m <module>`` and whose
# module path names a projection or a consumer is a writer or consumer process
# started beside the runtime instead of as a node inside it.
_WORKLOAD_MODULE = re.compile(r"(^|\.)(node_projection\w*|\w*consumer)(\.|$)")

# Services the real dev stack still runs outside the runtime container. Shrink
# only: move a service's node into the runtime, then delete its line here.
SEPARATE_SERVICES: tuple[str, ...] = (
    # Effects and worker processes.
    "runtime-effects",
    "runtime-worker",
    # A second kernel boot under its own profile.
    "tenant-projection-writer",
    "projection-api",
    # Event consumers.
    "agent-actions-consumer",
    "skill-lifecycle-consumer",
    "context-audit-consumer",
    "infra-routing-decisions-consumer",
    # Projection writers.
    "projection-delegation-writer",
    "projection-live-events-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-tenant-registry-writer",
)


# Compose merge tags (``!override``, ``!reset``) are directives to compose, not data.
_COMPOSE_TAG = re.compile(r"(?m)(:\s+|-\s+)!(?:override|reset)\b[ \t]*")


def _load(path: Path) -> dict[str, Any]:
    """Load a compose file, reading its merge tags as plain data."""
    loaded = yaml.safe_load(_COMPOSE_TAG.sub(r"\1", path.read_text(encoding="utf-8")))
    return loaded if isinstance(loaded, dict) else {}


def load_compose(path: Path) -> dict[str, dict[str, Any]]:
    """The ``services`` mapping of one compose file."""
    return dict(_load(path).get("services") or {})


def _env(service: Mapping[str, Any]) -> dict[str, str]:
    env = service.get("environment") or {}
    if isinstance(env, list):
        return dict(str(e).split("=", 1) for e in env if "=" in str(e))
    return {str(k): str(v) for k, v in env.items()}


def dev_stack_services() -> dict[str, dict[str, Any]]:
    """The dev stack: the base infra services with the dev-lane overlay merged on."""
    merged = load_compose(_DOCKER / "docker-compose.infra.yml")
    for name, overlay in load_compose(_DOCKER / "docker-compose.dev-lane.yml").items():
        base = dict(merged.get(name) or {})
        env = {**_env(base), **_env(overlay)}
        base.update(overlay)
        if env:
            base["environment"] = env
        merged[name] = base
    return merged


def _built_from_runtime_image(service: Mapping[str, Any]) -> bool:
    build = service.get("build")
    dockerfile = build.get("dockerfile") if isinstance(build, Mapping) else build
    return str(dockerfile or "").endswith(_RUNTIME_DOCKERFILE)


def find_out_of_runtime_workloads(
    services: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    """Services that run a writer, consumer or effects process outside the runtime.

    Returns service name -> reason. A backing service (its own third-party
    image) and a runtime-image service that is neither a second kernel profile
    nor a ``python -m`` projection/consumer command are not workloads here.
    """
    refused: dict[str, str] = {}
    for name, service in services.items():
        if not _built_from_runtime_image(service):
            continue
        profile = _env(service).get("RUNTIME_PROFILE", _RUNTIME_PROFILE)
        if profile != _RUNTIME_PROFILE:
            refused[name] = f"runs the runtime kernel under profile {profile!r}"
            continue
        command = service.get("command") or []
        argv = (
            command.split() if isinstance(command, str) else [str(c) for c in command]
        )
        if argv[:2] == ["python", "-m"] and len(argv) > 2:
            if _WORKLOAD_MODULE.search(argv[2]):
                refused[name] = f"runs {argv[2]} as its own process"
    return refused


def test_fixture_runtime_only_stack_is_clean() -> None:
    stack = load_compose(_FIXTURES / "stack_runtime_only.yml")
    assert find_out_of_runtime_workloads(stack) == {}


def test_fixture_separate_workloads_are_each_refused() -> None:
    stack = load_compose(_FIXTURES / "stack_separate_workloads.yml")
    refused = find_out_of_runtime_workloads(stack)
    assert sorted(refused) == [
        "effects-process",
        "event-consumer",
        "widget-projection-writer",
        "worker-process",
    ]


def test_real_dev_stack_runs_no_workload_outside_the_runtime() -> None:
    stack = dev_stack_services()
    refused = find_out_of_runtime_workloads(stack)
    new = sorted(set(refused) - set(SEPARATE_SERVICES))
    assert not new, f"services run outside the runtime container: {new}"
    stale = sorted(set(SEPARATE_SERVICES) - set(refused))
    assert not stale, f"delete these entries from SEPARATE_SERVICES: {stale}"


def test_real_dev_stack_keeps_the_runtime_itself_unflagged() -> None:
    """Positive control: the check reads the real stack and spares the runtime."""
    stack = dev_stack_services()
    assert "omninode-runtime" in stack
    assert "omninode-runtime" not in find_out_of_runtime_workloads(stack)
    assert find_out_of_runtime_workloads(stack), "the check read no workload at all"


def test_separate_services_list_names_each_service_once() -> None:
    assert len(set(SEPARATE_SERVICES)) == len(SEPARATE_SERVICES)
