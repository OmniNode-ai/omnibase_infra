# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose isolation for tests that start real containers (OMN-17427).

On 2026-09-29T23:57:58Z ``tests/integration/test_catalog_roundtrip.py`` ran on the
.201 lab host with no ``-p``, against a generated file whose top-level ``name:``
was ``omnibase-infra`` and whose ``container_name`` values were the dev lane's
own. Its ``up -d`` recreated the dev lane's postgres, redpanda, valkey, keycloak
and infisical containers, and its ``down`` removed them, which took the dev
lane's broker off the air.

A test that runs ``docker compose`` against a real daemon therefore uses a
project of its own and proves, before ``up``, that nothing it would create or
remove belongs to a declared lane. The lane set is read from
``deploy/lane-census/lane-manifest.yaml``, the one place lanes are declared, so a
new lane is covered the day it is declared.
"""

from __future__ import annotations

import copy
import os
import uuid
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
LANE_MANIFEST = REPO_ROOT / "deploy" / "lane-census" / "lane-manifest.yaml"
ISOLATED_PREFIX = "onex-test-"


@lru_cache(maxsize=1)
def _lanes() -> dict[str, dict[str, object]]:
    data = yaml.safe_load(LANE_MANIFEST.read_text(encoding="utf-8"))
    lanes = data.get("lanes") if isinstance(data, dict) else None
    if not isinstance(lanes, dict) or not lanes:
        raise AssertionError(f"{LANE_MANIFEST} declares no lanes")
    return {str(k): v for k, v in lanes.items() if isinstance(v, dict)}


def declared_lane_projects() -> frozenset[str]:
    """Every declared lane's compose project, from the lane manifest."""
    projects = {
        str(lane["compose_project"])
        for lane in _lanes().values()
        if lane.get("compose_project")
    }
    if not projects:
        raise AssertionError(f"{LANE_MANIFEST} declares no compose_project")
    return frozenset(projects)


def declared_lane_object_names() -> frozenset[str]:
    """Container and network names the lane manifest declares for any lane."""
    names: set[str] = set()
    for lane in _lanes().values():
        network = lane.get("network")
        if isinstance(network, str) and network:
            names.add(network)
        services = lane.get("services")
        if isinstance(services, list):
            for svc in services:
                if isinstance(svc, dict) and svc.get("name"):
                    names.add(str(svc["name"]))
    return frozenset(names)


def is_lane_owned(name: str) -> bool:
    """True when a Docker object name belongs to a declared lane.

    A lane's project name, any name the manifest declares, and any name that
    starts with a lane project followed by ``-`` or ``_`` (compose's own
    container, volume and network naming) all count.
    """
    if name.startswith(ISOLATED_PREFIX):
        return False
    if name in declared_lane_object_names():
        return True
    return any(
        name == project or name.startswith((f"{project}-", f"{project}_"))
        for project in declared_lane_projects()
    )


def isolated_project_name(purpose: str) -> str:
    """A compose project unique to one test run and never a lane's."""
    slug = "".join(c if c.isalnum() else "-" for c in purpose.lower()).strip("-")
    name = f"{ISOLATED_PREFIX}{slug}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    assert not is_lane_owned(name), name
    return name


def compose_isolation_violations(compose: dict[str, object]) -> list[str]:
    """Why a compose document would touch a declared lane; empty when isolated."""
    problems: list[str] = []
    project = compose.get("name")
    if not isinstance(project, str) or not project:
        problems.append("the compose document names no project")
    elif project in declared_lane_projects() or is_lane_owned(project):
        problems.append(f"project {project!r} is a declared lane")
    services = compose.get("services")
    for svc_name, svc in (services if isinstance(services, dict) else {}).items():
        if not isinstance(svc, dict):
            continue
        cname = svc.get("container_name")
        if isinstance(cname, str) and is_lane_owned(cname):
            problems.append(
                f"service {svc_name} container_name {cname!r} is a lane's container"
            )
    for kind in ("volumes", "networks"):
        block = compose.get(kind)
        for key, spec in (block if isinstance(block, dict) else {}).items():
            name = spec.get("name") if isinstance(spec, dict) else None
            if isinstance(name, str) and is_lane_owned(name):
                problems.append(f"{kind[:-1]} {key} is named {name!r}, a lane's")
    return problems


def assert_compose_isolated(compose: dict[str, object]) -> None:
    problems = compose_isolation_violations(compose)
    assert not problems, (
        "refusing to run docker compose on a document that touches a declared "
        "lane (OMN-17427): " + "; ".join(problems)
    )


def without_host_ports(compose: dict[str, object]) -> dict[str, object]:
    """A copy that publishes no host port, so it cannot collide with a lane's."""
    out = copy.deepcopy(compose)
    services = out.get("services")
    for svc in (services if isinstance(services, dict) else {}).values():
        if isinstance(svc, dict):
            svc.pop("ports", None)
    return out
