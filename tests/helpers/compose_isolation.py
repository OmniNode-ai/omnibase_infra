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
remove belongs to anything else.

OMN-20934: that proof is an ALLOWLIST, not a list of lanes. It used to read the
lane set from the lane manifest, which is private deployment config this
repository no longer ships, and a denylist is only as current as the list it
reads. Now every name a test's compose document owns (its project, and any
container, volume or network name it sets) must carry this module's
``onex-test-`` prefix, which no lane, lab or otherwise, ever uses. A name the
test did not mint is refused whatever it is, so a lane is covered without
anyone declaring it here.
"""

from __future__ import annotations

import copy
import os
import uuid

ISOLATED_PREFIX = "onex-test-"


def is_isolated(name: str) -> bool:
    """True when a Docker object name was minted by a test (carries the prefix)."""
    return name.startswith(ISOLATED_PREFIX)


def isolated_project_name(purpose: str) -> str:
    """A compose project unique to one test run and never a lane's."""
    slug = "".join(c if c.isalnum() else "-" for c in purpose.lower()).strip("-")
    name = f"{ISOLATED_PREFIX}{slug}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    assert is_isolated(name), name
    return name


def compose_isolation_violations(compose: dict[str, object]) -> list[str]:
    """Why a compose document could touch something it does not own; empty when isolated."""
    problems: list[str] = []
    project = compose.get("name")
    if not isinstance(project, str) or not project:
        problems.append("the compose document names no project")
    elif not is_isolated(project):
        problems.append(
            f"project {project!r} is not a test project (no {ISOLATED_PREFIX!r} prefix)"
        )
    services = compose.get("services")
    for svc_name, svc in (services if isinstance(services, dict) else {}).items():
        if not isinstance(svc, dict):
            continue
        cname = svc.get("container_name")
        if isinstance(cname, str) and not is_isolated(cname):
            problems.append(
                f"service {svc_name} container_name {cname!r} is not a test container"
            )
    for kind in ("volumes", "networks"):
        block = compose.get(kind)
        for key, spec in (block if isinstance(block, dict) else {}).items():
            name = spec.get("name") if isinstance(spec, dict) else None
            if isinstance(name, str) and not is_isolated(name):
                problems.append(
                    f"{kind[:-1]} {key} is named {name!r}, not a test {kind[:-1]}"
                )
    return problems


def assert_compose_isolated(compose: dict[str, object]) -> None:
    problems = compose_isolation_violations(compose)
    assert not problems, (
        "refusing to run docker compose on a document that names something a "
        "test did not mint (OMN-17427, OMN-20934): " + "; ".join(problems)
    )


def without_host_ports(compose: dict[str, object]) -> dict[str, object]:
    """A copy that publishes no host port, so it cannot collide with a lane's."""
    out = copy.deepcopy(compose)
    services = out.get("services")
    for svc in (services if isinstance(services, dict) else {}).values():
        if isinstance(svc, dict):
            svc.pop("ports", None)
    return out
