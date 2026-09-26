# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Every catalog-rendered runtime kernel names its lane and exactly one overlay source.

OMN-19749 (runtime lane overlays plan, task LO5). A runtime learns which lane it
is only from a ``runtime.lane`` overlay document that whoever runs it supplies.
It needs two things from its deployment to find that document:

* **a lane**: ``ONEX_RUNTIME_LANE``, the bootstrap identity naming the lane, and
* **a source**: exactly one of the two config overlay sources, never both and
  never neither. ``store`` is selected by a non-blank ``INFISICAL_ADDR`` (the
  stack's own config store); ``local-home`` by ``~/.onex/config.yaml`` saying
  ``config_source: local-home`` with the documents under
  ``~/.omninode/config/<environment>/<lane>/``.

This module renders every bundle in ``docker/catalog/bundles.yaml`` through the
catalog generator (the path ``onex up <bundle>`` and ``make up-local`` take) and
refuses a runtime kernel service that could render without a lane, or with no
source or two. A lane is satisfied only by a non-empty literal or a required
``${VAR:?...}`` reference: a soft ``${VAR:-default}`` is a lane the deployment
never chose, which is the silent path this plan removes.

It also pins the laptop env template to the shipped example document
(``omnibase_infra.examples.config_overlays``), which ``onex local init`` writes,
so the lane the laptop runtime names is the lane that document declares.

The module names no lane of any deployment: the one lane id it reads is the
example's own, taken from the example file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from omnibase_infra.docker.catalog.generator import generate_compose
from omnibase_infra.docker.catalog.resolver import CatalogResolver
from omnibase_infra.examples.config_overlays import read_runtime_lane_example

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_CATALOG_DIR = _REPO / "docker" / "catalog"
_BUNDLES = sorted(yaml.safe_load((_CATALOG_DIR / "bundles.yaml").read_text()))
_LAPTOP_ENV_TEMPLATE = _REPO / "docker" / "local.env.example"

_LANE_VAR = "ONEX_RUNTIME_LANE"
_STORE_VAR = "INFISICAL_ADDR"
# Where the runtime image's user (omniinfra, docker/Dockerfile.runtime) reads the
# local-home source from: its bootstrap file and its documents root.
_CONTAINER_HOME = "/home/omniinfra"
_LOCAL_HOME_TARGETS = frozenset(
    {f"{_CONTAINER_HOME}/.onex/config.yaml", f"{_CONTAINER_HOME}/.omninode/config"}
)

_REQUIRED_REF = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*:\?[^}]*\}$")
_ANY_REF = re.compile(r"\$\{")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _is_runtime_kernel(service: Mapping[str, object]) -> bool:
    """A service that boots the runtime kernel: the runtime image, default command.

    The image's CMD is the kernel (``onex-runtime``); a service that sets its own
    ``command`` runs something else in the same image (a projection runner, a
    consumer) and resolves no lane.
    """
    return str(service.get("image", "")).endswith("runtime:latest") and (
        "command" not in service
    )


def _env(service: Mapping[str, object]) -> dict[str, str]:
    raw = service.get("environment") or {}
    if isinstance(raw, list):
        return dict(str(item).split("=", 1) for item in raw)
    assert isinstance(raw, dict)
    return {str(k): str(v) for k, v in raw.items()}


def _volume_targets(service: Mapping[str, object]) -> set[str]:
    targets: set[str] = set()
    for volume in service.get("volumes") or []:  # type: ignore[attr-defined]
        # "<source>:<target>[:mode]"; a source may be a ${VAR:?msg} reference
        # whose message holds colons, so split after the closing brace.
        text = str(volume)
        rest = text[text.index("}") + 1 :] if text.startswith("${") else text
        parts = rest.split(":")
        target = parts[1] if len(parts) > 1 else parts[0]
        targets.add(target)
    return targets


def lane_problem(env: Mapping[str, str]) -> str | None:
    """Why this environment does not guarantee a lane, or ``None`` when it does."""
    if _LANE_VAR not in env:
        return f"{_LANE_VAR} is not set"
    value = env[_LANE_VAR].strip()
    if _REQUIRED_REF.match(value):
        return None
    if not value:
        return f"{_LANE_VAR} is empty"
    if _ANY_REF.search(value):
        return f"{_LANE_VAR}={value!r} can render without a lane (not a :? reference)"
    if not _SLUG.match(value):
        return f"{_LANE_VAR}={value!r} is not a lane id"
    return None


def sources(service: Mapping[str, object]) -> set[str]:
    """The overlay sources this service's render guarantees."""
    found: set[str] = set()
    store = _env(service).get(_STORE_VAR, "").strip()
    if store and (_REQUIRED_REF.match(store) or not _ANY_REF.search(store)):
        found.add("store")
    if _volume_targets(service) >= _LOCAL_HOME_TARGETS:
        found.add("local-home")
    return found


def kernel_problems(compose: Mapping[str, object]) -> list[str]:
    """Every runtime kernel in ``compose`` without a lane or with not one source."""
    services = compose.get("services") or {}
    assert isinstance(services, dict)
    problems: list[str] = []
    for name, service in services.items():
        if not _is_runtime_kernel(service):
            continue
        lane = lane_problem(_env(service))
        if lane is not None:
            problems.append(f"{name}: {lane}")
        found = sources(service)
        if len(found) != 1:
            problems.append(
                f"{name}: needs exactly one overlay source (store via a non-blank "
                f"{_STORE_VAR}, or local-home via mounts at "
                f"{sorted(_LOCAL_HOME_TARGETS)}); found {sorted(found) or 'none'}"
            )
    return problems


def _render(bundle: str) -> dict[str, object]:
    resolved = CatalogResolver(catalog_dir=str(_CATALOG_DIR)).resolve([bundle])
    return generate_compose(resolved)


@pytest.mark.parametrize("bundle", _BUNDLES)
def test_every_runtime_kernel_has_a_lane_and_one_source(bundle: str) -> None:
    assert kernel_problems(_render(bundle)) == []


def test_the_check_sees_runtime_kernels() -> None:
    """Guard against a vacuous pass: the product bundles do render kernels."""
    for bundle in ("local", "runtime"):
        services = _render(bundle)["services"]
        assert isinstance(services, dict)
        assert [n for n, s in services.items() if _is_runtime_kernel(s)], bundle


def test_laptop_kernels_read_local_home_and_the_full_stack_reads_the_store() -> None:
    for bundle, expected in (("local", {"local-home"}), ("runtime", {"store"})):
        services = _render(bundle)["services"]
        assert isinstance(services, dict)
        for name, service in services.items():
            if _is_runtime_kernel(service):
                assert sources(service) == expected, (bundle, name)


# --- the checker refuses what it must (positive controls) -------------------


def _kernel(env: dict[str, str], volumes: list[str] | None = None) -> dict[str, object]:
    service: dict[str, object] = {"image": "runtime:latest", "environment": env}
    if volumes is not None:
        service["volumes"] = volumes
    return {"services": {"kernel": service}}


_STORE_ENV = {_STORE_VAR: "http://config-store:8080"}
_LOCAL_HOME_VOLUMES = [
    "${HOME:?HOME must be set}/.onex/config.yaml:/home/omniinfra/.onex/config.yaml:ro",
    "${HOME:?HOME must be set}/.omninode/config:/home/omniinfra/.omninode/config:ro",
]


@pytest.mark.parametrize(
    ("env", "volumes", "fragment"),
    [
        (dict(_STORE_ENV), None, "is not set"),
        ({**_STORE_ENV, _LANE_VAR: ""}, None, "is empty"),
        ({**_STORE_ENV, _LANE_VAR: "${ONEX_RUNTIME_LANE:-x}"}, None, "without a lane"),
        ({**_STORE_ENV, _LANE_VAR: "Not A Lane"}, None, "not a lane id"),
        ({_LANE_VAR: "a-lane"}, None, "found none"),
        ({_LANE_VAR: "a-lane", _STORE_VAR: ""}, None, "found none"),
        ({_LANE_VAR: "a-lane", _STORE_VAR: "${X:-}"}, None, "found none"),
        ({_LANE_VAR: "a-lane", **_STORE_ENV}, _LOCAL_HOME_VOLUMES, "local-home"),
        ({_LANE_VAR: "a-lane"}, _LOCAL_HOME_VOLUMES[:1], "found none"),
    ],
)
def test_checker_refuses(
    env: dict[str, str], volumes: list[str] | None, fragment: str
) -> None:
    problems = kernel_problems(_kernel(env, volumes))
    assert problems, (env, volumes)
    assert any(fragment in p for p in problems), problems


@pytest.mark.parametrize(
    ("env", "volumes"),
    [
        ({_LANE_VAR: "a-lane", **_STORE_ENV}, None),
        ({_LANE_VAR: "${ONEX_RUNTIME_LANE:?name it}", **_STORE_ENV}, None),
        ({_LANE_VAR: "a-lane", _STORE_VAR: "${X:?set it}"}, None),
        ({_LANE_VAR: "a-lane", _STORE_VAR: ""}, _LOCAL_HOME_VOLUMES),
    ],
)
def test_checker_accepts(env: dict[str, str], volumes: list[str] | None) -> None:
    assert kernel_problems(_kernel(env, volumes)) == []


def test_checker_ignores_a_service_that_runs_its_own_command() -> None:
    compose: dict[str, object] = {
        "services": {"writer": {"image": "runtime:latest", "command": ["x"]}}
    }
    assert kernel_problems(compose) == []


# --- the laptop names the lane the shipped example declares -----------------


def test_laptop_env_template_names_the_example_lane() -> None:
    """``onex local init`` writes the shipped example; the laptop env names it."""
    example = json.loads(read_runtime_lane_example())
    lines = [
        line.split("=", 1)
        for line in _LAPTOP_ENV_TEMPLATE.read_text().splitlines()
        if line.startswith(f"{_LANE_VAR}=")
    ]
    assert lines == [[_LANE_VAR, example["lane_id"]]]
