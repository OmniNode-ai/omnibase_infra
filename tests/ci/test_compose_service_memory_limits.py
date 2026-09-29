# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19957: the two runtime consumers carry a 512 MiB memory limit, on every lane.

`context-audit-consumer` and `omninode-contract-resolver` ran pinned at their
256 MiB limit on the .202 dev lane (94-96% of the limit) and hit the same limit
on .201's dev lane (cgroup `memory.events` `max` counters in the hundreds). The
limit is the service's own memory need, so it lives in the base compose file,
`docker/docker-compose.infra.yml`, and every lane inherits it (RULING
2026-09-28T15:31:52Z, rolling work ledger; plan
`beta/plans/2026-09-28-202-memory-hardening-plan.md`, task 3).

Two layers, so the guard never goes quiet:

* The static layer reads every `docker/docker-compose*.yml` as YAML and needs
  no Docker daemon or CLI. It checks the base literal, and that no overlay
  declares a different memory limit for either service.
* The render layer runs `docker compose config --format json` over the base and
  over each lane stack, so compose's own merge semantics (`!override`,
  `!reset`, anchors) decide the effective limit. It supplies only `PATH`,
  `HOME`, and placeholder values for required variables referenced by the
  stack, so interpolation is deterministic without depending on the caller's
  environment. It needs the docker CLI (no daemon), and it skips only when the
  CLI is absent, in which case the static layer still holds the line.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOCKER_DIR = REPO_ROOT / "docker"
BASE_COMPOSE = "docker-compose.infra.yml"

SERVICES = ("context-audit-consumer", "omninode-contract-resolver")
EXPECTED_LIMIT_BYTES = 512 * 1024 * 1024  # 536870912

# Lane stacks as each lane renders them. The dev-202, dev-105, dev-200 and
# prepr overlays sit on top of the dev-lane overlay (see each file's header).
LANE_STACKS: dict[str, tuple[str, ...]] = {
    "base": (BASE_COMPOSE,),
    "dev-lane": (BASE_COMPOSE, "docker-compose.dev-lane.yml"),
    "dev-202": (
        BASE_COMPOSE,
        "docker-compose.dev-lane.yml",
        "docker-compose.dev-202.yml",
    ),
    "dev-105": (
        BASE_COMPOSE,
        "docker-compose.dev-lane.yml",
        "docker-compose.dev-105.yml",
    ),
    "dev-200": (
        BASE_COMPOSE,
        "docker-compose.dev-lane.yml",
        "docker-compose.dev-200.yml",
    ),
    "prepr": (
        BASE_COMPOSE,
        "docker-compose.dev-lane.yml",
        "docker-compose.prepr.yml",
    ),
    "stability-test": (BASE_COMPOSE, "docker-compose.stability-test.yml"),
}

# go-units RAMInBytes: binary multiples, case-insensitive, optional trailing b.
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp]?)i?b?\s*$", re.IGNORECASE)
_UNIT = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40, "p": 1 << 50}
_REQUIRED_VARIABLE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):?\?")


def _to_bytes(value: object) -> int:
    """Convert a compose memory value (int bytes or a size string) to bytes."""
    if isinstance(value, bool):
        raise AssertionError(f"unexpected boolean memory value: {value!r}")
    if isinstance(value, int):
        return value
    match = _SIZE_RE.match(str(value))
    if match is None:
        raise AssertionError(f"unparseable compose memory value: {value!r}")
    number, unit = match.groups()
    return int(float(number) * _UNIT[unit.lower()])


class _ComposeLoader(yaml.SafeLoader):
    """A SafeLoader that tolerates compose's custom tags (`!override`, `!reset`)."""


def _drop_tag(loader: yaml.Loader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    raise AssertionError(f"unhandled YAML node kind: {type(node).__name__}")


_ComposeLoader.add_multi_constructor("!", _drop_tag)  # type: ignore[no-untyped-call]


def _load(path: Path) -> Any:
    with path.open() as handle:
        return yaml.load(handle, Loader=_ComposeLoader)  # noqa: S506 - local tolerant loader


def _memory_limit(service: dict[str, Any]) -> object:
    return (
        (service.get("deploy") or {})
        .get("resources", {})
        .get("limits", {})
        .get("memory")
    )


# --------------------------------------------------------------- static layer


@pytest.mark.parametrize("service", SERVICES)
def test_base_compose_declares_512_mib_limit(service: str) -> None:
    services = _load(DOCKER_DIR / BASE_COMPOSE)["services"]
    assert service in services, f"{service} is missing from {BASE_COMPOSE}"
    limit = _memory_limit(services[service])
    assert limit is not None, f"{service} declares no memory limit in {BASE_COMPOSE}"
    assert _to_bytes(limit) == EXPECTED_LIMIT_BYTES, (
        f"{BASE_COMPOSE} {service} memory limit is {limit!r} "
        f"({_to_bytes(limit)} bytes); expected {EXPECTED_LIMIT_BYTES} (512M). OMN-19957"
    )


_OVERLAYS = sorted(
    p for p in DOCKER_DIR.glob("docker-compose*.yml") if p.name != BASE_COMPOSE
)
assert _OVERLAYS, f"no compose overlays found in {DOCKER_DIR}"


@pytest.mark.parametrize("overlay", _OVERLAYS, ids=lambda p: p.name)
def test_no_overlay_sets_a_different_limit(overlay: Path) -> None:
    services = (_load(overlay) or {}).get("services") or {}
    offenders = []
    for name in SERVICES:
        service = services.get(name)
        if not isinstance(service, dict):
            continue
        limit = _memory_limit(service)
        if limit is not None and _to_bytes(limit) != EXPECTED_LIMIT_BYTES:
            offenders.append(f"{name}: {limit!r}")
    assert not offenders, (
        f"{overlay.name} overrides the memory limit set in {BASE_COMPOSE}: "
        f"{offenders}. The limit is the service's own need and lives in the base "
        "(OMN-19957)."
    )


CATALOG_DIR = DOCKER_DIR / "catalog" / "services"
CATALOG_FILES = {
    "context-audit-consumer": CATALOG_DIR / "context-audit-consumer.yaml",
    "omninode-contract-resolver": CATALOG_DIR / "contract-resolver.yaml",
}


@pytest.mark.parametrize("service", SERVICES)
def test_service_catalog_declares_512_mib_limit(service: str) -> None:
    """The catalog generates `docker-compose.generated.yml`; it must agree with the base."""
    manifest = _load(CATALOG_FILES[service])
    assert manifest["name"] == service
    limit = (manifest.get("resources") or {}).get("memory")
    assert limit is not None, f"catalog {service} declares no memory limit"
    assert _to_bytes(limit) == EXPECTED_LIMIT_BYTES, (
        f"catalog {CATALOG_FILES[service].name} memory is {limit!r}; "
        f"expected {EXPECTED_LIMIT_BYTES} (512M), matching {BASE_COMPOSE}. OMN-19957"
    )


def test_size_parser_positive_controls() -> None:
    assert _to_bytes("256M") == 268435456
    assert _to_bytes("512M") == EXPECTED_LIMIT_BYTES
    assert _to_bytes("512m") == EXPECTED_LIMIT_BYTES
    assert _to_bytes("0.5g") == EXPECTED_LIMIT_BYTES
    assert _to_bytes(536870912) == EXPECTED_LIMIT_BYTES


# --------------------------------------------------------------- render layer


def _docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _compose_environment(stack: tuple[str, ...]) -> dict[str, str]:
    environment = {name: os.environ[name] for name in ("PATH", "HOME")}
    for name in stack:
        contents = (DOCKER_DIR / name).read_text()
        required_variables = _REQUIRED_VARIABLE_RE.findall(contents)
        environment.update(dict.fromkeys(required_variables, "placeholder"))
        for variable in required_variables:
            if variable.endswith(("_PORT", "_REPLICAS")):
                # Compose validates port and replica substitutions as integers.
                environment[variable] = "1"
            elif variable.endswith("_DIR") or variable == "OMNI_HOME":
                # Volume sources and container paths must be path-shaped.
                environment[variable] = str(REPO_ROOT / "placeholder")
    return environment


@pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose CLI absent; the static layer above still guards the limit",
)
@pytest.mark.parametrize("lane", sorted(LANE_STACKS))
def test_rendered_lane_stack_carries_512_mib(lane: str) -> None:
    command = ["docker", "compose", "--profile", "*"]
    for name in LANE_STACKS[lane]:
        command += ["-f", str(DOCKER_DIR / name)]
    command += ["config", "--format", "json"]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=_compose_environment(LANE_STACKS[lane]),
    )
    assert result.returncode == 0, (
        f"docker compose config failed for {lane}:\n{result.stderr}"
    )
    services = json.loads(result.stdout)["services"]
    for name in SERVICES:
        assert name in services, f"{name} missing from the rendered {lane} stack"
        limit = _memory_limit(services[name])
        assert limit is not None, f"{name} has no memory limit in the {lane} render"
        assert _to_bytes(limit) == EXPECTED_LIMIT_BYTES, (
            f"{lane} renders {name} with memory limit {limit!r}; "
            f"expected {EXPECTED_LIMIT_BYTES}. OMN-19957"
        )
