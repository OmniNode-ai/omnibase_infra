# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-19962: .202 Redpanda caps must leave at least 1 GiB above broker memory.

Check the floor of ``max(--memory + 1 GiB, ceil(1.25 * measured peak))``.
Prerequisite C supplies the measured peak used to size the committed caps;
these tests check the floor against committed defaults, ignoring operator env.
The static layer checks each lane's own overlay without Docker. The render
layer checks Compose's merged stack and skips only if its CLI is absent.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"
GIB = 1 << 30
LANE_STACKS: dict[str, tuple[str, ...]] = {
    "dev-202": (
        "docker-compose.infra.yml",
        "docker-compose.dev-lane.yml",
        "docker-compose.dev-202.yml",
    ),
    "sim-202": ("docker-compose.dogfood.yml", "docker-compose.sim-202.yml"),
}

# go-units RAMInBytes: binary multiples, case-insensitive, optional i/b suffix.
_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp]?)i?b?\s*$", re.IGNORECASE)
_UNIT = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": GIB, "t": 1 << 40, "p": 1 << 50}


def _to_bytes(value: object) -> int:
    """Convert a Compose memory value (integer bytes or size string) to bytes."""
    if isinstance(value, bool):
        raise AssertionError(f"unexpected boolean memory value: {value!r}")
    if isinstance(value, int):
        return value
    match = _SIZE_RE.fullmatch(str(value))
    assert match is not None, f"unparseable compose memory value: {value!r}"
    number, unit = match.groups()
    return int(float(number) * _UNIT[unit.lower()])


def _resolve_default(value: str) -> str:
    """Use the committed default, never the operator's environment value."""
    match = re.fullmatch(r"\$\{[A-Za-z_][A-Za-z0-9_]*:?-([^{}]*)\}", value)
    return match.group(1) if match else value


class _ComposeLoader(yaml.SafeLoader):
    """Tolerate Compose custom tags such as !override and !reset."""


def _drop_tag(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    raise AssertionError(f"unhandled YAML node kind: {type(node).__name__}")


_ComposeLoader.add_multi_constructor("!", _drop_tag)  # type: ignore[no-untyped-call]


def _broker_memory(command: object) -> str:
    """Extract --memory from either supported command-list spelling."""
    assert isinstance(command, list), "redpanda command must be a list"
    for index, item in enumerate(command):
        if item == "--memory":
            assert index + 1 < len(command), "redpanda --memory has no value"
            value = command[index + 1]
            assert isinstance(value, str), "redpanda --memory must be a string"
            return _resolve_default(value)
        if isinstance(item, str) and item.startswith("--memory="):
            return _resolve_default(item.partition("=")[2])
    raise AssertionError("redpanda command is missing --memory")


def _assert_memory_cap(service: dict[str, Any], lane: str, source: str) -> None:
    assert service.get("container_name") == f"omnibase-infra-{lane}-redpanda"
    memory = _broker_memory(service.get("command"))
    mem_limit = service.get("mem_limit")
    deploy_limit = (
        (service.get("deploy") or {})
        .get("resources", {})
        .get("limits", {})
        .get("memory")
    )
    if isinstance(mem_limit, str):
        mem_limit = _resolve_default(mem_limit)
    if isinstance(deploy_limit, str):
        deploy_limit = _resolve_default(deploy_limit)
    if mem_limit is not None and deploy_limit is not None:
        assert _to_bytes(mem_limit) == _to_bytes(deploy_limit), (
            f"{source} redpanda mem_limit and deploy.resources.limits.memory disagree"
        )
    cap = mem_limit if mem_limit is not None else deploy_limit
    assert cap is not None, (
        f"{source} services.redpanda has no memory cap (OMN-19962); "
        "the cap must be sized from the Prerequisite C measured peak"
    )
    assert _to_bytes(cap) >= _to_bytes(memory) + GIB, (
        f"{source} services.redpanda cap {cap!r} must be at least "
        f"--memory {memory!r} + 1 GiB (OMN-19962)"
    )


@pytest.mark.parametrize("lane", LANE_STACKS, ids=list(LANE_STACKS))
def test_overlay_redpanda_memory_cap(lane: str) -> None:
    overlay = DOCKER_DIR / LANE_STACKS[lane][-1]
    with overlay.open() as handle:
        config = yaml.load(handle, Loader=_ComposeLoader)  # noqa: S506 - SafeLoader subclass
    _assert_memory_cap(config["services"]["redpanda"], lane, overlay.name)


def test_size_parser_positive_controls() -> None:
    for value in ("1G", "1g", "1GiB", "1gb", "1024M", "1048576kB", GIB):
        assert _to_bytes(value) == GIB
    assert _to_bytes("0.5g") == GIB // 2
    assert _to_bytes("512") == 512


def test_default_resolver_positive_controls() -> None:
    assert _resolve_default("${X:-4G}") == "4G"
    assert _resolve_default("${X-2G}") == "2G"
    assert _resolve_default("3G") == "3G"
    assert _broker_memory(["--memory", "${X:-4G}"]) == "4G"
    assert _broker_memory(["--memory=${X-2G}"]) == "2G"


def test_missing_broker_memory_is_an_assertion() -> None:
    with pytest.raises(AssertionError, match="missing --memory"):
        _broker_memory(["redpanda", "start", "--reserve-memory", "0M"])


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


@pytest.mark.skipif(
    not _docker_compose_available(),
    reason="docker compose CLI absent; the static layer still guards the cap",
)
@pytest.mark.parametrize("lane", LANE_STACKS, ids=list(LANE_STACKS))
def test_rendered_redpanda_memory_cap(lane: str) -> None:
    command = ["docker", "compose"]
    for name in LANE_STACKS[lane]:
        command += ["-f", str(DOCKER_DIR / name)]
    command += ["config", "--format", "json", "--no-interpolate"]
    result = subprocess.run(
        command, cwd=REPO_ROOT, check=False, capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, (
        f"docker compose config failed for {lane}:\n{result.stderr}"
    )
    service = json.loads(result.stdout)["services"]["redpanda"]
    _assert_memory_cap(service, lane, f"{LANE_STACKS[lane][-1]} ({lane} render)")
