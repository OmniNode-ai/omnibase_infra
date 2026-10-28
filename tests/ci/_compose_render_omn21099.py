# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Compose readers shared by the OMN-21099 dev-lane tests.

Two layers, as in tests/ci/test_compose_service_memory_limits.py:

* ``load_compose`` reads one compose file as YAML with a loader that constructs
  compose's own tags (``!override``, ``!reset``) instead of deleting their text,
  so a tag inside a comment or a string value is never touched. It needs no
  Docker daemon or CLI.
* ``render_stack`` runs ``docker compose config --format json`` over an explicit
  lane stack, so compose's own merge semantics decide the effective service. It
  supplies only ``PATH``, ``HOME``, placeholder values for the stack's required
  variables and any overrides the caller names. It needs the docker CLI (no
  daemon); callers skip it only when the CLI is absent.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "docker"
BASE_COMPOSE = "docker-compose.infra.yml"
DEV_LANE_OVERLAY = "docker-compose.dev-lane.yml"

DEV_LANE_STACK: tuple[str, ...] = (BASE_COMPOSE, DEV_LANE_OVERLAY)

_REQUIRED_VARIABLE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):?\?")


class _ComposeLoader(yaml.SafeLoader):
    """A SafeLoader that constructs compose's custom tags (`!override`, `!reset`)."""


def _construct(loader: _ComposeLoader, tag_suffix: str, node: yaml.Node) -> Any:
    if tag_suffix not in ("override", "reset"):
        raise yaml.constructor.ConstructorError(
            None, None, f"unexpected compose tag !{tag_suffix}", node.start_mark
        )
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    raise AssertionError(f"unhandled YAML node kind: {type(node).__name__}")


yaml.add_multi_constructor("!", _construct, Loader=_ComposeLoader)


def load_compose(filename: str) -> dict[str, Any]:
    """One compose file under docker/, tags constructed, anchors merged."""
    with (DOCKER_DIR / filename).open(encoding="utf-8") as handle:
        document = _ComposeLoader(handle).get_single_data()
    assert isinstance(document, dict), filename
    return document


def _layers_the_dev_lane_overlay(filename: str, dev_lane_only: set[str]) -> bool:
    """Whether ``filename`` partially redeclares a service only the overlay defines.

    A service with no image, build or extends cannot render on its own, so a
    file holding one renders only on top of the dev-lane overlay. A lane that
    declares such a service in full (its own image or build) layers the base
    file alone and is not a borrower.
    """
    services = load_compose(filename).get("services") or {}
    return any(
        isinstance(services.get(name), dict)
        and not {"image", "build", "extends"}.intersection(services[name])
        for name in dev_lane_only
    )


def _borrower_stacks() -> dict[str, tuple[str, ...]]:
    overlay_services = set(load_compose(DEV_LANE_OVERLAY)["services"])
    dev_lane_only = overlay_services - set(load_compose(BASE_COMPOSE)["services"])
    return {
        path.name: (*DEV_LANE_STACK, path.name)
        for path in sorted(DOCKER_DIR.glob("docker-compose*.yml"))
        if path.name not in DEV_LANE_STACK
        and _layers_the_dev_lane_overlay(path.name, dev_lane_only)
    }


# Lane stacks that layer the dev-lane overlay, keyed by their own overlay file,
# found from compose content rather than file names or header prose.
BORROWER_STACKS: dict[str, tuple[str, ...]] = _borrower_stacks()


def docker_compose_available() -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _compose_environment(
    stack: tuple[str, ...], overrides: dict[str, str]
) -> dict[str, str]:
    environment = {name: os.environ[name] for name in ("PATH", "HOME")}
    for name in stack:
        contents = (DOCKER_DIR / name).read_text(encoding="utf-8")
        for variable in _REQUIRED_VARIABLE_RE.findall(contents):
            if variable.endswith(("_PORT", "_REPLICAS")):
                # Compose validates port and replica substitutions as integers.
                environment[variable] = "1"
            elif variable.endswith("_DIR") or variable == "OMNI_HOME":
                # Volume sources and container paths must be path-shaped.
                environment[variable] = str(REPO_ROOT / "placeholder")
            else:
                environment[variable] = "placeholder"
    policy = dotenv_values(DOCKER_DIR / "runtime-policy.env")
    for key in ("DEV_REDPANDA_MEMORY", "DEV_REDPANDA_MEM_LIMIT"):
        if key in environment and policy.get(key):
            environment[key] = str(policy[key])
    environment.update(overrides)
    return environment


def render_stack(
    stack: tuple[str, ...], overrides: dict[str, str] | None = None
) -> dict[str, Any]:
    """The services ``docker compose config`` renders for ``stack``."""
    command = ["docker", "compose", "--profile", "*"]
    for name in stack:
        command += ["-f", str(DOCKER_DIR / name)]
    command += ["config", "--format", "json"]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=_compose_environment(stack, overrides or {}),
    )
    assert result.returncode == 0, (
        f"docker compose config failed for {stack}:\n{result.stderr}"
    )
    services = json.loads(result.stdout)["services"]
    assert isinstance(services, dict)
    return services
