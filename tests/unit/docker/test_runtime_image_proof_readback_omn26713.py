# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The runtime image proposed for AWS Dev carries the gateway and delegation nodes (OMN-26713).

AWS Dev pins a runtime image built before the workflow gateway node existed. The proof
that a fresh dev build carries the gateway and delegation nodes and answers ``/health``
in the Dev-shaped settings was taken on a lab host and recorded in
``docs/readbacks/OMN-26713_runtime_image_proof.yaml``. These tests hold that readback to
the shape the pin proposal for the cloud owner quotes, and hold the build it names to
the declared composed-runtime build (the infra image, then the marketplace layer built
``FROM`` it).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = [pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[3]
_READBACK = _REPO / "docs" / "readbacks" / "OMN-26713_runtime_image_proof.yaml"
_DEV_LANE = _REPO / "docker" / "docker-compose.dev-lane.yml"

_REQUIRED_NODES = (
    "node_workflow_gateway_effect",
    "node_gateway_workflow_terminal_effect",
    "node_delegate_skill_orchestrator",
    "node_delegation_orchestrator",
)
_SHA40 = re.compile(r"[0-9a-f]{40}")
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class _ComposeLoader(yaml.SafeLoader):
    """SafeLoader that keeps the value under compose's ``!override``/``!reset`` tags."""


def _keep_value(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    raise AssertionError(f"unhandled YAML node kind: {type(node).__name__}")


_ComposeLoader.add_constructor("!override", _keep_value)
_ComposeLoader.add_constructor("!reset", _keep_value)


def _load_readback() -> dict[str, Any]:
    loaded = yaml.safe_load(_READBACK.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _readback_problems(readback: dict[str, Any]) -> list[str]:
    """Every way a readback fails to prove the image; empty when it proves it."""
    problems: list[str] = []
    if not _SHA40.fullmatch(str(readback.get("build_commit", ""))):
        problems.append("build_commit is not a 40-character commit")
    if not _IMAGE_DIGEST.fullmatch(str(readback.get("local_image_digest", ""))):
        problems.append("local_image_digest is not a sha256 digest")
    registry = readback.get("node_registry", {})
    present = set(registry.get("required_present", []))
    for node in _REQUIRED_NODES:
        if node not in present:
            problems.append(f"{node} is not recorded present")
    if registry.get("absent_name_control_matches") != 0:
        problems.append("the absent-name control did not match zero times")
    health = readback.get("health", {})
    if health.get("http_status") != 200:
        problems.append("health did not answer 200")
    if health.get("restart_count") != 0:
        problems.append("the container restarted before the health reading")
    if readback.get("known_bad", {}).get("http_status") == 200:
        problems.append("the known-bad run answered 200")
    return problems


def test_readback_proves_gateway_and_delegation_nodes_and_health() -> None:
    assert _readback_problems(_load_readback()) == []


def test_readback_check_refuses_a_missing_gateway_node() -> None:
    bad = _load_readback()
    bad["node_registry"]["required_present"] = [
        n
        for n in bad["node_registry"]["required_present"]
        if n != "node_workflow_gateway_effect"
    ]
    assert _readback_problems(bad) == [
        "node_workflow_gateway_effect is not recorded present"
    ]


def test_readback_check_refuses_an_unhealthy_run() -> None:
    bad = _load_readback()
    bad["health"]["http_status"] = 503
    assert _readback_problems(bad) == ["health did not answer 200"]


def test_composed_runtime_build_is_the_marketplace_layer_on_the_infra_image() -> None:
    compose = _ComposeLoader(_DEV_LANE.read_text(encoding="utf-8")).get_single_data()
    build = compose["x-dev-lane-composed-runtime-build"]
    assert build["context"] == "../workspace/sibling-repos/omnimarket"
    assert build["dockerfile"] == "Dockerfile.runtime"
    assert build["additional_contexts"]["infra-runtime"] == (
        "service:dev-lane-infra-runtime-image"
    )
    infra = compose["services"]["dev-lane-infra-runtime-image"]["build"]
    assert infra["dockerfile"] == "docker/Dockerfile.runtime"
