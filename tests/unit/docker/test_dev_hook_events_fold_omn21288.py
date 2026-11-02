# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21288: the claude hook-events projection runs inside the dev main runtime.

``node_projection_claude_hook_events`` had its own container beside
``omninode-runtime``, and the main runtime excluded the node so the consumer
group had one owner. The fold removes both halves together: no compose file
declares the writer, no compose file sets the exclusion, and the deploy handler
retires a previous revision's writer container before main is recreated.

The consumer group id carries ``KAFKA_INSTANCE_ID``; the retired writer set the
same ``runtime-main`` value as main, so main resumes from its committed offsets.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit.docker.test_dev_tenant_projection_fold_omn21288 import (
    dev_stack_services,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
DOCKER = ROOT / "docker"
DEPLOY = ROOT / "src/omnibase_infra/handlers/handler_runtime_deploy.sh"
FOLD_OVERLAY = DOCKER / "docker-compose.dev-tenant-fold.yml"
REFRESH = ROOT / "scripts/runtime_build/refresh_dev_lane.sh"
WRITER = "projection-claude-hook-events-writer"
NODE = "projection_claude_hook_events"


def _array(text: str, name: str) -> list[str]:
    match = re.search(rf"^readonly {name}=\((.*?)\)$", text, re.M | re.S)
    assert match, name
    return [
        token
        for line in match.group(1).splitlines()
        for token in line.split()
        if not line.strip().startswith("#")
    ]


def test_dev_lane_overlay_neither_starts_the_writer_nor_excludes_the_node() -> None:
    text = (DOCKER / "docker-compose.dev-lane.yml").read_text(encoding="utf-8")
    assert WRITER not in text
    assert "ONEX_RUNTIME_NODE_EXCLUSION" not in text
    assert NODE not in text


def test_fold_overlay_gates_the_writer_under_a_folded_profile() -> None:
    text = FOLD_OVERLAY.read_text(encoding="utf-8")
    assert f'  {WRITER}:\n    profiles: !override ["folded-hook-events"]' in text


def test_dev_main_runtime_keeps_the_instance_id_the_writer_shared() -> None:
    stack = dev_stack_services()
    assert WRITER not in stack
    env = stack["omninode-runtime"]["environment"]
    assert env["KAFKA_INSTANCE_ID"] == "runtime-main"
    assert NODE not in str(env)


def test_no_deploy_set_starts_or_builds_the_writer() -> None:
    deploy = DEPLOY.read_text(encoding="utf-8")
    for name in (
        "RUNTIME_SERVICES",
        "DEV_LANE_ONLY_RUNTIME_SERVICES",
        "DEV_LANE_EXTRA_BROKER_CLIENTS",
    ):
        assert WRITER not in _array(deploy, name), name
    assert WRITER not in REFRESH.read_text(encoding="utf-8")
    events = (ROOT / "scripts/deploy-agent/deploy_agent/events.py").read_text()
    assert WRITER not in events


def test_deploy_handler_retires_the_previous_writer_before_main_restarts() -> None:
    deploy = DEPLOY.read_text(encoding="utf-8")
    assert _array(deploy, "RETIRED_NODE_CARRIER_SERVICES") == [WRITER]
    retire = deploy[deploy.index("OMN-21288: retire folded roles") :]
    loop = retire[: retire.index("local cmd=(")]
    assert '"${RETIRED_NODE_CARRIER_SERVICES[@]}"' in loop
    assert "docker stop" in loop
    assert "docker rm" in loop
    assert "-v" not in re.findall(r"docker rm[^\n]*", loop)[0].split()
    # Retirement precedes the compose up that recreates main.
    assert retire.index("docker stop") < retire.index("up -d --no-deps")
