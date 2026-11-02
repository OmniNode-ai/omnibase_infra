# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21288: the claude hook-events projection runs inside the dev main runtime.

``node_projection_claude_hook_events`` had its own container beside
``omninode-runtime``, and the main runtime excluded the node so the consumer
group had one owner. The fold removes both active owners together: the dev stack keeps the writer
inert, main no longer excludes the node, and the deploy handler
retires a previous revision's writer container before main is recreated.

The consumer group id carries ``KAFKA_INSTANCE_ID``; the retired writer set the
same ``runtime-main`` value as main, so main resumes from its committed offsets.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

import pytest

from tests.scripts.test_deploy_runtime_folded_services_omn21288 import (
    DEV_PROJECT,
    _function,
    _harness,
)
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


@pytest.mark.parametrize("scope", ["omninode-runtime", "runtime-worker", ""])
@pytest.mark.parametrize("containers", ["writer-a\nwriter-b\n", ""])
@pytest.mark.parametrize("failure", ["", "ps", "stop", "rm"])
def test_deploy_handler_retires_the_previous_writer_before_main_restarts(
    tmp_path: Path, scope: str, containers: str, failure: str
) -> None:
    """Execute the handler against old labelled containers absent from compose."""
    calls = tmp_path / "calls"
    calls.touch()
    retired = _array(DEPLOY.read_text(), "RETIRED_NODE_CARRIER_SERVICES")
    assert retired == [WRITER]
    harness = (
        _harness("", DEV_PROJECT)
        + "\nreadonly RETIRED_NODE_CARRIER_SERVICES=("
        + " ".join(shlex.quote(service) for service in retired)
        + ")\n"
        + _function("restart_services")
        + f"\nCALLS={shlex.quote(str(calls))}"
        + f"\nWRITER={shlex.quote(WRITER)}"
        + f"\nCONTAINERS={shlex.quote(containers)}"
        + f"\nFAILURE={shlex.quote(failure)}"
        + f"\nRUNTIME_BUILD_SERVICES=({shlex.quote(scope) if scope else ''})"
        + "\nRUNTIME_BUILD_SERVICES_OVERRIDE=scoped"
        + r"""
log_step() { :; }
log_info() { :; }
log_cmd() { :; }
resolve_compose_file_args() { local -n out="$1"; out=(); }
exclude_cloud_database_services() { :; }
COMPOSE_PROFILE=runtime
RUNTIME_COMPOSE_WAIT_TIMEOUT_SECONDS=60
docker() {
    printf '%s\n' "$*" >> "$CALLS"
    [[ "$1" != "$FAILURE" ]] || return 9
    case "$1" in
        ps)
            [[ "$*" == "ps -aq --filter label=com.docker.compose.project=$PROJECT --filter label=com.docker.compose.service=$WRITER" ]] || return 8
            printf '%s' "$CONTAINERS" ;;
        stop | rm) ;;
        *) return 8 ;;
    esac
}
compose_up_bounded() { shift; printf 'UP %s\n' "$*" >> "$CALLS"; }
restart_services /tmp "$PROJECT"
"""
    )
    result = subprocess.run(
        ["bash", "-c", harness], capture_output=True, text=True, check=False
    )
    lines = calls.read_text().splitlines()
    if scope != "omninode-runtime":
        assert result.returncode == 0, result.stderr
        assert not any(line.startswith(("ps ", "stop ", "rm ")) for line in lines)
        assert len(lines) == bool(scope)
        if scope:
            assert lines[0].endswith(scope)
        return
    assert lines[0] == (
        f"ps -aq --filter label=com.docker.compose.project={DEV_PROJECT} "
        f"--filter label=com.docker.compose.service={WRITER}"
    )
    failed = failure == "ps" or (bool(containers) and failure in {"stop", "rm"})
    assert (result.returncode != 0) == failed, result.stderr
    if failed:
        assert not any(line.startswith("UP ") for line in lines)
        if failure == "stop":
            assert not any(line.startswith("rm ") for line in lines)
        return
    assert lines[-1].startswith("UP docker compose ")
    assert lines[-1].endswith("omninode-runtime")
    if containers:
        assert lines[1:3] == [
            "stop --time 30 writer-a writer-b",
            "rm writer-a writer-b",
        ]
    else:
        assert len(lines) == 2
    assert WRITER not in lines[-1]
