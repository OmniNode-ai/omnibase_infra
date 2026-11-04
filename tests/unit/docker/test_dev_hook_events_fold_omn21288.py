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
import yaml

from scripts.render_runtime_policy_env import load_contract, render_env
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


def _services(overlay: Path) -> dict[str, dict[str, object]]:
    """The ``services`` mapping of a compose overlay, parsed (comments ignored)."""
    # compose's ``!override`` tag only changes how overlays merge; the value it wraps is plain YAML.
    parsed = yaml.safe_load(
        re.sub(r"!override[ \t]*", "", overlay.read_text(encoding="utf-8"))
    )
    services = parsed["services"]
    assert isinstance(services, dict)
    return services


def _array(text: str, name: str) -> list[str]:
    """Members of ``readonly name=( ... )``; a ``#`` comment, whole-line or inline, is no member."""
    match = re.search(rf"^readonly {name}=\((.*?)\)$", text, re.M | re.S)
    assert match, name
    return [
        token
        for line in match.group(1).splitlines()
        for token in line.split("#", 1)[0].split()
    ]


def test_array_reader_skips_whole_line_and_inline_comments() -> None:
    text = (
        "readonly SAMPLE=(\n"
        f"    # {WRITER} named in prose only\n"
        f"    tenant-projection-writer  # unlike {WRITER}\n"
        "    onex-api\n"
        ")\n"
    )
    assert _array(text, "SAMPLE") == ["tenant-projection-writer", "onex-api"]


def test_dev_lane_overlay_neither_starts_the_writer_nor_excludes_the_node() -> None:
    services = _services(DOCKER / "docker-compose.dev-lane.yml")
    assert "omninode-runtime" in services  # positive control: the overlay really parsed
    assert WRITER not in services
    for name, service in services.items():
        rendered = yaml.safe_dump(service)
        assert "ONEX_RUNTIME_NODE_EXCLUSION" not in rendered, name
        assert NODE not in rendered, name


def test_fold_overlay_gates_the_writer_under_a_folded_profile() -> None:
    assert _services(FOLD_OVERLAY)[WRITER]["profiles"] == ["folded-hook-events"]


def test_dev_main_runtime_keeps_the_instance_id_the_writer_shared() -> None:
    stack = dev_stack_services()
    assert WRITER not in stack
    env = stack["omninode-runtime"]["environment"]
    assert env["KAFKA_INSTANCE_ID"] == "runtime-main"
    assert NODE not in str(env)


def test_no_deploy_set_starts_or_builds_the_writer() -> None:
    deploy = DEPLOY.read_text(encoding="utf-8")
    # Positive control: each set is read with a member it really carries, so an
    # empty parse cannot pass the absence check below.
    for name, member in (
        ("RUNTIME_SERVICES", "omninode-runtime"),
        ("DEV_LANE_ONLY_RUNTIME_SERVICES", "tenant-projection-writer"),
        ("DEV_LANE_EXTRA_BROKER_CLIENTS", "context-audit-consumer"),
    ):
        members = _array(deploy, name)
        assert member in members, name
        assert WRITER not in members, name
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
    # The retirement list is the policy's, rendered once: the handler declares none.
    assert "RETIRED_NODE_CARRIER_SERVICES" not in DEPLOY.read_text(encoding="utf-8")
    process = load_contract().profiles["dev"].processes["main"]
    assert list(process.retired_services) == [WRITER]
    assert render_env(load_contract())["DEV_RUNTIME_MAIN_RETIRED_SERVICES"] == WRITER
    harness = (
        _harness("", DEV_PROJECT, dev_profiles="projection-writer-savings")
        + f"\nDEV_RUNTIME_MAIN_RETIRED_SERVICES={shlex.quote(WRITER)}\n"
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
    # The folded savings writer has no old container here; only the retired
    # hook-events writer's census is recorded and made to fail.
    [[ "$1" != ps || "$*" == *"service=$WRITER" ]] || return 0
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
