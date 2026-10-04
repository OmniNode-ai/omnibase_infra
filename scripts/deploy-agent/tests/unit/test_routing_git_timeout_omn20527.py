# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-20527 AC3: a git timeout in the routing table read takes the default route.

Measured on .200, 2026-10-01 17:06:46 local: ``GitTableAtRef.__call__`` ran
``git fetch`` under a 30-second ``subprocess.run`` timeout, the fetch outran it,
and the ``subprocess.TimeoutExpired`` propagated out of ``poll_and_accept`` and
killed the dev-200 deploy agent (launchd last exit 1). An unreadable table
already has a defined outcome -- ``DeployRouter.decide`` takes the pinned
default with a ``routing_table_unreadable_at_ref`` warning -- so a slow git call
must land there too instead of ending the process.
"""

from __future__ import annotations

import subprocess
from typing import Any
from uuid import uuid4

import pytest
from deploy_agent.events import ModelRebuildRequested
from deploy_agent.routing import (
    PINNED_DEFAULT_INSTANCE,
    DeployRouter,
    GitTableAtRef,
    RoutingTableError,
    parse_routing_table,
)

pytestmark = pytest.mark.unit

SHA = "d" * 40

TABLE = """
default_instance: dev-201
instances:
  dev-201:
    hostnames: [omninode-pc]
    consumer_group: onex-deploy-agent
  dev-202:
    hostnames: [omnipc2]
    consumer_group: onex-deploy-agent-dev-202
routes:
  - runtime_lane: dev
    requester_repository: omnimarket
    instance: dev-202
"""


def _timing_out_on(verb: str) -> Any:
    """A runner whose ``git <verb>`` times out; every other call says "missing"."""

    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        if argv[3] == verb:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="")

    return run


@pytest.mark.parametrize("verb", ["cat-file", "fetch"])
def test_a_git_timeout_becomes_a_routing_table_error(verb: str) -> None:
    reader = GitTableAtRef("/nonexistent", run=_timing_out_on(verb))
    with pytest.raises(RoutingTableError, match="timed out"):
        reader(SHA)


def test_a_git_timeout_on_show_becomes_a_routing_table_error() -> None:
    def run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        if argv[3] == "show":
            raise subprocess.TimeoutExpired(cmd=argv, timeout=timeout)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    with pytest.raises(RoutingTableError, match="timed out"):
        GitTableAtRef("/nonexistent", run=run)(SHA)


def test_a_timing_out_fetch_routes_the_command_to_the_default() -> None:
    table = parse_routing_table(TABLE)
    router = DeployRouter(
        table,
        table.instances["dev-202"],
        GitTableAtRef("/nonexistent", run=_timing_out_on("fetch")),
    )
    cmd = ModelRebuildRequested.model_validate(
        {
            "correlation_id": str(uuid4()),
            "requested_by": "gha/omnimarket/pr-1",
            "scope": "full",
            "runtime_lane": "dev",
            "build_source": "workspace",
            "git_ref": SHA,
        }
    )

    decision = router.decide(cmd)

    assert decision.instance == PINNED_DEFAULT_INSTANCE
    assert "unreadable" in decision.basis
