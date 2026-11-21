# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26122 -- ``/health`` names the lane the agent's job rolls.

The one agent serves every lane single-flight. A probe that
grades one lane has to tell that lane's deploy from another's, and ``/health``
carried neither the active job's lane nor the last result's. The name is the
only addition: ``/job/{id}`` and ``/health`` still expose no command or payload.

Hermetic: an in-memory aiohttp client over ``create_health_app`` and a real
``JobStore`` on a tmp state dir.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from aiohttp.test_utils import TestClient, TestServer
from deploy_agent.health import create_health_app
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

ACTIVE = UUID("c80367de-9020-4d9a-a14b-a002daad3012")
DONE = UUID("c8fd5fda-5f5b-4af1-8a84-77e1fc8c2a64")


async def _health(store: JobStore) -> dict[str, Any]:
    client = TestClient(TestServer(create_health_app(store, lambda: "deploying")))
    await client.start_server()
    try:
        resp = await client.get("/health")
        body: dict[str, Any] = await resp.json()
        return body
    finally:
        await client.close()


async def test_active_job_and_last_result_name_their_lane(tmp_path: Path) -> None:
    store = JobStore(state_dir=tmp_path / "state")
    store.accept(DONE, {"runtime_lane": "lane-b", "requested_by": "agent/x"})
    store.complete(DONE, status="failed", errors=["e"])
    store.accept(
        ACTIVE, {"runtime_lane": "lane-a", "requested_by": "agent/idle-restore"}
    )

    body = await _health(store)

    assert body["active_job"]["runtime_lane"] == "lane-a"
    assert body["last_result"]["runtime_lane"] == "lane-b"
    assert "command" not in body["active_job"]


async def test_a_command_without_a_lane_reads_empty_never_a_guess(
    tmp_path: Path,
) -> None:
    store = JobStore(state_dir=tmp_path / "state")
    store.accept(ACTIVE, {"requested_by": "agent/idle-restore"})

    body = await _health(store)

    assert body["active_job"]["runtime_lane"] == ""
