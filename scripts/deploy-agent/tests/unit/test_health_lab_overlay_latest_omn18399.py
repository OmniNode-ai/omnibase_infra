# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18399 -- the ``/lab-overlay-latest`` endpoint.

The exact-sha ``/lab-overlay/{sha}`` endpoint has no way to answer "what did
the agent last apply" -- and under a busy ``dev`` branch the agent's own
``_current_git_sha`` can resolve to a LATER head than the merge that
triggered a given run, so an intermediate sha's exact record is never
written at all. This endpoint exists so the CI reader has something to
compare the requested sha against via GitHub's compare API (the CI job's own
checkout is depth-1 and cannot resolve ancestry locally).

Hermetic: an in-memory aiohttp test client over ``create_health_app``, a real
``JobStore`` pointed at a tmp state dir, and real files written under
``lab-overlay/``. No network, no deploy agent process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from deploy_agent.health import create_health_app
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

SHA_OLDER = "a" * 40
SHA_NEWER = "0" * 40  # sorts BEFORE SHA_OLDER as a filename; must not win on that basis


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


async def _client(state_dir: Path) -> TestClient:
    store = JobStore(state_dir=state_dir)
    app = create_health_app(store, lambda: "idle")
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def test_404_when_no_record_has_ever_been_written(state_dir: Path) -> None:
    client = await _client(state_dir)
    try:
        resp = await client.get("/lab-overlay-latest")
        assert resp.status == 404
    finally:
        await client.close()


async def test_returns_the_record_with_the_newest_finished_at(state_dir: Path) -> None:
    directory = state_dir / "lab-overlay"
    directory.mkdir(parents=True)
    (directory / f"{SHA_OLDER}.json").write_text(
        json.dumps(
            {"sha": SHA_OLDER, "finished_at": "2026-09-15T08:00:00Z", "checks": []}
        )
    )
    (directory / f"{SHA_NEWER}.json").write_text(
        json.dumps(
            {"sha": SHA_NEWER, "finished_at": "2026-09-15T09:00:00Z", "checks": []}
        )
    )
    client = await _client(state_dir)
    try:
        resp = await client.get("/lab-overlay-latest")
        assert resp.status == 200
        body = await resp.json()
        assert body["sha"] == SHA_NEWER
    finally:
        await client.close()


async def test_500_on_a_malformed_record_not_a_silent_404(state_dir: Path) -> None:
    directory = state_dir / "lab-overlay"
    directory.mkdir(parents=True)
    (directory / f"{SHA_OLDER}.json").write_text("[]")
    client = await _client(state_dir)
    try:
        resp = await client.get("/lab-overlay-latest")
        assert resp.status == 500
    finally:
        await client.close()
