# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24397 -- ``/health`` carries ``hmac_key_check``, the check value of the signing key.

Hermetic: an in-memory aiohttp client over ``create_health_app`` and a real
``JobStore`` on a tmp state dir. The key comes from the process environment the way
the agent reads it; no network and no agent process.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from deploy_agent.health import create_health_app
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

# test_auth_key_check.py holds the same vector against the function itself.
GOLDEN_SECRET = "fixture-key"
GOLDEN_CHECK = "6dce9bc99db04417"

# Begins with a character no payload text carries, so every prefix is a fair probe.
DISTINCT_SECRET = "Q|Zk#9-nothing-else-says-this"


async def _health(state_dir: Path) -> tuple[int, str]:
    app = create_health_app(JobStore(state_dir=state_dir), lambda: "idle")
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        response = await client.get("/health")
        return response.status, await response.text()
    finally:
        await client.close()


async def test_health_carries_the_golden_check_for_the_key_in_its_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", GOLDEN_SECRET)
    status, body = await _health(tmp_path)
    assert status == 200
    assert json.loads(body)["hmac_key_check"] == GOLDEN_CHECK


@pytest.mark.parametrize("blank", [None, ""])
async def test_health_check_is_null_when_no_key_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blank: str | None
) -> None:
    if blank is None:
        monkeypatch.delenv("DEPLOY_AGENT_HMAC_SECRET", raising=False)
    else:
        monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", blank)
    _, body = await _health(tmp_path)
    payload = json.loads(body)
    assert "hmac_key_check" in payload
    assert payload["hmac_key_check"] is None


async def test_health_never_exposes_the_key_or_any_prefix_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", DISTINCT_SECRET)
    _, body = await _health(tmp_path)
    assert json.loads(body)["hmac_key_check"] is not None
    leaked = [
        DISTINCT_SECRET[:length]
        for length in range(1, len(DISTINCT_SECRET) + 1)
        if DISTINCT_SECRET[:length] in body
    ]
    assert leaked == []
