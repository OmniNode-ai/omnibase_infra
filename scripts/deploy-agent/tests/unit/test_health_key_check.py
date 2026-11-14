# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-24397 -- the ``/health`` payload carries ``hmac_key_check``.

Hermetic: an in-memory aiohttp test client over ``create_health_app`` with a
real ``JobStore`` pointed at a tmp state dir. No network, no deploy agent
process.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from deploy_agent.health import create_health_app
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit

_GOLDEN_SECRET = "fixture-key"
_GOLDEN_VALUE = "6dce9bc99db04417"


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    return tmp_path / "state"


async def _client(state_dir: Path) -> TestClient:
    store = JobStore(state_dir=state_dir)
    app = create_health_app(store, lambda: "idle")
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def test_health_carries_the_golden_check_value(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", _GOLDEN_SECRET)
    client = await _client(state_dir)
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        body = await resp.json()
        assert body["hmac_key_check"] == _GOLDEN_VALUE
    finally:
        await client.close()


async def test_health_check_is_null_without_a_key(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEPLOY_AGENT_HMAC_SECRET", raising=False)
    client = await _client(state_dir)
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        body = await resp.json()
        assert body["hmac_key_check"] is None
    finally:
        await client.close()


async def test_health_never_exposes_the_key(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "test-secret-abc123"
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", secret)
    client = await _client(state_dir)
    try:
        resp = await client.get("/health")
        text = await resp.text()
        assert secret not in text
        assert secret[:8] not in text
        assert secret.encode().hex() not in text
    finally:
        await client.close()


async def test_health_status_and_other_fields_unchanged(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEPLOY_AGENT_HMAC_SECRET", _GOLDEN_SECRET)
    client = await _client(state_dir)
    try:
        resp = await client.get("/health")
        assert resp.status == 200
        body = await resp.json()
        assert "state" in body
        assert "version" in body
    finally:
        await client.close()
