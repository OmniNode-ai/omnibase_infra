# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""GET /job/{correlation_id} serves the job's not_started reason (OMN-21272).

On 2026-10-21 deploy-agent job f19ec60c ended ``failed`` with
``not_started=lane_lock_contended``: an out-of-band refresh held the lane lock,
so the job never ran a phase. The route the post-merge lab guard polls did not
serve that reason, and the guard read a deploy that never happened as a
convergence failure. The reason is served here, and is ``None`` for a job that
started.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from deploy_agent.events import EnumJobNotStarted
from deploy_agent.health import _job_handler
from deploy_agent.job_state import JobStore

pytestmark = pytest.mark.unit


def _serve(store: JobStore, cid: object) -> dict[str, Any]:
    request = SimpleNamespace(
        app={"job_store": store}, match_info={"correlation_id": str(cid)}
    )
    return cast(
        "dict[str, Any]",
        json.loads(asyncio.run(_job_handler(cast("Any", request))).text),
    )


def test_job_endpoint_serves_not_started_reason(tmp_path: Path) -> None:
    store = JobStore(state_dir=tmp_path / "jobs")
    cid = uuid4()
    store.accept(cid, {"scope": "runtime", "runtime_lane": "dev"})
    store.complete(
        cid, status="failed", not_started=EnumJobNotStarted.LANE_LOCK_CONTENDED
    )

    body = _serve(store, cid)
    assert body["status"] == "failed"
    assert body["not_started"] == "lane_lock_contended"


def test_job_endpoint_serves_not_started_null_when_job_started(
    tmp_path: Path,
) -> None:
    store = JobStore(state_dir=tmp_path / "jobs")
    cid = uuid4()
    store.accept(cid, {"scope": "runtime", "runtime_lane": "dev"})
    store.complete(cid, status="success")

    body = _serve(store, cid)
    assert "not_started" in body
    assert body["not_started"] is None


def test_job_endpoint_serves_not_started_null_while_in_flight(
    tmp_path: Path,
) -> None:
    store = JobStore(state_dir=tmp_path / "jobs")
    cid = uuid4()
    store.accept(cid, {"scope": "runtime", "runtime_lane": "dev"})

    body = _serve(store, cid)
    assert body["status"] == "accepted"
    assert body["not_started"] is None
