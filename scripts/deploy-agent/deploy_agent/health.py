# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Correlation-aware health and job API."""

from __future__ import annotations

import time
from collections.abc import Callable
from uuid import UUID

from aiohttp import web

from deploy_agent.job_state import JobStore

_start_time = time.monotonic()


def create_health_app(
    job_store: JobStore,
    get_agent_state: Callable[[], str],
) -> web.Application:
    app = web.Application()
    app["job_store"] = job_store
    app["get_agent_state"] = get_agent_state

    app.router.add_get("/health", _health_handler)
    app.router.add_get("/job/{correlation_id}", _job_handler)
    return app


async def _health_handler(request: web.Request) -> web.Response:
    store: JobStore = request.app["job_store"]
    state = request.app["get_agent_state"]()

    active = store.load_active()
    active_job = None
    if active:
        active_job = {
            "correlation_id": str(active.correlation_id),
            "current_phase": str(active.current_phase),
            "started_at": active.accepted_at.isoformat(),
        }

    # Find the most recent completed job for last_result
    last_result = None
    completed_jobs = []
    for path in store.state_dir.glob("*.json"):
        try:
            from deploy_agent.job_state import JobState

            job = JobState.model_validate_json(path.read_text())
            if job.status in ("success", "failed") and job.completed_at:
                completed_jobs.append(job)
        except Exception:  # noqa: BLE001
            continue
    if completed_jobs:
        completed_jobs.sort(key=lambda j: j.completed_at, reverse=True)
        latest = completed_jobs[0]
        last_result = {
            "correlation_id": str(latest.correlation_id),
            "status": latest.status,
            "completed_at": latest.completed_at.isoformat()
            if latest.completed_at
            else None,
            "phase_results": {str(k): str(v) for k, v in latest.phase_results.items()},
        }

    pending_publish = len(store.get_pending_publish())

    return web.json_response(
        {
            "state": state,
            "version": "0.1.0",
            "uptime_seconds": int(time.monotonic() - _start_time),
            "active_job": active_job,
            "last_result": last_result,
            "pending_publish_count": pending_publish,
        }
    )


async def _job_handler(request: web.Request) -> web.Response:
    store: JobStore = request.app["job_store"]
    cid_str = request.match_info["correlation_id"]

    try:
        cid = UUID(cid_str)
    except ValueError:
        return web.json_response({"error": "invalid correlation_id"}, status=400)

    job = store.load(cid)
    if job is None:
        return web.json_response({"error": "not found"}, status=404)

    return web.json_response(
        {
            "correlation_id": str(job.correlation_id),
            "status": job.status,
            "current_phase": str(job.current_phase),
            "phase_results": {str(k): str(v) for k, v in job.phase_results.items()},
            "errors": job.errors,
            "accepted_at": job.accepted_at.isoformat(),
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "result_publish_pending": job.result_publish_pending,
        }
    )
