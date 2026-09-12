# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Correlation-aware health and job API."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from uuid import UUID

from aiohttp import web

from deploy_agent.job_state import JobStore
from deploy_agent.lab_overlay import load_record

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
    # OMN-18200 AC5. The onex-lab overlay record for one merged sha.
    #
    # This endpoint exists because a GitHub Actions artifact can only be created
    # by a job inside a run: the apply happens here, on the lab host, where the
    # images and the k3s socket are, and the receipt must be published there. The
    # `verify-lab-overlay-converged` job in runtime-rebuild-trigger.yml reads
    # this, then emits and uploads the receipt.
    #
    # Read-only, and it serves nothing but the record: the record carries check
    # names, verdicts and evidence, never a credential — the store binding is
    # written to a 0600 file and shredded, and no value from it reaches here.
    app.router.add_get("/lab-overlay/{sha}", _lab_overlay_handler)
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


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


async def _lab_overlay_handler(request: web.Request) -> web.Response:
    """Serve the onex-lab overlay record for one merged sha.

    Three distinct answers, because the reader must not collapse them:

    * ``400`` -- the sha is not an exact 40-character lowercase commit. An
      abbreviated ref could match more than one commit, and a record resolved by
      prefix is a record about the wrong change.
    * ``404`` -- no record exists. "The apply never ran" is a real finding and
      the reader turns it into a FAIL receipt naming the sha; it is never a pass.
    * ``500`` -- a record exists and is unparseable. Distinct from ``404`` on
      purpose: a corrupt record and an absent one point at different defects.
    """
    store: JobStore = request.app["job_store"]
    sha = request.match_info["sha"]
    if not _SHA_RE.match(sha):
        return web.json_response(
            {"error": "sha must be 40 lowercase hex characters", "sha": sha},
            status=400,
        )
    try:
        record = load_record(store.state_dir, sha)
    except (OSError, ValueError) as exc:
        return web.json_response(
            {"error": f"record is unreadable: {exc}", "sha": sha}, status=500
        )
    if record is None:
        return web.json_response({"error": "no record", "sha": sha}, status=404)
    return web.json_response(record)
