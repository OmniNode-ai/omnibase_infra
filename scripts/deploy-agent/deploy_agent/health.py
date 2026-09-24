# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Correlation-aware health and job API."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from datetime import datetime
from uuid import UUID

from aiohttp import web

from deploy_agent.accept_backlog import (
    EnumAcceptBacklogStatus,
    ModelAcceptBacklogVerdict,
)
from deploy_agent.job_state import JobState, JobStore
from deploy_agent.lab_overlay import load_latest_record, load_record
from deploy_agent.loaded_code import loaded_code_sha_if_recorded
from deploy_agent.queue_depth import ModelControlTopicLag, compute_queue_snapshot

_start_time = time.monotonic()


def create_health_app(
    job_store: JobStore,
    get_agent_state: Callable[[], str],
    get_accept_backlog: Callable[[], ModelAcceptBacklogVerdict | None] | None = None,
    get_control_topic_lag: Callable[[], ModelControlTopicLag] | None = None,
) -> web.Application:
    """Build the agent's HTTP surface.

    ``get_accept_backlog`` is the watchdog's latest verdict (OMN-18636 AC4). It
    is optional because the watchdog is a property of a running agent and this
    app is also built by tests that are not exercising it; where it is absent
    the payload says INDETERMINATE rather than claiming a healthy queue.

    ``get_control_topic_lag`` is the consumer's latest lag sample (OMN-18144),
    optional on the same terms and unknown rather than zero where it is absent.
    """
    app = web.Application()
    app["job_store"] = job_store
    app["get_agent_state"] = get_agent_state
    app["get_accept_backlog"] = get_accept_backlog
    app["get_control_topic_lag"] = get_control_topic_lag

    app.router.add_get("/health", _health_handler)
    app.router.add_get("/job/{correlation_id}", _job_handler)
    # OMN-18144. What is ahead of a command, and how fast this agent has been
    # draining it. The post-merge lab-pass guard reads this to bound its wait
    # by queue position instead of a clock: on 2026-09-18 a merge that was
    # third in line receipted FAIL against a healthy lane because nothing
    # reachable from CI could say it was third in line.
    #
    # READ-ONLY, and deliberately so. It exposes no command, no payload and no
    # credential -- correlation ids, counts and durations only -- and nothing
    # here can cancel, reorder or curtail a job. A reader that could shorten a
    # rebuild to fit its own window would manufacture the false FAIL this
    # endpoint exists to remove.
    app.router.add_get("/queue", _queue_handler)
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
    # OMN-18399. The most recently written record, any sha. Exists so the CI
    # reader can ask "what did the agent last apply" when the exact sha it
    # requested was never written -- the deploy agent keys its record by
    # whatever `dev` HEAD it resolved to when it reached the job, which on a
    # busy branch can be a later descendant of the triggering merge. The
    # reader compares the requested sha against this one via GitHub's compare
    # API, never locally: this endpoint carries no ancestry logic itself.
    app.router.add_get("/lab-overlay-latest", _lab_overlay_latest_handler)
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

    # Find the most recent completed job for last_result.
    #
    # OMN-18640: the completion timestamp is carried ALONGSIDE the job rather
    # than read back out of it inside the sort key. ``completed_at`` is
    # ``datetime | None`` on the model and the guard above is what makes it
    # non-None here; a lambda reaching back into the model cannot express that,
    # so the sort was typed as ordering by a possibly-``None`` key -- a real
    # ``TypeError`` waiting for the day the guard is loosened, not a nuisance.
    last_result = None
    completed_jobs: list[tuple[datetime, JobState]] = []
    for path in store.state_dir.glob("*.json"):
        try:
            job = JobState.model_validate_json(path.read_text())
            completed_at = job.completed_at
            if job.status in ("success", "failed") and completed_at is not None:
                completed_jobs.append((completed_at, job))
        except Exception:  # noqa: BLE001
            continue
    if completed_jobs:
        completed_jobs.sort(key=lambda pair: pair[0], reverse=True)
        latest = completed_jobs[0][1]
        last_result = {
            "correlation_id": str(latest.correlation_id),
            "status": latest.status,
            "completed_at": latest.completed_at.isoformat()
            if latest.completed_at
            else None,
            "phase_results": {str(k): str(v) for k, v in latest.phase_results.items()},
            # OMN-18636 AC5. A reader that only looks at /health must be able to
            # tell a finished job from one whose post-terminal work is still
            # running on this agent's job thread.
            "settling": latest.settling_stage is not None,
            "settling_stage": (
                latest.settling_stage.value if latest.settling_stage else None
            ),
        }

    pending_publish = len(store.get_pending_publish())

    # OMN-18636 AC4. An accept queue that nothing has drained past the declared
    # bound means this process is not serving its own socket -- which, to every
    # reader downstream, has until now looked like INDETERMINATE rather than a
    # failure. The status code goes with it: RED rather than silent.
    backlog = _accept_backlog_block(request)
    status_code = (
        503 if backlog["status"] == EnumAcceptBacklogStatus.UNHEALTHY.value else 200
    )

    return web.json_response(
        {
            "state": state,
            "version": "0.1.0",
            # OMN-18200. ``version`` is a static string that names no commit, so
            # until this field existed the health payload could not answer "which
            # code is this process running" -- and on 2026-09-14 the answer was
            # four days older than the clone on the same disk. ``null`` here is a
            # real finding (the process never reached its startup step), not a
            # formatting quirk, so it is reported rather than defaulted.
            "loaded_code_sha": loaded_code_sha_if_recorded(),
            "uptime_seconds": int(time.monotonic() - _start_time),
            "active_job": active_job,
            "last_result": last_result,
            "pending_publish_count": pending_publish,
            "accept_backlog": backlog,
        },
        status=status_code,
    )


def _accept_backlog_block(request: web.Request) -> dict[str, object]:
    """The watchdog's latest verdict, or an honest statement that there is none.

    An absent watchdog is INDETERMINATE, never healthy: a payload that reported
    a healthy accept queue on the strength of nobody having looked would assert
    exactly the fact this block exists to establish.
    """
    getter = request.app.get("get_accept_backlog")
    verdict = getter() if getter is not None else None
    if verdict is None:
        return {
            "status": EnumAcceptBacklogStatus.INDETERMINATE.value,
            "queue_depth": None,
            "undrained_seconds": 0.0,
            "bound_seconds": None,
            "observed_at": None,
            "evidence": (
                "no accept-backlog verdict has been produced in this process "
                "yet; the queue depth is unknown, which is not the same as empty"
            ),
        }
    return {
        "status": verdict.status.value,
        "queue_depth": verdict.queue_depth,
        "undrained_seconds": verdict.undrained_seconds,
        "bound_seconds": verdict.bound_seconds,
        "observed_at": verdict.observed_at.isoformat(),
        "evidence": verdict.evidence,
    }


async def _queue_handler(request: web.Request) -> web.Response:
    """Serve the queue snapshot (OMN-18144).

    Always 200 with a body, including when a half is unreadable: the
    unreadability IS the answer the caller needs, and an error status would be
    indistinguishable to a CI reader from an agent too old to serve this route
    at all -- which is a distinction that decides whether it falls back or
    retries.
    """
    store: JobStore = request.app["job_store"]
    getter = request.app.get("get_control_topic_lag")
    lag = (
        getter()
        if getter is not None
        else ModelControlTopicLag.unknown(
            "this agent was built with no control-topic lag sampler, so the "
            "records queued beyond it are unknown -- not zero"
        )
    )
    snapshot = compute_queue_snapshot(store, lag)
    return web.json_response(snapshot.to_payload())


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
            # OMN-18636 AC5. "Terminal" and "the agent is done with this job"
            # were the same fact on this payload, and for 3m38s on 2026-09-17
            # they were not the same fact in reality. ``settling`` is the
            # boolean a reader branches on; ``settling_stage`` names which
            # post-terminal host mutation is in flight.
            "settling": job.settling_stage is not None,
            "settling_stage": (
                job.settling_stage.value if job.settling_stage else None
            ),
            # OMN-18143. The supersession, served where the post-merge lab
            # guard already reads -- it polls this exact route for the
            # acceptance timestamp (OMN-18573). A superseded command's own CI
            # run has no other way to learn that the lane converged on a
            # newer sha rather than on the one that run is about, and a
            # receipt that cannot say so would read as a plain pass for a
            # commit whose tree the lane never built.
            "superseded_by_sha": job.superseded_by_sha,
            "superseded_by_correlation_id": (
                str(job.superseded_by_correlation_id)
                if job.superseded_by_correlation_id
                else None
            ),
            "superseded_count": job.superseded_count,
            "superseded_correlation_ids": [
                str(cid) for cid in job.superseded_correlation_ids
            ],
            # OMN-19374. Served where the post-merge lab guard reads, so a
            # receipt can say the container generation moved because THIS job
            # recreated a runtime during its own verification.
            "verify_recreate": [
                record.model_dump(mode="json") for record in job.verify_recreate
            ],
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


async def _lab_overlay_latest_handler(request: web.Request) -> web.Response:
    """Serve the most recently written onex-lab overlay record, any sha.

    OMN-18399. Two answers, mirroring ``_lab_overlay_handler``'s two failure
    modes for the same reason -- they are different facts:

    * ``404`` -- no lab-overlay record has ever been written on this agent.
      Distinct from "the one I asked for is missing" (that is ``/lab-overlay/
      {sha}``'s 404): this means the apply path has never completed once.
    * ``500`` -- a record exists and is unparseable.
    """
    store: JobStore = request.app["job_store"]
    try:
        record = load_latest_record(store.state_dir)
    except (OSError, ValueError) as exc:
        return web.json_response({"error": f"record is unreadable: {exc}"}, status=500)
    if record is None:
        return web.json_response(
            {"error": "no lab-overlay record exists yet"}, status=404
        )
    return web.json_response(record)
