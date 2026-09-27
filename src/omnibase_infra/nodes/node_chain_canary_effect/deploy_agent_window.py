# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Read the lane's deploy agent, and decide whether a deploy hit the window (OMN-19811).

WHY
    Chain-canary run 36202173467 (2026-09-25T23:44Z) reported
    ``terminal_missing`` because deploy-agent job 193cfeda recreated
    ``omninode-runtime`` inside the probe's 120 s budget. The canary had no way
    to know a deploy was in progress, so every scheduled run that lands on a
    redeploy goes RED against a healthy chain, and C15 with it.

THE SOURCE
    The deploy agent's own HTTP surface (``scripts/deploy-agent/deploy_agent/
    health.py``): ``/health`` carries ``state``, ``active_job`` and
    ``last_result``, ``/job/{correlation_id}`` carries a job's
    ``accepted_at``, and ``/queue`` carries ``commands_ahead``, the deploy
    commands waiting behind the in-flight one. The ``verify-lane-converged`` job in
    ``runtime-rebuild-trigger.yml`` reads the same surface through
    ``scripts/ci/check_dev_lane_staleness.py``, and the chain-canary workflow
    resolves the URL through the same routing declaration
    (``config/deploy_lane_routing.yaml``, ``verify.deploy_agent_url``).

WHAT IT NEVER DOES
    It never turns an unreadable agent into "no deploy", and never turns "no
    deploy" into a retry. Both readers below return a value instead of raising,
    and an unreadable value carries ``readable=False``, which the handler reads
    as "no evidence" and fails closed on.

KNOWN LIMIT
    ``/health`` names the in-flight job and the MOST RECENT completed job, not
    every job. Two jobs completing inside one probe window are both seen only
    if the later one is still the most recent when the agent is read; the
    agent runs jobs serially and a window is a few minutes, so a missed first
    job is always followed by a second one that is itself in the window.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID

import httpx

from omnibase_infra.nodes.node_chain_canary_effect.models.model_deploy_agent_snapshot import (
    ModelDeployAgentSnapshot,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

# The same bound scripts/ci/check_dev_lane_staleness.py applies to /queue
# (MAX_QUEUE_LAG_AGE_SECONDS, OMN-18990): the agent samples its control-topic
# lag only when it polls, and it does not poll while a rebuild runs, so an
# older sample describes the queue before that rebuild started.
MAX_QUEUE_LAG_AGE_SECONDS = 120.0

# Clock tolerance between the runner and the agent. Both run on the same lab
# host today, but the probe must not miss a deploy that completed a second
# before its window by the agent's clock.
WINDOW_SKEW = timedelta(seconds=5)


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_uuid(value: object) -> UUID | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return UUID(value.strip())
    except ValueError:
        return None


def _as_dict(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def queued_commands_from_payload(queue: dict[str, object] | None) -> int | None:
    """``commands_ahead`` from a ``/queue`` body, or None when it is not a fact.

    None for an absent body, a depth the agent itself reports unknown, a value
    that is not a non-negative count, or a lag sample older than
    ``MAX_QUEUE_LAG_AGE_SECONDS``. An absent age is an agent that predates the
    age field (OMN-18990) and is read as current, as the staleness guard does.
    """
    if not queue:
        return None
    ahead = queue.get("commands_ahead")
    if not isinstance(ahead, int) or isinstance(ahead, bool) or ahead < 0:
        return None
    age = queue.get("control_topic_lag_age_seconds")
    if age is not None and (
        isinstance(age, bool)
        or not isinstance(age, (int, float))
        or age > MAX_QUEUE_LAG_AGE_SECONDS
    ):
        return None
    return ahead


def snapshot_from_payloads(
    *,
    observed_at: datetime,
    health: dict[str, object],
    last_job: dict[str, object] | None,
    queue: dict[str, object] | None = None,
) -> ModelDeployAgentSnapshot:
    """Build a snapshot from the agent's ``/health``, ``/job`` and ``/queue`` bodies.

    Pure, so the handler's decisions are testable against real payload shapes.
    """
    active = _as_dict(health.get("active_job"))
    last = _as_dict(health.get("last_result"))
    return ModelDeployAgentSnapshot(
        observed_at=observed_at,
        readable=True,
        state=str(health.get("state") or ""),
        active_correlation_id=_parse_uuid(active.get("correlation_id")),
        active_accepted_at=_parse_ts(active.get("started_at")),
        last_correlation_id=_parse_uuid(last.get("correlation_id")),
        last_accepted_at=_parse_ts((last_job or {}).get("accepted_at")),
        last_completed_at=_parse_ts(last.get("completed_at")),
        last_settling=bool(last.get("settling", False)),
        queued_commands=queued_commands_from_payload(queue),
    )


def deploys_in_window(
    snapshot: ModelDeployAgentSnapshot,
    started_at: datetime,
    ended_at: datetime,
) -> tuple[UUID, ...]:
    """Deploy job ids whose run overlapped ``[started_at, ended_at]``.

    A job overlaps when it was accepted no later than the window's end and had
    not completed before the window's start. The in-flight job has not
    completed at all, so it overlaps whenever it was accepted before the end
    (a missing ``accepted_at`` counts as overlapping: the agent says it is
    running now). The last completed job overlaps when it completed at or
    after the start and was accepted no later than the end (a missing
    ``accepted_at`` counts as overlapping on the same terms).

    An unreadable snapshot has no jobs. The caller must check ``readable``
    before reading an empty result as "no deploy".
    """
    if not snapshot.readable:
        return ()
    start = started_at - WINDOW_SKEW
    end = ended_at + WINDOW_SKEW
    found: list[UUID] = []
    if snapshot.active_correlation_id is not None and (
        snapshot.active_accepted_at is None or snapshot.active_accepted_at <= end
    ):
        found.append(snapshot.active_correlation_id)
    if (
        snapshot.last_correlation_id is not None
        and snapshot.last_completed_at is not None
        and snapshot.last_completed_at >= start
        and (snapshot.last_accepted_at is None or snapshot.last_accepted_at <= end)
        and snapshot.last_correlation_id not in found
    ):
        found.append(snapshot.last_correlation_id)
    return tuple(found)


async def read_deploy_agent_via_httpx(
    agent_url: str, timeout_s: float, observed_at: datetime
) -> ModelDeployAgentSnapshot:
    """Read ``/health``, the last job's ``/job/{id}`` and ``/queue``. Never raises.

    ``/health`` answers 503 with a full body when the agent's accept backlog is
    unhealthy (OMN-18636 AC4). The body is still the agent's account of its
    jobs, so any JSON body is read regardless of status.
    """
    base = agent_url.rstrip("/")
    try:
        # See module docstring "THE SOURCE": this reads the deploy agent's own
        # ops-plane HTTP surface, the same one check_dev_lane_staleness.py
        # already polls, resolved from config/deploy_lane_routing.yaml -- not
        # a domain transport a node contract should own or inject.
        client = httpx.AsyncClient(timeout=timeout_s)  # no-contract-check: the seam
        async with client:
            response = await client.get(f"{base}/health")
            try:
                health = response.json()
            except ValueError:
                return ModelDeployAgentSnapshot(
                    observed_at=observed_at,
                    readable=False,
                    error=(
                        f"{base}/health answered HTTP {response.status_code} "
                        "with a body that is not JSON"
                    ),
                )
            if not isinstance(health, dict):
                return ModelDeployAgentSnapshot(
                    observed_at=observed_at,
                    readable=False,
                    error=f"{base}/health answered a JSON body that is not an object",
                )
            last_id = _parse_uuid(
                _as_dict(health.get("last_result")).get("correlation_id")
            )
            last_job: dict[str, object] | None = None
            if last_id is not None:
                job_response = await client.get(f"{base}/job/{last_id}")
                if job_response.status_code == 200:
                    try:
                        body = job_response.json()
                    except ValueError:
                        body = None
                    last_job = body if isinstance(body, dict) else None
            # /queue is additive: a 404 (an agent that predates OMN-18144) or
            # an unreadable body leaves the queue unread, never the snapshot.
            queue: dict[str, object] | None = None
            try:
                queue_response = await client.get(f"{base}/queue")
                queue_body = (
                    queue_response.json() if queue_response.status_code == 200 else None
                )
            except Exception:  # noqa: BLE001 — an unread queue is not an empty one
                queue_body = None
            queue = queue_body if isinstance(queue_body, dict) else None
    except Exception as exc:  # noqa: BLE001 — any failure is "unreadable", never "no deploy"
        return ModelDeployAgentSnapshot(
            observed_at=observed_at,
            readable=False,
            error=sanitize_error_message(exc),
        )
    return snapshot_from_payloads(
        observed_at=observed_at, health=health, last_job=last_job, queue=queue
    )


async def lane_ready_via_httpx(url: str, timeout_s: float) -> bool:
    """``GET {url}/health`` answers 200. Never raises."""
    try:
        # See module docstring "THE SOURCE": same ops-plane deploy-agent read
        # as read_deploy_agent_via_httpx above, not a domain transport.
        client = httpx.AsyncClient(timeout=timeout_s)  # no-contract-check: the seam
        async with client:
            response = await client.get(f"{url.rstrip('/')}/health")
    except Exception:  # noqa: BLE001 — not ready is the only answer a failure gives
        return False
    return response.status_code == 200


__all__ = [
    "MAX_QUEUE_LAG_AGE_SECONDS",
    "WINDOW_SKEW",
    "deploys_in_window",
    "lane_ready_via_httpx",
    "queued_commands_from_payload",
    "read_deploy_agent_via_httpx",
    "snapshot_from_payloads",
]
