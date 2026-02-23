# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler for GitHub PR API polling — core logic for node_github_pr_poller_effect.

Implements the ``github.poll.prs`` operation declared in contract.yaml.

Triage State Logic
------------------
``compute_triage_state(pr, stale_hours)`` is a **pure, deterministic function**
that maps a raw GitHub PR payload dict to one of 8 triage states:

    draft | stale | ci_failing | changes_requested | ready_to_merge |
    approved_pending_ci | needs_review | blocked

Evaluation order (first match wins):
    1. draft           — ``pr["draft"] is True``
    2. blocked         — any label in {"blocked", "do-not-merge", "wip"}
    3. stale           — updated_at older than ``stale_hours``
    4. ci_failing      — ``combined_status == "failure"``
    5. changes_requested — at least one reviewer requested changes
    6. ready_to_merge  — CI passing AND at least 1 approval AND no changes_requested
    7. approved_pending_ci — at least 1 approval but CI still pending/running
    8. needs_review    — fallback (no reviews yet)

Non-blocking Design
-------------------
The handler uses ``httpx.AsyncClient`` with a configurable timeout.
GitHub API errors are logged and surfaced in ``ModelGitHubPollerResult.errors``
rather than raising — the poller must not block the runtime tick loop.

Related Tickets:
    - OMN-2656: Phase 2 — Effect Nodes & CLIs (omnibase_infra)
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Protocol

import httpx

from omnibase_core.types import JsonType


class ProtocolEventPublisherBase(Protocol):
    """Minimal protocol for the event publisher used by HandlerGitHubApiPoll."""

    async def publish(
        self,
        event_type: str,
        payload: JsonType,
        partition_key: str | None = None,
    ) -> bool:
        """Publish an event; returns True on success."""
        ...


from omnibase_infra.nodes.node_github_pr_poller_effect.models.model_github_poller_config import (
    ModelGitHubPollerConfig,
)
from omnibase_infra.nodes.node_github_pr_poller_effect.models.model_github_poller_result import (
    ModelGitHubPollerResult,
)

logger = logging.getLogger(__name__)

# GitHub API base URL — injectable for testing
GITHUB_API_BASE = "https://api.github.com"

# Triage state type alias (matches TriageState in omnibase_core PR model)
TriageState = str

_BLOCKING_LABELS = frozenset({"blocked", "do-not-merge", "wip"})

__all__ = ["HandlerGitHubApiPoll", "ProtocolEventPublisherBase", "compute_triage_state"]


def compute_triage_state(
    pr: dict[str, JsonType],
    stale_hours: int = 48,
) -> TriageState:
    """Compute the triage state for a single GitHub PR payload.

    This is a **pure, deterministic function**. It does not call any external
    service; all required data must be present in ``pr``.

    The ``pr`` dict is expected to mirror the GitHub REST API
    ``GET /repos/{owner}/{repo}/pulls/{pull_number}`` response, augmented
    with:
        - ``pr["combined_status"]`` — string: "success" | "failure" | "pending"
          (derived from the commits statuses/check-runs endpoint by the caller)
        - ``pr["review_states"]`` — list[str]: review states from
          ``GET /repos/{owner}/{repo}/pulls/{pull_number}/reviews``
          (e.g. ["APPROVED", "CHANGES_REQUESTED"])

    Evaluation order (first-match wins):
        1. draft
        2. blocked  (label-based)
        3. stale    (time-based)
        4. ci_failing
        5. changes_requested
        6. ready_to_merge
        7. approved_pending_ci
        8. needs_review (fallback)

    Args:
        pr: GitHub PR payload dict (see above).
        stale_hours: Hours of inactivity that qualify as stale.

    Returns:
        One of the 8 triage state strings.
    """
    # 1. draft
    if pr.get("draft") is True:
        return "draft"

    # 2. blocked — any blocking label present
    raw_labels = pr.get("labels", [])
    labels = raw_labels if isinstance(raw_labels, list) else []
    label_names = {
        str(lbl.get("name", "")).lower() for lbl in labels if isinstance(lbl, dict)
    }
    if label_names & _BLOCKING_LABELS:
        return "blocked"

    # 3. stale — no activity in stale_hours
    updated_at_str = pr.get("updated_at")
    if isinstance(updated_at_str, str):
        try:
            updated_at = datetime.fromisoformat(updated_at_str.rstrip("Z")).replace(
                tzinfo=UTC
            )
            if datetime.now(tz=UTC) - updated_at > timedelta(hours=stale_hours):
                return "stale"
        except ValueError:
            pass

    combined_status = pr.get("combined_status", "pending")
    raw_review_states = pr.get("review_states", [])
    review_states: list[str] = (
        [str(s) for s in raw_review_states]
        if isinstance(raw_review_states, list)
        else []
    )
    has_approval = any(s == "APPROVED" for s in review_states)
    has_changes_requested = any(s == "CHANGES_REQUESTED" for s in review_states)

    # 4. ci_failing
    if combined_status == "failure":
        return "ci_failing"

    # 5. changes_requested
    if has_changes_requested:
        return "changes_requested"

    ci_passing = combined_status == "success"

    # 6. ready_to_merge — CI green + approved + no changes requested
    if ci_passing and has_approval:
        return "ready_to_merge"

    # 7. approved_pending_ci — approved but CI not yet green
    if has_approval:
        return "approved_pending_ci"

    # 8. needs_review (fallback)
    return "needs_review"


class HandlerGitHubApiPoll:
    """Handler for the ``github.poll.prs`` operation.

    Fetches open PRs from the GitHub REST API for each configured repository,
    classifies their triage state via ``compute_triage_state``, and publishes
    ``ModelGitHubPRStatusEvent`` instances to ``onex.evt.github.pr-status.v1``.

    Non-blocking contract:
        Errors from individual repo/PR lookups are collected in
        ``ModelGitHubPollerResult.errors`` rather than raised. The handler
        always returns a result, even on partial failure.

    Args:
        publisher: An object implementing ``async def publish(event_type, payload,
            partition_key)`` — typically
            ``AdapterProtocolEventPublisherKafka``.
        api_base: GitHub API base URL (default: https://api.github.com).
            Override in tests.
        http_timeout: HTTP client timeout in seconds (default: 15).
    """

    def __init__(
        self,
        publisher: ProtocolEventPublisherBase,
        api_base: str = GITHUB_API_BASE,
        http_timeout: float = 15.0,
    ) -> None:
        self._publisher = publisher
        self._api_base = api_base
        self._http_timeout = http_timeout

    async def handle(
        self,
        config: ModelGitHubPollerConfig,
    ) -> ModelGitHubPollerResult:
        """Execute one poll cycle for all configured repositories.

        Args:
            config: Poller configuration (repos, interval, stale threshold,
                token env var).

        Returns:
            ``ModelGitHubPollerResult`` with counts and any non-fatal errors.
        """
        token = os.environ.get(config.github_token_env_var, "")
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        errors: list[str] = []
        events_published = 0
        repos_polled: list[str] = []
        prs_polled = 0

        async with httpx.AsyncClient(
            headers=headers, timeout=self._http_timeout
        ) as client:
            for repo in config.repos:
                try:
                    pr_events, repo_prs, repo_errors = await self._poll_repo(
                        client=client,
                        repo=repo,
                        stale_hours=config.stale_threshold_hours,
                    )
                    repos_polled.append(repo)
                    prs_polled += repo_prs
                    errors.extend(repo_errors)

                    for event_payload, partition_key in pr_events:
                        published = await self._publish_event(
                            event_payload=event_payload,
                            partition_key=partition_key,
                        )
                        if published:
                            events_published += 1
                        else:
                            errors.append(
                                f"Failed to publish event for {partition_key}"
                            )
                except Exception as exc:
                    msg = f"Error polling repo {repo}: {exc}"
                    logger.warning(msg)
                    errors.append(msg)

        return ModelGitHubPollerResult(
            events_published=events_published,
            repos_polled=repos_polled,
            prs_polled=prs_polled,
            errors=errors,
        )

    async def _publish_event(
        self,
        event_payload: JsonType,
        partition_key: str,
    ) -> bool:
        """Publish a single event via the publisher."""
        result = await self._publisher.publish(
            event_type="onex.evt.github.pr-status.v1",
            payload=event_payload,
            partition_key=partition_key,
        )
        return bool(result)

    async def _poll_repo(
        self,
        client: httpx.AsyncClient,
        repo: str,
        stale_hours: int,
    ) -> tuple[list[tuple[JsonType, str]], int, list[str]]:
        """Poll all open PRs for a single repository.

        Returns:
            Tuple of (event_payload_and_key_pairs, total_prs_polled, errors).
        """
        errors: list[str] = []
        pr_events: list[tuple[JsonType, str]] = []

        # Fetch open PRs
        prs_url = f"{self._api_base}/repos/{repo}/pulls"
        response = await client.get(prs_url, params={"state": "open", "per_page": 100})
        response.raise_for_status()
        prs: list[dict[str, JsonType]] = response.json()

        for pr in prs:
            pr_number = pr["number"]
            if not isinstance(pr_number, int):
                continue
            try:
                # Augment with combined status and review states
                head = pr.get("head", {})
                sha = str(head.get("sha", "")) if isinstance(head, dict) else ""
                pr["combined_status"] = await self._get_combined_status(
                    client, repo, sha
                )
                review_states_raw = await self._get_review_states(
                    client, repo, pr_number
                )
                pr["review_states"] = list(review_states_raw)

                triage = compute_triage_state(pr, stale_hours)
                partition_key = f"{repo}:{pr_number}"
                event_payload: JsonType = {
                    "event_type": "onex.evt.github.pr-status.v1",
                    "repo": repo,
                    "pr_number": pr_number,
                    "triage_state": triage,
                    "title": str(pr.get("title", "")),
                }
                pr_events.append((event_payload, partition_key))
            except Exception as exc:
                msg = f"Error processing PR {repo}#{pr_number}: {exc}"
                logger.warning(msg)
                errors.append(msg)

        return pr_events, len(prs), errors

    async def _get_combined_status(
        self, client: httpx.AsyncClient, repo: str, sha: str
    ) -> str:
        """Fetch combined commit status for a given SHA.

        Returns: "success" | "failure" | "pending"
        """
        try:
            url = f"{self._api_base}/repos/{repo}/commits/{sha}/status"
            resp = await client.get(url)
            resp.raise_for_status()
            data: dict[str, JsonType] = resp.json()
            return str(data.get("state", "pending"))
        except Exception:
            return "pending"

    async def _get_review_states(
        self, client: httpx.AsyncClient, repo: str, pr_number: int
    ) -> list[str]:
        """Fetch review states for a PR.

        Returns list of review state strings (e.g. ["APPROVED", "CHANGES_REQUESTED"]).
        """
        try:
            url = f"{self._api_base}/repos/{repo}/pulls/{pr_number}/reviews"
            resp = await client.get(url)
            resp.raise_for_status()
            reviews: list[dict[str, JsonType]] = resp.json()
            # Deduplicate by user — last review per user wins
            latest: dict[str, str] = {}
            for review in reviews:
                user_data = review.get("user", {})
                user = (
                    str(user_data.get("login", "unknown"))
                    if isinstance(user_data, dict)
                    else "unknown"
                )
                state = str(review.get("state", ""))
                if state in {"APPROVED", "CHANGES_REQUESTED"}:
                    latest[user] = state
            return list(latest.values())
        except Exception:
            return []
