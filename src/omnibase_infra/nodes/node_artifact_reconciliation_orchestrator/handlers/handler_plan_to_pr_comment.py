# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""Handler that posts an artifact update plan as a PR comment.

Receives a ModelUpdatePlan and formats it as a markdown table, then posts
or updates a GitHub PR comment using the GITHUB_TOKEN environment variable.

Idempotency:
    Uses a hidden HTML anchor comment (<!-- onex-artifact-plan:{plan_id} -->)
    embedded in the comment body. Before posting, fetches existing PR comments
    and searches for the anchor. If found, PATCHes the existing comment.
    If not found, POSTs a new one.

PR Trigger Guard:
    Only posts for PR-triggered plans (trigger_type in pr_opened, pr_updated,
    pr_merged). Other trigger types (e.g. manual_plan_request, contract_changed)
    are skipped with a log message.

Error Handling:
    On HTTP error: logs the error with response body, returns ModelPRCommentResult
    with posted=False and error message. NO automatic retry.

Tracking:
    - OMN-3944: Task 7 — Reconciliation ORCHESTRATOR Node
    - OMN-3925: Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

import logging
import os
from uuid import UUID

import httpx

from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
)
from omnibase_infra.nodes.node_artifact_reconciliation_orchestrator.models.model_pr_comment_result import (
    ModelPRCommentResult,
)
from omnibase_infra.nodes.node_update_plan_reducer.models.model_update_plan import (
    ModelUpdatePlan,
)

logger = logging.getLogger(__name__)

# PR trigger types that should receive a comment
_PR_TRIGGER_TYPES: frozenset[str] = frozenset({"pr_opened", "pr_updated", "pr_merged"})

# Hidden HTML anchor template embedded in each posted comment
_ANCHOR_TEMPLATE: str = "<!-- onex-artifact-plan:{plan_id} -->"

# GitHub API base URL (overridable in tests)
_GITHUB_API_BASE: str = "https://api.github.com"


def _build_markdown_table(plan: ModelUpdatePlan) -> str:
    """Build a markdown table summarizing the update plan.

    Columns: Artifact | Type | Strength | Action | Owner

    Args:
        plan: The update plan to render.

    Returns:
        Markdown string suitable for a PR comment body.
    """
    lines: list[str] = [
        f"## Artifact Update Plan — {plan.summary}",
        "",
        "| Artifact | Type | Strength | Action | Owner |",
        "|----------|------|----------|--------|-------|",
    ]
    for artifact in plan.impacted_artifacts:
        strength_pct = f"{artifact.impact_strength * 100:.0f}%"
        owner = ""
        # Find the corresponding task's owner_hint if available
        for task in plan.tasks:
            if task.target_artifact_id == artifact.artifact_id:
                owner = task.owner_hint or ""
                break
        lines.append(
            f"| `{artifact.path}` "
            f"| {artifact.artifact_type} "
            f"| {strength_pct} "
            f"| {artifact.required_action} "
            f"| {owner} |"
        )

    lines.extend(
        [
            "",
            f"**Merge Policy**: `{plan.merge_policy}`",
            f"**Tasks**: {len(plan.tasks)} required",
            "",
        ]
    )
    return "\n".join(lines)


def _anchor_for_plan(plan_id: UUID) -> str:
    """Return the idempotency anchor string for this plan_id."""
    return _ANCHOR_TEMPLATE.format(plan_id=str(plan_id))


def _parse_owner_repo(source_entity_ref: str) -> tuple[str, str, int] | None:
    """Parse source_entity_ref 'pr/<owner>/<repo>/<number>' into parts.

    Returns:
        Tuple of (owner, repo, pr_number) or None if not parseable.
    """
    parts = source_entity_ref.split("/")
    if len(parts) < 4 or parts[0] != "pr":
        return None
    try:
        pr_number = int(parts[-1])
        repo = parts[-2]
        owner = "/".join(parts[1:-2])
        return owner, repo, pr_number
    except (ValueError, IndexError):
        return None


class HandlerPlanToPRComment:
    """Post an artifact update plan as a GitHub PR comment.

    Fetches existing comments to check for the idempotency anchor, then
    either updates the existing comment or creates a new one. Only posts
    for PR-triggered plans.

    Attributes:
        _github_api_base: Base URL for GitHub API (injectable for testing).
        _github_token: Bearer token for GitHub API authentication.
    """

    def __init__(
        self,
        github_api_base: str = _GITHUB_API_BASE,
        github_token: str | None = None,
    ) -> None:
        """Initialize with optional API base override and token.

        Args:
            github_api_base: Base URL for GitHub API. Defaults to
                https://api.github.com. Override in tests to point at a
                mock server.
            github_token: Bearer token for GitHub API. When None, falls back
                to the GITHUB_TOKEN environment variable at construction time.
        """
        self._github_api_base = github_api_base
        _env_token = os.environ.get("GITHUB_TOKEN", "")  # ONEX_EXCLUDE: constructor injection fallback  # fmt: skip
        self._github_token: str = github_token if github_token is not None else _env_token  # fmt: skip

    @property
    def handler_id(self) -> str:
        """Unique handler identifier."""
        return "handler-plan-to-pr-comment"

    @property
    def handler_type(self) -> EnumHandlerType:
        """Architectural role: infrastructure handler (EFFECT — makes HTTP calls)."""
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        """Behavioral classification: effect (external I/O via GitHub API)."""
        return EnumHandlerTypeCategory.EFFECT

    async def post_plan_comment(
        self,
        plan: ModelUpdatePlan,
        trigger_type: str,
        correlation_id: UUID | None = None,
    ) -> ModelPRCommentResult:
        """Post or update the PR comment for an update plan.

        Checks whether the trigger type is a PR trigger, parses the
        source_entity_ref to extract owner/repo/PR number, fetches existing
        comments to check for the idempotency anchor, and either POSTs a new
        comment or PATCHes an existing one.

        Args:
            plan: The update plan to post as a PR comment.
            trigger_type: The trigger type from the originating event.
            correlation_id: Correlation ID for tracing.

        Returns:
            ModelPRCommentResult indicating whether the comment was posted,
            updated, skipped, or failed.
        """
        if trigger_type not in _PR_TRIGGER_TYPES:
            logger.info(
                "Skipping PR comment for trigger_type=%r (not a PR trigger)",
                trigger_type,
                extra={"plan_id": str(plan.plan_id)},
            )
            return ModelPRCommentResult(posted=False, skipped=True)

        parsed = _parse_owner_repo(plan.source_entity_ref)
        if parsed is None:
            logger.warning(
                "Cannot parse source_entity_ref=%r as PR reference; skipping",
                plan.source_entity_ref,
                extra={"plan_id": str(plan.plan_id)},
            )
            return ModelPRCommentResult(
                posted=False,
                error=f"Unparseable source_entity_ref: {plan.source_entity_ref!r}",
                skipped=True,
            )

        owner, repo, pr_number = parsed
        anchor = _anchor_for_plan(plan.plan_id)
        markdown_body = _build_markdown_table(plan)
        comment_body = f"{anchor}\n\n{markdown_body}"

        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._github_token:
            headers["Authorization"] = f"Bearer {self._github_token}"

        async with httpx.AsyncClient(base_url=self._github_api_base) as client:
            # Fetch existing comments and search for our anchor
            existing_comment_id = await self._find_existing_comment(
                client=client,
                headers=headers,
                owner=owner,
                repo=repo,
                pr_number=pr_number,
                anchor=anchor,
            )

            if existing_comment_id is not None:
                # Update existing comment
                return await self._update_comment(
                    client=client,
                    headers=headers,
                    owner=owner,
                    repo=repo,
                    comment_id=existing_comment_id,
                    body=comment_body,
                    plan_id=plan.plan_id,
                )
            else:
                # Post new comment
                return await self._post_comment(
                    client=client,
                    headers=headers,
                    owner=owner,
                    repo=repo,
                    pr_number=pr_number,
                    body=comment_body,
                    plan_id=plan.plan_id,
                )

    async def _find_existing_comment(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        owner: str,
        repo: str,
        pr_number: int,
        anchor: str,
    ) -> int | None:
        """Fetch PR comments and find one containing our anchor.

        Args:
            client: Configured httpx async client.
            headers: Request headers including auth.
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            anchor: The idempotency anchor string to search for.

        Returns:
            GitHub comment ID if found, else None.
        """
        url = f"/repos/{owner}/{repo}/issues/{pr_number}/comments"
        try:
            response = await client.get(url, headers=headers, params={"per_page": 100})
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.exception(
                "Failed to fetch PR comments: status=%d body=%s",
                exc.response.status_code,
                exc.response.text,
                extra={"owner": owner, "repo": repo, "pr": pr_number},
            )
            return None
        except httpx.RequestError as exc:
            logger.exception(
                "Network error fetching PR comments: %s",
                exc,
                extra={"owner": owner, "repo": repo, "pr": pr_number},
            )
            return None

        comments: list[dict[str, str | int]] = response.json()
        for comment in comments:
            body = str(comment.get("body", ""))
            if anchor in body:
                return int(comment["id"])
        return None

    async def _post_comment(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        owner: str,
        repo: str,
        pr_number: int,
        body: str,
        plan_id: UUID,
    ) -> ModelPRCommentResult:
        """Post a new PR comment.

        Args:
            client: Configured httpx async client.
            headers: Request headers including auth.
            owner: Repository owner.
            repo: Repository name.
            pr_number: Pull request number.
            body: Comment body text.
            plan_id: Plan UUID for logging.

        Returns:
            ModelPRCommentResult indicating success or failure.
        """
        url = f"/repos/{owner}/{repo}/issues/{pr_number}/comments"
        try:
            response = await client.post(url, headers=headers, json={"body": body})
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.exception(
                "Failed to post PR comment: status=%d body=%s plan_id=%s",
                exc.response.status_code,
                exc.response.text,
                str(plan_id),
            )
            return ModelPRCommentResult(
                posted=False,
                error=f"HTTP {exc.response.status_code}: {exc.response.text}",
            )
        except httpx.RequestError as exc:
            logger.exception(
                "Network error posting PR comment: %s plan_id=%s",
                exc,
                str(plan_id),
            )
            return ModelPRCommentResult(posted=False, error=str(exc))

        comment_data: dict[str, str | int] = response.json()
        comment_id = int(comment_data["id"])
        logger.info(
            "Posted PR comment: comment_id=%d plan_id=%s",
            comment_id,
            str(plan_id),
        )
        return ModelPRCommentResult(posted=True, comment_id=comment_id)

    async def _update_comment(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        owner: str,
        repo: str,
        comment_id: int,
        body: str,
        plan_id: UUID,
    ) -> ModelPRCommentResult:
        """Update an existing PR comment.

        Args:
            client: Configured httpx async client.
            headers: Request headers including auth.
            owner: Repository owner.
            repo: Repository name.
            comment_id: GitHub comment ID to update.
            body: New comment body text.
            plan_id: Plan UUID for logging.

        Returns:
            ModelPRCommentResult indicating success or failure.
        """
        url = f"/repos/{owner}/{repo}/issues/comments/{comment_id}"
        try:
            response = await client.patch(url, headers=headers, json={"body": body})
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.exception(
                "Failed to update PR comment: status=%d body=%s plan_id=%s",
                exc.response.status_code,
                exc.response.text,
                str(plan_id),
            )
            return ModelPRCommentResult(
                posted=False,
                error=f"HTTP {exc.response.status_code}: {exc.response.text}",
            )
        except httpx.RequestError as exc:
            logger.exception(
                "Network error updating PR comment: %s plan_id=%s",
                exc,
                str(plan_id),
            )
            return ModelPRCommentResult(posted=False, error=str(exc))

        logger.info(
            "Updated PR comment: comment_id=%d plan_id=%s",
            comment_id,
            str(plan_id),
        )
        return ModelPRCommentResult(posted=True, comment_id=comment_id)


__all__: list[str] = ["HandlerPlanToPRComment"]
