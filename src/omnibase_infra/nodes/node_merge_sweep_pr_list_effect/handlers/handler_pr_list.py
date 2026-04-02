# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that lists open PRs from GitHub via the gh CLI.

This is an EFFECT handler - it performs external I/O (GitHub API).
"""

from __future__ import annotations

import asyncio
import json
import logging
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_info import (
    ModelPRInfo,
)
from omnibase_infra.nodes.node_merge_sweep_pr_list_effect.models.model_pr_list_result import (
    ModelPRListResult,
)

logger = logging.getLogger(__name__)

_GH_JSON_FIELDS = (
    "number,title,headRefName,baseRefName,author,isDraft,"
    "mergeable,reviewDecision,statusCheckRollup,autoMergeRequest,"
    "labels,updatedAt"
)


def _extract_ci_status(pr_json: dict) -> str:
    """Extract rollup CI status from gh pr list JSON."""
    rollup = pr_json.get("statusCheckRollup") or []
    if not rollup:
        return "UNKNOWN"
    states = {item.get("conclusion") or item.get("state", "") for item in rollup}
    if "FAILURE" in states or "failure" in states:
        return "FAILURE"
    if "PENDING" in states or "pending" in states or "" in states:
        return "PENDING"
    return "SUCCESS"


def _pr_json_to_model(pr_json: dict, repo: str) -> ModelPRInfo:
    """Convert a single gh pr JSON object to ModelPRInfo."""
    author_obj = pr_json.get("author") or {}
    labels_raw = pr_json.get("labels") or []
    return ModelPRInfo(
        number=pr_json["number"],
        repo=repo,
        title=pr_json.get("title", ""),
        head_ref=pr_json.get("headRefName", ""),
        base_ref=pr_json.get("baseRefName", "main"),
        author=author_obj.get("login", ""),
        is_draft=pr_json.get("isDraft", False),
        mergeable=pr_json.get("mergeable", "UNKNOWN"),
        review_decision=pr_json.get("reviewDecision", ""),
        ci_status=_extract_ci_status(pr_json),
        has_auto_merge=pr_json.get("autoMergeRequest") is not None,
        labels=tuple(
            label.get("name", "") if isinstance(label, dict) else str(label)
            for label in labels_raw
        ),
        updated_at=pr_json.get("updatedAt", ""),
    )


class HandlerPRList:
    """Lists open PRs from GitHub repositories using gh CLI."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def _scan_repo(self, repo: str) -> tuple[str, list[ModelPRInfo] | None, str]:
        """Scan a single repo for open PRs.

        Returns:
            Tuple of (repo, prs_or_none, error_message).
        """
        cmd = [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            _GH_JSON_FIELDS,
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except TimeoutError:
            return (repo, None, f"Timeout scanning {repo}")
        except OSError as e:
            return (repo, None, f"OS error scanning {repo}: {e}")

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            return (repo, None, f"gh pr list failed for {repo}: {err}")

        try:
            raw_prs = json.loads(stdout.decode())
        except json.JSONDecodeError as e:
            return (repo, None, f"Invalid JSON from {repo}: {e}")

        prs = [_pr_json_to_model(pr, repo) for pr in raw_prs]
        return (repo, prs, "")

    async def handle(
        self,
        repos: tuple[str, ...],
        correlation_id: UUID,
        authors: tuple[str, ...] = (),
        labels: tuple[str, ...] = (),
        since: str = "",
    ) -> ModelPRListResult:
        """List open PRs across all specified repos.

        Args:
            repos: GitHub repos to scan.
            correlation_id: Workflow correlation ID.
            authors: Filter by author login (empty = all).
            labels: Filter by label name (empty = all, any-match).
            since: ISO 8601 date filter for updatedAt.

        Returns:
            ModelPRListResult with all discovered PRs.
        """
        logger.info(
            "Scanning %d repos for open PRs (correlation_id=%s)",
            len(repos),
            correlation_id,
        )

        tasks = [self._scan_repo(repo) for repo in repos]
        results = await asyncio.gather(*tasks)

        all_prs: list[ModelPRInfo] = []
        scanned: list[str] = []
        failed: list[str] = []

        for repo, prs, error in results:
            if prs is None:
                logger.warning("Scan failed for %s: %s", repo, error)
                failed.append(repo)
                continue
            scanned.append(repo)
            all_prs.extend(prs)

        # Apply filters
        if authors:
            author_set = frozenset(authors)
            all_prs = [pr for pr in all_prs if pr.author in author_set]

        if labels:
            label_set = frozenset(labels)
            all_prs = [pr for pr in all_prs if label_set & frozenset(pr.labels)]

        if since:
            all_prs = [pr for pr in all_prs if pr.updated_at >= since]

        logger.info(
            "Scan complete: %d PRs from %d repos (%d failed)",
            len(all_prs),
            len(scanned),
            len(failed),
        )

        return ModelPRListResult(
            correlation_id=correlation_id,
            prs=tuple(all_prs),
            repos_scanned=tuple(scanned),
            repos_failed=tuple(failed),
            success=len(failed) == 0,
            error_message=f"{len(failed)} repo(s) failed to scan" if failed else "",
        )
