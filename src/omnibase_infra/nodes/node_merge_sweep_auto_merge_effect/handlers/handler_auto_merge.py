# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that enables GitHub auto-merge and enqueues PRs via gh CLI.

On merge-queue repos, ``gh pr merge --auto`` arms auto-merge but does NOT cause
the PR to be enqueued into the merge queue. The PR must additionally be
enqueued via the ``enqueuePullRequest`` GraphQL mutation. Without this second
step, CLEAN auto-merge-enabled PRs sit idle forever while the merge-sweep loop
reports "success" (OMN-9276).

This is an EFFECT handler - it performs external I/O (GitHub API).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal
from uuid import UUID

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.models.model_auto_merge_outcome import (
    ModelAutoMergeOutcome,
)
from omnibase_infra.nodes.node_merge_sweep_auto_merge_effect.models.model_auto_merge_result import (
    ModelAutoMergeResult,
)

logger = logging.getLogger(__name__)

_ENQUEUE_MUTATION = (
    "mutation($id: ID!) { enqueuePullRequest(input: {pullRequestId: $id}) "
    "{ mergeQueueEntry { position } } }"
)

_NO_MERGE_QUEUE_MARKERS = (
    "does not have a merge queue",
    "merge queue is not enabled",
    "merge_queue_not_enabled",
)


class HandlerAutoMerge:
    """Enables GitHub auto-merge AND enqueues PRs on merge-queue repos."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def _run_gh(
        self,
        cmd: list[str],
        timeout: float = 30.0,
    ) -> tuple[int, bytes, bytes]:
        """Run a gh CLI command, returning (returncode, stdout, stderr).

        On timeout, kills and reaps the subprocess before re-raising so we
        never leak gh processes — under long-running runtimes a stuck gh on
        a flaky network would otherwise accumulate zombie children.
        """
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            try:
                await proc.wait()
            except (ProcessLookupError, OSError):
                # Process already reaped or vanished (PID reused, kernel raced the kill).
                # Nothing to wait on — safe to swallow and re-raise the original TimeoutError.
                pass
            raise
        assert proc.returncode is not None
        return proc.returncode, stdout, stderr

    async def _fetch_pr_node_id(self, repo: str, pr_number: int) -> tuple[str, str]:
        """Return (node_id, error). One of the two is always empty."""
        try:
            rc, stdout, stderr = await self._run_gh(
                [
                    "gh",
                    "pr",
                    "view",
                    str(pr_number),
                    "--repo",
                    repo,
                    "--json",
                    "id",
                ],
            )
        except TimeoutError:
            return "", f"Timeout fetching node_id for {repo}#{pr_number}"
        except OSError as e:
            return "", f"OS error fetching node_id: {e}"

        if rc != 0:
            return "", f"gh pr view failed: {stderr.decode(errors='replace').strip()}"

        try:
            payload = json.loads(stdout.decode(errors="replace"))
        except json.JSONDecodeError as e:
            return "", f"failed to parse gh pr view JSON: {e}"

        if not isinstance(payload, dict):
            return "", "gh pr view returned non-object JSON"

        node_id = payload.get("id", "")
        if not isinstance(node_id, str) or not node_id:
            return "", "gh pr view returned empty or non-string id"
        return node_id, ""

    async def _enqueue_pr(self, node_id: str) -> tuple[bool, int | None, str, bool]:
        """Call enqueuePullRequest GraphQL mutation.

        Returns (enqueued, position, error, skipped_no_queue).
        ``skipped_no_queue`` is True when the repo has no merge queue configured —
        this is an expected outcome for legacy repos, not an error.
        """
        try:
            rc, stdout, stderr = await self._run_gh(
                [
                    "gh",
                    "api",
                    "graphql",
                    "-f",
                    f"query={_ENQUEUE_MUTATION}",
                    "-F",
                    f"id={node_id}",
                ],
            )
        except TimeoutError:
            return False, None, "Timeout enqueuing PR", False
        except OSError as e:
            return False, None, f"OS error enqueuing: {e}", False

        stderr_text = stderr.decode(errors="replace").strip()
        stdout_text = stdout.decode(errors="replace").strip()

        if rc != 0:
            lowered = (stderr_text + " " + stdout_text).lower()
            if any(marker in lowered for marker in _NO_MERGE_QUEUE_MARKERS):
                return False, None, "", True
            return (
                False,
                None,
                f"enqueuePullRequest failed: {stderr_text or stdout_text}",
                False,
            )

        try:
            payload = json.loads(stdout_text or "{}")
        except json.JSONDecodeError as e:
            return False, None, f"failed to parse enqueue response: {e}", False

        if not isinstance(payload, dict):
            return False, None, "enqueue response was not a JSON object", False

        data = payload.get("data")
        if not isinstance(data, dict):
            return False, None, "enqueue response missing data object", False

        enqueue_result = data.get("enqueuePullRequest")
        if not isinstance(enqueue_result, dict):
            return (
                False,
                None,
                "enqueue response missing enqueuePullRequest object",
                False,
            )

        entry = enqueue_result.get("mergeQueueEntry")
        if entry is None:
            return (
                False,
                None,
                "enqueuePullRequest returned null mergeQueueEntry",
                False,
            )
        if not isinstance(entry, dict):
            return (
                False,
                None,
                "enqueuePullRequest returned non-object mergeQueueEntry",
                False,
            )

        position = entry.get("position")
        if not isinstance(position, int):
            return (
                False,
                None,
                "enqueuePullRequest returned non-integer position",
                False,
            )

        return True, position, "", False

    async def _enable_and_enqueue(
        self,
        repo: str,
        pr_number: int,
        merge_method: str,
        dry_run: bool,
    ) -> ModelAutoMergeOutcome:
        """Enable auto-merge and enqueue a single PR."""
        if dry_run:
            logger.info(
                "[DRY RUN] Would enable auto-merge + enqueue %s#%d", repo, pr_number
            )
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=True,
                dry_run=True,
                enqueue_skipped=True,
            )

        cmd = [
            "gh",
            "pr",
            "merge",
            str(pr_number),
            "--repo",
            repo,
            "--auto",
            f"--{merge_method}",
        ]

        try:
            rc, _, stderr = await self._run_gh(cmd)
        except TimeoutError:
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=False,
                error_message=f"Timeout enabling auto-merge on {repo}#{pr_number}",
                enqueue_skipped=True,
            )
        except OSError as e:
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=False,
                error_message=f"OS error: {e}",
                enqueue_skipped=True,
            )

        if rc != 0:
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=False,
                error_message=(
                    "gh pr merge --auto failed: "
                    f"{stderr.decode(errors='replace').strip()}"
                ),
                enqueue_skipped=True,
            )

        logger.info("Auto-merge enabled on %s#%d (%s)", repo, pr_number, merge_method)

        node_id, node_err = await self._fetch_pr_node_id(repo, pr_number)
        if node_err:
            logger.warning(
                "Auto-merge armed but node_id fetch failed for %s#%d: %s",
                repo,
                pr_number,
                node_err,
            )
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=True,
                enqueue_error=node_err,
            )

        enqueued, position, enqueue_err, skipped_no_queue = await self._enqueue_pr(
            node_id
        )

        if skipped_no_queue:
            logger.info(
                "%s#%d: repo has no merge queue — auto-merge will handle merge",
                repo,
                pr_number,
            )
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=True,
                enqueue_skipped=True,
            )

        if enqueued:
            logger.info("Enqueued %s#%d at position %s", repo, pr_number, position)
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=True,
                enqueued=True,
                queue_position=position,
            )

        logger.warning(
            "Auto-merge armed but enqueuePullRequest failed for %s#%d: %s",
            repo,
            pr_number,
            enqueue_err,
        )
        return ModelAutoMergeOutcome(
            repo=repo,
            pr_number=pr_number,
            success=True,
            enqueue_error=enqueue_err,
        )

    async def handle(
        self,
        prs: tuple[tuple[str, int], ...],
        correlation_id: UUID,
        merge_method: Literal["squash", "merge", "rebase"] = "squash",
        dry_run: bool = False,
    ) -> ModelAutoMergeResult:
        """Enable auto-merge and enqueue all specified PRs.

        Args:
            prs: Tuples of (repo, pr_number).
            correlation_id: Workflow correlation ID.
            merge_method: Merge strategy.
            dry_run: If true, do not actually enable auto-merge.

        Returns:
            ModelAutoMergeResult with per-PR outcomes.
        """
        logger.info(
            "Enabling auto-merge + enqueue on %d PRs "
            "(method=%s, dry_run=%s, correlation_id=%s)",
            len(prs),
            merge_method,
            dry_run,
            correlation_id,
        )

        tasks = [
            self._enable_and_enqueue(repo, pr_num, merge_method, dry_run)
            for repo, pr_num in prs
        ]
        outcomes = await asyncio.gather(*tasks)

        enabled = sum(1 for o in outcomes if o.success and not o.dry_run)
        enqueued = sum(1 for o in outcomes if o.enqueued)
        failed = sum(1 for o in outcomes if not o.success)

        return ModelAutoMergeResult(
            correlation_id=correlation_id,
            outcomes=tuple(outcomes),
            total_enabled=enabled,
            total_enqueued=enqueued,
            total_failed=failed,
            success=failed == 0,
        )
