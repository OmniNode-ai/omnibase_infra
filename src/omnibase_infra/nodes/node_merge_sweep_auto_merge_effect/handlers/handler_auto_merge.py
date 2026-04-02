# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Handler that enables GitHub auto-merge on PRs via gh CLI.

This is an EFFECT handler - it performs external I/O (GitHub API).
"""

from __future__ import annotations

import asyncio
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


class HandlerAutoMerge:
    """Enables GitHub auto-merge on Track A PRs using gh CLI."""

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def _enable_auto_merge(
        self,
        repo: str,
        pr_number: int,
        merge_method: str,
        dry_run: bool,
    ) -> ModelAutoMergeOutcome:
        """Enable auto-merge on a single PR."""
        if dry_run:
            logger.info("[DRY RUN] Would enable auto-merge on %s#%d", repo, pr_number)
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=True,
                dry_run=True,
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
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except TimeoutError:
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=False,
                error_message=f"Timeout enabling auto-merge on {repo}#{pr_number}",
            )
        except OSError as e:
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=False,
                error_message=f"OS error: {e}",
            )

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            return ModelAutoMergeOutcome(
                repo=repo,
                pr_number=pr_number,
                success=False,
                error_message=f"gh pr merge --auto failed: {err}",
            )

        logger.info("Auto-merge enabled on %s#%d (%s)", repo, pr_number, merge_method)
        return ModelAutoMergeOutcome(
            repo=repo,
            pr_number=pr_number,
            success=True,
        )

    async def handle(
        self,
        prs: tuple[tuple[str, int], ...],
        correlation_id: UUID,
        merge_method: Literal["squash", "merge", "rebase"] = "squash",
        dry_run: bool = False,
    ) -> ModelAutoMergeResult:
        """Enable auto-merge on all specified PRs.

        Args:
            prs: Tuples of (repo, pr_number).
            correlation_id: Workflow correlation ID.
            merge_method: Merge strategy.
            dry_run: If true, do not actually enable auto-merge.

        Returns:
            ModelAutoMergeResult with per-PR outcomes.
        """
        logger.info(
            "Enabling auto-merge on %d PRs (method=%s, dry_run=%s, correlation_id=%s)",
            len(prs),
            merge_method,
            dry_run,
            correlation_id,
        )

        tasks = [
            self._enable_auto_merge(repo, pr_num, merge_method, dry_run)
            for repo, pr_num in prs
        ]
        outcomes = await asyncio.gather(*tasks)

        enabled = sum(1 for o in outcomes if o.success and not o.dry_run)
        failed = sum(1 for o in outcomes if not o.success)

        return ModelAutoMergeResult(
            correlation_id=correlation_id,
            outcomes=tuple(outcomes),
            total_enabled=enabled,
            total_failed=failed,
            success=failed == 0,
        )
