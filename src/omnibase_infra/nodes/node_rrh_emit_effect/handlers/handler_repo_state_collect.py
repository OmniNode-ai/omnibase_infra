# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Handler that collects git repository state for RRH validation.

Uses ``asyncio.create_subprocess_exec`` (not shell) to capture branch,
HEAD SHA, dirty status, repo root, and remote URL.  All errors are
captured in the result — this handler does not raise to the caller.
"""

from __future__ import annotations

import asyncio
import logging

from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.rrh.model_rrh_environment_data import ModelRRHRepoState

logger = logging.getLogger(__name__)


class HandlerRepoStateCollect:
    """Collect git repository state.

    Gathers: branch, head_sha, is_dirty, repo_root, remote_url.

    Attributes:
        handler_type: ``INFRA_HANDLER``
        handler_category: ``EFFECT``
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, repo_path: str) -> ModelRRHRepoState:
        """Collect git state from the given repository path.

        Args:
            repo_path: Absolute path to the repository root.

        Returns:
            Populated ``ModelRRHRepoState`` with git information.
            On error, fields default to empty/unknown values.
        """
        branch = await self._git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
        head_sha = await self._git(repo_path, "rev-parse", "HEAD")
        dirty_output = await self._git(repo_path, "status", "--porcelain")
        is_dirty = len(dirty_output.strip()) > 0
        root = await self._git(repo_path, "rev-parse", "--show-toplevel")
        remote_url = await self._git(repo_path, "remote", "get-url", "origin")

        return ModelRRHRepoState(
            branch=branch.strip(),
            head_sha=head_sha.strip(),
            is_dirty=is_dirty,
            repo_root=root.strip(),
            remote_url=remote_url.strip(),
        )

    @staticmethod
    async def _git(repo_path: str, *args: str) -> str:
        """Run a git command via create_subprocess_exec and return stdout.

        Uses create_subprocess_exec (not shell) for safety — arguments
        are passed as a list, preventing shell injection.

        Returns empty string on any failure.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                repo_path,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10.0)
            if proc.returncode != 0:
                logger.debug(
                    "git %s failed (rc=%d): %s",
                    " ".join(args),
                    proc.returncode,
                    stderr.decode(errors="replace").strip(),
                )
                return ""
            return stdout.decode(errors="replace")
        except (TimeoutError, FileNotFoundError, OSError) as exc:
            logger.debug("git %s error: %s", " ".join(args), exc)
            return ""


__all__: list[str] = ["HandlerRepoStateCollect"]
