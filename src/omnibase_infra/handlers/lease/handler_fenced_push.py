# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A fix worker's push, with fence checks before and after acceptance (OMN-21426).

The pre-check refuses stale, expired and unknown grants without pushing.
An accepted push is re-checked and reports ``fence_held_after_push`` so callers
can quarantine it if the grant was lost or cannot be read. Git and etcd cannot
commit atomically: a stale push is reported, not prevented. The intervening
window is bounded by the push timeout, and ``--force-with-lease`` conditions
the Git write on the branch head the worker built on, not on the etcd fence.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Awaitable, Callable

from omnibase_core.validators.no_unguarded_git_subprocess import scrub_git_location_env
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.handlers.lease.handler_lease_fence import check_fence
from omnibase_infra.handlers.placement.handler_placement_etcd import (
    HandlerPlacementEtcd,
)
from omnibase_infra.models.lease import (
    EnumLeaseFenceStatus,
    FencedPushRefusal,
    ModelFencedPushRequest,
    ModelFencedPushResult,
)

_REFUSAL: dict[EnumLeaseFenceStatus, FencedPushRefusal] = {
    EnumLeaseFenceStatus.STALE: "LEASE_FENCE_STALE",
    EnumLeaseFenceStatus.EXPIRED: "LEASE_EXPIRED",
    EnumLeaseFenceStatus.UNKNOWN: "LEASE_UNREADABLE",
}


#: Performs the push once the fence reads ``ok``; returns ``(accepted, detail)``.
FencedPushTransport = Callable[[ModelFencedPushRequest], Awaitable[tuple[bool, str]]]


class GitCliPushTransport:
    """``git push --force-with-lease=<branch>:<expected sha>`` from the worktree.

    The push runs with the git location variables scrubbed, so a ``GIT_DIR``
    inherited from a hook cannot retarget it away from ``request.worktree``.
    """

    def __init__(self, timeout_s: float = 300.0) -> None:
        self._timeout_s = timeout_s

    async def push(self, request: ModelFencedPushRequest) -> tuple[bool, str]:
        ref = f"refs/heads/{request.branch}"
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            request.worktree,
            "push",
            "--porcelain",
            f"--force-with-lease={ref}:{request.expected_remote_sha}",
            request.remote,
            f"HEAD:{ref}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=scrub_git_location_env(),
            start_new_session=True,
        )
        try:
            output, _ = await asyncio.wait_for(proc.communicate(), self._timeout_s)
        except (TimeoutError, asyncio.CancelledError) as exc:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await proc.wait()
            if isinstance(exc, asyncio.CancelledError):
                raise
            return False, f"git push timed out after {self._timeout_s:g} s"
        return proc.returncode == 0, output.decode(errors="replace").strip()


class HandlerFencedPush:
    """Check before pushing and after acceptance; report a grant lost during the push."""

    def __init__(
        self, adapter: HandlerPlacementEtcd, transport: FencedPushTransport
    ) -> None:
        self._adapter = adapter
        self._transport = transport

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(self, request: ModelFencedPushRequest) -> ModelFencedPushResult:
        status = await check_fence(
            self._adapter,
            kind=request.kind,
            resource=request.resource,
            fence=request.fence,
        )
        if status is not EnumLeaseFenceStatus.OK:
            return ModelFencedPushResult(
                outcome="refused",
                reason=_REFUSAL[status],
                kind=request.kind,
                resource=request.resource,
                fence=request.fence,
                fence_held_after_push=False,
                detail=f"check_fence read {status.value}; no push was made",
            )
        accepted, detail = await self._transport(request)
        fence_held_after_push = False
        if accepted:
            after = await check_fence(
                self._adapter,
                kind=request.kind,
                resource=request.resource,
                fence=request.fence,
            )
            fence_held_after_push = after is EnumLeaseFenceStatus.OK
        return ModelFencedPushResult(
            outcome="pushed" if accepted else "refused",
            reason="" if accepted else "PUSH_REJECTED",
            kind=request.kind,
            resource=request.resource,
            fence=request.fence,
            fence_held_after_push=fence_held_after_push,
            detail=detail,
        )


__all__ = ["FencedPushTransport", "GitCliPushTransport", "HandlerFencedPush"]
