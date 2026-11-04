# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Settle the subscriber dispatches still running at shutdown (OMN-20117)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

__all__ = ["ModelLiveDispatchSettler"]

logger = logging.getLogger(__name__)


class ModelLiveDispatchSettler(BaseModel):
    """The two bounds of the shutdown drain, and the drain itself."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    drain_seconds: float
    grace_seconds: float

    async def settle(
        self,
        live_dispatches: Iterable[asyncio.Future[None]],
        record_tasks: Iterable[asyncio.Task[None]],
    ) -> None:
        """Let running dispatches finish, then cancel the rest (OMN-20117).

        Runs with the producer open. First waits up to
        ``consumer_shutdown_drain_seconds`` for every live subscriber dispatch
        to return, so a handler that finishes in that window publishes its
        terminal and has its record committed. Then cancels whatever is still
        running and waits up to ``consumer_shutdown_cancel_grace_seconds`` for
        those to settle: a handler that answers its cancellation with a typed
        failure terminal publishes it here, so its caller is told instead of
        waiting out its window. A cancelled dispatch that returns nothing is
        never committed past, so the next consumer redelivers its record.

        Only the INNER dispatch futures are cancelled, never the consume
        tasks awaiting them. Those tasks see the dispatch settle through their
        own ``asyncio.wait`` and account for it exactly as for any other
        outcome: a returned dispatch is finished and committed, a cancelled
        one is not.
        """
        live = {dispatch for dispatch in live_dispatches if not dispatch.done()}
        drain = self.drain_seconds
        grace = self.grace_seconds
        if live:
            logger.info(
                "shutdown_drain_started live_dispatches=%d drain_seconds=%.1f "
                "-- publishing stays open until they settle (OMN-20117)",
                len(live),
                drain,
            )
            if drain > 0:
                _, live = await asyncio.wait(live, timeout=drain)
        if live:
            logger.warning(
                "shutdown_drain_expired cancelling=%d cancel_grace_seconds=%.1f "
                "-- a handler may answer its cancellation with a failure "
                "terminal; a record whose handler does not is left uncommitted "
                "and is redelivered to the next consumer (OMN-20117)",
                len(live),
                grace,
            )
            for dispatch in live:
                dispatch.cancel()
            if grace > 0:
                _, unsettled = await asyncio.wait(live, timeout=grace)
                if unsettled:
                    logger.warning(
                        "shutdown_cancel_grace_expired unsettled=%d -- these "
                        "dispatches ignored cancellation; their records stay "
                        "uncommitted (OMN-20117)",
                        len(unsettled),
                    )
        # Let the record tasks awaiting the settled dispatches run their
        # accounting (finish, commit) before the consumers are stopped.
        accounting = {task for task in record_tasks if not task.done()}
        if accounting:
            await asyncio.wait(accounting, timeout=max(grace, 1.0))
