# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The lag refresh runs while the job pool is busy (OMN-19018).

WHY THIS FILE EXISTS SEPARATELY FROM OMN-18990'S
-------------------------------------------------
OMN-18990 built ``LagRefresher`` to keep the control-topic lag current during
a rebuild, and its tests call ``refresh()`` directly and assert what lands in
the sampler. They test the refresher. They do not test that anything REACHES
it while a rebuild is running, and the wiring did not: the periodic task
submitted through ``_offload``, which runs on ``_job_pool``, whose
``max_workers`` is 1 and whose one worker is the rebuild. The refresh sat in
the queue behind the job it was measuring.

Measured live on the dev lane 2026-09-21, agent on
``939823681dd9254962c1395f6c5771566db82a93`` which carries OMN-18990's merge:
two reads of the queue route 130 seconds apart, during one rebuild, against a
60 second interval, returned the SAME ``control_topic_lag_observed_at`` of
``12:21:13.924Z``. The age went 28.2s to 158.6s and ``commands_ahead`` went
``1`` to ``null`` -- the fail-closed half working, the freshness half inert.

So the property under test here is a SCHEDULING one and is written
behaviourally. Asserting that the call site does not name ``_offload`` would
pass against any other single-worker pool someone wired in later.
"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from deploy_agent.agent import JOB_POOL_MAX_WORKERS, LAG_POOL_MAX_WORKERS

pytestmark = pytest.mark.unit

#: Generous next to the microseconds the refresh itself needs, and tiny next
#: to the 20-40 minutes a real rebuild occupies the job pool for.
_BOUND_SECONDS = 2.0


class _Agent:
    """The two pools and the periodic task, lifted out of the agent class.

    Constructing the real agent needs a kafka config, a state dir and a git
    clone. What is under test is which executor the periodic task submits to,
    so the pools and the loop are reproduced exactly and nothing else is.
    """

    def __init__(self, *, use_job_pool: bool) -> None:
        self._shutdown = False
        self._use_job_pool = use_job_pool
        self._job_pool = ThreadPoolExecutor(
            max_workers=JOB_POOL_MAX_WORKERS, thread_name_prefix="deploy-agent-job"
        )
        self._lag_pool = ThreadPoolExecutor(
            max_workers=LAG_POOL_MAX_WORKERS, thread_name_prefix="deploy-agent-lag"
        )

    async def _offload(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._job_pool, fn)

    async def refresh_forever(self, refresh, interval: float) -> None:  # type: ignore[no-untyped-def]
        loop = asyncio.get_running_loop()
        while not self._shutdown:
            if self._use_job_pool:
                await self._offload(refresh)
            else:
                await loop.run_in_executor(self._lag_pool, refresh)
            await asyncio.sleep(interval)

    def close(self) -> None:
        self._job_pool.shutdown(wait=False)
        self._lag_pool.shutdown(wait=False)


async def _drive(*, use_job_pool: bool) -> bool:
    """Occupy the job pool, then see whether one refresh lands inside the bound."""
    agent = _Agent(use_job_pool=use_job_pool)
    release = threading.Event()
    refreshed = threading.Event()

    def _rebuild() -> None:
        # Stands in for `_execute_command`, which holds the job pool's only
        # worker for the whole rebuild.
        release.wait(timeout=30.0)

    def _refresh() -> None:
        refreshed.set()

    try:
        loop = asyncio.get_running_loop()
        rebuild = loop.run_in_executor(agent._job_pool, _rebuild)
        # Let the rebuild actually occupy the worker before the task starts,
        # so this measures queueing and not a race for the first slot.
        await asyncio.sleep(0.1)
        task = asyncio.create_task(agent.refresh_forever(_refresh, 0.05))
        deadline = time.monotonic() + _BOUND_SECONDS
        while not refreshed.is_set() and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        landed = refreshed.is_set()
        task.cancel()
        release.set()
        await asyncio.gather(rebuild, return_exceptions=True)
        await asyncio.gather(task, return_exceptions=True)
        return landed
    finally:
        release.set()
        agent.close()


class TestTheRefreshIsNotQueuedBehindTheRebuild:
    async def test_it_lands_while_the_job_pool_worker_is_occupied(self) -> None:
        """AC1. The repair."""
        assert await _drive(use_job_pool=False), (
            "the refresh must reach the sampler while a rebuild holds the job "
            "pool; that window is the only one it exists for"
        )

    async def test_the_positive_control_reproduces_the_defect(self) -> None:
        """AC1's falsifier, kept in the tree rather than described in prose.

        This is the shipped-and-broken wiring. If it ever starts passing, the
        test above has stopped discriminating and proves nothing.
        """
        assert not await _drive(use_job_pool=True), (
            "submitting to the single-worker job pool must still be observably "
            "blocked; a pass here means this file no longer tests anything"
        )


class TestTheSerialisationPropertyIsNotRelaxed:
    def test_the_job_pool_is_still_single_worker(self) -> None:
        """AC2. The fix must not be a pool widening."""
        assert JOB_POOL_MAX_WORKERS == 1, (
            "one worker is what makes two overlapping deploys impossible; a "
            "stale count is a far smaller failure than a second rebuild"
        )

    def test_the_lag_pool_is_its_own_and_is_also_single_worker(self) -> None:
        assert LAG_POOL_MAX_WORKERS == 1


class TestTheRefreshNeverReachesTheJobPool:
    async def test_the_job_pool_sees_the_deploy_and_not_the_refresh(self) -> None:
        """AC3, with its own positive control in the same test.

        A test that only asserted "the job pool was never used" would pass
        against a wiring where nothing ran at all.
        """
        agent = _Agent(use_job_pool=False)
        submitted: list[str] = []
        real_submit = agent._job_pool.submit

        def _recording_submit(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
            submitted.append(getattr(fn, "__name__", repr(fn)))
            return real_submit(fn, *args, **kwargs)

        agent._job_pool.submit = _recording_submit  # type: ignore[method-assign]
        try:
            refreshed = threading.Event()
            task = asyncio.create_task(agent.refresh_forever(refreshed.set, 0.05))
            deadline = time.monotonic() + _BOUND_SECONDS
            while not refreshed.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            assert refreshed.is_set()
            assert submitted == [], (
                f"the refresh must not reach the job pool; it submitted {submitted}"
            )

            # Positive control: a deploy still does reach it.
            await agent._offload(lambda: None)
            assert submitted, "an offloaded deploy must still use the job pool"

            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        finally:
            agent.close()


class TestTeardown:
    def test_both_pools_shut_down_and_leave_no_live_worker(self) -> None:
        """AC4."""
        agent = _Agent(use_job_pool=False)
        threads: list[threading.Thread] = []
        agent._lag_pool.submit(lambda: threads.append(threading.current_thread()))
        agent._lag_pool.shutdown(wait=True)
        agent._job_pool.shutdown(wait=True)
        assert threads, "the lag pool must actually have run something"
        assert not any(t.is_alive() for t in threads)
