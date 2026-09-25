# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Main orchestrator. Single-job concurrency. Runs consumer + health + publisher concurrently."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import signal
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiohttp import web

from deploy_agent.accept_backlog import (
    DEFAULT_SAMPLE_INTERVAL_SECONDS,
    DEFAULT_UNDRAINED_BOUND_SECONDS,
    AcceptBacklogWatchdog,
    read_accept_queue,
)
from deploy_agent.coalesce import GitAncestryResolver, ModelSupersession
from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    TOPIC_REBUILD_REJECTED,
    TOPIC_REBUILD_REQUESTED,
    DeployInProgressError,
    EnumOnexApiDeliveryResult,
    EnumRejectionReason,
    EnumRuntimeLane,
    EnumSelfUpdateBoundary,
    ModelOnexApiDelivery,
    ModelRebuildRejected,
    ModelRebuildRequested,
    ModelRejectionNotice,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import (
    DEPLOY_AGENT_DIR,
    REPO_DIR,
    SCOPE_BUNDLES,
    DeployExecutor,
    DevLaneMigrationPreflightError,
    EnumInstancePhase,
    active_dev_instance,
    assert_prod_request_has_stability_digest,
    lane_config_for,
    lane_runs_phase,
    resolve_prod_target_service,
    select_dev_instance,
)
from deploy_agent.health import create_health_app
from deploy_agent.job_state import EnumJobSettlingStage, JobState, JobStore
from deploy_agent.kafka_config import load_deploy_agent_kafka_config_from_env
from deploy_agent.lab_overlay import (
    DEFAULT_APPLY_BUDGET_SECONDS,
    LabOverlayApplier,
)
from deploy_agent.lag_refresher import (
    DEFAULT_REFRESH_INTERVAL_SECONDS,
    LagRefresher,
)
from deploy_agent.lane_lock_client import (
    DEFAULT_LANE_LOCK_TIMEOUT_SECONDS,
    LaneLockContendedError,
    lane_lock,
)
from deploy_agent.lane_policy import load_allowed_lanes_from_env
from deploy_agent.lineage_fence import DockerProvenanceReader, GitRefResolver
from deploy_agent.loaded_code import record_loaded_code_sha
from deploy_agent.lock import single_flight_lock
from deploy_agent.publisher import (
    PublishCircuitBreaker,
    build_completion_payload,
    publish_result,
)
from deploy_agent.queue_depth import LagSampler
from deploy_agent.routing import ROUTED_LANES, build_router_from_env
from deploy_agent.tracking_ref import load_tracking_remote_ref_from_env

logger = logging.getLogger(__name__)


def _runtime_container_for_lane(lane: EnumRuntimeLane) -> str:
    """The lane's main runtime container, whose image records its build (OMN-19270)."""
    return lane_config_for(lane).main_runtime_container


STATE_DIR = Path(
    os.environ.get("DEPLOY_AGENT_STATE_DIR", "/data/omninode/deploy-agent/state/jobs")
)
HEALTH_PORT = int(os.environ.get("DEPLOY_AGENT_PORT", "8099"))
PUBLISH_RETRY_INTERVAL = 30

#: OMN-18636 AC4. The accept-backlog watchdog's declared bound and cadence.
#:
#: Named here, at module scope, rather than passed from a call site, because the
#: bound is the criterion: "the queue has been non-empty and undrained for
#: longer than a declared bound". See ``accept_backlog`` for what the numbers
#: mean and why thirty seconds is the bound on the DEFECT rather than on a
#: deploy.
ACCEPT_BACKLOG_BOUND_SECONDS = DEFAULT_UNDRAINED_BOUND_SECONDS
ACCEPT_BACKLOG_INTERVAL_SECONDS = DEFAULT_SAMPLE_INTERVAL_SECONDS

#: OMN-18200. How often an IDLE agent re-checks whether the code it loaded is
#: still the code in its clone.
#:
#: The two OMN-16442 boundaries are both job-driven, so an agent that nobody
#: sends a job to cannot pick up a fix to itself -- and on 2026-09-14 the fix
#: that needed picking up WAS the fix to this agent, whose merge published no
#: rebuild command at all. Five minutes is a git fetch against one branch, far
#: below the cadence of the external reconciler that moves this clone, and well
#: inside the window in which a merged agent fix should start running.
#:
#: A cadence is a tuning knob, not a deployment identity, so unlike the tracking
#: ref and the lane fence it carries a declared default rather than refusing to
#: start without one.
SELF_UPDATE_IDLE_INTERVAL_SECONDS = int(
    os.environ.get("DEPLOY_AGENT_SELF_UPDATE_IDLE_INTERVAL", "300")
)

#: OMN-18200 AC5. The k3s ``onex-lab`` overlay re-apply, ON by default.
#:
#: Default ON deliberately, per rule 5 ("enforcement, not detection"): an opt-in
#: lab re-apply is one nobody turns on, which is exactly how the lane reached
#: three days of staleness under a green trigger. ``off`` is an incident kill
#: switch for the case where a wedged apply is holding this agent's single-flight
#: lock, and a disabled run says so in the journal rather than being silent.
LAB_OVERLAY_ENABLED = os.environ.get("DEPLOY_AGENT_LAB_OVERLAY", "on").lower() != "off"
LAB_OVERLAY_BUDGET_SECONDS = int(
    os.environ.get("DEPLOY_AGENT_LAB_OVERLAY_BUDGET", str(DEFAULT_APPLY_BUDGET_SECONDS))
)
#: The ``omninode_infra`` clone the overlay is archived from. Derived from
#: ``REPO_DIR``'s parent rather than written out, because the two are siblings in
#: the same ``omni_home`` clone by construction and a second literal path is a
#: second thing to keep in step (rule 6).
LAB_OVERLAY_SOURCE_DIR = Path(
    os.environ.get(
        "DEPLOY_AGENT_LAB_OVERLAY_SOURCE", str(Path(REPO_DIR).parent / "omninode_infra")
    )
)

# NOTE (OMN-13760): no systemd watchdog. _run_deploy runs minutes-long
# synchronous subprocess.run() rebuilds — and a background pinger would only
# mask real hangs. The unit therefore declares no WatchdogSec; Restart=on-failure
# recovers genuine crashes. See deploy/deploy-agent.service for the rationale.
#
# OMN-18636 amends the PREMISE of that note without changing its conclusion. The
# rebuilds are still minutes-long synchronous subprocesses, but they no longer
# run on the event loop thread, so "a periodic ping task cannot fire during a
# rebuild" is no longer true. The unit still declares no WatchdogSec, now for the
# remaining reason only: a liveness ping that CAN fire throughout a rebuild
# proves the loop is turning, which is not the same fact as the deploy making
# progress, and Restart=on-failure already recovers a crash. Re-deciding that is
# a separate change with its own evidence.

#: OMN-18636. How many deploy jobs this agent runs at once. ONE, and the number
#: is the contract rather than a tuning knob.
#:
#: Every phase of a deploy is a synchronous ``subprocess.run`` against a shared
#: host: one compose project, one git clone, one image cache, one lane lock. The
#: agent has always run exactly one job at a time — ``single_flight_lock`` and
#: the consumer's ``busy`` rejection both say so — and moving the work off the
#: event loop thread must not quietly buy concurrency the rest of the design
#: refuses. A pool of one preserves the existing ordering exactly: the poll, the
#: job, the self-update boundaries and the publish retries all run on the same
#: single worker, in the order they were submitted, which is the order they ran
#: in when they all ran on the loop thread.
#:
#: Raising this is not a performance knob, it is a change to the concurrency
#: contract, and ``tests/unit/test_agent_offloads_phases_omn18636.py`` fails if
#: it moves.
JOB_POOL_MAX_WORKERS = 1

#: Workers on the lag-refresh pool (OMN-19018). One is enough: the refresher
#: serialises itself under its own lock and a second thread would only let two
#: ListOffsets round trips overlap. It is separate from the job pool rather
#: than larger than it, so nothing here can make two deploys possible.
LAG_POOL_MAX_WORKERS = 1

#: Milliseconds a rejection publish may spend resolving broker metadata
#: (OMN-18143). See the call site in ``_publish_rejection_event`` for the
#: measurement this replaces and for why the cause is recorded rather than
#: worked around here.
REJECTION_PUBLISH_MAX_BLOCK_MS = 10_000


class DeployAgent:
    def __init__(self, *, skip_self_update: bool = False):
        self.job_store = JobStore(state_dir=STATE_DIR)
        self.executor = DeployExecutor()
        self._state = "idle"
        self._shutdown = False
        self._current_git_sha = ""
        #: OMN-18572: this job's onex-api pin delivery verdict, read by the
        #: terminal payload. ``None`` means the delivery was never reached,
        #: which is a different fact from one that ran and refused.
        self._onex_api_delivery: ModelOnexApiDelivery | None = None
        self._skip_self_update = skip_self_update
        self._publish_cb = PublishCircuitBreaker()
        # Stamped at the top of the poll loop so the first idle check happens
        # one interval AFTER startup -- a process that has just recorded its own
        # identity has nothing to compare yet.
        #
        # ``None`` means "never checked, due now", and it is None rather than
        # 0.0 because ``time.monotonic()``'s zero is an arbitrary reference
        # point, not a time. On Linux it is the boot instant, so on a runner
        # that has been up for less than the interval a 0.0 sentinel reads as
        # "checked recently" and the check never fires -- which is exactly what
        # happened to this file's own tests in CI while they passed on a
        # long-running workstation. A sentinel that means "never" must not be
        # a value the clock can produce.
        self._last_idle_self_update: float | None = None
        self._kafka_config = load_deploy_agent_kafka_config_from_env()
        # OMN-16939: fail closed at process construction, before the health
        # port binds and long before a command is polled. An agent that has
        # not declared which lanes it may deploy must not start at all.
        self._allowed_lanes = load_allowed_lanes_from_env()
        # OMN-19506. A dev-lane agent is one of the dev instances in
        # config/deploy_lane_routing.yaml, and it refuses to start when the
        # table has no default, names an unknown instance, or cannot say which
        # instance this host is (AC2). An agent fenced to other lanes routes
        # nothing.
        self._router = (
            build_router_from_env(REPO_DIR)
            if self._allowed_lanes & ROUTED_LANES
            else None
        )
        if self._router is not None:
            # OMN-19522: the instance picks the dev lane's composition too.
            # Selected here, once, before anything reads a lane config; an
            # instance with no composition refuses start (ValueError).
            select_dev_instance(self._router.instance.name)
            logger.info(
                "Deploy agent routing instance: %s, consumer group %s, %d route(s), "
                "default %s",
                self._router.instance.name,
                self._router.consumer_group,
                len(self._router.table.routes),
                self._router.table.default_instance,
            )
        # OMN-18636. The one thread every blocking call in this process runs on.
        # See JOB_POOL_MAX_WORKERS and _offload for why it is one, and why the
        # event loop thread must be left with nothing to do but serve HTTP.
        self._job_pool = ThreadPoolExecutor(
            max_workers=JOB_POOL_MAX_WORKERS,
            thread_name_prefix="deploy-agent-job",
        )
        # OMN-19018. The lag refresh gets a thread of its OWN, and borrowing
        # the pool above is the defect this repairs. That pool is deliberately
        # single-worker so deploys serialise, so a refresh submitted to it
        # queues behind the rebuild it exists to measure and runs only once
        # that rebuild is over -- inert in exactly the window OMN-18990 built
        # it for. Measured live on the dev lane 2026-09-21: the sample's
        # observation time did not move across 130s of a running rebuild
        # against a 60s interval.
        #
        # Widening the job pool instead is refused. Its single worker is what
        # makes two overlapping deploys impossible, and a stale count is a far
        # smaller failure than a second rebuild on the same lane.
        self._lag_pool = ThreadPoolExecutor(
            max_workers=LAG_POOL_MAX_WORKERS,
            thread_name_prefix="deploy-agent-lag",
        )
        # OMN-18636 AC4. Started once the socket is bound (see ``run``), stopped
        # in the same ``finally`` that tears the site down. It holds no handle on
        # the pool above and cannot curtail anything: it samples the listen
        # socket from its own thread and writes a verdict where a reader can
        # find it even when this loop is not running.
        self._accept_backlog = AcceptBacklogWatchdog(
            port=HEALTH_PORT,
            state_dir=STATE_DIR,
            bound_seconds=ACCEPT_BACKLOG_BOUND_SECONDS,
            interval_seconds=ACCEPT_BACKLOG_INTERVAL_SECONDS,
            probe=lambda: read_accept_queue(HEALTH_PORT),
        )

    def _get_state(self) -> str:
        return self._state

    async def _offload(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run one blocking call on the job thread, not on the event loop.

        OMN-18636. THE WHOLE POINT OF THIS METHOD IS WHAT IT LEAVES BEHIND: an
        event loop thread with nothing on it but the aiohttp application, so the
        listening socket is polled and ``accept()`` is called while a deploy
        runs.

        The defect it removes, measured 2026-09-17 (lane
        ``deploy-agent-http-hang-diag-2105``, 121 samples): the health app and
        the job executor shared one loop, and every executor phase was a plain
        synchronous ``subprocess.run``. For the length of every phase the loop
        thread sat inside ``_communicate``'s selector poll, nothing called
        ``accept()``, and ``/health`` returned curl code ``000`` — 107 of those
        121 samples. The listen backlog climbed to its 128 limit, after which
        the kernel silently drops SYNs, so the unreachability OUTLIVED the call
        that caused it. Downstream, ``check_dev_lane_staleness.py`` could not
        read ``deployed_revision`` off ``/job/{correlation_id}`` and the
        compose-dev lab-pass receipt read ``indeterminate`` for a lane that had
        in fact converged — which closes rule 24(b) delivery for a good sha.

        Two properties this deliberately does NOT change:

        * **Serialization.** One worker, so submissions run one at a time in
          submission order. A job still excludes the next poll, the self-update
          boundaries still fall between jobs, and a publish retry still waits
          for an in-flight deploy. Nothing here makes two deploys possible.
        * **Job duration.** Nothing cancels, times out or interrupts the call.
          The caller ``await``s it for as long as it takes. A health surface
          that stayed responsive by curtailing a rebuild would manufacture the
          false FAIL receipt this work exists to remove (OMN-18636 AC6).

        One consequence worth naming rather than discovering: the self-update
        boundaries re-exec with ``os.execv``, which now runs on this worker
        thread. That is defined: POSIX ``execve`` terminates every other thread
        in the process and the calling thread becomes the new image's initial
        thread, so a re-exec from here replaces the process exactly as it did
        from the loop thread. ``test_agent_offloads_phases_omn18636.py`` pins
        that a boundary reached through an offload still re-execs.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._job_pool, functools.partial(fn, *args, **kwargs)
        )

    async def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logger.info(
            "Deploy agent starting (state_dir=%s, kafka=%s, allowed_lanes=%s)",
            STATE_DIR,
            self._kafka_config.bootstrap_servers,
            ",".join(sorted(lane.value for lane in self._allowed_lanes)),
        )

        # Step 0: record which code this process actually loaded, before
        # anything can move the clone underneath it, and NAME IT IN THE JOURNAL
        # (OMN-18200). Until this line existed there was no artifact anywhere --
        # not the journal, not the health payload, not a job record -- that said
        # which commit a running agent was executing, which is why a process
        # four days stale beside a fixed clone looked exactly like a healthy
        # one. self_update compares against this value, not against the remote.
        loaded_sha = record_loaded_code_sha(
            os.environ.get("DEPLOY_AGENT_DIR", DEPLOY_AGENT_DIR)
        )
        logger.info("Deploy agent loaded code sha: %s", loaded_sha)

        # Step 1: Recover crashed jobs
        recovered = self.job_store.recover_crashed_jobs()
        if recovered:
            logger.info("Recovered %d crashed job(s)", len(recovered))

        # Step 2: Prune old completed jobs
        pruned = self.job_store.prune_completed()
        if pruned:
            logger.info("Pruned %d old job(s)", pruned)

        # Step 3: Retry pending publishes. Called directly rather than through
        # _offload: nothing is listening yet (the site starts at step 4), so
        # there is no HTTP surface for a blocking call to deny here.
        self._retry_pending_publishes()

        # Step 4: Start health endpoint
        # OMN-18144. Created BEFORE the health app and handed to both sides:
        # the consumer (below) writes into it from the worker thread, this
        # surface reads it from the loop thread. Built here rather than beside
        # the consumer because the health app must be able to answer /queue
        # from the moment it binds -- before the first poll it answers
        # "never sampled", which is true, rather than an empty queue.
        self._lag_sampler = LagSampler()
        health_app = create_health_app(
            job_store=self.job_store,
            get_agent_state=self._get_state,
            get_accept_backlog=self._accept_backlog.latest,
            get_control_topic_lag=self._lag_sampler.latest,
        )
        runner = web.AppRunner(health_app)
        await runner.setup()
        site = web.TCPSite(
            runner,
            "0.0.0.0",  # noqa: S104
            HEALTH_PORT,
            reuse_address=True,
            reuse_port=True,
        )
        await site.start()
        logger.info("Health endpoint listening on port %d", HEALTH_PORT)

        # AFTER the bind, because before it there is no listening socket to
        # sample and the probe would read "no such socket" as indeterminate for
        # as long as startup took.
        self._accept_backlog.start()
        logger.info(
            "Accept-backlog watchdog sampling port %d every %.1fs, bound %.1fs",
            HEALTH_PORT,
            ACCEPT_BACKLOG_INTERVAL_SECONDS,
            ACCEPT_BACKLOG_BOUND_SECONDS,
        )

        # Step 4b: converge the lane's deps BEFORE the consumer exists
        # (OMN-18692). This agent's control bus IS the dev lane's redpanda, so
        # a half-recreated lane is not merely a lane this process cannot
        # deploy to -- it is a broker this process cannot CONNECT to. On
        # 2026-09-18 the CORE phase removed that container, the agent
        # crash-looped on NoBrokersAvailable against the broker it had just
        # destroyed, and systemd gave up after seven restarts; the lane then sat
        # broken for 22 minutes until an operator ran the deps-only `up -d` by
        # hand. Nothing in this startup path read the lane first.
        self._converge_deps_before_consuming()

        # Step 5+6: Main loop
        consumer = DeployConsumer(
            kafka_config=self._kafka_config,
            job_store=self.job_store,
            allowed_lanes=self._allowed_lanes,
            self_update_hook=self._self_update_pre_accept,
            lag_sampler=self._lag_sampler,
            # OMN-18143. Both halves of coalescing are injected rather than
            # constructed inside the consumer: the ancestry fact comes from
            # the deploy-source clone, which is the executor's concern, and
            # the terminal event needs a producer, which is this class's. A
            # consumer built without either -- every test that does not ask
            # for coalescing -- folds nothing.
            ancestry_resolver=GitAncestryResolver(REPO_DIR),
            on_superseded=self._publish_superseded,
            on_rejected=self._publish_rejection_notice,
            # OMN-19270. The lineage fence compares each command with the
            # provenance the lane's runtime image was built from, and with the
            # job that built it. Infra ancestry asks the resolver above, so it
            # and coalescing share one fetch cooldown.
            running_build=DockerProvenanceReader(_runtime_container_for_lane),
            ref_resolver=GitRefResolver(REPO_DIR),
            tracking_ref=load_tracking_remote_ref_from_env(),
            router=self._router,
        )

        # Step 6b: keep the lag sample current DURING a rebuild (OMN-18990).
        # The loop below is serial -- poll, then execute -- so the consumer's
        # own sampler, which only writes from inside `poll_and_accept`, goes
        # untouched for the whole 20-40 minutes of a command. That is exactly
        # when later merges queue, so `/queue` reported a pre-rebuild zero as
        # this moment's count. This refresher owns a separate, group-less
        # client and never touches the one above.
        lag_refresher = LagRefresher(
            self._kafka_config,
            self._lag_sampler,
            TOPIC_REBUILD_REQUESTED,
        )
        lag_refresh_task = asyncio.create_task(self._refresh_lag_forever(lag_refresher))

        # Handle signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

        publish_retry_task = asyncio.create_task(self._publish_retry_loop())
        self._last_idle_self_update = time.monotonic()

        try:
            while not self._shutdown:
                # OMN-18636. EVERY branch of this loop is offloaded, not just
                # the deploy. `poll_and_accept` blocks for up to its 1000 ms
                # kafka poll and can re-exec inside its self-update hook;
                # `_maybe_self_update_idle` shells out to git. Both are short
                # next to a rebuild and both are long next to the 2 s bound the
                # receipt reader needs, and the diagnosis measured `000`s with
                # no child process at all — the short phases deny the surface
                # exactly as effectively as `docker build` does. A loop that
                # offloaded only the obvious minutes-long call would still fail
                # the bound it is here to hold.
                cmd, reason = await self._offload(consumer.poll_and_accept)
                if cmd is not None:
                    await self._offload(self._execute_command, cmd)
                elif reason:
                    logger.info("Rejected command: %s", reason)
                else:
                    await self._offload(self._maybe_self_update_idle)
                    await asyncio.sleep(1)
        finally:
            publish_retry_task.cancel()
            lag_refresh_task.cancel()
            lag_refresher.close()
            consumer.close()
            # Stopped before the site goes away: once the socket is closed the
            # probe reads nothing and the last verdict would be overwritten with
            # an indeterminate one, erasing the evidence a reader came for.
            self._accept_backlog.stop()
            await runner.cleanup()
            # Last, and waiting. The loop above only exits once its own awaited
            # offload has returned, so nothing of this agent's own work is in
            # flight here; the wait is for the publish-retry task, whose
            # asyncio-level cancel does not reach a function already running on
            # the worker. Abandoning a publish mid-flight is how a terminal
            # result goes missing.
            self._job_pool.shutdown(wait=True)
            # Not waited on, unlike the job pool above. An observation in
            # flight owes nobody a result, and a ListOffsets round trip to an
            # unreachable broker must not hold up shutdown.
            self._lag_pool.shutdown(wait=False)
            logger.info("Deploy agent stopped")

    async def _refresh_lag_forever(self, refresher: LagRefresher) -> None:
        """Sample the control-topic lag on a timer, off the polling client.

        Runs on ``_lag_pool``, this class's own thread, and NOT on the job
        pool: the job pool has one worker by design and it is occupied by the
        rebuild for the whole window this refresh exists to measure
        (OMN-19018). It stays off the event loop for the same reason every
        other blocking call here does -- the listening socket must keep being
        accepted while a deploy runs.

        Never lets a failed observation end the task. A refresher that stopped
        on its first transient error would leave the sampler ageing silently,
        which is the shape of the defect rather than a repair of it -- and the
        staleness bound one layer down is what turns that into an unreadable
        answer instead of a stale number.
        """
        loop = asyncio.get_running_loop()
        while not self._shutdown:
            try:
                # NOT ``_offload`` (OMN-19018). That submits to the
                # single-worker job pool, where this call would sit behind the
                # rebuild it is measuring.
                await loop.run_in_executor(self._lag_pool, refresher.refresh)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never fatal to the agent
                logger.debug("lag refresh task iteration failed: %s", exc)
            await asyncio.sleep(DEFAULT_REFRESH_INTERVAL_SECONDS)

    def _converge_deps_before_consuming(self) -> bool:
        """Bring a half-recreated lane's deps up before the consumer is built.

        Returns whether convergence was attempted, so a test can assert the
        fence below rather than infer it from the absence of a docker call.

        THE FENCE IS THE POINT, and it is fail-closed. This runs ONLY when the
        agent's declared lane set is exactly ``{dev}`` -- the lane
        ``omni_home/CLAUDE.md``'s lane table calls a "fully mutable test
        platform". An agent that also carries ``stability-test`` or ``prod``
        does nothing here, because bringing containers up on a governed lane
        without an accepted command is an unattributed lane mutation, which is
        precisely what the OMN-15243 raw-bypass signature set and the OMN-15218
        attribution interlock exist to refuse. A recovery that had to violate a
        deploy gate to run would be a worse defect than the one it repairs.

        Deliberately NON-FATAL. A convergence that fails must not stop the agent
        from starting: the process still has a job store to recover, a health
        surface to serve and pending publishes to retry, and an agent that
        refused to start because its lane was broken is an agent that cannot
        report that its lane is broken.
        """
        if self._allowed_lanes != {EnumRuntimeLane.DEV}:
            logger.info(
                "Skipping the startup deps convergence: this agent's declared "
                "lanes are %s, and convergence runs only for an agent fenced to "
                "the dev lane alone -- an un-commanded container start on a "
                "governed lane is an unattributed lane mutation",
                ",".join(sorted(lane.value for lane in self._allowed_lanes)),
            )
            return False
        try:
            converged, was_down = self.executor.converge_deps(lane=EnumRuntimeLane.DEV)
        except Exception:
            logger.exception(
                "Startup deps convergence raised; continuing to start. The lane "
                "may still be missing its broker, which this process will "
                "report through /health rather than by crash-looping."
            )
            return True
        if was_down:
            logger.warning(
                "Startup deps convergence found %s not running and %s",
                was_down,
                "converged the lane" if converged else "COULD NOT converge the lane",
            )
        return True

    def _handle_shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._shutdown = True

    def _self_update_pre_accept(self, rewind_offset: Callable[[], None]) -> None:
        """Self-update boundary before a polled command is marked started.

        Passed to the consumer, which calls it once a command has cleared every
        acceptance check and before ``job_store.accept``. ``rewind_offset``
        re-points the committed offset at that command so the replacement
        process re-reads it (OMN-16442).
        """
        self.executor.self_update(
            boundary=EnumSelfUpdateBoundary.PRE_ACCEPT,
            skip=self._skip_self_update,
            on_before_reexec=rewind_offset,
        )

    def _self_update_post_terminal(self) -> None:
        """Self-update boundary after a job reaches a terminal, published state.

        This is the deferral half of OMN-16442: an agent that finds itself
        behind while a deploy is in flight does NOT interrupt that deploy --
        the deploy completes on the version it started on, and the update fires
        here, at the next boundary.
        """
        try:
            self.executor.self_update(
                boundary=EnumSelfUpdateBoundary.POST_TERMINAL,
                skip=self._skip_self_update,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(  # noqa: TRY400
                "Self-update at the post-terminal boundary failed, "
                "staying on the current image: %s "
                "friction_type=self_update_boundary_failed",
                e,
            )

    def _maybe_self_update_idle(self) -> None:
        """Self-update boundary on the poll loop's idle branch (OMN-18200).

        The third boundary, and the only one that is not job-driven. Both
        OMN-16442 boundaries fire inside the handling of a command, so an agent
        with no traffic can never pick up a fix -- including, and especially, a
        fix to itself. That is not hypothetical: on 2026-09-14 the change that
        needed to reach this process was the change to this process, and the
        only merge that would have published a rebuild command was that same
        merge.

        Two guards, both asserted rather than assumed, because "the poll
        returned nothing" is not the same fact as "nothing is in flight":

        * an accepted or in-progress job means a deploy is running and a
          re-exec would abort it -- the OMN-16442 defect, arrived at from a
          different direction;
        * a job awaiting its terminal publish means a result is owed to the
          bus. It IS durable (``result_publish_pending`` on disk, replayed by
          ``_retry_pending_publishes`` at startup), so a re-exec here would not
          lose it -- but it would delay it by a process restart for no reason,
          and the next idle tick is one interval away.

        The interval is stamped whether or not the guards let the check through,
        so a guarded agent re-checks on the next interval rather than probing
        the job store on every one-second poll.
        """
        now = time.monotonic()
        if (
            self._last_idle_self_update is not None
            and now - self._last_idle_self_update < SELF_UPDATE_IDLE_INTERVAL_SECONDS
        ):
            return
        self._last_idle_self_update = now

        if self.job_store.has_active_job():
            logger.info(
                "self_update[boundary=%s]: a job is in flight, deferring",
                EnumSelfUpdateBoundary.IDLE_HEARTBEAT.value,
            )
            return
        pending = self.job_store.get_pending_publish()
        if pending:
            logger.info(
                "self_update[boundary=%s]: %d result(s) still awaiting publish, "
                "deferring",
                EnumSelfUpdateBoundary.IDLE_HEARTBEAT.value,
                len(pending),
            )
            return

        try:
            self.executor.self_update(
                boundary=EnumSelfUpdateBoundary.IDLE_HEARTBEAT,
                skip=self._skip_self_update,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(  # noqa: TRY400
                "Self-update at the idle boundary failed, "
                "staying on the current image: %s "
                "friction_type=self_update_boundary_failed",
                e,
            )

    def _execute_command(self, cmd: ModelRebuildRequested) -> None:
        """Run one accepted command to its terminal state. BLOCKING, by design.

        OMN-18636 made this synchronous rather than ``async``. It never awaited
        anything: every statement in it and in ``_run_deploy`` is a blocking
        call, and declaring that ``async`` said the opposite of what the body
        did — which is precisely how the whole deploy came to run on the event
        loop thread. It is now called through ``DeployAgent._offload``, and its
        signature is the honest one for what it is.
        """
        try:
            with single_flight_lock():
                self._run_deploy(cmd)
        except DeployInProgressError:
            logger.warning(
                "Single-flight lock held — rejecting %s (in_progress)",
                cmd.correlation_id,
            )
            self._publish_rejected(cmd, reason=EnumRejectionReason.IN_PROGRESS)
            return

        # OMN-16442 job boundary: the job has a terminal status, its result has
        # been published, and the single-flight lock is released. Deliberately
        # outside the `with` block above — a re-exec must not happen while this
        # process holds the deploy lock.
        self._self_update_post_terminal()

    def _run_deploy(self, cmd: ModelRebuildRequested) -> None:
        """Execute every phase of one deploy. BLOCKING — see ``_execute_command``."""
        self._state = "deploying"
        cid = cmd.correlation_id
        health_checks = []
        # OMN-18057: what the deploy ACTUALLY did, for the terminal event.
        # rebuild_scope has always returned the services it brought up and this
        # method has always discarded the return, which is why every terminal
        # event reported services_restarted=[] for a scope-default deploy.
        services_restarted: list[str] = []
        # OMN-18545: the resolved sha is a PROPERTY OF THIS JOB, cleared here
        # like the two above. It lives on the agent, which outlives the job, and
        # the lab overlay is now reachable from the failing path too (see the
        # repair build in the `except` below) -- so a job that dies before
        # git_pull would otherwise build images and stamp a record for the
        # PREVIOUS job's commit, which this job never deployed. An unresolved sha
        # must read as unresolved.
        self._current_git_sha = ""
        # OMN-18572: cleared per job for the same reason the sha is -- a
        # verdict carried from the previous job would attach the previous
        # merge's delivery to this merge's terminal event.
        self._onex_api_delivery = None

        def on_phase_update(phase: Phase, status: PhaseStatus) -> None:
            self.job_store.update_phase(cid, phase, status)

        try:
            # OMN-18572. THE LANE'S OWN LOCK, HELD FOR THE WHOLE JOB.
            #
            # OMN-18124 put this lock around `git_pull` alone, because the
            # deploy-source clone was the shared thing it was reasoning about.
            # Every phase after it -- the build, the compose up, the verify,
            # the lab-overlay apply -- then ran with the lane UNLOCKED, so a
            # concurrent `scripts/deploy-runtime.sh` took the same lock
            # uncontended and recreated the project underneath a job that was
            # mid-flight. Measured 2026-09-17: a hand deploy entered at 11:20Z
            # against job 746a118a, accepted 11:07:45Z and still in its runtime
            # phase; the lane answered 000 on all three published ports for
            # about six minutes with containers stranded in `Created`.
            #
            # `single_flight_lock` above does NOT cover this. It serializes this
            # agent against itself and says nothing about any other writer on
            # the host, which is why nothing refused.
            #
            # Re-entrant by construction: `git_pull` takes the same project and
            # the client's ONEX_LANE_LOCK_HELD token makes that a no-op, the
            # same rule `lane_lock.sh` applies to nested shell callers.
            #
            # The lock wraps the phases and NOT the terminal publish below. A
            # publish is a bus write, not a lane mutation, and holding a lane
            # lock across a retrying publish would serialize the next merge
            # behind a broker problem that has nothing to do with the lane.
            with lane_lock(
                lane_config_for(cmd.runtime_lane).compose_project,
                lane=cmd.runtime_lane.value,
                ref=cmd.git_ref,
                timeout=DEFAULT_LANE_LOCK_TIMEOUT_SECONDS,
            ):
                # OMN-15181: boundary-level guard — a prod request may only
                # deploy a digest already proven in stability-test. Must run
                # before ANY deploy effect (preflight included), not just before
                # health checks. Previously implemented
                # (assert_prod_request_has_stability_digest) but never called
                # from the live consume/execute path — dead code, unit-tested in
                # isolation only.
                #
                # Round 4 (Finding 11): the guard is resolved PER-SERVICE, not
                # per-lane — a runtime-effects request must be compared against
                # the effects stability container, not the runtime one
                # (resolve_prod_target_service / resolve_stability_ready_digest).
                # Previously every prod request was compared against the RUNTIME
                # stability digest unconditionally, wrongly rejecting a
                # runtime-effects command carrying its own stability-proven
                # digest.
                if cmd.runtime_lane == EnumRuntimeLane.PROD:
                    target_service = resolve_prod_target_service(cmd)
                    stability_digest = self.executor.resolve_stability_ready_digest(
                        target_service
                    )
                    assert_prod_request_has_stability_digest(
                        cmd, stability_ready_digest=stability_digest
                    )

                # Preflight
                self.executor.preflight(on_phase_update=on_phase_update)

                # Git pull -- OMN-18124: under the lane's host lock, the same
                # per-compose-project lock refresh_dev_lane.sh takes. The
                # deploy-source clone is shared with that script, and this agent
                # took nothing until now.
                self._current_git_sha = self.executor.git_pull(
                    cmd.git_ref,
                    lane=cmd.runtime_lane,
                    on_phase_update=on_phase_update,
                )

                # Regenerate compose from catalog (non-fatal — logs warning on failure)
                self.executor.compose_gen(
                    SCOPE_BUNDLES.get(cmd.scope, ["core", "runtime"]),
                    on_phase_update=on_phase_update,
                    lane=cmd.runtime_lane,
                )

                # Seed Infisical before containers start (non-fatal). OMN-19522:
                # the executor records it SKIPPED on an instance with none.
                self.executor.seed_infisical(
                    on_phase_update=on_phase_update, lane=cmd.runtime_lane
                )

                # Runtime/full deploys must not start with stale endpoint env values.
                if cmd.scope in (Scope.RUNTIME, Scope.FULL):
                    self.executor.validate_llm_endpoint_env_contract()

                # Rebuild — pass git_sha so _compose_build can bust the COPY src/ layer
                # cache. prod pulls the pinned digest instead of rebuilding from a ref.
                services_restarted = self.executor.rebuild_scope(
                    cmd.scope,
                    cmd.services,
                    on_phase_update=on_phase_update,
                    git_sha=self._current_git_sha,
                    # OMN-16442/OMN-17291: the command's own pin, carried through to
                    # stage_workspace.sh as DEPLOY_REF for workspace-mode builds.
                    git_ref=cmd.git_ref,
                    build_source=cmd.build_source,
                    lane=cmd.runtime_lane,
                    image_digest=cmd.image_digest,
                )

                # prod must serve exactly the pinned digest: verify the running
                # container image digest equals the requested digest BEFORE any
                # health check, failing closed on mismatch.
                if cmd.runtime_lane == EnumRuntimeLane.PROD and cmd.image_digest:
                    health_checks = self.executor.deploy_and_verify(
                        lane=cmd.runtime_lane,
                        expected_digest=cmd.image_digest,
                        on_phase_update=on_phase_update,
                        service=resolve_prod_target_service(cmd),
                    )
                else:
                    health_checks = self.executor.verify(
                        on_phase_update=on_phase_update, lane=cmd.runtime_lane
                    )

                # Complete -- AND SAY WHAT IS STILL RUNNING (OMN-18636 AC5).
                #
                # The verdict below is final and is deliberately not affected by
                # anything that follows it. What follows it is nevertheless this
                # job's work, on this job's thread: the lab-overlay apply, the
                # onex-api pin delivery, the terminal publish. On 2026-09-17 that
                # tail ran for 3m38s after a record that read `success`, and
                # nothing the agent served could tell the two apart.
                #
                # The stage travels IN this write. A `complete` followed by a
                # separate `set_settling` would reopen the window by exactly the
                # gap between two file writes.
                self.job_store.complete(
                    cid,
                    status="success",
                    settling_stage=EnumJobSettlingStage.LAB_OVERLAY,
                    verify_recreate=self.executor.verify_recreate,
                )
                logger.info(
                    "Job %s completed successfully; settling (lab overlay)", cid
                )

                # OMN-18200 AC5 -- the k3s onex-lab overlay's half of rule 24(a).
                #
                # AFTER the compose lane is verified and the job is marked complete,
                # and deliberately NOT part of its verdict. The compose lane
                # converged on its own merits by this point; a lab-overlay failure
                # must not report a lane that IS running the merged sha as broken.
                # The lab verdict travels in its own sha-keyed receipt, on its own
                # lane value, emitted by the workflow job that reads the record this
                # writes.
                #
                # Only the dev lane. A merge to `main` targets stability-test, which
                # is a governed lane this agent's fence already refuses, and the lab
                # overlay is not a stability surface.
                # OMN-18572: the apply is also where the DELIVERABLE LINEAGE
                # comes from. The four lab image tags carry the overlay's own
                # omninode_infra commit, never the merged omnibase_infra sha
                # this job is keyed by, so the delivery below needs what the
                # apply resolved rather than what the job is named after.
                manifest_sha = self._apply_lab_overlay(cmd)

                self.job_store.set_settling(cid, EnumJobSettlingStage.ONEX_API_PIN)

                # OMN-18572. The applier above BUILT a fresh onex-api image;
                # this is what DELIVERS it. Inside the lock, because it
                # recreates a container on this lane, and inside the `try`
                # rather than after it, because a delivery attempted on a job
                # that has already failed would pin an image the lane was never
                # proven able to run -- the same argument OMN-18545 made for the
                # repair build taking a narrower path than the apply.
                self._deliver_onex_api_pin(cmd, manifest_sha=manifest_sha)

        except LaneLockContendedError as e:
            # Named separately from a build failure because the two lead to
            # different actions: this one is retried later by whoever holds the
            # lane, and NOTHING on this lane was touched -- the contended
            # acquire happens before the first phase runs.
            logger.error(  # noqa: TRY400
                "Job %s did not start: %s friction_type=lane_lock_contended",
                cid,
                e,
            )
            self.job_store.complete(cid, status="failed", errors=[str(e)])
        except Exception as e:
            logger.exception("Job %s failed: %s", cid, e)
            # OMN-18636 AC5: the failing path has a tail of its own -- the
            # terminal publish always, and the repair build below on the one
            # failure a fresh image can fix -- so its terminal write names a
            # settling stage for the same reason the success path does.
            self.job_store.complete(
                cid,
                status="failed",
                errors=[str(e)],
                settling_stage=EnumJobSettlingStage.PUBLISH,
                verify_recreate=self.executor.verify_recreate,
            )
            # OMN-18545 -- THE REPAIR BUILD, AND WHY THE FAILING PATH NEEDED ONE
            # AT ALL.
            #
            # Until now the ONLY call to the lab overlay was the one above, the
            # last statement of the `try`. That closed a loop the agent could
            # not open. The compose dev lane runs whatever
            # ONEX_CLOUD_MIGRATE_IMAGE names; the dev-lane migration preflight
            # (executor._ensure_runtime_migrations_ready, reached from
            # _compose_up for the RUNTIME phase, i.e. from inside rebuild_scope)
            # requires the cloud-migration one-shots to exit 0 USING THAT IMAGE;
            # and lab_overlay is the only thing in this repository that builds a
            # replacement CLOUD-migrate image from the archived omninode_infra
            # overlay tree. (build-and-push-migrate-image.yml builds the same
            # Dockerfile against THIS repo's tree and pushes the INFRA migrate
            # image to ECR -- a different image, which does not satisfy this
            # pin.) So while the
            # pinned image was broken the preflight raised, control jumped HERE,
            # and the build never ran -- the agent could not produce the image
            # that would let the preflight pass. Measured three times on
            # 2026-09-16; job 6d8316f0 reached terminal `failed` at 20:44:18Z
            # with `verification: skipped`, and the newest
            # onex-lab/omninode-cloud-migrate tag on the host stayed the 17:33Z
            # pre-merge one throughout, which is the mechanical proof the
            # applier never ran.
            #
            # This is a REPAIR BUILD, not the apply above. It builds one image
            # and touches nothing else -- no runtime promotion, no lane apply.
            # Running the full apply here would promote
            # omnibase-infra-omninode-runtime:latest on a premise a failed job
            # can falsify, rolling the persistent k3s lane to a tag NAMING the
            # merged sha while it ran the previous commit's binary, and report
            # PASS. build_repair_migrate_image's docstring carries the full
            # reasoning and the cost argument.
            #
            # The honest limit, stated rather than implied: this makes a
            # replacement image EXIST on the host. It does not DELIVER it --
            # ONEX_CLOUD_MIGRATE_IMAGE is operator-held and nothing in this
            # repository writes it. That half is deliberately out of scope.
            #
            # TARGETED, not unconditional. Only a dev-lane migration preflight
            # failure triggers it, because that is the one deploy failure a fresh
            # cloud-migrate image can actually fix. A gateway refusal, an
            # out-of-memory build or an unset compose variable would otherwise
            # each spend up to sixteen minutes rebuilding an unrelated image
            # under this agent's single-flight lock -- which rejects every
            # concurrent rebuild command outright -- on the path that is by
            # construction the busy one while the lane is broken.
            #
            # The isolation from the job's verdict is structural: the verdict is
            # already written on the line above, and this method swallows. Both
            # properties are pinned by tests, not by this comment --
            # tests/unit/test_lab_overlay_build_order_omn18545.py.
            if isinstance(e, DevLaneMigrationPreflightError):
                self.job_store.set_settling(cid, EnumJobSettlingStage.REPAIR_BUILD)
                self._build_lab_repair_image(cmd)

        # Publish result (don't use on_phase_update — job is already completed,
        # and update_phase would revert status to in_progress)
        self.job_store.set_settling(cid, EnumJobSettlingStage.PUBLISH)
        job = self.job_store.load(cid)
        if job:
            job.phase_results[Phase.PUBLISH] = PhaseStatus.IN_PROGRESS
            job.current_phase = Phase.PUBLISH
            self.job_store._save(job)
            payload = build_completion_payload(
                job,
                self._current_git_sha,
                # OMN-18640 AC8: the local above is the target of the
                # assignment that raises when verification refuses, so it is
                # empty on precisely the job whose probe readings matter. The
                # executor records them before it raises.
                health_checks or self.executor.health_checks,
                services_restarted=services_restarted,
                container_residue=self.executor.container_residue,
                # OMN-18692: what the deps-phase ceiling did -- the deferral it
                # took before touching the lane, or the wait it held rather
                # than cancelling a live recreate.
                recreate_supervision=self.executor.recreate_supervision,
                deps_convergence=self.executor.deps_convergence,
                compose_invocations=self.executor.compose_invocations,
                # OMN-18640 AC7: the runtime containers verification found dead
                # and recreated, and whether that repaired them.
                verify_recreate=self.executor.verify_recreate,
                # OMN-17135: which sibling commits this build actually vendored.
                # The command's git_ref pins omnibase_infra alone.
                sibling_refs=self.executor.sibling_source_refs,
                # OMN-18572: what the onex-api pin delivery did on this job's
                # tail. It never affects `status` -- the compose verdict is
                # already settled above -- but until it rode this event a
                # reader had no way to tell a lane running the merged image
                # from one running a two-day-old pin.
                onex_api_delivery=self._onex_api_delivery,
            )
            if publish_result(payload, self._kafka_config):
                job.phase_results[Phase.PUBLISH] = PhaseStatus.SUCCESS
                self.job_store._save(job)
                self._publish_cb.record_success(str(cid))
            else:
                job.phase_results[Phase.PUBLISH] = PhaseStatus.FAILED
                job.result_publish_pending = True
                self.job_store._save(job)
                logger.warning("Publish failed for %s, marked pending", cid)

        # OMN-18636 AC5. Nothing further runs for this job, so it stops
        # reporting itself as settling. A flag that is only ever set makes every
        # finished job look like a working one, which distinguishes nothing.
        # This is cleared whether the publish succeeded or not: a publish still
        # owed to the bus is durable in `result_publish_pending` and is replayed
        # by the retry loop, which is a different fact from "this job's own work
        # is still executing on the job thread".
        self.job_store.clear_settling(cid)

        self._state = "idle"

    def _apply_lab_overlay(self, cmd: ModelRebuildRequested) -> str | None:
        """Re-apply the k3s onex-lab overlay for this merge (OMN-18200 AC5).

        Called on the SUCCESS path only. The failing path takes the narrower
        ``_build_lab_repair_image`` instead (OMN-18545), because the apply
        promotes the runtime pin on a premise a failed job can falsify.

        Swallows every exception by design. The record the applier writes is the
        channel this result travels on; a raise here would convert a lab finding
        into a failed compose deploy, which is the opposite of what the two
        separate lane values exist to keep apart. An exception that escapes the
        applier is itself logged and then dropped, because the applier's own
        contract is that it writes a record on both outcomes -- so an escape is a
        defect in the applier, reported as one, not a reason to lose the deploy.

        Swallowing also protects the TERMINAL PUBLISH. This runs inside the
        deploy job's ``try``/``except`` and the publish block sits after it, so
        an exception escaping here would skip the publish entirely: the job would
        be durably ``failed`` on disk with nothing on the bus and
        ``result_publish_pending`` never set, so the retry loop would not replay
        it either.
        """
        sha = self._resolve_lab_overlay_sha(cmd, action="re-apply")
        if sha is None:
            return None
        applier: LabOverlayApplier | None = None
        try:
            applier = self._lab_overlay_applier()
            path = applier.apply(
                sha=sha,
                stamp=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                correlation_id=str(cmd.correlation_id),
            )
            logger.info("lab overlay re-apply recorded at %s", path)
        except Exception:
            logger.exception(
                "lab overlay re-apply raised instead of recording for %s; the "
                "onex-lab-k3s receipt for this sha will report a missing record",
                sha,
            )
        # Read AFTER the try/except, deliberately: an apply that raised partway
        # may still have resolved its source, and the image built from that
        # lineage is on the host either way. Returning it is what lets the
        # delivery say "refused for THIS lineage" instead of nothing.
        #
        # In a try of its own for the same reason the apply is: this method
        # runs inside the deploy job's own `try`, with the terminal publish
        # after it, so ANY escape from here costs a published result. An
        # applier that cannot report the commit it resolved is a defect in the
        # applier, reported as one -- never a reason to lose the deploy, and
        # never a reason to deliver an image of unknown lineage.
        if applier is None:
            return None
        try:
            return applier.manifest_sha
        except Exception:
            logger.exception(
                "the lab-overlay applier did not report the omninode_infra "
                "commit it resolved for %s; the onex-api pin delivery will "
                "record a named non-attempt rather than pin an image whose "
                "lineage this job cannot name",
                sha,
            )
            return None

    def _deliver_onex_api_pin(
        self, cmd: ModelRebuildRequested, *, manifest_sha: str | None
    ) -> ModelOnexApiDelivery | None:
        """Advance ``ONEX_API_IMAGE`` to the image the apply just built (OMN-18572).

        The applier makes a correct onex-api image EXIST on this host. This is
        the step that makes the lane RUN it. Without it the two facts diverge
        silently and stay diverged until somebody delivers by hand -- which is
        what a fix merged at 09:00:53Z on 2026-09-17 waited until 11:20Z for,
        with tenant creation on the lab impossible throughout.

        ``manifest_sha`` IS THE OMNINODE_INFRA COMMIT, AND THAT IS THE WHOLE
        CORRECTION HERE. ``_resolve_lab_overlay_sha`` answers a different
        question -- may this job touch the lab at all -- and its answer is the
        merged **omnibase_infra** sha the record is keyed by. The lab image tags
        are ``<omninode_infra sha8>-<stamp>``, so passing the fence's sha as
        ``--sha`` searched for a lineage no image has ever carried. Every
        delivery between the merge on 2026-09-17 and 2026-09-19 refused on that,
        thirty consecutive canary runs stayed red behind it, and the refusal
        text read "none of them for omninode_infra sha <an omnibase_infra sha>"
        -- which describes an applier that produced nothing, not a caller that
        asked for the wrong repository. The fence is still consulted, for the
        question it actually answers; the lineage now comes from the apply.

        A MISSING LINEAGE IS A NAMED NON-DELIVERY, NEVER A FALLBACK. The repoint
        script will happily take the newest resident image of any lineage when
        ``--sha`` is omitted, and reaching for that here would deliver some
        other merge's image and record it as this one's. ``NOT_ATTEMPTED`` says
        what happened instead.

        STILL DOES NOT RAISE, AND NO LONGER STAYS QUIET. The delivery runs after
        the compose lane's verdict is written; converting its failure into a
        failed deploy would report a lane that converged as broken, and an
        escape here would skip the terminal publish that sits after this
        method's caller. Both of those remain true. What changes is that every
        outcome is now a typed record on the job and on the terminal event, and
        a failing one is logged at ``ERROR``: the refusal above was emitted at
        ``INFO`` thirty times and read exactly like a successful no-op.
        """
        if not lane_runs_phase(cmd.runtime_lane, EnumInstancePhase.ONEX_API_PIN):
            # OMN-19522: no onex-api runs on this instance (its overlay
            # disables the service), so there is no pin to deliver.
            logger.info(
                "onex-api delivery skipped: instance %s runs no onex-api",
                active_dev_instance(),
            )
            return None
        fence_sha = self._resolve_lab_overlay_sha(cmd, action="onex-api delivery")
        if fence_sha is None:
            return None

        if manifest_sha is None:
            return self._record_onex_api_delivery(
                cmd,
                ModelOnexApiDelivery(
                    result=EnumOnexApiDeliveryResult.NOT_ATTEMPTED,
                    raw_result=EnumOnexApiDeliveryResult.NOT_ATTEMPTED.value,
                    reason=(
                        "the lab-overlay apply for omnibase_infra "
                        f"{fence_sha} resolved no omninode_infra overlay "
                        "commit, so no image was built from any lineage this "
                        "job can name. Delivering the newest resident image "
                        "regardless of lineage would pin another merge's "
                        "build; nothing was attempted."
                    ),
                    requested_sha=None,
                ),
            )

        try:
            record = self.executor.deliver_onex_api_pin(
                sha=manifest_sha,
                omninode_clone=LAB_OVERLAY_SOURCE_DIR,
                lane=cmd.runtime_lane,
            )
        except Exception as exc:
            logger.exception(
                "onex-api delivery raised instead of returning a verdict for "
                "omninode_infra %s; the lane may still be running the previous "
                "pin",
                manifest_sha,
            )
            return self._record_onex_api_delivery(
                cmd,
                ModelOnexApiDelivery(
                    result=EnumOnexApiDeliveryResult.RAISED,
                    raw_result=EnumOnexApiDeliveryResult.RAISED.value,
                    reason=f"{type(exc).__name__}: {exc}",
                    requested_sha=manifest_sha,
                ),
            )

        return self._record_onex_api_delivery(
            cmd,
            ModelOnexApiDelivery.from_record(record, requested_sha=manifest_sha),
        )

    def _record_onex_api_delivery(
        self, cmd: ModelRebuildRequested, delivery: ModelOnexApiDelivery
    ) -> ModelOnexApiDelivery:
        """Log the verdict at a level that matches it, and make it durable.

        Held on the agent as well as written to the job record because the
        terminal payload is built after this runs and reads it from here; the
        job record is what an operator reads on the host afterwards.

        The log level is the point. A delivery that did not deliver is an
        ``ERROR`` even though it does not fail the job, because the whole defect
        this closes was legible only as thirty identical ``INFO`` lines.
        """
        self._onex_api_delivery = delivery
        log = logger.error if delivery.is_failure else logger.info
        log(
            "onex-api delivery for omninode_infra %s: result=%s "
            "tag_advanced=%s recreated=%s pin=%s -> %s reason=%s",
            delivery.requested_sha,
            delivery.raw_result,
            delivery.tag_advanced,
            delivery.recreated,
            delivery.pin_before,
            delivery.pin_after,
            delivery.reason,
        )
        self.job_store.record_onex_api_delivery(cmd.correlation_id, delivery)
        return delivery

    def _build_lab_repair_image(self, cmd: ModelRebuildRequested) -> None:
        """Build the cloud-migrate repair image after a FAILED deploy (OMN-18545).

        The narrow half of the overlay path: one image build, no runtime
        promotion, no lane apply. See ``LabOverlayApplier.build_repair_migrate_image``
        for why the failing path must not take the full apply, and the call site
        for the loop this opens.

        Swallows for the same two reasons ``_apply_lab_overlay`` does, and the
        second one binds harder here: this runs inside the deploy job's
        ``except`` block, so an escape would skip the terminal publish of a job
        that has ALREADY been recorded as failed.
        """
        sha = self._resolve_lab_overlay_sha(cmd, action="repair build")
        if sha is None:
            return
        try:
            applier = self._lab_overlay_applier()
            path = applier.build_repair_migrate_image(
                sha=sha,
                stamp=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                correlation_id=str(cmd.correlation_id),
            )
            logger.info("lab overlay repair build recorded at %s", path)
        except Exception:
            logger.exception(
                "lab overlay repair build raised instead of recording for %s; no "
                "replacement cloud-migrate image was produced for this sha",
                sha,
            )

    def _resolve_lab_overlay_sha(
        self, cmd: ModelRebuildRequested, *, action: str
    ) -> str | None:
        """The fence both lab-overlay paths pass through, or ``None`` to skip.

        Dev lane only. A merge to ``main`` targets stability-test, which is a
        governed lane this agent's fence already refuses, and the lab overlay is
        not a stability surface.

        The sha is per-job -- cleared at the top of ``_run_deploy`` -- and that
        clearing is load-bearing now that the failing path reaches here at all: a
        job that died before ``git_pull`` would otherwise build images for the
        PREVIOUS job's commit and stamp a record naming it.
        """
        if cmd.runtime_lane != EnumRuntimeLane.DEV:
            return None
        if not lane_runs_phase(cmd.runtime_lane, EnumInstancePhase.LAB_OVERLAY):
            # OMN-19522: the k3s onex-lab overlay lives on the .201 host only.
            logger.info(
                "lab overlay %s skipped: instance %s has no k3s onex-lab overlay",
                action,
                active_dev_instance(),
            )
            return None
        if not LAB_OVERLAY_ENABLED:
            logger.info(
                "lab overlay %s DISABLED by DEPLOY_AGENT_LAB_OVERLAY=off; "
                "no onex-lab-k3s record will exist for %s",
                action,
                self._current_git_sha,
            )
            return None
        sha = self._current_git_sha
        if not sha or len(sha) != 40:
            logger.warning(
                "lab overlay %s skipped: the resolved deploy sha is %r, and a "
                "record must be keyed by an exact 40-character sha",
                action,
                sha,
            )
            return None
        return sha

    def _lab_overlay_applier(self) -> LabOverlayApplier:
        return LabOverlayApplier(
            state_dir=STATE_DIR,
            repo_dir=Path(REPO_DIR),
            overlay_source_dir=LAB_OVERLAY_SOURCE_DIR,
            env=os.environ,
            budget_seconds=LAB_OVERLAY_BUDGET_SECONDS,
        )

    def _publish_rejected(
        self, cmd: ModelRebuildRequested, *, reason: EnumRejectionReason
    ) -> None:
        self._publish_rejection_event(
            ModelRebuildRejected(
                correlation_id=cmd.correlation_id,
                reason=reason,
                scope=cmd.scope,
            )
        )

    @staticmethod
    def publish_rejection_notice(
        notice: ModelRejectionNotice,
        *,
        publish: Callable[[ModelRebuildRejected], bool],
    ) -> bool:
        """Route one consumer-resolved refusal to the single publish helper (OMN-17079).

        THE DEFECT THIS CLOSES. ``EnumRejectionReason`` has eight members and exactly
        two of them ever reached the topic: ``IN_PROGRESS`` from this module and
        ``SUPERSEDED`` from OMN-18143 AC7. The other six are decided inside
        ``consumer._process_message``, which committed the offset, logged a line and
        returned a bare reason string that the agent loop only logged again. LD-11,
        filed 2026-08-30, still true twenty days later.

        It matters now because the topic is about to have a reader. A widget wired to a
        topic carrying two of eight refusal reasons looks quiet while six kinds of
        refusal happen, and a quiet errors widget is read as "nothing was refused" --
        the exact invisibility the observability work exists to remove.

        WHY THIS IS A STATICMETHOD TAKING ITS PUBLISHER. The decision it makes -- publish
        or withhold -- is the part worth testing, and it is pure. Threading the publisher
        in keeps that decision exercisable without a broker, a job store or an agent
        instance, and keeps the one publish helper the only thing that talks to Kafka.

        WHY AN UNATTRIBUTABLE NOTICE PUBLISHES NOTHING. Three of the six reasons are
        decided before ``ModelRebuildRequested`` validation, so the correlation id and
        scope may be unresolvable. A rejection published with a fabricated id is worse
        than no rejection: it is a durable record pointing at a command that never
        existed, and a reader cannot tell it from a real one. The quarantine record
        already written on those paths stays the durable evidence.

        Returns whether an event was published. ``SUPERSEDED`` is REFUSED here rather
        than silently mishandled: it carries two further required fields and keeps its
        own builder, and routing it through this path would drop them.
        """
        if notice.reason is EnumRejectionReason.SUPERSEDED:
            msg = (
                "a superseded rejection names the commit and job that replaced it and "
                "must be built by _publish_superseded; routing it through the notice "
                "path would drop superseded_by_sha and superseded_by_correlation_id"
            )
            raise ValueError(msg)

        if notice.correlation_id is None or notice.scope is None:
            logger.warning(
                "Rejection %s is unattributable and will NOT be published "
                "(correlation_id=%s scope=%s); the quarantine record is the durable "
                "evidence for this refusal",
                notice.reason.value,
                notice.correlation_id,
                notice.scope,
            )
            return False

        return publish(
            ModelRebuildRejected(
                correlation_id=notice.correlation_id,
                reason=notice.reason,
                scope=notice.scope,
            )
        )

    def _publish_rejection_notice(self, notice: ModelRejectionNotice) -> None:
        """Instance seam the consumer's ``on_rejected`` hook is wired to.

        A supersession by the running build (OMN-19270) also has a job record
        owing this event, so a publish that landed clears that debt here, and
        the retry loop pays it otherwise.
        """
        published = self.publish_rejection_notice(
            notice, publish=self._publish_rejection_event
        )
        if (
            published
            and notice.reason is EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD
            and notice.correlation_id is not None
        ):
            self.job_store.mark_published(notice.correlation_id)

    def _publish_superseded(self, supersession: ModelSupersession) -> None:
        """AC6's terminal event: this command will not run, and here is what did.

        On the SAME topic as every other "will not run" outcome, with its own
        reason token and the two fields that make it actionable. A new topic
        was weighed and refused: it would need a consumer nobody has, while
        the rejection topic already carries the class of outcome this belongs
        to -- and ``ModelRebuildRejected`` is what keeps the three cases AC6
        names apart, since a timeout and a rollback cannot carry
        ``reason=superseded`` and a supersession cannot omit the sha.
        """
        cmd = supersession.superseded.command
        published = self._publish_rejection_event(
            ModelRebuildRejected(
                correlation_id=cmd.correlation_id,
                reason=EnumRejectionReason.SUPERSEDED,
                scope=cmd.scope,
                superseded_by_sha=supersession.superseded_by_sha,
                superseded_by_correlation_id=supersession.superseded_by_correlation_id,
            )
        )
        if published:
            self.job_store.mark_published(cmd.correlation_id)

    def _publish_superseded_for_job(self, job: JobState) -> bool:
        """Re-publish a superseded record's terminal event from the record alone.

        The retry loop must be able to pay this debt without the in-memory
        ``ModelSupersession`` the scan built, because the process that built
        it may be gone -- the job store outlives it, which is the reason the
        record carries both fields rather than only the log line naming them.
        """
        if job.superseded_by_running_build:
            # OMN-19270. The running build replaced it, and a running build is
            # not a command: the event names no replacement.
            return self._publish_rejection_event(
                ModelRebuildRejected(
                    correlation_id=job.correlation_id,
                    reason=EnumRejectionReason.SUPERSEDED_BY_RUNNING_BUILD,
                    scope=Scope(job.command["scope"]),
                )
            )
        if job.superseded_by_sha is None or job.superseded_by_correlation_id is None:
            # Refused by JobState's own validator, so this is unreachable
            # through any write path; it is here because an unreachable branch
            # that silently publishes a malformed event is worse than one that
            # says the record is unusable.
            logger.error(
                "Superseded job %s names no replacement, so no terminal event "
                "can be built for it",
                job.correlation_id,
            )
            return False
        return self._publish_rejection_event(
            ModelRebuildRejected(
                correlation_id=job.correlation_id,
                reason=EnumRejectionReason.SUPERSEDED,
                scope=Scope(job.command["scope"]),
                superseded_by_sha=job.superseded_by_sha,
                superseded_by_correlation_id=job.superseded_by_correlation_id,
            )
        )

    def _publish_rejection_event(self, event: ModelRebuildRejected) -> bool:
        """Publish one rejection. Returns whether the broker took it.

        The return value is new and the swallow is not: every existing caller
        ignores it and behaves exactly as before, while the supersession path
        needs to know, because a superseded record's publish debt is only
        cleared when the event actually landed.
        """
        from kafka import KafkaProducer

        payload = json.dumps(event.to_wire()).encode()
        try:
            producer = KafkaProducer(
                **self._kafka_config.producer_kwargs(),
                value_serializer=lambda v: v,
                # OMN-18143. Bound the metadata wait. `send()` blocks up to
                # kafka-python's `max_block_ms` (default 60_000) resolving the
                # topic, so a publish to a topic the broker does not have
                # occupies this process's SINGLE job thread for a full minute
                # before raising -- and the retry loop then does it again every
                # 30 s until the circuit breaker trips at ten consecutive
                # failures. That is ten minutes of job-thread time per record,
                # spent on a send that cannot land.
                #
                # Measured on the .201 dev lane 2026-09-19: of 1714 topics,
                # `onex.evt.deploy.rebuild-requested.v1` and
                # `...rebuild-completed.v1` are present and
                # `...rebuild-rejected.v1` is ABSENT, so every rejection this
                # agent publishes takes exactly that path. Two attempts for
                # correlation 63858212 each took 60 s, at 11:50:49Z and
                # 11:51:49Z, while a completion publish to the present topic
                # succeeded at 11:49:34Z on the same config.
                #
                # This bounds the COST, and deliberately does not pretend to
                # fix the cause: the topic's absence, and the fact that nothing
                # in any repository consumes it, are recorded on OMN-18143 and
                # are not the agent's to resolve. Ten seconds is far above the
                # sub-second publish this lane achieves when the topic exists,
                # so a healthy publish is unaffected.
                max_block_ms=REJECTION_PUBLISH_MAX_BLOCK_MS,
            )
            producer.send(TOPIC_REBUILD_REJECTED, payload)
            producer.flush(timeout=5)
            producer.close()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Failed to publish rebuild-rejected (reason=%s) for %s",
                event.reason.value,
                event.correlation_id,
            )
            return False
        return True

    def _retry_pending_publishes(self) -> None:
        """Replay any result still owed to the bus. BLOCKING — see ``_offload``."""
        pending = self.job_store.get_pending_publish()
        for job in pending:
            cid_str = str(job.correlation_id)
            if self._publish_cb.is_tripped(cid_str):
                logger.critical(
                    "Publish circuit breaker tripped for job %s — "
                    "dropping from pending queue after repeated failures. "
                    "Manual investigation required. "
                    "friction_type=publish_circuit_breaker_tripped",
                    cid_str,
                )
                self.job_store.mark_published(job.correlation_id)
                self._publish_cb.clear(cid_str)
                continue

            # OMN-18143. A superseded job owes a REJECTION event, not a
            # completion one. Routing it through the completion builder below
            # would publish a ModelRebuildCompleted whose phases are all
            # SKIPPED onto the completed topic -- an event asserting that a
            # rebuild finished for a command that never started, on the topic
            # the redeploy effect waits on.
            if job.status == "superseded":
                if self._publish_superseded_for_job(job):
                    self.job_store.mark_published(job.correlation_id)
                    self._publish_cb.record_success(cid_str)
                    logger.info("Retried superseded publish for %s: success", cid_str)
                else:
                    self._publish_cb.record_failure(cid_str)
                    logger.warning(
                        "Retried superseded publish for %s: still failing", cid_str
                    )
                continue

            payload = build_completion_payload(job, "")
            if publish_result(payload, self._kafka_config):
                self.job_store.mark_published(job.correlation_id)
                self._publish_cb.record_success(cid_str)
                logger.info("Retried publish for %s: success", cid_str)
            else:
                self._publish_cb.record_failure(cid_str)
                logger.warning("Retried publish for %s: still failing", cid_str)

    async def _publish_retry_loop(self) -> None:
        while True:
            await asyncio.sleep(PUBLISH_RETRY_INTERVAL)
            # OMN-18636: onto the job thread, not the loop. A kafka publish
            # blocks for its own timeouts, and this task fires every 30 s for
            # the life of the process, so leaving it here would deny the health
            # surface on a cadence even with no job running at all. The single
            # worker means a retry that lands mid-deploy simply waits for the
            # deploy, which is what it did when the loop was blocked.
            await self._offload(self._retry_pending_publishes)
