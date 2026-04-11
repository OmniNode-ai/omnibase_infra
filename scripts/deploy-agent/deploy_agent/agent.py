# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Main orchestrator. Single-job concurrency. Runs consumer + health + publisher concurrently."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from aiohttp import web

from deploy_agent.consumer import DeployConsumer
from deploy_agent.events import (
    ModelRebuildRequested,
    Phase,
    PhaseStatus,
    Scope,
)
from deploy_agent.executor import DeployExecutor
from deploy_agent.health import create_health_app
from deploy_agent.job_state import JobStore
from deploy_agent.publisher import build_completion_payload, publish_result

logger = logging.getLogger(__name__)

STATE_DIR = Path(
    os.environ.get("DEPLOY_AGENT_STATE_DIR", "/data/omninode/deploy-agent/state/jobs")
)
BOOTSTRAP_SERVERS = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
HEALTH_PORT = int(os.environ.get("DEPLOY_AGENT_PORT", "8099"))
PUBLISH_RETRY_INTERVAL = 30


class DeployAgent:
    def __init__(self):
        self.job_store = JobStore(state_dir=STATE_DIR)
        self.executor = DeployExecutor()
        self._state = "idle"
        self._shutdown = False
        self._current_git_sha = ""

    def _get_state(self) -> str:
        return self._state

    async def run(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logger.info(
            "Deploy agent starting (state_dir=%s, kafka=%s)",
            STATE_DIR,
            BOOTSTRAP_SERVERS,
        )

        # Step 1: Recover crashed jobs
        recovered = self.job_store.recover_crashed_jobs()
        if recovered:
            logger.info("Recovered %d crashed job(s)", len(recovered))

        # Step 2: Prune old completed jobs
        pruned = self.job_store.prune_completed()
        if pruned:
            logger.info("Pruned %d old job(s)", pruned)

        # Step 3: Retry pending publishes
        await self._retry_pending_publishes()

        # Step 4: Start health endpoint
        health_app = create_health_app(
            job_store=self.job_store,
            get_agent_state=self._get_state,
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

        # Step 5+6: Main loop
        consumer = DeployConsumer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            job_store=self.job_store,
        )

        # Handle signals
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown)

        publish_retry_task = asyncio.create_task(self._publish_retry_loop())

        try:
            while not self._shutdown:
                cmd, reason = consumer.poll_and_accept()
                if cmd is not None:
                    await self._execute_command(cmd)
                elif reason:
                    logger.info("Rejected command: %s", reason)
                else:
                    await asyncio.sleep(1)
        finally:
            publish_retry_task.cancel()
            consumer.close()
            await runner.cleanup()
            logger.info("Deploy agent stopped")

    def _handle_shutdown(self) -> None:
        logger.info("Shutdown signal received")
        self._shutdown = True

    async def _execute_command(self, cmd: ModelRebuildRequested) -> None:
        self._state = "deploying"
        cid = cmd.correlation_id
        health_checks = []

        def on_phase_update(phase: Phase, status: PhaseStatus) -> None:
            self.job_store.update_phase(cid, phase, status)

        try:
            # Preflight
            self.executor.preflight(on_phase_update=on_phase_update)

            # Git pull
            self._current_git_sha = self.executor.git_pull(
                cmd.git_ref, on_phase_update=on_phase_update
            )

            # Seed Infisical before containers start (non-fatal)
            self.executor.seed_infisical(on_phase_update=on_phase_update)

            # Rebuild — pass git_sha so _compose_build can bust the COPY src/ layer cache
            self.executor.rebuild_scope(
                cmd.scope,
                cmd.services,
                on_phase_update=on_phase_update,
                git_sha=self._current_git_sha,
            )

            # Verify
            health_checks = self.executor.verify(on_phase_update=on_phase_update)

            # Complete
            self.job_store.complete(cid, status="success")
            logger.info("Job %s completed successfully", cid)

        except Exception as e:
            logger.exception("Job %s failed: %s", cid, e)
            self.job_store.complete(cid, status="failed", errors=[str(e)])

        # Publish result (don't use on_phase_update — job is already completed,
        # and update_phase would revert status to in_progress)
        job = self.job_store.load(cid)
        if job:
            job.phase_results[Phase.PUBLISH] = PhaseStatus.IN_PROGRESS
            job.current_phase = Phase.PUBLISH
            self.job_store._save(job)
            payload = build_completion_payload(
                job, self._current_git_sha, health_checks
            )
            if publish_result(payload):
                job.phase_results[Phase.PUBLISH] = PhaseStatus.SUCCESS
                self.job_store._save(job)
            else:
                job.phase_results[Phase.PUBLISH] = PhaseStatus.FAILED
                job.result_publish_pending = True
                self.job_store._save(job)
                logger.warning("Publish failed for %s, marked pending", cid)

        self._state = "idle"

    async def _retry_pending_publishes(self) -> None:
        pending = self.job_store.get_pending_publish()
        for job in pending:
            payload = build_completion_payload(job, "")
            if publish_result(payload):
                self.job_store.mark_published(job.correlation_id)
                logger.info("Retried publish for %s: success", job.correlation_id)
            else:
                logger.warning(
                    "Retried publish for %s: still failing", job.correlation_id
                )

    async def _publish_retry_loop(self) -> None:
        while True:
            await asyncio.sleep(PUBLISH_RETRY_INTERVAL)
            await self._retry_pending_publishes()
