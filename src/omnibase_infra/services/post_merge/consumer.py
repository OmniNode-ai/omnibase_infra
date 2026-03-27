# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Async Kafka consumer for post-merge quality checks.

Subscribes to ``onex.evt.github.pr-merged.v1`` and dispatches a chain of
quality checks for each merged PR:

1. **Hostile review** -- scans the merged diff for security concerns,
   missing error handling, and naming convention violations.
2. **Contract sweep** -- runs ``check_topic_drift.py`` against the merged
   commit to detect contract/topic drift.
3. **Integration check** -- analyses changed files for cross-repo boundary
   violations (topic renames, enum changes, public API surface changes).

Findings above the configured severity threshold are auto-ticketed in Linear.
Results are published to ``onex.evt.github.post-merge-result.v1``.

Architecture::

    Kafka: onex.evt.github.pr-merged.v1
           |
           v
    PostMergeConsumer
           |
           +-- run_hostile_review()
           +-- run_contract_sweep()
           +-- run_integration_check()
           |
           v
    Findings -> Linear tickets (auto-create)
           |
           v
    Kafka: onex.evt.github.post-merge-result.v1

Related Tickets:
    - OMN-6727: post-merge consumer chain
    - OMN-6726: GitHub merge event producer (upstream)
    - OMN-6725: contract_sweep skill wrapper

Example::

    >>> from omnibase_infra.services.post_merge import (
    ...     ConfigPostMergeConsumer,
    ...     PostMergeConsumer,
    ... )
    >>> config = ConfigPostMergeConsumer(
    ...     kafka_bootstrap_servers="localhost:19092",
    ... )
    >>> consumer = PostMergeConsumer(config)
    >>> await consumer.start()
    >>> await consumer.run()

    # Or run as a module:
    # python -m omnibase_infra.services.post_merge.consumer
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
from datetime import UTC, datetime
from uuid import uuid4

from aiohttp import web
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.errors import KafkaError
from pydantic import ValidationError

from omnibase_infra.models.github.model_pr_merged_event import ModelPRMergedEvent
from omnibase_infra.services.post_merge.checks import (
    run_contract_sweep,
    run_hostile_review,
    run_integration_check,
)
from omnibase_infra.services.post_merge.config import ConfigPostMergeConsumer
from omnibase_infra.services.post_merge.enum_check_stage import EnumCheckStage
from omnibase_infra.services.post_merge.enum_finding_severity import (
    EnumFindingSeverity,
)
from omnibase_infra.services.post_merge.model_post_merge_finding import (
    ModelPostMergeFinding,
)
from omnibase_infra.services.post_merge.model_post_merge_result import (
    ModelPostMergeResult,
)
from omnibase_infra.topics.platform_topic_suffixes import (
    SUFFIX_GITHUB_POST_MERGE_RESULT,
)

logger = logging.getLogger(__name__)

RESULT_TOPIC = SUFFIX_GITHUB_POST_MERGE_RESULT

# Severity ordering for filtering
_SEVERITY_ORDER: dict[str, int] = {
    EnumFindingSeverity.CRITICAL: 0,
    EnumFindingSeverity.HIGH: 1,
    EnumFindingSeverity.MEDIUM: 2,
    EnumFindingSeverity.LOW: 3,
    EnumFindingSeverity.INFO: 4,
}


class PostMergeConsumer:
    """Async Kafka consumer for post-merge quality check chain.

    Consumes PR merged events and dispatches hostile review, contract sweep,
    and integration checks. Findings above the configured severity threshold
    are auto-ticketed in Linear.

    Related Tickets:
        - OMN-6727: post-merge consumer chain
    """

    def __init__(self, config: ConfigPostMergeConsumer) -> None:
        self._config = config
        self._consumer: AIOKafkaConsumer | None = None
        self._producer: AIOKafkaProducer | None = None
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._consumer_id = f"post-merge-consumer-{uuid4().hex[:8]}"
        self._last_process_time: datetime | None = None

        # Health check server
        self._health_app: web.Application | None = None
        self._health_runner: web.AppRunner | None = None

        logger.info(
            "PostMergeConsumer initialized",
            extra={
                "consumer_id": self._consumer_id,
                "input_topic": self._config.input_topic,
                "group_id": self._config.kafka_group_id,
                "bootstrap_servers": self._config.kafka_bootstrap_servers,
                "stages_enabled": {
                    "hostile_review": self._config.hostile_review_enabled,
                    "contract_sweep": self._config.contract_sweep_enabled,
                    "integration_check": self._config.integration_check_enabled,
                },
            },
        )

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the Kafka consumer, producer, and health check server."""
        if self._running:
            return

        self._consumer = AIOKafkaConsumer(
            self._config.input_topic,
            bootstrap_servers=self._config.kafka_bootstrap_servers,
            group_id=self._config.kafka_group_id,
            auto_offset_reset=self._config.auto_offset_reset,
            enable_auto_commit=False,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await self._consumer.start()

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._config.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
        )
        await self._producer.start()

        # Start health check server
        await self._start_health_server()

        self._running = True
        logger.info(
            "PostMergeConsumer started",
            extra={"consumer_id": self._consumer_id},
        )

    async def stop(self) -> None:
        """Stop the consumer, producer, and health check server."""
        self._running = False
        self._shutdown_event.set()

        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

        if self._producer:
            await self._producer.stop()
            self._producer = None

        if self._health_runner:
            await self._health_runner.cleanup()
            self._health_runner = None

        logger.info(
            "PostMergeConsumer stopped",
            extra={"consumer_id": self._consumer_id},
        )

    async def run(self) -> None:
        """Main consumer loop. Blocks until shutdown signal received."""
        if not self._consumer:
            raise RuntimeError("Consumer not started. Call start() first.")

        # Register signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._shutdown_event.set)

        logger.info(
            "PostMergeConsumer entering run loop",
            extra={"consumer_id": self._consumer_id},
        )

        try:
            async for msg in self._consumer:
                if self._shutdown_event.is_set():
                    break

                try:
                    await self._process_message(msg.value)
                    await self._consumer.commit()
                except Exception:
                    logger.exception(
                        "Failed to process PR merged event",
                        extra={
                            "consumer_id": self._consumer_id,
                            "topic": msg.topic,
                            "partition": msg.partition,
                            "offset": msg.offset,
                        },
                    )
                    # Commit anyway to avoid reprocessing poison pills
                    await self._consumer.commit()
        except KafkaError:
            logger.exception(
                "Kafka error in consumer loop",
                extra={"consumer_id": self._consumer_id},
            )
        finally:
            await self.stop()

    # =========================================================================
    # Message Processing
    # =========================================================================

    async def _process_message(self, raw: dict[str, object]) -> None:
        """Process a single PR merged event through the check chain."""
        try:
            event = ModelPRMergedEvent.model_validate(raw)
        except ValidationError:
            logger.exception(
                "Invalid PR merged event payload",
                extra={"consumer_id": self._consumer_id, "raw_keys": list(raw.keys())},
            )
            return

        logger.info(
            "Processing PR merged event",
            extra={
                "consumer_id": self._consumer_id,
                "repo": event.repo,
                "pr_number": event.pr_number,
                "merge_sha": event.merge_sha,
                "changed_files_count": len(event.changed_files),
            },
        )

        started_at = datetime.now(tz=UTC)
        all_findings: list[ModelPostMergeFinding] = []
        stages_completed: list[EnumCheckStage] = []
        stages_failed: list[EnumCheckStage] = []

        # Stage 1: Hostile Review
        if self._config.hostile_review_enabled:
            try:
                findings = await run_hostile_review(
                    event, github_token=self._config.github_token
                )
                all_findings.extend(findings)
                stages_completed.append(EnumCheckStage.HOSTILE_REVIEW)
            except Exception:
                logger.exception(
                    "Hostile review stage failed",
                    extra={"repo": event.repo, "pr_number": event.pr_number},
                )
                stages_failed.append(EnumCheckStage.HOSTILE_REVIEW)

        # Stage 2: Contract Sweep
        if self._config.contract_sweep_enabled:
            try:
                findings = await run_contract_sweep(
                    event,
                    contracts_dir=self._config.contract_sweep_contracts_dir,
                )
                all_findings.extend(findings)
                stages_completed.append(EnumCheckStage.CONTRACT_SWEEP)
            except Exception:
                logger.exception(
                    "Contract sweep stage failed",
                    extra={"repo": event.repo, "pr_number": event.pr_number},
                )
                stages_failed.append(EnumCheckStage.CONTRACT_SWEEP)

        # Stage 3: Integration Check
        if self._config.integration_check_enabled:
            try:
                findings = await run_integration_check(event)
                all_findings.extend(findings)
                stages_completed.append(EnumCheckStage.INTEGRATION_CHECK)
            except Exception:
                logger.exception(
                    "Integration check stage failed",
                    extra={"repo": event.repo, "pr_number": event.pr_number},
                )
                stages_failed.append(EnumCheckStage.INTEGRATION_CHECK)

        # Auto-ticket creation for findings above threshold
        tickets_created: list[str] = []
        if all_findings and not self._config.dry_run:
            tickets_created = await self._create_tickets(event, all_findings)

        completed_at = datetime.now(tz=UTC)

        # Build and publish result
        result = ModelPostMergeResult(
            repo=event.repo,
            pr_number=event.pr_number,
            merge_sha=event.merge_sha,
            findings=all_findings,
            stages_completed=stages_completed,
            stages_failed=stages_failed,
            tickets_created=tickets_created,
            started_at=started_at,
            completed_at=completed_at,
        )

        await self._publish_result(result)
        self._last_process_time = completed_at

        logger.info(
            "Post-merge check chain completed",
            extra={
                "consumer_id": self._consumer_id,
                "repo": event.repo,
                "pr_number": event.pr_number,
                "findings_count": len(all_findings),
                "stages_completed": [s.value for s in stages_completed],
                "stages_failed": [s.value for s in stages_failed],
                "tickets_created": tickets_created,
                "duration_seconds": (completed_at - started_at).total_seconds(),
            },
        )

    # =========================================================================
    # Ticket Creation
    # =========================================================================

    async def _create_tickets(
        self,
        event: ModelPRMergedEvent,
        findings: list[ModelPostMergeFinding],
    ) -> list[str]:
        """Create Linear tickets for findings above the severity threshold.

        Returns:
            List of created ticket identifiers.
        """
        if not self._config.linear_api_key or not self._config.linear_team_id:
            logger.warning(
                "Linear ticket creation skipped: missing API key or team ID",
                extra={"consumer_id": self._consumer_id},
            )
            return []

        min_severity_rank = _SEVERITY_ORDER.get(
            self._config.auto_ticket_min_severity, 2
        )
        ticketable = [
            f
            for f in findings
            if _SEVERITY_ORDER.get(f.severity, 4) <= min_severity_rank
        ]

        if not ticketable:
            return []

        tickets: list[str] = []
        for finding in ticketable:
            try:
                ticket_id = await self._create_linear_ticket(event, finding)
                if ticket_id:
                    tickets.append(ticket_id)
            except Exception:
                logger.exception(
                    "Failed to create Linear ticket for finding",
                    extra={
                        "consumer_id": self._consumer_id,
                        "finding_title": finding.title,
                    },
                )

        return tickets

    async def _create_linear_ticket(
        self,
        event: ModelPRMergedEvent,
        finding: ModelPostMergeFinding,
    ) -> str | None:
        """Create a single Linear ticket via the Linear API.

        Returns:
            The created issue identifier (e.g. 'OMN-1234'), or None on failure.
        """
        import httpx

        title = f"[post-merge] {finding.title} (PR #{event.pr_number})"
        body_parts = [
            "**Auto-created by post-merge consumer** ([OMN-6727])",
            "",
            f"**Repository:** {event.repo}",
            f"**PR:** #{event.pr_number} — {event.title}",
            f"**Merge SHA:** `{event.merge_sha}`",
            f"**Stage:** {finding.stage.value}",
            f"**Severity:** {finding.severity.value}",
            "",
            "## Finding",
            "",
            finding.description,
        ]
        if finding.file_path:
            body_parts.append(f"\n**File:** `{finding.file_path}`")
        if finding.line_number:
            body_parts.append(f"**Line:** {finding.line_number}")

        body = "\n".join(body_parts)

        # Linear GraphQL API
        mutation = """
        mutation CreateIssue($title: String!, $description: String!, $teamId: String!) {
            issueCreate(input: {
                title: $title
                description: $description
                teamId: $teamId
            }) {
                success
                issue {
                    identifier
                }
            }
        }
        """

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.linear.app/graphql",
                headers={
                    "Authorization": self._config.linear_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "query": mutation,
                    "variables": {
                        "title": title,
                        "description": body,
                        "teamId": self._config.linear_team_id,
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

        issue_data = data.get("data", {}).get("issueCreate", {})
        if issue_data.get("success"):
            identifier = issue_data.get("issue", {}).get("identifier", "")
            logger.info(
                "Linear ticket created",
                extra={
                    "consumer_id": self._consumer_id,
                    "ticket": identifier,
                    "finding_title": finding.title,
                },
            )
            return identifier

        logger.warning(
            "Linear ticket creation returned success=false",
            extra={"consumer_id": self._consumer_id, "response": data},
        )
        return None

    # =========================================================================
    # Result Publishing
    # =========================================================================

    async def _publish_result(self, result: ModelPostMergeResult) -> None:
        """Publish the aggregated result to the output Kafka topic."""
        if not self._producer:
            logger.warning("Producer not available, skipping result publish")
            return

        try:
            payload = json.loads(result.model_dump_json())
            key = f"{result.repo}/pr/{result.pr_number}".encode()
            await self._producer.send_and_wait(
                RESULT_TOPIC,
                key=key,
                value=payload,
            )
            logger.info(
                "Published post-merge result",
                extra={
                    "consumer_id": self._consumer_id,
                    "topic": RESULT_TOPIC,
                    "repo": result.repo,
                    "pr_number": result.pr_number,
                },
            )
        except Exception:
            logger.exception(
                "Failed to publish post-merge result",
                extra={"consumer_id": self._consumer_id},
            )

    # =========================================================================
    # Health Check
    # =========================================================================

    async def _start_health_server(self) -> None:
        """Start a minimal HTTP health check server."""
        self._health_app = web.Application()
        self._health_app.router.add_get("/health", self._health_handler)
        self._health_runner = web.AppRunner(self._health_app)
        await self._health_runner.setup()
        site = web.TCPSite(
            self._health_runner,
            self._config.health_check_host,
            self._config.health_check_port,
        )
        await site.start()
        logger.info(
            "Health check server started",
            extra={
                "consumer_id": self._consumer_id,
                "host": self._config.health_check_host,
                "port": self._config.health_check_port,
            },
        )

    async def _health_handler(self, _request: web.Request) -> web.Response:
        """HTTP health check handler."""
        status = {
            "status": "healthy" if self._running else "stopped",
            "consumer_id": self._consumer_id,
            "last_process_time": (
                self._last_process_time.isoformat() if self._last_process_time else None
            ),
        }
        return web.json_response(status)


# =============================================================================
# CLI Entry Point
# =============================================================================


async def _main() -> None:
    """Run the post-merge consumer as a standalone process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = ConfigPostMergeConsumer()
    consumer = PostMergeConsumer(config)

    await consumer.start()
    try:
        await consumer.run()
    finally:
        await consumer.stop()


if __name__ == "__main__":
    asyncio.run(_main())
