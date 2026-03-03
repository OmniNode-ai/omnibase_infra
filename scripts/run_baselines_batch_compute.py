#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 OmniNode Team
"""One-shot runner for NodeBaselinesBatchCompute.

Connects to PostgreSQL and Kafka, runs the 3-phase baselines batch computation,
emits a baselines-computed snapshot event, then exits.

Intended to be invoked by a scheduler (cron, GitHub Actions) rather than running
as a long-lived service.

Usage:
    uv run python scripts/run_baselines_batch_compute.py

Environment Variables:
    OMNIBASE_INFRA_DB_URL (required)
        Full PostgreSQL DSN, e.g.:
        postgresql://postgres:pass@localhost:5436/omnibase_infra

    KAFKA_BOOTSTRAP_SERVERS (optional, default: localhost:19092)
        Kafka bootstrap address. Set to empty string to skip event emission
        (DB computation still runs, but no baselines-computed event is emitted).

Exit Codes:
    0  Success
    1  Configuration or runtime error

Ticket: OMN-3335
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from uuid import uuid4

import asyncpg
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from omnibase_infra.nodes.node_baselines_batch_compute.handlers.handler_baselines_batch_compute import (
    HandlerBaselinesBatchCompute,
)
from omnibase_infra.nodes.node_baselines_batch_compute.models.model_baselines_batch_compute_command import (
    ModelBaselinesBatchComputeCommand,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("run_baselines_batch_compute")


def _make_publisher(
    producer: AIOKafkaProducer,
) -> HandlerBaselinesBatchCompute.ProtocolPublisher:  # type: ignore[attr-defined]
    """Wrap AIOKafkaProducer as the ProtocolPublisher callable expected by the handler."""

    async def publish(
        event_type: str,
        payload: object,
        topic: str | None,
        correlation_id: object,
        **kwargs: object,
    ) -> bool:
        if topic is None:
            logger.warning("publish called with topic=None, skipping emission")
            return False
        body = json.dumps(
            {
                "event_type": event_type,
                "payload": payload,
                "correlation_id": str(correlation_id),
            }
        ).encode("utf-8")
        await producer.send_and_wait(topic, body)
        logger.info("Emitted event %s to topic %s", event_type, topic)
        return True

    return publish  # type: ignore[return-value]


async def _run() -> int:
    db_url = os.environ.get("OMNIBASE_INFRA_DB_URL", "").strip()
    if not db_url:
        logger.error("OMNIBASE_INFRA_DB_URL is required but not set")
        return 1

    kafka_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092").strip()
    use_kafka = bool(kafka_servers)

    correlation_id = uuid4()
    logger.info(
        "Starting baselines batch compute run (correlation_id=%s)", correlation_id
    )

    pool: asyncpg.Pool | None = None
    producer: AIOKafkaProducer | None = None

    try:
        pool = await asyncpg.create_pool(
            db_url, min_size=1, max_size=3, command_timeout=60
        )

        publisher = None
        if use_kafka:
            producer = AIOKafkaProducer(
                bootstrap_servers=kafka_servers,
                acks="all",
                enable_idempotence=True,
            )
            try:
                await producer.start()
                publisher = _make_publisher(producer)
                logger.info("Kafka producer connected to %s", kafka_servers)
            except KafkaError:
                logger.warning(
                    "Kafka unavailable (%s) — computation will run but no event emitted",
                    kafka_servers,
                    exc_info=True,
                )
                producer = None

        handler = HandlerBaselinesBatchCompute(pool=pool, publisher=publisher)
        command = ModelBaselinesBatchComputeCommand(correlation_id=correlation_id)
        result = await handler.handle(command)

        logger.info(
            "Baselines computation complete: comparisons=%d trend=%d breakdown=%d snapshot_emitted=%s",
            result.result.comparisons_rows,
            result.result.trend_rows,
            result.result.breakdown_rows,
            result.snapshot_emitted,
        )

        if result.result.has_errors:
            logger.warning("Computation finished with errors: %s", result.result.errors)
            return 1

        return 0

    except Exception:
        logger.exception("Baselines batch compute failed")
        return 1

    finally:
        if producer is not None:
            await producer.stop()
        if pool is not None:
            await pool.close()


def main() -> None:
    sys.exit(asyncio.run(_run()))


if __name__ == "__main__":
    main()
