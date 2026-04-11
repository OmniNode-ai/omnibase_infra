# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka consumer with acceptance protocol."""

from __future__ import annotations

import json
import logging
from typing import Any

from kafka import KafkaConsumer

from deploy_agent.events import (
    TOPIC_REBUILD_REQUESTED,
    ModelRebuildRequested,
)
from deploy_agent.job_state import JobStore

logger = logging.getLogger(__name__)


class DeployConsumer:
    def __init__(self, bootstrap_servers: str, job_store: JobStore):
        self.consumer = KafkaConsumer(
            TOPIC_REBUILD_REQUESTED,
            bootstrap_servers=bootstrap_servers,
            group_id="onex-deploy-agent",
            auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        self.job_store = job_store

    def poll_and_accept(self) -> tuple[ModelRebuildRequested | None, str | None]:
        """Poll for one command.

        Returns (command, None) on accept.
        Returns (None, reason) on reject.
        Returns (None, None) on no message.

        Protocol:
        1. Poll message
        2. Validate payload (schema, scope, services legality)
        3. Check busy (has_active_job) -> reject "busy"
        4. Check dedup (is_duplicate) -> reject "duplicate"
        5. Persist job state (accepted)
        6. Commit Kafka offset
        7. Return (command, None)
        """
        records = self.consumer.poll(timeout_ms=1000)
        if not records:
            return None, None

        # Process first message only
        for topic_partition, messages in records.items():
            for msg in messages:
                return self._process_message(msg)

        return None, None

    def _process_message(
        self, msg: Any
    ) -> tuple[ModelRebuildRequested | None, str | None]:
        payload = msg.value
        correlation_id_str = payload.get("correlation_id", "unknown")

        # Step 2: Validate payload
        try:
            cmd = ModelRebuildRequested.model_validate(payload)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Invalid payload (correlation_id=%s): %s",
                correlation_id_str,
                e,
            )
            self.consumer.commit()
            return None, "invalid_payload"

        # Step 3: Check busy
        if self.job_store.has_active_job():
            logger.info("Rejecting command %s: agent busy", cmd.correlation_id)
            self.consumer.commit()
            return None, "busy"

        # Step 4: Check dedup
        if self.job_store.is_duplicate(cmd.correlation_id):
            logger.info("Rejecting command %s: duplicate", cmd.correlation_id)
            self.consumer.commit()
            return None, "duplicate"

        # Step 5: Persist job state
        self.job_store.accept(
            correlation_id=cmd.correlation_id,
            command=payload,
        )

        # Step 6: Commit offset
        self.consumer.commit()

        # Step 7: Return accepted command
        logger.info("Accepted command %s (scope=%s)", cmd.correlation_id, cmd.scope)
        return cmd, None

    def close(self) -> None:
        self.consumer.close()
