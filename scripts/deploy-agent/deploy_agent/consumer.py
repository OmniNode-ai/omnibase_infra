# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka consumer with acceptance protocol."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from kafka import KafkaConsumer, TopicPartition

from deploy_agent.auth import verify_command
from deploy_agent.events import (
    TOPIC_REBUILD_REQUESTED,
    EnumRuntimeLane,
    ModelRebuildRequested,
)
from deploy_agent.job_state import JobStore
from deploy_agent.kafka_config import ModelDeployAgentKafkaConfig
from deploy_agent.lane_policy import (
    LaneNotAllowedError,
    assert_lane_allowed,
)

logger = logging.getLogger(__name__)

# OMN-16442. Invoked at the PRE_ACCEPT job boundary with a callback that rewinds
# this consumer's committed offset to the command being examined. The hook is
# expected not to return when it decides to update: it replaces the process
# image, and the rewound offset is what makes the replacement process re-read
# the command instead of skipping it.
SelfUpdateHook = Callable[[Callable[[], None]], None]


class DeployConsumer:
    def __init__(
        self,
        kafka_config: ModelDeployAgentKafkaConfig,
        job_store: JobStore,
        allowed_lanes: frozenset[EnumRuntimeLane],
        self_update_hook: SelfUpdateHook,
    ) -> None:
        self.consumer = KafkaConsumer(
            TOPIC_REBUILD_REQUESTED,
            **kafka_config.consumer_kwargs(),
            group_id="onex-deploy-agent",
            auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )
        self.job_store = job_store
        self.allowed_lanes = allowed_lanes
        self.self_update_hook = self_update_hook
        logger.info(
            "Deploy agent lane fence: %s",
            ",".join(sorted(lane.value for lane in self.allowed_lanes)),
        )

    def poll_and_accept(self) -> tuple[ModelRebuildRequested | None, str | None]:
        """Poll for one command.

        Returns (command, None) on accept.
        Returns (None, reason) on reject.
        Returns (None, None) on no message.

        Protocol:
        1. Poll message
        2. Verify HMAC signature
        3. Validate payload (schema, scope, services legality)
        4. Check the lane fence -> reject "lane_not_allowed"
        5. Check busy (has_active_job) -> reject "busy"
        6. Check dedup (is_duplicate) -> reject "duplicate"
        7. Self-update boundary (OMN-16442) -- may not return
        8. Persist job state (accepted)
        9. Commit Kafka offset
        10. Return (command, None)
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

        # Step 2: Verify HMAC signature
        if not verify_command(payload):
            logger.warning(
                "Rejecting command (correlation_id=%s): invalid_signature",
                correlation_id_str,
            )
            self.consumer.commit()
            return None, "invalid_signature"

        # Step 3: Validate payload. The signature is transport metadata, not
        # part of the command contract itself.
        try:
            command_payload = {k: v for k, v in payload.items() if k != "_signature"}
            cmd = ModelRebuildRequested.model_validate(command_payload)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Invalid payload (correlation_id=%s): %s",
                correlation_id_str,
                e,
            )
            self.consumer.commit()
            return None, "invalid_payload"

        # Step 4: Lane fence (OMN-16939). The dev control bus carries both dev
        # and stability-test rebuild commands, so "which bus am I on" does not
        # bound which lane this process may mutate. The offset is committed:
        # the command is not for this agent and re-reading it forever would
        # stall every command behind it.
        try:
            assert_lane_allowed(cmd.runtime_lane, self.allowed_lanes)
        except LaneNotAllowedError as e:
            logger.warning(
                "Rejecting command %s: lane_not_allowed (%s)", cmd.correlation_id, e
            )
            self.consumer.commit()
            return None, "lane_not_allowed"

        # Step 5: Check busy
        if self.job_store.has_active_job():
            logger.info("Rejecting command %s: agent busy", cmd.correlation_id)
            self.consumer.commit()
            return None, "busy"

        # Step 6: Check dedup
        if self.job_store.is_duplicate(cmd.correlation_id):
            logger.info("Rejecting command %s: duplicate", cmd.correlation_id)
            self.consumer.commit()
            return None, "duplicate"

        # Step 7: Self-update boundary (OMN-16442). This is the last point at
        # which nothing is in flight: the command has passed every acceptance
        # check but is not yet marked started and its offset is not yet
        # committed. If the agent is behind its tracking ref, the hook replaces
        # the process image here and does not return -- having first rewound
        # the committed offset to this message, so the replacement process
        # re-reads it. Update-then-process, rather than the process-then-die
        # that killed command 8d0c861a-f91e-4ca2-954e-a073759dd39d when the
        # same check ran mid-deploy.
        #
        # A self-update failure must not cost a valid command: it is logged and
        # the command is processed on the current image, exactly as the
        # method's own dirty-tree and fetch-failure rails already do.
        try:
            self.self_update_hook(lambda: self._rewind_committed_offset_to(msg))
        except Exception as e:  # noqa: BLE001
            logger.error(  # noqa: TRY400
                "Self-update at the pre-accept boundary failed for %s, "
                "proceeding on the current image: %s "
                "friction_type=self_update_boundary_failed",
                cmd.correlation_id,
                e,
            )

        # Step 8: Persist job state
        self.job_store.accept(
            correlation_id=cmd.correlation_id,
            command=command_payload,
        )

        # Step 9: Commit offset
        self.consumer.commit()

        # Step 10: Return accepted command
        logger.info("Accepted command %s (scope=%s)", cmd.correlation_id, cmd.scope)
        return cmd, None

    def _rewind_committed_offset_to(self, msg: Any) -> None:
        """Commit this message's own offset so it is re-read, not skipped.

        Called immediately before the process image is replaced. Seeking to
        ``msg.offset`` and committing the resulting position makes the
        committed offset point AT this command rather than past it, so the
        replacement process fetches the same command again.

        Relying on the message simply being uncommitted is not enough: this
        consumer is configured ``auto_offset_reset="latest"``, so a group with
        no committed offset yet -- the first command a freshly created group
        ever sees -- would resume past the message and lose it.
        """
        topic_partition = TopicPartition(msg.topic, msg.partition)
        self.consumer.seek(topic_partition, msg.offset)
        self.consumer.commit()
        logger.info(
            "Rewound committed offset to %s@%d:%d before self-update re-exec",
            msg.topic,
            msg.partition,
            msg.offset,
        )

    def close(self) -> None:
        self.consumer.close()
