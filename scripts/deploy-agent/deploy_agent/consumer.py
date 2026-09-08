# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka consumer with acceptance protocol.

POISON PILLS ARE QUARANTINED, NEVER RE-READ FOREVER (OMN-16442)
---------------------------------------------------------------

``onex.cmd.deploy.rebuild-requested.v1`` is a control topic with one consumer:
this agent. A record it cannot decode used to take the whole control plane
down, because the failure happened *before* the offset commit and redelivery
reproduced it exactly. Measured on the .201 dev lane on 2026-09-08: one record
published with snappy compression by an ``rpk topic produce`` default, against
a ``kafka-python`` client with no snappy codec installed, raised
``UnsupportedCodecError`` on every poll -- twelve crashes in sixty seconds, six
systemd restarts, then the start limit, ``ActiveState=failed``. Nothing on the
host could recover it: every restart re-read the same offset.

Two distinct classes reach the agent, and they surface in different places:

* **Fetch-level** -- the batch cannot be decompressed or is corrupt on the
  wire. ``kafka-python`` raises inside ``poll()``; no record object exists, so
  the offset must be recovered from the consumer's own position.
* **Record-level** -- the bytes decompress but are not a JSON object, or are a
  JSON object that ``ModelRebuildRequested`` refuses. A record object exists
  and names its own partition and offset.

Both are quarantined the same way: a durable local record, a best-effort
dead-letter publish, a CRITICAL log, and **the offset committed past it**.

WHY COMMITTING IS RIGHT HERE, WHERE THE BOUNDARY DLQ WITHHOLDS
--------------------------------------------------------------

``omnibase_infra.runtime.auto_wiring`` deliberately withholds the offset when a
dead-letter write is not durable (``BoundaryDlqNotPersistedError``): there, the
record still exists on the source topic, so withholding preserves it for the
rewind path. That argument does not transfer. A command this agent cannot
decode can never be decoded by redelivery -- retrying is not recovery, it is
the outage -- and the cost of withholding is not one lost record but every
subsequent deploy command on the partition, which is what actually happened.
So the commit is unconditional and the durability is best-effort *and*
recorded, in that order. The gateway-delivery path made the same call for the
same reason (OMN-15748).

WHAT IS DELIBERATELY *NOT* QUARANTINED
--------------------------------------

Only the two decode-failure error classes are caught at fetch level. A broker
that is down, a SASL handshake that fails, a rebalance -- all of those are
transient and must NOT advance an offset, so they propagate. Skipping records
because the network hiccuped would be a worse failure than the one this fixes.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer
from kafka.errors import CorruptRecordError, UnsupportedCodecError
from kafka.structs import OffsetAndMetadata

from deploy_agent.auth import verify_command
from deploy_agent.events import (
    TOPIC_DEPLOY_COMMAND_DLQ,
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

ENV_QUARANTINE_DIR = "DEPLOY_AGENT_QUARANTINE_DIR"

# Fetch-level failures that redelivery can never fix. Kept deliberately narrow:
# anything transient must propagate rather than advance an offset.
UNDECODABLE_FETCH_ERRORS: tuple[type[BaseException], ...] = (
    UnsupportedCodecError,
    CorruptRecordError,
)

# Bytes of a rejected payload kept in the quarantine record, so an operator can
# recognise the message without the record becoming a copy of an arbitrary blob.
RAW_PREVIEW_BYTES = 512

# Stamped on the committed offset so `rpk group describe` shows WHY the group
# is past a record it never processed, without anyone reading the journal.
QUARANTINE_COMMIT_METADATA = "onex-deploy-agent:quarantined-undecodable"

# kafka-python requires an explicit leader epoch; -1 is its "unknown" sentinel.
UNKNOWN_LEADER_EPOCH = -1


@dataclass(frozen=True)
class UndecodableValue:
    """Marker returned by the deserializer instead of raising inside ``poll()``.

    Raising in a ``value_deserializer`` is the record-level shape of the same
    stall: the exception escapes ``poll()``, so the record object that knows its
    own offset is never handed back and the agent cannot commit past it.
    """

    error: str
    raw_preview: str
    raw_bytes: int


def deserialize_command_value(raw: bytes) -> Any:
    """Decode one command record, returning a marker rather than raising."""
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — any decode failure is one class here
        return UndecodableValue(
            error=f"{type(exc).__name__}: {exc}",
            raw_preview=repr(raw[:RAW_PREVIEW_BYTES]),
            raw_bytes=len(raw),
        )
    if not isinstance(decoded, dict):
        return UndecodableValue(
            error=(
                "command payload decoded to "
                f"{type(decoded).__name__}, expected a JSON object"
            ),
            raw_preview=repr(raw[:RAW_PREVIEW_BYTES]),
            raw_bytes=len(raw),
        )
    return decoded


class DeployConsumer:
    def __init__(
        self,
        kafka_config: ModelDeployAgentKafkaConfig,
        job_store: JobStore,
        allowed_lanes: frozenset[EnumRuntimeLane],
        quarantine_dir: Path | None = None,
    ) -> None:
        self.consumer = KafkaConsumer(
            TOPIC_REBUILD_REQUESTED,
            **kafka_config.consumer_kwargs(),
            group_id="onex-deploy-agent",
            auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=deserialize_command_value,
        )
        self.kafka_config = kafka_config
        self.job_store = job_store
        self.allowed_lanes = allowed_lanes
        self.quarantine_dir = Path(
            quarantine_dir
            or os.environ.get(ENV_QUARANTINE_DIR)
            or job_store.state_dir.parent / "quarantine"
        )
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
        1. Poll message — an undecodable batch is quarantined, not re-read
        2. Verify HMAC signature
        3. Validate payload (schema, scope, services legality)
        4. Check the lane fence -> reject "lane_not_allowed"
        5. Check busy (has_active_job) -> reject "busy"
        6. Check dedup (is_duplicate) -> reject "duplicate"
        7. Persist job state (accepted)
        8. Commit Kafka offset
        9. Return (command, None)
        """
        try:
            records = self.consumer.poll(timeout_ms=1000)
        except UNDECODABLE_FETCH_ERRORS as exc:
            return None, self._quarantine_undecodable_fetch(exc)

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

        # Step 1b: the deserializer hands back a marker rather than raising, so
        # this record's own offset is available to commit past.
        if isinstance(payload, UndecodableValue):
            self._quarantine_record(
                msg,
                reason="undecodable_payload",
                detail=payload.error,
                raw_preview=payload.raw_preview,
            )
            self.consumer.commit()
            return None, "undecodable_payload"

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
            # A signed command the contract refuses is as permanent as an
            # undecodable one -- the bytes will not change on redelivery -- so
            # it is quarantined with the reason rather than merely logged. This
            # is the class that took the dev lane's operator entry point out:
            # the trigger script published `reason` and omitted `runtime_lane`
            # for weeks and the only trace was a warning line.
            self._quarantine_record(
                msg,
                reason="invalid_payload",
                detail=str(e),
                raw_preview=None,
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

        # Step 7: Persist job state
        self.job_store.accept(
            correlation_id=cmd.correlation_id,
            command=command_payload,
        )

        # Step 8: Commit offset
        self.consumer.commit()

        # Step 9: Return accepted command
        logger.info("Accepted command %s (scope=%s)", cmd.correlation_id, cmd.scope)
        return cmd, None

    # ── quarantine ───────────────────────────────────────────────────────────

    def _quarantine_undecodable_fetch(self, exc: BaseException) -> str:
        """Advance past a batch the client cannot decode, one offset at a time.

        No record object exists, so the poison offset is read from the
        consumer's own position. Only partitions with an unconsumed record
        (``position < highwater``) are candidates: a caught-up partition has
        nothing pending and advancing it would skip a future command.

        Exactly ONE offset is skipped per candidate partition per event, never
        a bulk seek to the end. On the single-partition control topic that is
        precisely the poison record; were the topic ever repartitioned, the
        worst case is one good command per extra candidate partition, and every
        skipped coordinate is written to the quarantine record rather than
        being lost silently.
        """
        advanced: list[dict[str, Any]] = []
        for tp in sorted(
            self.consumer.assignment(), key=lambda p: (p.topic, p.partition)
        ):
            position = self.consumer.position(tp)
            highwater = self.consumer.highwater(tp)
            if position is None or highwater is None or position >= highwater:
                continue
            self.consumer.seek(tp, position + 1)
            self.consumer.commit(
                {
                    tp: OffsetAndMetadata(
                        position + 1, QUARANTINE_COMMIT_METADATA, UNKNOWN_LEADER_EPOCH
                    )
                }
            )
            advanced.append(
                {
                    "topic": tp.topic,
                    "partition": tp.partition,
                    "skipped_offset": position,
                    "committed_offset": position + 1,
                }
            )

        record = {
            "quarantined_at": datetime.now(UTC).isoformat(),
            "class": "undecodable_fetch",
            "reason": f"{type(exc).__name__}: {exc}",
            "advanced": advanced,
        }

        if not advanced:
            # Honest rather than silent: the stall was NOT cleared, and saying
            # so is the only way the next restart is diagnosable.
            logger.critical(
                "metric_name=deploy_command_quarantine_failed class=undecodable_fetch "
                "error=%s: no assigned partition had a pending record to skip, so "
                "the offset could not be advanced and this fetch will fail again. "
                "friction_type=deploy_command_poison_pill_unrecoverable",
                record["reason"],
            )
            self._write_quarantine_file(record)
            return "undecodable_fetch_unrecovered"

        logger.critical(
            "metric_name=deploy_command_quarantined class=undecodable_fetch "
            "error=%s advanced=%s — offset committed past the undecodable record. "
            "friction_type=deploy_command_poison_pill",
            record["reason"],
            json.dumps(advanced),
        )
        self._write_quarantine_file(record)
        self._publish_dlq(record)
        return "undecodable_fetch"

    def _quarantine_record(
        self,
        msg: Any,
        *,
        reason: str,
        detail: str,
        raw_preview: str | None,
    ) -> None:
        """Record one permanently-unprocessable message before committing past it."""
        record = {
            "quarantined_at": datetime.now(UTC).isoformat(),
            "class": reason,
            "reason": detail,
            "topic": msg.topic,
            "partition": msg.partition,
            "skipped_offset": msg.offset,
            "committed_offset": msg.offset + 1,
            "key": None if msg.key is None else str(msg.key),
            "raw_preview": raw_preview,
        }
        logger.critical(
            "metric_name=deploy_command_quarantined class=%s topic=%s partition=%s "
            "offset=%s error=%s — offset committed past it. "
            "friction_type=deploy_command_poison_pill",
            reason,
            msg.topic,
            msg.partition,
            msg.offset,
            detail,
        )
        self._write_quarantine_file(record)
        self._publish_dlq(record)

    def _write_quarantine_file(self, record: dict[str, Any]) -> None:
        """Write the durable local copy. Best-effort, and loud when it fails."""
        try:
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            stamp = record["quarantined_at"].replace(":", "").replace("-", "")
            path = self.quarantine_dir / f"{stamp}-{record['class']}.json"
            fd, tmp = tempfile.mkstemp(dir=self.quarantine_dir, suffix=".tmp")
            with os.fdopen(fd, "w") as handle:
                json.dump(record, handle, indent=2, sort_keys=True)
            Path(tmp).replace(path)
            logger.info("Quarantine record written: %s", path)
        except Exception as exc:
            logger.exception(
                "metric_name=deploy_command_quarantine_write_failed error=%s record=%s",
                exc,
                json.dumps(record, default=str),
            )

    def _publish_dlq(self, record: dict[str, Any]) -> None:
        """Dead-letter the quarantine record. Best-effort: the commit already stands."""
        from kafka import KafkaProducer

        try:
            producer = KafkaProducer(
                **self.kafka_config.producer_kwargs(),
                compression_type=None,
                value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            )
            try:
                producer.send(TOPIC_DEPLOY_COMMAND_DLQ, value=record)
                producer.flush(timeout=10)
            finally:
                producer.close()
            logger.info(
                "Dead-lettered quarantine record to %s", TOPIC_DEPLOY_COMMAND_DLQ
            )
        except Exception as exc:
            logger.exception(
                "metric_name=deploy_command_dlq_publish_failed topic=%s error=%s — "
                "the durable local quarantine record is the surviving evidence",
                TOPIC_DEPLOY_COMMAND_DLQ,
                exc,
            )

    def close(self) -> None:
        self.consumer.close()
