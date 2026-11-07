# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replay headers a dead letter carries back onto its source topic.

``node_dlq_replay_effect`` stamps these on a record it republishes, and the
Kafka bus reads them on consume. Both sides import the names from here, so a
header is spelled in exactly one place.

``x-replay-count`` is the replay lineage counter (OMN-14551). OMN-22822 adds
``x-replay-target-group``: the one consumer group whose subscriber failed the
record, so the replay returns to it alone instead of to every subscriber of the
topic.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from omnibase_infra.event_bus.models.model_event_headers import ModelEventHeaders

logger = logging.getLogger(__name__)

# Header ``node_dlq_replay_effect.engine_dlq_replay.DLQProducer.replay_message``
# stamps onto a message it republishes to its original topic. Must match that
# producer's header name verbatim -- this is the sole read side of that
# contract for the raw Kafka header shape. ``EventBusKafka._kafka_headers_to_model``
# reads the same constant to populate ``ModelEventHeaders.retry_count`` for the
# parsed-message shape (see the isinstance branch below).
REPLAY_COUNT_HEADER: str = "x-replay-count"

# OMN-14551 defect 4: a malformed/negative/absent-value replay-count header
# must NOT reset replay lineage to 0 -- that would silently reopen the
# unbounded-replay amplification the header exists to bound. Any
# ``max_replay_count`` in practice (default 5) is tiny; this sentinel
# guarantees ``retry_count >= max_replay_count`` trips on the very next
# replay-eligibility check, forcing quarantine instead of blind replay.
REPLAY_COUNT_PARSE_FAILURE_SENTINEL: int = 2**31 - 1


def extract_replay_count_from_raw_headers(raw_msg: object) -> int:
    """Read the replay lineage count off a DLQ-bound message, if present.

    OMN-14551: a message replayed by ``node_dlq_replay_effect`` carries an
    ``x-replay-count`` header when it lands back on its original topic. If it
    fails again there, ``_publish_raw_to_dlq``/``_publish_to_dlq`` must carry
    that count forward into the new DLQ record's ``retry_count`` field so
    ``should_replay``'s ``retry_count >= max_replay_count`` guard is
    reachable -- otherwise every re-published copy resets to 0 and the guard
    can never trip (the live DLQ amplification incident this fixes).

    Two ``raw_msg.headers`` shapes are recognized, matching the two shapes
    the four production ``_publish_raw_to_dlq`` call sites actually pass:

    1. Raw Kafka ``list[tuple[str, bytes]]`` header shape (a real
       ``ConsumerRecord`` or a test double of one) -- the deserialization-
       error call site (``EventBusKafka._consume_loop``). Read directly off
       the ``x-replay-count`` entry.
    2. A parsed ``ModelEventHeaders`` instance -- the ``handler_exception``
       call site (``handler_wiring.py``) and the five
       ``event_bus_subcontract_wiring.py`` call sites, all of which pass a
       ``ModelEventMessage``/``ProtocolEventMessage`` as ``raw_msg`` once the
       consume loop has already deserialized it. Its ``.headers.retry_count``
       was itself already populated from ``x-replay-count`` by
       ``EventBusKafka._kafka_headers_to_model`` at consume time, so it is
       read directly here rather than re-parsed.

    Any other ``raw_msg.headers`` shape (missing, unrecognized type) has no
    replay lineage to read and defaults to 0 -- the pre-existing
    first-failure behavior.
    """
    headers = getattr(raw_msg, "headers", None)

    if isinstance(headers, ModelEventHeaders):
        retry_count = headers.retry_count
        if isinstance(retry_count, int) and not isinstance(retry_count, bool):
            if retry_count >= 0:
                return retry_count
            logger.warning(
                "Negative ModelEventHeaders.retry_count %d on raw DLQ "
                "message, failing closed to stop replay",
                retry_count,
            )
            return REPLAY_COUNT_PARSE_FAILURE_SENTINEL
        logger.warning(
            "Invalid ModelEventHeaders.retry_count %r on raw DLQ message, "
            "failing closed to stop replay",
            retry_count,
        )
        return REPLAY_COUNT_PARSE_FAILURE_SENTINEL

    if not isinstance(headers, list | tuple):
        return 0
    for entry in headers:
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            continue
        key, value = entry
        if key != REPLAY_COUNT_HEADER:
            continue
        if value is None:
            logger.warning(
                "Empty %s header value on raw DLQ message, failing closed "
                "to stop replay",
                REPLAY_COUNT_HEADER,
            )
            return REPLAY_COUNT_PARSE_FAILURE_SENTINEL
        try:
            decoded = value.decode("utf-8") if isinstance(value, bytes) else str(value)
            parsed = int(decoded)
        except (ValueError, TypeError, UnicodeDecodeError):
            logger.warning(
                "Malformed %s header %r on raw DLQ message, failing closed "
                "to stop replay",
                REPLAY_COUNT_HEADER,
                value,
            )
            return REPLAY_COUNT_PARSE_FAILURE_SENTINEL
        if parsed < 0:
            logger.warning(
                "Negative %s header value %d on raw DLQ message, failing "
                "closed to stop replay",
                REPLAY_COUNT_HEADER,
                parsed,
            )
            return REPLAY_COUNT_PARSE_FAILURE_SENTINEL
        return parsed
    return 0


def resolve_replay_count(replay_count_str: str | None, default: int) -> int:
    """The replay lineage count for a consumed record's ``x-replay-count`` header.

    OMN-14551: a message republished by ``node_dlq_replay_effect`` carries
    ``x-replay-count`` instead of ``retry_count``; when present it is the
    authoritative lineage counter and wins over any stale ``retry_count``.
    Absent returns ``default``. A malformed or negative value fails CLOSED to
    ``REPLAY_COUNT_PARSE_FAILURE_SENTINEL`` so ``should_replay`` quarantines
    instead of blind-replaying (the 2026-08-05 dev DLQ amplification incident).
    """
    if replay_count_str is None:
        return default
    try:
        parsed = int(replay_count_str)
    except (ValueError, TypeError):
        logger.warning(
            "Malformed %s header %r, failing closed to stop replay",
            REPLAY_COUNT_HEADER,
            replay_count_str,
        )
        return REPLAY_COUNT_PARSE_FAILURE_SENTINEL
    if parsed < 0:
        logger.warning(
            "Negative %s header value %d, failing closed to stop replay",
            REPLAY_COUNT_HEADER,
            parsed,
        )
        return REPLAY_COUNT_PARSE_FAILURE_SENTINEL
    return parsed


# OMN-22822: header ``DLQProducer.replay_message`` stamps on a replayed record to
# address it to the ONE consumer group whose subscriber failed it. The same
# constant is the consume-side read in ``EventBusKafka._process_consumed_record``.
REPLAY_TARGET_GROUP_HEADER: str = "x-replay-target-group"

# OMN-22822: consumer-group LABELS that name no subscriber. Callers that have no
# real subscription group (the "unknown" fallback when a record arrives with no
# subscriber registered, the auto-wiring boundary built without a group, and the
# sync publisher, which consumes nothing) pass one of these so the DLQ call stays
# well formed. Recording one on the envelope as the replay target would address
# the replay to a group nobody is in, and every consumer would skip it: a lost
# record. They are therefore never recorded as ``consumer_group``.
DLQ_NON_SUBSCRIBER_CONSUMER_GROUPS: frozenset[str] = frozenset(
    {"unknown", "auto-wiring", "auto-wiring-sync-publisher"}
)


def envelope_consumer_group(consumer_group: object) -> str | None:
    """The consumer group to record on a dead-letter envelope, or None.

    A blank value or a label from ``DLQ_NON_SUBSCRIBER_CONSUMER_GROUPS`` names
    no subscriber and is not recorded; an envelope without a group replays to
    every subscriber, as it always did.
    """
    group = consumer_group.strip() if isinstance(consumer_group, str) else ""
    if not group or group in DLQ_NON_SUBSCRIBER_CONSUMER_GROUPS:
        return None
    return group


def extract_replay_target_group_from_raw_headers(raw_msg: object) -> str | None:
    """Read the ``x-replay-target-group`` header off a consumed Kafka record.

    Returns the named consumer group, or ``None`` when the record carries no
    target: no headers, no such header, or a value that is empty, null or not
    UTF-8. ``None`` means "deliver to everyone", the behaviour of every record
    that was not replayed to a specific subscriber. An unreadable target fails
    OPEN (the record is processed): a skip is a commit, and committing past a
    record on a guess is a silent drop.
    """
    headers = getattr(raw_msg, "headers", None)
    if not isinstance(headers, list | tuple):
        return None
    for entry in headers:
        if not isinstance(entry, list | tuple) or len(entry) != 2:
            continue
        key, value = entry
        if key != REPLAY_TARGET_GROUP_HEADER:
            continue
        if value is None:
            return None
        try:
            decoded = value.decode("utf-8") if isinstance(value, bytes) else str(value)
        except UnicodeDecodeError:
            logger.warning(
                "Unreadable %s header on consumed record, processing it as untargeted",
                REPLAY_TARGET_GROUP_HEADER,
            )
            return None
        return decoded.strip() or None
    return None


class ModelDlqReplayTarget(BaseModel):
    """Which consumer group a replayed record is addressed to (OMN-22822)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    consumer_group: str | None = Field(
        default=None,
        description="Addressed group, or None for a record addressed to everyone.",
    )

    @classmethod
    def from_raw_msg(cls, raw_msg: object) -> ModelDlqReplayTarget:
        return cls(consumer_group=extract_replay_target_group_from_raw_headers(raw_msg))

    def excludes(self, group_id: str) -> bool:
        """True when the record is addressed to a group other than ``group_id``."""
        return self.consumer_group is not None and self.consumer_group != group_id


__all__ = [
    "DLQ_NON_SUBSCRIBER_CONSUMER_GROUPS",
    "REPLAY_COUNT_HEADER",
    "REPLAY_COUNT_PARSE_FAILURE_SENTINEL",
    "REPLAY_TARGET_GROUP_HEADER",
    "ModelDlqReplayTarget",
    "envelope_consumer_group",
    "resolve_replay_count",
    "extract_replay_count_from_raw_headers",
    "extract_replay_target_group_from_raw_headers",
]
