# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka consumer for the ONEX Status TUI.

Subscribes to three topics and posts messages to the app via the
textual ``post_message`` mechanism. The consumer runs in a background
asyncio task spawned by StatusApp.

Topics consumed:
    - onex.evt.github.pr-status.v1  → PRStatusReceived
    - onex.evt.git.hook.v1          → HookEventReceived
    - onex.evt.linear.snapshot.v1   → SnapshotReceived

No imports from application-specific modules (omniclaude, omnidash, etc.).
Topic names are sourced from module-level constants only.

Related Tickets:
    - OMN-2657: Phase 3 — TUI ONEX Status Terminal View (omnibase_infra)
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError
from textual.message import Message

if TYPE_CHECKING:
    from omnibase_core.types import JsonType

logger = logging.getLogger(__name__)

# Topic constants — sourced from event contracts, not hardcoded per-app
TOPIC_PR_STATUS = "onex.evt.github.pr-status.v1"
TOPIC_GIT_HOOK = "onex.evt.git.hook.v1"
TOPIC_LINEAR_SNAPSHOT = "onex.evt.linear.snapshot.v1"

_ALL_TOPICS = (TOPIC_PR_STATUS, TOPIC_GIT_HOOK, TOPIC_LINEAR_SNAPSHOT)


# ---------------------------------------------------------------------------
# Message types posted to the TUI app
# ---------------------------------------------------------------------------


class PRStatusReceived(Message):
    """Posted when a PR status event arrives from Kafka.

    Fields mirror the ModelGitHubPRStatusEvent schema:
        event_type, pr_number, triage_state, title, partition_key, repo.
    """

    def __init__(self, payload: dict[str, JsonType]) -> None:
        super().__init__()
        self.payload = payload


class HookEventReceived(Message):
    """Posted when a git hook event arrives from Kafka.

    Fields mirror the ModelGitHookEvent schema:
        event_type, hook, repo, branch, author, outcome, gates, emitted_at.
    """

    def __init__(self, payload: dict[str, JsonType]) -> None:
        super().__init__()
        self.payload = payload


class SnapshotReceived(Message):
    """Posted when a Linear snapshot event arrives from Kafka.

    Fields mirror the ModelLinearSnapshotEvent schema:
        event_type, snapshot_id, workstreams, snapshot, emitted_at.
    """

    def __init__(self, payload: dict[str, JsonType]) -> None:
        super().__init__()
        self.payload = payload


# ---------------------------------------------------------------------------
# Consumer coroutine
# ---------------------------------------------------------------------------


async def consume_all(app: object) -> None:
    """Subscribe to all three topics and dispatch messages to ``app``.

    ``app`` must implement ``post_message(message)`` (standard textual App).

    Gracefully handles:
    - Kafka unreachable (logs warning, returns immediately)
    - JSON decode errors (logs and skips message)
    - Unknown topics (logs and skips)

    Args:
        app: The textual App instance to receive messages.
    """
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:29092")
    consumer: AIOKafkaConsumer | None = None
    try:
        consumer = AIOKafkaConsumer(
            *_ALL_TOPICS,
            bootstrap_servers=bootstrap_servers,
            group_id="onex-tui-status",
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: v,  # raw bytes; we JSON-decode below
            request_timeout_ms=5000,
            connections_max_idle_ms=10000,
        )
        await consumer.start()
        logger.info(
            "TUI consumer started. Topics: %s. Servers: %s",
            _ALL_TOPICS,
            bootstrap_servers,
        )
        async for msg in consumer:
            topic = msg.topic
            try:
                payload: dict[str, JsonType] = json.loads(msg.value.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                logger.warning("Failed to decode message from %s: %s", topic, exc)
                continue

            if topic == TOPIC_PR_STATUS:
                app.post_message(PRStatusReceived(payload))  # type: ignore[attr-defined]
            elif topic == TOPIC_GIT_HOOK:
                app.post_message(HookEventReceived(payload))  # type: ignore[attr-defined]
            elif topic == TOPIC_LINEAR_SNAPSHOT:
                app.post_message(SnapshotReceived(payload))  # type: ignore[attr-defined]
            else:
                logger.debug("Ignoring unknown topic: %s", topic)

    except KafkaError as exc:
        logger.warning("Kafka unavailable or connection failed: %s", exc)
    except Exception as exc:
        logger.exception("Consumer error: %s", exc)
    finally:
        if consumer is not None:
            try:
                await consumer.stop()
            except Exception:
                pass


__all__ = [
    "PRStatusReceived",
    "HookEventReceived",
    "SnapshotReceived",
    "consume_all",
    "TOPIC_GIT_HOOK",
    "TOPIC_LINEAR_SNAPSHOT",
    "TOPIC_PR_STATUS",
]
