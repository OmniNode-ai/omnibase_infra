# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Debounced Slack alerter for ONEX topic format violations.

Fires a Slack notification the first time a non-conforming topic is seen,
then suppresses duplicate alerts for the same topic within a configurable
debounce window (default 15 minutes).

Design decisions:
- Debounce is **topic-centric** — one debounce entry per unique topic name.
- In-memory dict; no persistence (acceptable for runtime alerting).
- Uses ``SLACK_BOT_TOKEN`` with ``httpx`` for direct Slack API calls.
- Missing token results in a logged warning and no-op (never raises).
- All exceptions are caught and logged, never propagated.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_DEBOUNCE_SECONDS = 900  # 15 minutes
_SLACK_POST_URL = "https://slack.com/api/chat.postMessage"


@dataclass
class DebounceEntry:
    """Tracks debounce state for a single topic."""

    last_alert_time: float = 0.0
    suppressed_count: int = 0


class TopicViolationAlerter:
    """Debounced Slack alerter for topic format violations.

    Parameters
    ----------
    debounce_seconds:
        Minimum interval between alerts for the same topic.  Defaults to
        ``ONEX_TOPIC_ALERT_DEBOUNCE_SECONDS`` env var or 900 (15 min).
    channel:
        Slack channel to post to.  Defaults to
        ``ONEX_TOPIC_ALERT_SLACK_CHANNEL`` env var.
    slack_token:
        Slack bot token.  Defaults to ``SLACK_BOT_TOKEN`` env var.
    """

    def __init__(
        self,
        debounce_seconds: int | None = None,
        channel: str | None = None,
        slack_token: str | None = None,
    ) -> None:
        self._debounce_seconds = debounce_seconds or int(
            os.environ.get(
                "ONEX_TOPIC_ALERT_DEBOUNCE_SECONDS", str(_DEFAULT_DEBOUNCE_SECONDS)
            )
        )
        self._channel = channel or os.environ.get("ONEX_TOPIC_ALERT_SLACK_CHANNEL", "")
        self._token = slack_token or os.environ.get("SLACK_BOT_TOKEN", "")
        self._entries: dict[str, DebounceEntry] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def maybe_alert(self, topic: str, reason: str) -> bool:
        """Send a Slack alert if *topic* has not been alerted within the debounce window.

        Returns ``True`` if the alert was dispatched, ``False`` if suppressed.
        """
        now = time.monotonic()
        entry = self._entries.setdefault(topic, DebounceEntry())

        if (
            entry.last_alert_time > 0
            and (now - entry.last_alert_time) < self._debounce_seconds
        ):
            entry.suppressed_count += 1
            logger.debug(
                "Suppressed topic violation alert for %r (suppressed %d times)",
                topic,
                entry.suppressed_count,
            )
            return False

        entry.last_alert_time = now
        suppressed = entry.suppressed_count
        entry.suppressed_count = 0

        # Fire-and-forget — caller should not await the Slack POST.
        # Store reference to prevent GC of the task (RUF006).
        task = asyncio.create_task(self._send_alert(topic, reason, suppressed))
        task.add_done_callback(
            lambda t: t.result() if not t.cancelled() and not t.exception() else None
        )
        return True

    def get_suppressed_count(self, topic: str) -> int:
        """Return how many alerts have been suppressed for *topic* since the last sent alert."""
        entry = self._entries.get(topic)
        return entry.suppressed_count if entry else 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _send_alert(
        self, topic: str, reason: str, suppressed_since_last: int
    ) -> None:
        """POST a message to the Slack channel.  Never raises."""
        if not self._token:
            logger.warning(
                "SLACK_BOT_TOKEN not set — skipping topic violation alert for %r", topic
            )
            return
        if not self._channel:
            logger.warning(
                "ONEX_TOPIC_ALERT_SLACK_CHANNEL not set — skipping topic violation alert for %r",
                topic,
            )
            return

        caller_hint = _best_effort_caller()
        suppression_note = (
            f"\n({suppressed_since_last} alerts suppressed since last notification)"
            if suppressed_since_last > 0
            else ""
        )

        text = (
            f":warning: *ONEX Topic Format Violation*\n"
            f"*Topic:* `{topic}`\n"
            f"*Reason:* {reason}\n"
            f"*Caller:* `{caller_hint}`{suppression_note}"
        )

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    _SLACK_POST_URL,
                    headers={"Authorization": f"Bearer {self._token}"},
                    json={"channel": self._channel, "text": text},
                )
                if resp.status_code != 200:
                    logger.warning(
                        "Slack API returned %d for topic violation alert: %s",
                        resp.status_code,
                        resp.text[:200],
                    )
        except (httpx.HTTPError, OSError, ValueError):
            logger.warning(
                "Failed to send Slack topic violation alert for %r",
                topic,
                exc_info=True,
            )


def _best_effort_caller() -> str:
    """Walk the call stack to find the most likely application caller."""
    for frame_info in traceback.extract_stack():
        if frame_info.filename and "omnibase_infra" in frame_info.filename:
            if "topic_violation_alerter" not in frame_info.filename:
                return f"{frame_info.filename}:{frame_info.lineno}"
    return "unknown"
