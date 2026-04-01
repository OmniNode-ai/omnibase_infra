# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Kafka consumer that sends Slack notifications for waitlist signups.

Subscribes to onex.evt.omniweb.waitlist-signup.v1 and posts a Slack message
for each new signup using the HandlerSlackWebhook infrastructure.

Privacy: The event payload contains only the email domain (no PII).
The Slack notification reflects this — it shows the domain, not the full address.

Design:
    - Single-message processing (no batching needed for low-volume signups)
    - Fail-open: Slack delivery failures are logged, never block consumption
    - Health check endpoint for container orchestration
    - Graceful shutdown on SIGTERM/SIGINT

Related Tickets:
    - OMN-7199: Slack notification on new waitlist signup
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import signal
import time
from datetime import UTC, datetime
from uuid import uuid4

from aiohttp import web
from aiokafka import AIOKafkaConsumer

from omnibase_infra.handlers.handler_slack_webhook import HandlerSlackWebhook
from omnibase_infra.handlers.models.model_slack_alert import (
    EnumAlertSeverity,
    ModelSlackAlert,
)
from omnibase_infra.services.waitlist_signup_notifier.config import (
    ConfigWaitlistSignupNotifier,
)

logger = logging.getLogger(__name__)


class WaitlistSignupNotifier:
    """Kafka consumer that posts Slack notifications for waitlist signups."""

    def __init__(self, config: ConfigWaitlistSignupNotifier) -> None:
        self._config = config
        self._consumer: AIOKafkaConsumer | None = None
        self._running = False
        self._healthy = False
        self._last_message_at: float | None = None
        self._health_runner: web.AppRunner | None = None

        self._slack_handler = HandlerSlackWebhook(
            bot_token=config.slack_bot_token or None,
            default_channel=config.slack_channel_id or None,
        )

    async def start(self) -> None:
        """Start the Kafka consumer and health check server."""
        self._consumer = AIOKafkaConsumer(
            self._config.kafka_topic,
            bootstrap_servers=self._config.kafka_bootstrap_servers,
            group_id=self._config.kafka_group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        await self._consumer.start()
        self._healthy = True
        self._running = True
        logger.info(
            "Waitlist signup notifier started, subscribed to %s",
            self._config.kafka_topic,
        )

    async def stop(self) -> None:
        """Stop the consumer and health server gracefully."""
        self._running = False
        self._healthy = False
        if self._health_runner:
            await self._health_runner.cleanup()
            self._health_runner = None
        if self._consumer:
            await self._consumer.stop()
            self._consumer = None
        logger.info("Waitlist signup notifier stopped")

    async def run(self) -> None:
        """Main consume loop — processes messages until stopped."""
        if not self._consumer:
            msg = "Consumer not started — call start() first"
            raise RuntimeError(msg)

        while self._running:
            try:
                # Poll with a timeout so we can check _running periodically
                messages = await self._consumer.getmany(timeout_ms=1000, max_records=10)
                for _tp, batch in messages.items():
                    for message in batch:
                        await self._handle_message(message.value)
            except Exception:
                logger.exception("Error in consume loop")
                await asyncio.sleep(1.0)

    async def _handle_message(self, value: dict) -> None:  # type: ignore[type-arg]
        """Process a single waitlist signup event and post to Slack.

        Fail-open: Slack errors are logged but never re-raised.
        """
        self._last_message_at = time.monotonic()

        try:
            # Extract fields from the ONEX event envelope
            payload = value.get("payload", {})
            email_domain = payload.get("email_domain", "unknown")
            occurred_at = value.get("occurred_at", datetime.now(tz=UTC).isoformat())

            # Format a human-friendly timestamp
            try:
                dt = datetime.fromisoformat(occurred_at)
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M UTC")
            except (ValueError, TypeError):
                timestamp_str = occurred_at

            message = (
                f":tada: *New waitlist signup* from `{email_domain}` domain "
                f"-- signed up at `{timestamp_str}`"
            )

            alert = ModelSlackAlert(
                severity=EnumAlertSeverity.INFO,
                message=message,
                title="Waitlist Signup",
                correlation_id=uuid4(),
            )

            result = await self._slack_handler.handle(alert)
            if result.success:
                logger.info(
                    "Slack notification sent for waitlist signup (domain=%s)",
                    email_domain,
                )
            else:
                logger.warning(
                    "Slack notification failed for waitlist signup: %s (code=%s)",
                    result.error,
                    result.error_code,
                )

        except Exception:
            # Fail-open: never block consumption on Slack failures
            logger.exception("Failed to process waitlist signup event")

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def _health_handler(self, _request: web.Request) -> web.Response:
        """Health check endpoint for container probes."""
        if self._healthy:
            return web.json_response(
                {"status": "healthy", "last_message_at": self._last_message_at}
            )
        return web.json_response({"status": "unhealthy"}, status=503)

    async def run_health_server(self) -> None:
        """Start the health check HTTP server."""
        app = web.Application()
        app.router.add_get("/health", self._health_handler)
        runner = web.AppRunner(app)
        self._health_runner = runner
        await runner.setup()
        site = web.TCPSite(
            runner,
            self._config.health_check_host,
            self._config.health_check_port,
        )
        await site.start()
        logger.info(
            "Health check listening on %s:%d",
            self._config.health_check_host,
            self._config.health_check_port,
        )


async def _main() -> None:
    """Entry point for running the consumer as a module."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = ConfigWaitlistSignupNotifier()
    notifier = WaitlistSignupNotifier(config)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _signal_handler)

    await notifier.start()
    await notifier.run_health_server()

    # Run until shutdown signal
    consume_task = asyncio.create_task(notifier.run())
    await stop_event.wait()

    # stop() sets _running=False which causes run() to exit naturally
    await notifier.stop()
    consume_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consume_task


if __name__ == "__main__":
    asyncio.run(_main())
