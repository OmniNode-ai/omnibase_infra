# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Kafka testing utilities for integration tests.

This module provides shared utilities for Kafka-based integration tests,
including consumer readiness polling and topic management helpers.

Available Utilities:
    - wait_for_consumer_ready: Poll for Kafka consumer readiness with exponential backoff
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omnibase_infra.event_bus.kafka_event_bus import KafkaEventBus

# Module-level logger for diagnostics
logger = logging.getLogger(__name__)


async def wait_for_consumer_ready(
    event_bus: KafkaEventBus,
    topic: str,
    max_wait: float = 10.0,
    initial_backoff: float = 0.1,
    max_backoff: float = 1.0,
    backoff_multiplier: float = 1.5,
) -> bool:
    """Wait for Kafka consumer to be ready to receive messages using polling.

    This is a **best-effort** readiness check that always returns True. It attempts
    to detect when the consumer is ready by polling health checks, but falls back
    gracefully on timeout to avoid blocking tests indefinitely.

    Kafka consumers require time to join the consumer group and start receiving
    messages after subscription. This helper polls the event bus health check
    until the consumer count increases, indicating the consumer task is running.

    Behavior Summary:
        1. Polls event_bus.health_check() with exponential backoff
        2. If consumer_count increases within max_wait: returns True (early exit)
        3. If max_wait exceeded: returns True anyway (graceful fallback)

    Why Always Return True?
        The purpose is to REDUCE flakiness by waiting for actual readiness when
        possible, not to DETECT failures. Test assertions should verify expected
        outcomes, not this helper's return value.

    Implementation:
        Uses exponential backoff polling (initial_backoff * backoff_multiplier^n)
        to check consumer registration, capped at max_backoff per iteration.
        This is more reliable than a fixed sleep as it:
        - Returns early when consumer is ready (reduces test time)
        - Adapts to variable Kafka/Redpanda startup times
        - Reduces flakiness compared to fixed-duration sleeps

    Args:
        event_bus: The KafkaEventBus instance to check for readiness.
        topic: The topic to wait for (used for logging only, not filtering).
        max_wait: Maximum time in seconds to poll before giving up. The function
            will return True regardless of whether consumer became ready.
            Default: 10.0s. Actual wait may exceed max_wait by up to max_backoff
            (on timeout) or +0.1s stabilization delay (on success).
        initial_backoff: Initial polling delay in seconds (default 0.1s).
        max_backoff: Maximum polling delay cap in seconds (default 1.0s).
        backoff_multiplier: Multiplier for exponential backoff (default 1.5).

    Returns:
        Always True. Do not use return value for failure detection.
        Use test assertions to verify expected outcomes.

    Example:
        # Best-effort wait for consumer readiness (default max_wait=10.0s)
        await wait_for_consumer_ready(bus, topic)

        # Shorter wait for fast tests
        await wait_for_consumer_ready(bus, topic, max_wait=2.0)

        # Consumer MAY be ready here, but test should not rely on this
        # Use assertions on actual test outcomes instead
    """
    start_time = asyncio.get_running_loop().time()
    current_backoff = initial_backoff

    # Get initial consumer count for comparison
    initial_health = await event_bus.health_check()
    initial_consumer_count = initial_health.get("consumer_count", 0)

    # Poll until consumer count increases or timeout
    while (asyncio.get_running_loop().time() - start_time) < max_wait:
        health = await event_bus.health_check()
        consumer_count = health.get("consumer_count", 0)

        # If consumer count has increased, the subscription is active
        if consumer_count > initial_consumer_count:
            # Add a small additional delay for the consumer loop to start
            # processing messages after registration
            await asyncio.sleep(0.1)
            return True

        # Check if we've timed out after health check (prevents unnecessary sleep)
        elapsed = asyncio.get_running_loop().time() - start_time
        if elapsed >= max_wait:
            break

        # Exponential backoff with cap
        await asyncio.sleep(current_backoff)
        current_backoff = min(current_backoff * backoff_multiplier, max_backoff)

    # Return True even on timeout (graceful fallback)
    # Log at debug level for diagnostics
    logger.debug(
        "wait_for_consumer_ready timed out after %.2fs for topic %s",
        max_wait,
        topic,
    )
    return True


__all__ = [
    "wait_for_consumer_ready",
]
