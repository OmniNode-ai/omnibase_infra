# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Process-level restart-with-backoff for standalone Kafka consumers.

Provides a shared ``run_with_restart`` coroutine that wraps a consumer's
main loop in an exponential-backoff restart cycle with:

- **Fatal error classification**: Config, auth, and programming errors are
  re-raised immediately (no retry).
- **Graceful shutdown**: An optional ``asyncio.Event`` parameter allows
  signal handlers to interrupt the restart loop within milliseconds.
- **Backoff reset**: If a run is stable for longer than ``stable_threshold_s``,
  the backoff counter resets on the next failure.

Not to be confused with ``util_retry_optimistic.py`` which handles database-level
optimistic locking retries, not process-level restarts.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# Import fatal exception types with graceful fallback.
# These are checked by isinstance() for reliable classification.
_FATAL_TYPES: list[type[BaseException]] = [TypeError, AttributeError]

try:
    from pydantic import ValidationError as _PydanticValidationError

    _FATAL_TYPES.append(_PydanticValidationError)
except ImportError:
    pass

try:
    from aiokafka.errors import (
        AuthenticationFailedError as _AiokafkaAuthError,
    )

    _FATAL_TYPES.append(_AiokafkaAuthError)
except ImportError:
    pass

try:
    from asyncpg.exceptions import (
        InvalidAuthorizationSpecificationError as _AsyncpgAuthError,
    )

    _FATAL_TYPES.append(_AsyncpgAuthError)
except ImportError:
    pass

_FATAL_EXCEPTION_TUPLE: tuple[type[BaseException], ...] = tuple(_FATAL_TYPES)


def _is_fatal(exc: BaseException) -> bool:
    """Check if an exception indicates a permanent failure that should not be retried.

    Fatal categories:
    - Programming errors: TypeError, AttributeError
    - Config errors: pydantic.ValidationError
    - Auth errors: aiokafka AuthenticationFailedError, asyncpg auth errors

    NOT fatal: ValueError (aiokafka uses it for transient "no broker" conditions).

    Args:
        exc: The exception to classify.

    Returns:
        True if the exception is fatal and should not be retried.
    """
    return isinstance(exc, _FATAL_EXCEPTION_TUPLE)


async def _shutdown_aware_sleep(
    seconds: float,
    shutdown_event: asyncio.Event | None,
) -> bool:
    """Sleep for ``seconds``, but return early if shutdown_event is set.

    Args:
        seconds: Duration to sleep.
        shutdown_event: If set during sleep, returns early.

    Returns:
        True if shutdown was requested during sleep, False otherwise.
    """
    if shutdown_event is None:
        await asyncio.sleep(seconds)
        return False
    try:
        await asyncio.wait_for(shutdown_event.wait(), timeout=seconds)
        return True  # shutdown_event was set
    except TimeoutError:
        return False  # normal timeout, no shutdown


async def run_with_restart(
    coro_factory: Callable[[], Awaitable[None]],
    *,
    name: str,
    shutdown_event: asyncio.Event | None = None,
    initial_backoff_s: float = 1.0,
    max_backoff_s: float = 60.0,
    backoff_factor: float = 2.0,
    stable_threshold_s: float = 60.0,
) -> None:
    """Run a coroutine in a restart loop with exponential backoff.

    Intended for standalone consumer entry points that should self-heal
    after transient failures (Kafka/Postgres unavailable, network blips).

    Fatal errors (config, auth, programming) are re-raised immediately.
    Transient errors trigger backoff and retry.

    Backoff resets to initial after a run lasting > stable_threshold_s,
    indicating the process was stable before the failure.

    Args:
        coro_factory: Zero-arg async callable that runs the consumer.
        name: Consumer name for logging.
        shutdown_event: If set, exit cleanly instead of restarting.
            Signal handlers should set this AND call consumer.stop()
            to interrupt blocked Kafka polls.
        initial_backoff_s: Starting backoff delay.
        max_backoff_s: Maximum backoff delay.
        backoff_factor: Multiplier per consecutive failure.
        stable_threshold_s: Duration after which backoff resets.

    Raises:
        KeyboardInterrupt: Re-raised immediately.
        SystemExit: Re-raised immediately.
        asyncio.CancelledError: Re-raised immediately.
        Exception: Re-raised if classified as fatal (config, auth, programming error).
    """
    attempt = 0
    backoff = initial_backoff_s

    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            logger.info("%s shutdown requested, exiting restart loop", name)
            return

        attempt += 1
        start = time.monotonic()
        try:
            logger.info("%s starting (attempt %d)", name, attempt)
            await coro_factory()
            # Clean exit — check if shutdown was requested
            if shutdown_event is not None and shutdown_event.is_set():
                logger.info("%s stopped gracefully", name)
                return
            logger.warning("%s exited unexpectedly without error, restarting", name)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except Exception as exc:
            if _is_fatal(exc):
                logger.exception(
                    "%s hit fatal error (not retryable): %s",
                    name,
                    exc,
                )
                raise

            elapsed = time.monotonic() - start
            # Reset backoff if the run was stable before failing
            if elapsed > stable_threshold_s:
                backoff = initial_backoff_s
                attempt = 0

            jitter = backoff * random.uniform(0.8, 1.2)
            logger.exception(
                "%s crashed after %.1fs (attempt %d), retrying in %.1fs",
                name,
                elapsed,
                attempt,
                jitter,
            )

            # Shutdown-aware sleep: exits early on SIGTERM instead of blocking up to 60s
            if await _shutdown_aware_sleep(jitter, shutdown_event):
                logger.info("%s shutdown during backoff, exiting", name)
                return
            backoff = min(backoff * backoff_factor, max_backoff_s)
            continue

        # Clean exit path (no exception) — brief pause before restart
        if await _shutdown_aware_sleep(initial_backoff_s, shutdown_event):
            return
