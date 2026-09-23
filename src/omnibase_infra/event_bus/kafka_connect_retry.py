# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bounded, backed-off retry around a Kafka connect (OMN-18925).

One helper, consumed by both producer-opening surfaces, so the retry SHAPE
cannot drift between them even though their per-attempt deadlines differ. The
reasoning for there being exactly one of these lives on
:class:`~omnibase_infra.event_bus.models.config.model_kafka_connect_retry_policy.ModelKafkaConnectRetryPolicy`.

**The contract in one line:** every attempt is made under its own deadline,
every attempt that fails is cleaned up before the next one starts, a failure
that retrying cannot fix is raised at once, and the LAST failure is raised
unchanged once the attempts are exhausted.

That last clause is the load-bearing one. This helper never converts, wraps or
reclassifies the exception it re-raises, because both call sites already
translate transport exceptions into their own typed errors and a wrapper here
would either duplicate that or silently outrank it. What the caller gets back
is the same exception type it got before this helper existed — the change is
only that it arrives after N attempts instead of one.

.. versionadded:: OMN-18925
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from functools import lru_cache

from omnibase_infra.event_bus.models.config.model_kafka_connect_retry_policy import (
    ModelKafkaConnectRetryPolicy,
)

logger = logging.getLogger(__name__)

__all__ = ["connect_with_bounded_retry"]


@lru_cache(maxsize=1)
def _permanent_connect_errors() -> tuple[type[BaseException], ...]:
    """Connect failures that retrying cannot fix, so they are never retried.

    A retry loop exists for a broker that is not answering YET. An
    authentication rejection is the broker answering, clearly, that it will
    not serve this client -- the same answer arrives every time, so retrying
    it only delays the report. Against the event bus's own worst case that
    delay is the full bound, which would turn an instant, actionable "your
    credentials are wrong" into a two-minute wait before the identical
    message.

    Membership is deliberately narrow: only SASL and version negotiation,
    where the broker has stated a verdict about this client rather than
    failed to respond. A connection refused, a timeout and a reset are all
    absent on purpose, because each of those is exactly the case the retry
    exists for.

    ``InvalidConfigurationError`` is absent too, and that is a judgement
    rather than an oversight: aiokafka raises it for some conditions a
    rejoining cluster clears on its own, so excluding it would refuse a
    connect that a second attempt would have made.

    Imported lazily and cached: this module is imported by the DI provider,
    which defers its own aiokafka import so that merely materializing a
    contract does not pay for the client library.

    .. note::
       This is the CONNECT-side answer to "is this failure permanent". The
       publish side already has its own, in ``EventBusKafka``'s handling of
       ``TopicAuthorizationFailedError``, and a lane is widening that under a
       child of OMN-18627. Two answers to one question is the drift this
       change exists to remove elsewhere, so when a shared predicate lands,
       this should consume it and stop being a second list.
    """
    from aiokafka.errors import (
        AuthenticationFailedError,
        AuthenticationMethodNotSupported,
        IllegalSaslStateError,
        UnsupportedSaslMechanismError,
        UnsupportedVersionError,
    )

    return (
        AuthenticationFailedError,
        AuthenticationMethodNotSupported,
        IllegalSaslStateError,
        UnsupportedSaslMechanismError,
        UnsupportedVersionError,
    )


async def connect_with_bounded_retry(
    *,
    policy: ModelKafkaConnectRetryPolicy,
    connect: Callable[[], Awaitable[None]],
    cleanup: Callable[[], Awaitable[None]],
    target: str,
) -> int:
    """Run ``connect`` under ``policy``, returning the attempt number that won.

    Args:
        policy: How many attempts, how long each may take, how long between.
        connect: Opens the producer. Awaited afresh on every attempt, so it
            must be a factory-style callable rather than a single coroutine
            object -- a coroutine cannot be awaited twice, and a helper that
            accepted one would retry exactly once and then raise a confusing
            ``RuntimeError`` instead of the transport error.
        cleanup: Best-effort teardown of a half-open producer, awaited after
            every failed attempt INCLUDING the last. Skipping it between
            attempts is how a retry loop leaks a socket per attempt.
        target: Broker address or equivalent, for the log line only.

    Returns:
        The 1-based attempt number that succeeded. Returned rather than
        discarded because a connect that needed three tries and one that
        needed one are different facts about the broker, and the caller
        records it on the terminal it may later have to write.

    Raises:
        BaseException: The final attempt's own exception, unchanged and
            un-wrapped, once the policy's attempts are exhausted -- or the
            FIRST attempt's exception, immediately, when it is one retrying
            cannot fix (see :func:`_permanent_connect_errors`).
    """
    delays = policy.backoff_delays()
    last_error: BaseException | None = None

    for attempt in range(1, policy.total_attempts + 1):
        try:
            await asyncio.wait_for(
                connect(),
                timeout=policy.attempt_timeout_seconds,
            )
        except asyncio.CancelledError:
            # Never retried and never swallowed: cancellation is the caller
            # withdrawing the request, not the broker failing to answer. A
            # retry loop that treats it as a transport error keeps working
            # for a caller that has already walked away, and on the delegate
            # path that caller is a SIGALRM backstop whose whole job is to
            # stop the run.
            await _cleanup_quietly(cleanup, target=target)
            raise
        except BaseException as exc:
            last_error = exc
            await _cleanup_quietly(cleanup, target=target)

            if isinstance(exc, _permanent_connect_errors()):
                # Raised unchanged and at once. The caller's error contract is
                # identical to the pre-retry behaviour for this class, which
                # is the point: a credential that is wrong should report as
                # fast as it did before this retry loop existed.
                logger.warning(
                    "Kafka connect to %s failed with %s, which retrying cannot "
                    "fix; raising on attempt %d of %d without further attempts",
                    target,
                    type(exc).__name__,
                    attempt,
                    policy.total_attempts,
                    extra={
                        "target": target,
                        "attempt": attempt,
                        "total_attempts": policy.total_attempts,
                        "error_type": type(exc).__name__,
                        "permanent": True,
                    },
                )
                raise

            if attempt >= policy.total_attempts:
                logger.warning(
                    "Kafka connect exhausted %d attempt(s) against %s within "
                    "%.3fs bound; re-raising %s",
                    policy.total_attempts,
                    target,
                    policy.total_bound_seconds,
                    type(exc).__name__,
                    extra={
                        "target": target,
                        "attempts": policy.total_attempts,
                        "bound_seconds": policy.total_bound_seconds,
                        "error_type": type(exc).__name__,
                    },
                )
                raise

            delay = delays[attempt - 1]
            logger.warning(
                "Kafka connect attempt %d/%d against %s failed with %s; "
                "retrying in %.3fs",
                attempt,
                policy.total_attempts,
                target,
                type(exc).__name__,
                delay,
                extra={
                    "target": target,
                    "attempt": attempt,
                    "total_attempts": policy.total_attempts,
                    "error_type": type(exc).__name__,
                    "retry_delay_seconds": delay,
                },
            )
            await asyncio.sleep(delay)
        else:
            if attempt > 1:
                logger.info(
                    "Kafka connect to %s succeeded on attempt %d/%d",
                    target,
                    attempt,
                    policy.total_attempts,
                    extra={
                        "target": target,
                        "attempt": attempt,
                        "total_attempts": policy.total_attempts,
                    },
                )
            return attempt

    # Unreachable: the loop either returns on success or raises on the final
    # attempt. Asserted rather than ignored so a future edit to the bounds
    # cannot turn "exhausted" into "returned None" silently.
    raise AssertionError(  # pragma: no cover - defensive
        f"connect retry loop fell through for {target}: {last_error!r}"
    )


async def _cleanup_quietly(
    cleanup: Callable[[], Awaitable[None]], *, target: str
) -> None:
    """Run teardown, never letting its failure outrank the connect failure."""
    try:
        await cleanup()
    except Exception as cleanup_error:  # noqa: BLE001 — boundary: logs and degrades
        logger.warning(
            "Cleanup after failed Kafka connect to %s raised: %s",
            target,
            cleanup_error,
            exc_info=True,
        )
