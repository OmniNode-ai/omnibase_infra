# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Bounded connect retry, shared by both producer surfaces (OMN-18925).

The stall these tests are built from is measured, not imagined: the dev
Redpanda emitted ``Reactor stalled for 17222 ms on shard 0`` at
2026-09-21T14:35:54Z under host CPU starvation, against a producer whose
connect deadline was 10 seconds. A single ``wait_for`` cannot survive that at
any sane deadline, which is why the fix is a second attempt rather than a
bigger number.

AC-2 is the retry; AC-3 is that both surfaces read ONE declaration of it; AC-4
is the negative control, which uses a real closed TCP port rather than a mock
so that "fails fast and typed" is measured against the operating system rather
than against a fixture that was told to fail.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import time
from unittest.mock import AsyncMock, patch

import pytest

from omnibase_infra.errors import InfraConnectionError, InfraTimeoutError
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.kafka_connect_retry import connect_with_bounded_retry
from omnibase_infra.event_bus.models.config.model_kafka_connect_retry_policy import (
    ModelKafkaConnectRetryPolicy,
)
from omnibase_infra.event_bus.models.config.model_kafka_event_bus_config import (
    ModelKafkaEventBusConfig,
)
from omnibase_infra.runtime.models.model_kafka_producer_config import (
    ModelKafkaProducerConfig,
)
from omnibase_infra.runtime.providers.provider_kafka_producer import (
    ProviderKafkaProducer,
)

# The measured stall, scaled: the shape that matters is "the first attempt's
# deadline expires, a later one lands", not the literal 17.2 seconds, which
# would make the suite take 17.2 seconds to prove a branch.
_STALL_RATIO = 1.72


class _StallThenSucceed:
    """A producer whose first N ``start()`` calls outrun the per-attempt deadline."""

    def __init__(self, *, stall_seconds: float, failures: int = 1) -> None:
        self.calls = 0
        self._stall_seconds = stall_seconds
        self._failures = failures
        self.stop = AsyncMock()

    async def start(self) -> None:
        self.calls += 1
        if self.calls <= self._failures:
            await asyncio.sleep(self._stall_seconds)


@pytest.fixture(autouse=True)
def _collapse_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the retry SHAPE and drop the wall clock.

    The backoff base is a declared, env-resolved dial precisely so a test can
    do this without patching the clock or the helper -- the code under test
    runs its real schedule, just a microsecond-scaled one.
    """
    monkeypatch.setenv("KAFKA_RETRY_BACKOFF_BASE", "0.001")


class TestConnectRetriesAcrossAStall:
    """AC-2: a stall longer than one attempt's deadline is survivable."""

    @pytest.mark.asyncio
    async def test_provider_connects_on_the_second_attempt(self) -> None:
        producer = _StallThenSucceed(stall_seconds=_STALL_RATIO)

        with patch("aiokafka.AIOKafkaProducer", return_value=producer):
            config = ModelKafkaProducerConfig(
                bootstrap_servers="stalled-broker:19092",
                timeout_seconds=1.0,
            )
            result = await ProviderKafkaProducer(config).create()

        assert producer.calls == 2
        assert result is producer

    @pytest.mark.asyncio
    async def test_event_bus_connects_on_the_second_attempt(self) -> None:
        """The surface the delegate CLI actually resolves through.

        Worth stating plainly because the original diagnosis did not: the
        CLI never reaches ProviderKafkaProducer. Its transport is this class,
        built by RuntimeLocal through the onex.backends entry point, and
        before this change its ``start()`` had no retry either -- despite a
        docstring promising "connection retry" and a real retry loop that
        wraps publish only.
        """
        producer = _StallThenSucceed(stall_seconds=_STALL_RATIO)
        bus = EventBusKafka(
            config=ModelKafkaEventBusConfig.default().model_copy(
                update={
                    "bootstrap_servers": "stalled-broker:19092",
                    "timeout_seconds": 1,
                }
            )
        )

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=producer,
        ):
            await bus.start()

        assert producer.calls == 2

    @pytest.mark.asyncio
    async def test_the_failed_attempt_is_cleaned_up_before_the_next(self) -> None:
        """A retry loop that skips teardown leaks a socket per attempt."""
        producer = _StallThenSucceed(stall_seconds=_STALL_RATIO)

        with patch("aiokafka.AIOKafkaProducer", return_value=producer):
            await ProviderKafkaProducer(
                ModelKafkaProducerConfig(
                    bootstrap_servers="stalled-broker:19092",
                    timeout_seconds=1.0,
                )
            ).create()

        assert producer.stop.await_count == 1

    @pytest.mark.asyncio
    async def test_exhaustion_reraises_the_original_exception_type(self) -> None:
        """The caller's error contract is unchanged; only the timing moved.

        Both call sites translate transport exceptions into their own typed
        errors, so a wrapper in the retry helper would either duplicate that
        or silently outrank it.
        """
        producer = _StallThenSucceed(stall_seconds=_STALL_RATIO, failures=99)

        with patch("aiokafka.AIOKafkaProducer", return_value=producer):
            with pytest.raises(asyncio.TimeoutError):
                await ProviderKafkaProducer(
                    ModelKafkaProducerConfig(
                        bootstrap_servers="stalled-broker:19092",
                        timeout_seconds=1.0,
                    )
                ).create()

        assert producer.calls == 4  # 1 + the declared 3 retries


class TestBothSurfacesResolveOneRetryDeclaration:
    """AC-3: no second set of defaults for the same broker.

    Falsifier named on the ticket: two independent defaults. The pairing this
    guards has drifted once already -- ``max_request_size`` had to be pinned
    across these same two surfaces under OMN-16267 after they disagreed about
    the broker's own limit.
    """

    def test_both_surfaces_read_the_same_env_dials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One declaration, reached through the overlay-resolved bus config.

        The operator sets the same two ``KAFKA_*`` variables they already set
        for publish-side retry, and BOTH connect surfaces see them, because
        both resolve through ``ModelKafkaEventBusConfig`` rather than each
        reading the environment for itself. A second direct env read would
        agree with this one only until somebody changed how the overlay
        resolves, which is the drift AC-3 names.
        """
        monkeypatch.setenv("KAFKA_MAX_RETRY_ATTEMPTS", "7")
        monkeypatch.setenv("KAFKA_RETRY_BACKOFF_BASE", "0.25")

        config = ModelKafkaEventBusConfig.default()
        from_bus = ModelKafkaConnectRetryPolicy.from_bus_config(
            config, attempt_timeout_seconds=30.0
        )
        from_provider = ModelKafkaConnectRetryPolicy.from_bus_config(
            config, attempt_timeout_seconds=10.0
        )

        assert from_bus.max_retry_attempts == from_provider.max_retry_attempts == 7
        assert from_bus.retry_backoff_base == from_provider.retry_backoff_base == 0.25
        # The per-attempt deadline is the ONE thing that legitimately differs:
        # flattening 30s and 10s would be a behaviour change smuggled into a
        # shared declaration.
        assert from_bus.attempt_timeout_seconds != from_provider.attempt_timeout_seconds

    def test_the_bound_is_computed_not_asserted(self) -> None:
        policy = ModelKafkaConnectRetryPolicy(
            max_retry_attempts=3,
            retry_backoff_base=1.0,
            attempt_timeout_seconds=10.0,
        )

        assert policy.total_attempts == 4
        assert policy.backoff_delays() == (1.0, 2.0, 4.0)
        assert policy.total_bound_seconds == pytest.approx(4 * 10.0 + 7.0)

    def test_the_default_bound_stays_under_the_delegate_cli_deadline(self) -> None:
        """The retry must not outlive the caller waiting on it.

        The delegate CLI's default terminal wait is 240 + 60 seconds, with a
        10-second hard backstop after that. A connect policy that could
        outrun it would turn a recoverable stall into the timeout refusal
        this change exists to avoid producing.
        """
        bus_policy = ModelKafkaConnectRetryPolicy(
            attempt_timeout_seconds=float(
                ModelKafkaEventBusConfig.default().timeout_seconds
            )
        )
        provider_policy = ModelKafkaConnectRetryPolicy(
            attempt_timeout_seconds=ModelKafkaProducerConfig(
                bootstrap_servers="broker:19092"
            ).timeout_seconds
        )

        assert bus_policy.total_bound_seconds < 300.0
        assert provider_policy.total_bound_seconds < 300.0

    def test_the_default_bound_survives_the_measured_stall(self) -> None:
        """AC-2, stated as the number the incident actually produced."""
        measured_stall_seconds = 17.222
        provider_policy = ModelKafkaConnectRetryPolicy(
            attempt_timeout_seconds=ModelKafkaProducerConfig(
                bootstrap_servers="broker:19092"
            ).timeout_seconds
        )

        assert provider_policy.total_bound_seconds > measured_stall_seconds

    def test_an_out_of_range_dial_is_refused_at_the_boundary(self) -> None:
        """A resolvable but impossible value never becomes a policy."""
        with pytest.raises(ValueError, match="Invalid Kafka connect retry"):
            ModelKafkaConnectRetryPolicy.from_bus_config(
                ModelKafkaEventBusConfig.default().model_copy(
                    update={"max_retry_attempts": 99}
                ),
                attempt_timeout_seconds=10.0,
            )

    def test_a_non_numeric_dial_keeps_the_default_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Documented, not endorsed: an unparseable dial warns and defaults.

        Asserted rather than left implicit because it is the one place the
        shared declaration behaves differently from what this model would
        have chosen on its own. ``ModelKafkaEventBusConfig`` logs and keeps
        its default for any dial it cannot parse, and it does that for every
        dial it resolves, not just these two. Making the connect-retry pair
        refuse instead would give two fields of one config a different error
        policy from their neighbours -- a new disagreement in the place this
        change exists to remove one. So the behaviour is inherited, pinned
        here so a future reader knows it was seen and decided rather than
        missed, and the warning is what an operator has to notice.
        """
        monkeypatch.setenv("KAFKA_MAX_RETRY_ATTEMPTS", "not-a-number")

        with caplog.at_level(logging.WARNING):
            policy = ModelKafkaConnectRetryPolicy.from_bus_config(
                ModelKafkaEventBusConfig.default(),
                attempt_timeout_seconds=10.0,
            )

        assert policy.max_retry_attempts == 3
        assert "KAFKA_MAX_RETRY_ATTEMPTS" in caplog.text

    def test_zero_retries_restores_single_shot_exactly(self) -> None:
        policy = ModelKafkaConnectRetryPolicy(
            max_retry_attempts=0, attempt_timeout_seconds=10.0
        )

        assert policy.total_attempts == 1
        assert policy.backoff_delays() == ()
        assert policy.total_bound_seconds == pytest.approx(10.0)


class TestUnreachableBrokerFailsFastAndTyped:
    """AC-4: a genuinely unreachable broker, against a real closed port.

    A mock told to raise proves the branch; it does not prove the behaviour
    against an operating system that refuses a connection. The falsifier on
    the ticket is a hang, so the assertion is a measured wall clock.
    """

    @staticmethod
    def _closed_port() -> int:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return int(probe.getsockname()[1])

    @pytest.mark.asyncio
    async def test_event_bus_refuses_typed_and_well_inside_its_bound(self) -> None:
        port = self._closed_port()
        policy = ModelKafkaConnectRetryPolicy.from_bus_config(
            ModelKafkaEventBusConfig.default(), attempt_timeout_seconds=2.0
        )
        bus = EventBusKafka(
            config=ModelKafkaEventBusConfig.default().model_copy(
                update={
                    "bootstrap_servers": f"127.0.0.1:{port}",
                    "timeout_seconds": 2,
                }
            )
        )

        started = time.monotonic()
        with pytest.raises((InfraConnectionError, InfraTimeoutError)):
            await bus.start()
        elapsed = time.monotonic() - started

        # Typed, bounded, and NOT a hang: the whole point of AC-4.
        assert elapsed < policy.total_bound_seconds

    @pytest.mark.asyncio
    async def test_provider_refuses_typed_and_well_inside_its_bound(self) -> None:
        port = self._closed_port()
        policy = ModelKafkaConnectRetryPolicy.from_bus_config(
            ModelKafkaEventBusConfig.default(), attempt_timeout_seconds=2.0
        )

        started = time.monotonic()
        with pytest.raises(Exception) as raised:
            await ProviderKafkaProducer(
                ModelKafkaProducerConfig(
                    bootstrap_servers=f"127.0.0.1:{port}",
                    timeout_seconds=2.0,
                )
            ).create()
        elapsed = time.monotonic() - started

        assert not isinstance(raised.value, asyncio.CancelledError)
        assert elapsed < policy.total_bound_seconds


class TestAPermanentFailureIsNotRetried:
    """A broker that has answered "no" is not asked again.

    The retry loop exists for a broker that is not answering YET. An
    authentication rejection is the broker answering clearly, and the same
    answer arrives every time, so retrying it only delays the report -- by
    the full bound in the worst case, which would turn an instant "your
    credentials are wrong" into a two-minute wait before the identical
    message.
    """

    @pytest.mark.asyncio
    async def test_an_auth_rejection_raises_on_the_first_attempt(self) -> None:
        from aiokafka.errors import AuthenticationFailedError

        calls = 0

        async def _connect() -> None:
            nonlocal calls
            calls += 1
            raise AuthenticationFailedError("bad credentials")

        with pytest.raises(AuthenticationFailedError):
            await connect_with_bounded_retry(
                policy=ModelKafkaConnectRetryPolicy(attempt_timeout_seconds=1.0),
                connect=_connect,
                cleanup=AsyncMock(),
                target="broker:19092",
            )

        assert calls == 1, (
            "an authentication rejection was retried; the same answer comes "
            "back every time, so the only effect is to delay the report"
        )

    @pytest.mark.asyncio
    async def test_the_failed_attempt_is_still_cleaned_up(self) -> None:
        """Raising early must not skip teardown, or it leaks a socket."""
        from aiokafka.errors import AuthenticationFailedError

        cleanup = AsyncMock()

        async def _connect() -> None:
            raise AuthenticationFailedError("bad credentials")

        with pytest.raises(AuthenticationFailedError):
            await connect_with_bounded_retry(
                policy=ModelKafkaConnectRetryPolicy(attempt_timeout_seconds=1.0),
                connect=_connect,
                cleanup=cleanup,
                target="broker:19092",
            )

        cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_transient_failure_is_still_retried(self) -> None:
        """The positive control, and the one that keeps the set narrow.

        Without this, widening the permanent set until it swallowed the
        transient cases would pass every test above while removing the whole
        feature. A connection refused is exactly what the retry is for.
        """
        from aiokafka.errors import KafkaConnectionError

        calls = 0

        async def _connect() -> None:
            nonlocal calls
            calls += 1
            raise KafkaConnectionError("connection refused")

        with pytest.raises(KafkaConnectionError):
            await connect_with_bounded_retry(
                policy=ModelKafkaConnectRetryPolicy(attempt_timeout_seconds=1.0),
                connect=_connect,
                cleanup=AsyncMock(),
                target="broker:19092",
            )

        assert calls == 4, "a refused connection is the case the retry exists for"


class TestRetryHelperBoundaries:
    """Properties of the shared helper that neither call site should re-derive."""

    @pytest.mark.asyncio
    async def test_cancellation_is_never_retried(self) -> None:
        """Cancellation is the caller withdrawing, not the broker failing.

        On the delegate path that caller is a SIGALRM backstop whose entire
        job is to stop the run. A loop that retried through it would keep
        working for someone who has already walked away.
        """
        calls = 0

        async def _connect() -> None:
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError

        cleanup = AsyncMock()

        with pytest.raises(asyncio.CancelledError):
            await connect_with_bounded_retry(
                policy=ModelKafkaConnectRetryPolicy(attempt_timeout_seconds=1.0),
                connect=_connect,
                cleanup=cleanup,
                target="broker:19092",
            )

        assert calls == 1
        cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_failing_cleanup_never_outranks_the_connect_failure(self) -> None:
        """The reader needs to know the broker failed, not that teardown did."""

        async def _connect() -> None:
            raise InfraConnectionError("broker refused")

        async def _cleanup() -> None:
            raise RuntimeError("teardown exploded")

        with pytest.raises(InfraConnectionError, match="broker refused"):
            await connect_with_bounded_retry(
                policy=ModelKafkaConnectRetryPolicy(
                    max_retry_attempts=1,
                    retry_backoff_base=0.001,
                    attempt_timeout_seconds=1.0,
                ),
                connect=_connect,
                cleanup=_cleanup,
                target="broker:19092",
            )

    @pytest.mark.asyncio
    async def test_the_winning_attempt_number_is_returned(self) -> None:
        """A broker needing three tries and one needing one are different facts."""
        calls = 0

        async def _connect() -> None:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise InfraConnectionError("still stalled")

        attempt = await connect_with_bounded_retry(
            policy=ModelKafkaConnectRetryPolicy(
                max_retry_attempts=3,
                retry_backoff_base=0.001,
                attempt_timeout_seconds=1.0,
            ),
            connect=_connect,
            cleanup=AsyncMock(),
            target="broker:19092",
        )

        assert attempt == 3


class TestCircuitBreakerCountsStartsNotAttempts:
    """A threshold of 5 must keep meaning five failed starts.

    Recording one circuit failure per ATTEMPT would let a single exhausted
    start() burn four of the five, so a breaker sized for five outages would
    trip on the second. The retry is meant to absorb a transient, not to
    accelerate the breaker.
    """

    @pytest.mark.asyncio
    async def test_one_exhausted_start_records_one_failure(self) -> None:
        producer = _StallThenSucceed(stall_seconds=_STALL_RATIO, failures=99)
        bus = EventBusKafka(
            config=ModelKafkaEventBusConfig.default().model_copy(
                update={
                    "bootstrap_servers": "stalled-broker:19092",
                    "timeout_seconds": 1,
                }
            )
        )

        with patch(
            "omnibase_infra.event_bus.event_bus_kafka.AIOKafkaProducer",
            return_value=producer,
        ):
            with pytest.raises((InfraConnectionError, InfraTimeoutError)):
                await bus.start()

        assert producer.calls == 4  # the retry really did run
        assert bus._circuit_breaker_failures == 1  # and the breaker saw ONE start
