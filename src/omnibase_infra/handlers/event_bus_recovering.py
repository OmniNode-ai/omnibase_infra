# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime event bus policy while Kafka is lost or recovering.

Moved out of ``event_bus_kafka`` (a baselined file that may only shrink) so the
recovery policy lives in the canonical handlers location. See
:class:`EventBusRecovering`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from uuid import uuid4

from omnibase_infra.enums import (
    EnumConsumerGroupPurpose,
    EnumEventBusLostBrokerPolicy,
)
from omnibase_infra.errors import (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.models import (
    ModelEventBusReadiness,
    ModelEventHeaders,
    ModelEventMessage,
    ModelPublishReceipt,
)
from omnibase_infra.event_bus.models.config import ModelKafkaEventBusConfig
from omnibase_infra.models import ModelNodeIdentity
from omnibase_infra.models.event_bus.model_subscription_handle import (
    ModelSubscriptionHandle,
)
from omnibase_infra.models.health.model_consumer_rejoin_alarm import (
    ModelConsumerRejoinAlarm,
)
from omnibase_infra.models.model_event_bus_recovery_config import (
    ModelEventBusRecoveryConfig,
)
from omnibase_infra.protocols.protocol_pattern_b_broker_transport import (
    ProtocolPatternBBrokerTransport,
)
from omnibase_infra.utils.util_error_sanitization import sanitize_error_message

logger = logging.getLogger(__name__)


_Unsubscribe = Callable[[], Awaitable[None]]
_Attach = Callable[[bool], Awaitable[_Unsubscribe]]
_CONNECT_ERRORS = (
    InfraConnectionError,
    InfraTimeoutError,
    InfraUnavailableError,
    TimeoutError,
)


class EventBusRecovering(EventBusKafka):
    """Runtime policy while Kafka is lost: refuse readiness or alarmed local delivery.

    ``recovery.lost_broker_policy`` decides. ``refuse_ready`` (the shipped
    default, for a runtime that serves other processes) never routes traffic
    into a private in-memory bus no other process can reach: the bus reports
    not ready, fails publishes, retains subscription intent and keeps retrying
    Kafka. ``local_fallback`` is for a runtime declared single-process and
    behaves as described below (OMN-20275).

    The kernel injects this object once. Its identity and subscription intent
    survive transport changes. Local receipts identify the in-memory transport;
    local deliveries are never replayed to Kafka. Shared-bus readiness remains
    false until producer startup and live subscription restoration complete.
    The kernel supplies local delivery through the existing broker transport
    protocol and binds lifecycle callbacks on its infra adapter. Kafka recovery
    therefore does not import or construct the local adapter.
    Standalone EventBusKafka clients keep their existing startup policy.
    """

    def __init__(
        self,
        *,
        config: ModelKafkaEventBusConfig,
        recovery: ModelEventBusRecoveryConfig,
        local_bus: ProtocolPatternBBrokerTransport,
        local_start: Callable[[], Awaitable[None]],
        local_close: Callable[[], Awaitable[None]],
        local_health_check: Callable[[], Awaitable[dict[str, object]]],
        alarm_topic: str,
        consumer_group: str,
    ) -> None:
        super().__init__(config=config)
        self._recovery_policy = recovery
        self._local_bus = local_bus
        self._local_start = local_start
        self._local_close = local_close
        self._local_health_check = local_health_check
        self._alarm_topic = alarm_topic
        self._recovery_group = consumer_group
        self._fallback_active = False
        # refuse_ready: Kafka was lost and nothing local stands in for it.
        self._broker_lost = False
        # refuse_ready: retained attachments that have no live Kafka consumer
        # yet after a restore; readiness stays false until this is empty.
        self._pending_attachments: set[str] = set()
        # A join of this recovery episode has succeeded on the current transport.
        self._kafka_joined_any = False
        self._recovery_closed = False
        self._recovery_task: asyncio.Task[None] | None = None
        self._kafka_close_task: asyncio.Task[None] | None = None
        self._transport_lock = asyncio.Lock()
        # OMN-18087: advanced under the transport lock each time the transport
        # switches to local delivery. A Kafka attach runs outside the lock, so
        # it reads this before joining and compares after: a change means the
        # consumer it made belongs to a transport that has been replaced.
        self._transport_generation = 0
        # Kafka joins still running outside the lock. Recovery restores only
        # once they have settled, so a join begun on the old transport cannot
        # hold a consumer reservation that a restore would then skip.
        self._kafka_joins_in_flight = 0
        self._kafka_joins_settled = asyncio.Event()
        self._kafka_joins_settled.set()
        self._attachments: dict[str, _Attach] = {}
        self._unsubscribes: dict[str, _Unsubscribe] = {}
        # Consumers attached on the current Kafka transport that have not yet
        # taken ownership (a restore that is still missing other attachments).
        # They survive between recovery attempts so a retry joins only the
        # attachments that failed, never the ones that already attached.
        self._restored: dict[str, _Unsubscribe] = {}
        # True once a recovery attempt has reconnected Kafka and the transport
        # has not been closed since; the next attempt then retries only the
        # failed attachments instead of tearing the transport down again.
        self._kafka_reconnected = False
        # Topic of each retained subscription, for the recovery log lines.
        self._attachment_topics: dict[str, str] = {}
        self._recovery_stopped = asyncio.Event()

    async def start(self) -> None:
        """Start Kafka or continue locally with exactly one reconnect worker."""
        async with self._transport_lock:
            if self._fallback_active or self._broker_lost:
                return
            self._recovery_closed = False
            self._recovery_stopped.clear()
            if self._local_fallback_allowed:
                await self._local_start()
            try:
                async with asyncio.timeout(
                    self._recovery_policy.attempt_timeout_seconds
                ):
                    await super().start()
            except _CONNECT_ERRORS as exc:
                await self._enter_fallback(exc)

    async def _enter_fallback(self, error: Exception) -> None:
        """Called under the transport lock; retain and signal one alarm per episode."""
        if self._fallback_active or self._broker_lost or self._recovery_closed:
            return
        self._transport_generation += 1
        self._kafka_reconnected = False
        self._kafka_joined_any = False
        self._restored.clear()
        if self._local_fallback_allowed:
            self._fallback_active = True
            await self._local_start()
            for key, attach in self._attachments.items():
                self._unsubscribes[key] = await attach(True)
            message = (
                "Runtime event bus degraded: using in-memory delivery while Kafka "
                f"recovers: {sanitize_error_message(error)}"
            )
        else:
            # Shared runtime: other processes cannot reach a private in-memory
            # bus, so nothing is delivered locally. The consumers of the lost
            # transport are closed with it; the subscription intent is kept.
            self._broker_lost = True
            self._unsubscribes.clear()
            self._pending_attachments = set(self._attachments)
            message = (
                "Runtime event bus lost Kafka: refusing readiness and retrying "
                "Kafka; no in-memory delivery for a runtime serving other "
                f"processes: {sanitize_error_message(error)}"
            )
        logger.critical(message)
        alarm = ModelConsumerRejoinAlarm(
            topic=self._alarm_topic,
            consumer_group=self._recovery_group,
            broker=self.bootstrap_servers,
            reason="runtime_event_bus_unavailable",
            coordinator_unavailable_seconds=0,
            failed_rejoins=1,
            message=message,
        )
        # Existing alarm path retains a bounded local record even when the
        # broker-backed health emitter cannot publish its mirror.
        try:
            async with asyncio.timeout(self._recovery_policy.attempt_timeout_seconds):
                await self._record_consumer_rejoin_alarm(alarm)
        except (OSError, RuntimeError, TimeoutError):
            logger.warning("Runtime bus alarm mirror failed", exc_info=True)
        self._recovery_task = asyncio.create_task(
            self._recover(), name="runtime-event-bus-recovery"
        )
        self._recovery_task.add_done_callback(self._log_recovery_task_end)

    @staticmethod
    def _log_recovery_task_end(task: asyncio.Task[None]) -> None:
        """Say so when the recovery worker ends by an error nobody awaited."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.critical(
                "Runtime event bus recovery worker ended with an error and is "
                "no longer retrying Kafka: %s",
                sanitize_error_message(
                    exc if isinstance(exc, Exception) else Exception(repr(exc))
                ),
                exc_info=exc,
            )

    _BROKER_LOST_PUBLISH_REFUSAL = (
        "Runtime event bus has lost Kafka and this runtime serves other "
        "processes: publish refused, not delivered to a private in-memory bus"
    )

    @property
    def _local_fallback_allowed(self) -> bool:
        """Only a runtime declared single-process may deliver in memory."""
        return (
            self._recovery_policy.lost_broker_policy
            is EnumEventBusLostBrokerPolicy.LOCAL_FALLBACK
        )

    def _unsubscribes_live(self) -> set[str]:
        """Keys whose Kafka consumer is already live and owned (refuse_ready only)."""
        if self._local_fallback_allowed:
            return set()
        return set(self._unsubscribes)

    async def _sleep_before_retry(self, delay: float) -> None:
        """Wait between bounded attempts; cancellation interrupts shutdown."""
        await asyncio.sleep(delay)

    async def _recover(self) -> None:
        delay = self._recovery_policy.initial_backoff_seconds
        episode_started = time.monotonic()
        attempt = 0
        while not self._recovery_closed:
            await self._sleep_before_retry(delay)
            attempt += 1
            attempt_started = time.monotonic()
            logger.info(
                "Runtime event bus recovery attempt %d starting "
                "(episode_seconds=%.0f, kafka_reconnected=%s, next_delay_seconds=%s)",
                attempt,
                attempt_started - episode_started,
                self._kafka_reconnected,
                delay,
            )
            try:
                if not self._kafka_reconnected:
                    async with asyncio.timeout(
                        self._recovery_policy.attempt_timeout_seconds
                    ):
                        # Do network I/O outside the transport lock: new local
                        # subscriptions must work even while a connect/join hangs.
                        await self._close_kafka_transport()
                        self._restored.clear()
                        self._kafka_joined_any = False
                        await super().start()
                        await self._kafka_joins_settled.wait()
                    self._kafka_reconnected = True
                failed = await self._restore_subscriptions()
                if failed:
                    # Kafka answered but some attachments did not join: keep
                    # the transport and retry only those on the next attempt.
                    if not self._kafka_joined_any:
                        self._kafka_reconnected = False
                    raise InfraUnavailableError(
                        f"{len(failed)} subscription(s) did not attach: "
                        + "; ".join(
                            f"{self._attachment_topics.get(key, key)}: "
                            f"{type(error).__name__}"
                            for key, error in failed.items()
                        )
                    )
                logger.info(
                    "Runtime event bus recovered; Kafka subscriptions restored "
                    "(attempt=%d, attempt_seconds=%.1f, episode_seconds=%.0f, "
                    "subscriptions=%d)",
                    attempt,
                    time.monotonic() - attempt_started,
                    time.monotonic() - episode_started,
                    len(self._unsubscribes),
                )
                return
            except asyncio.CancelledError:
                raise
            except (*_CONNECT_ERRORS, OSError, RuntimeError, ExceptionGroup) as exc:
                logger.warning(
                    "Runtime event bus recovery attempt %d failed after %.1fs "
                    "(%s): %s; retrying in %ss",
                    attempt,
                    time.monotonic() - attempt_started,
                    type(exc).__name__,
                    sanitize_error_message(exc),
                    self._next_delay(delay),
                )
            except Exception as exc:
                # Anything else is a defect, but a recovery loop that dies
                # silently leaves a shared runtime cut off: log it with its
                # traceback and keep retrying.
                logger.exception(
                    "Runtime event bus recovery attempt %d failed after %.1fs "
                    "with an unexpected %s; retrying in %ss",
                    attempt,
                    time.monotonic() - attempt_started,
                    type(exc).__name__,
                    self._next_delay(delay),
                )
            self._kafka_reconnected = self._kafka_reconnected and self._kafka_joined_any
            delay = self._next_delay(delay)

    def _next_delay(self, delay: float) -> float:
        """Exponential backoff capped by the recovery policy."""
        return min(
            delay * self._recovery_policy.backoff_multiplier,
            self._recovery_policy.max_backoff_seconds,
        )

    async def _restore_subscriptions(self) -> dict[str, Exception]:
        """Bound each join, isolate each failure, then atomically switch ownership.

        Each join runs on its own with its own deadline: one that fails or times
        out is logged and returned, and the others still attach and are kept for
        the next attempt, so a single bad join never fails the rest. Ownership
        switches off local delivery only once every retained subscription has a
        live Kafka consumer; until then the failed set is returned.

        Hundreds of contracts may legitimately take longer than one join's
        deadline in aggregate. New local subscriptions and removals are folded
        into the next snapshot without holding a lock across broker I/O.
        """
        semaphore = asyncio.Semaphore(self.config.consumer_start_concurrency)
        restored = self._restored
        failed: dict[str, Exception] = {}

        async def restore(key: str, attach: _Attach) -> None:
            async with semaphore:
                try:
                    async with asyncio.timeout(
                        self._recovery_policy.attempt_timeout_seconds
                    ):
                        restored[key] = await attach(False)
                        self._kafka_joined_any = True
                except asyncio.CancelledError:
                    raise
                except (*_CONNECT_ERRORS, OSError, RuntimeError) as exc:
                    failed[key] = exc
                    logger.warning(
                        "Runtime event bus subscription restore failed "
                        "(topic=%s, %s): %s",
                        self._attachment_topics.get(key, key),
                        type(exc).__name__,
                        sanitize_error_message(exc),
                    )
                except Exception as exc:
                    # One unexpected failure is recorded with its traceback and
                    # never fails the other joins.
                    failed[key] = exc
                    logger.exception(
                        "Runtime event bus subscription restore failed "
                        "unexpectedly (topic=%s, %s)",
                        self._attachment_topics.get(key, key),
                        type(exc).__name__,
                    )

        while True:
            async with self._transport_lock:
                attachments = dict(self._attachments)
            async with asyncio.TaskGroup() as group:
                for key, attach in attachments.items():
                    if key not in restored and key not in self._unsubscribes_live():
                        group.create_task(restore(key, attach))
            async with self._transport_lock:
                for key in failed.keys() - self._attachments.keys():
                    del failed[key]
                removed = set(restored) - self._attachments.keys()
                missing = (
                    self._attachments.keys()
                    - restored.keys()
                    - failed.keys()
                    - self._unsubscribes_live()
                )
                if not removed and not missing:
                    if not self._local_fallback_allowed:
                        # Shared runtime: nothing local owns these keys, so what
                        # attached is committed now and Kafka carries traffic;
                        # readiness stays false until the failed ones attach.
                        self._unsubscribes.update(restored)
                        restored.clear()
                        self._pending_attachments = set(failed)
                        self._broker_lost = False
                        return failed
                    if failed:
                        return failed
                    for unsubscribe in self._unsubscribes.values():
                        await unsubscribe()
                    self._unsubscribes = restored
                    self._restored = {}
                    self._fallback_active = False
                    return {}
            for key in removed:
                async with asyncio.timeout(
                    self._recovery_policy.attempt_timeout_seconds
                ):
                    await restored.pop(key)()

    async def _close_kafka_transport(self) -> None:
        """Let one teardown finish even if the recovery worker is cancelled.

        EventBusKafka.close marks shutdown before settling its resources; a
        second call cannot resume a cancelled first call. Keep that teardown
        owned here so runtime shutdown can await it after cancelling recovery.
        """
        if self._kafka_close_task is None or self._kafka_close_task.done():
            self._kafka_close_task = asyncio.create_task(
                super().close(), name="runtime-event-bus-close"
            )
        await asyncio.shield(self._kafka_close_task)

    async def _publish_locally(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None,
    ) -> ModelPublishReceipt:
        """Require the fallback transport to report its actual delivery coordinate."""
        receipt = await self._local_bus.publish(topic, key, value, headers)
        if receipt is None:
            raise InfraUnavailableError("Local fallback transport returned no receipt")
        return receipt

    async def publish(
        self,
        topic: str,
        key: bytes | None,
        value: bytes,
        headers: ModelEventHeaders | None = None,
    ) -> ModelPublishReceipt:
        """Deliver locally during an outage only when declared single-process.

        A shared runtime that has lost Kafka refuses the publish instead of
        routing it into an in-memory bus no other process can reach.
        """
        if self._fallback_active:
            return await self._publish_locally(topic, key, value, headers)
        if self._broker_lost:
            raise InfraUnavailableError(self._BROKER_LOST_PUBLISH_REFUSAL)
        try:
            return await super().publish(topic, key, value, headers)
        except _CONNECT_ERRORS as exc:
            if self._recovery_closed:
                raise
            async with self._transport_lock:
                await self._enter_fallback(exc)
            if self._fallback_active:
                return await self._publish_locally(topic, key, value, headers)
            raise

    async def subscribe(
        self,
        topic: str,
        node_identity: ModelNodeIdentity | None = None,
        on_message: Callable[[ModelEventMessage], Awaitable[None]] | None = None,
        *,
        group_id: str | None = None,
        purpose: EnumConsumerGroupPurpose = EnumConsumerGroupPurpose.CONSUME,
        required_for_readiness: bool = False,
        auto_offset_reset: str | None = None,
    ) -> ModelSubscriptionHandle:
        """Retain subscription intent and return an unsubscribe that survives recovery.

        Raises:
            InfraConnectionError, InfraTimeoutError: When Kafka is live and this
                subscription's consumer cannot start. The bus stays on Kafka.
        """

        async def attach(local: bool) -> _Unsubscribe:
            if local:
                return await self._local_bus.subscribe(
                    topic,
                    node_identity,
                    on_message,
                    group_id=group_id,
                    purpose=purpose,
                    required_for_readiness=required_for_readiness,
                )
            return await super(EventBusRecovering, self).subscribe(
                topic=topic,
                node_identity=node_identity,
                on_message=on_message,
                group_id=group_id,
                purpose=purpose,
                required_for_readiness=required_for_readiness,
                auto_offset_reset=auto_offset_reset,
            )

        key = str(uuid4())
        while True:
            async with self._transport_lock:
                if self._broker_lost:
                    # Shared runtime with Kafka lost: keep the intent only; the
                    # recovery loop attaches it when Kafka answers.
                    self._attachments[key] = attach
                    self._attachment_topics[key] = topic
                    self._pending_attachments.add(key)
                    break
                if self._fallback_active:
                    # Local delivery is in-memory, so attaching under the lock
                    # costs nothing and keeps the switch atomic.
                    self._unsubscribes[key] = await attach(True)
                    self._attachments[key] = attach
                    self._attachment_topics[key] = topic
                    break
                generation = self._transport_generation
                self._kafka_joins_in_flight += 1
                self._kafka_joins_settled.clear()
            try:
                unsubscribe = await self._join_kafka(attach, generation, key, topic)
            finally:
                async with self._transport_lock:
                    self._kafka_joins_in_flight -= 1
                    if self._kafka_joins_in_flight == 0:
                        self._kafka_joins_settled.set()
            if unsubscribe is not None:
                break

        async def pause() -> None:
            current = self._unsubscribes.get(key)
            if isinstance(current, ModelSubscriptionHandle):
                await current.pause()

        async def drain() -> None:
            current = self._unsubscribes.get(key)
            if isinstance(current, ModelSubscriptionHandle):
                await current.drain()

        async def stop() -> None:
            async with self._transport_lock:
                self._attachments.pop(key, None)
                self._attachment_topics.pop(key, None)
                self._pending_attachments.discard(key)
                current = self._unsubscribes.pop(key, None)
                if current is not None:
                    await current()

        return ModelSubscriptionHandle.bind(
            topic=topic,
            consumer_group=group_id or "",
            subscription_key=key,
            pause=pause,
            drain=drain,
            stop=stop,
        )

    async def _join_kafka(
        self,
        attach: _Attach,
        generation: int,
        key: str,
        topic: str,
    ) -> _Unsubscribe | None:
        """Join on Kafka outside the lock; record ownership if the transport held.

        Returns the unsubscribe once recorded, or None when the transport was
        replaced during the join and the caller must attach again.
        """
        # OMN-18087: the Kafka attach is a consumer-group join (about 3 s on
        # the dev broker, its group_initial_rebalance_delay). Holding the
        # transport lock across it serialised every subscription in the
        # process: 522 boot subscriptions took 2207 s on the dev lane on 2026-10-28
        # although auto-wiring attaches contracts concurrently. The join
        # runs outside the lock; ownership is recorded under it.
        #
        # With Kafka live, a failed attach is this subscription's own
        # failure (e.g. an unprovisioned topic): raise it to the caller as
        # EventBusKafka does. Degrading the whole bus would retain an
        # attachment that recovery can never restore.
        try:
            unsubscribe = await attach(False)
        except Exception:
            async with self._transport_lock:
                replaced = self._transport_generation != generation
                closed = self._recovery_closed
            if not replaced or closed:
                raise
            return None  # the transport changed under the join; attach again
        async with self._transport_lock:
            if self._transport_generation == generation:
                self._attachments[key] = attach
                self._attachment_topics[key] = topic
                self._unsubscribes[key] = unsubscribe
                return unsubscribe
            closed = self._recovery_closed
        # The transport was replaced (fallback) or closed while this join
        # ran, and whatever re-attached the retained subscriptions did not
        # see this one. Its consumer belongs to the old transport.
        try:
            await unsubscribe()
        except (*_CONNECT_ERRORS, OSError, RuntimeError):
            # A stale consumer's teardown never masks the outcome below.
            logger.warning(
                "Stale Kafka subscription teardown failed (topic=%s)",
                topic,
                exc_info=True,
            )
        if closed:
            raise InfraUnavailableError(
                f"Runtime event bus closed while subscribing to {topic}"
            )
        return None

    async def health_check(self) -> dict[str, object]:
        """Local delivery is live but explicitly degraded, never Kafka-ready."""
        if self._broker_lost or (
            self._pending_attachments and not self._local_fallback_allowed
        ):
            # Liveness stays green (the process is running and retrying) while
            # readiness is refused: the runtime keeps its place for the retry.
            return {
                "healthy": True,
                "degraded": True,
                "transport": "kafka",
                "configured_transport": "kafka",
                "broker_lost": self._broker_lost,
                "pending_attachments": len(self._pending_attachments),
                "recovery_running": self._recovery_task is not None
                and not self._recovery_task.done(),
            }
        if self._fallback_active:
            return {
                **await self._local_health_check(),
                "degraded": True,
                "transport": "inmemory",
                "configured_transport": "kafka",
                "recovery_running": self._recovery_task is not None
                and not self._recovery_task.done(),
            }
        return await super().health_check()

    async def get_readiness_status(self) -> ModelEventBusReadiness:
        """Do not advertise shared-bus readiness during fallback or a lost broker."""
        if self._broker_lost or (
            self._pending_attachments and not self._local_fallback_allowed
        ):
            return ModelEventBusReadiness(
                is_ready=False,
                consumers_started=False,
                required_topics_ready=False,
                last_error=(
                    "runtime event bus lost Kafka: not ready, retrying Kafka "
                    f"({len(self._pending_attachments)} subscription(s) pending)"
                ),
            )
        if self._fallback_active:
            return ModelEventBusReadiness(
                is_ready=False,
                consumers_started=False,
                required_topics_ready=False,
                last_error="runtime event bus degraded: in-memory fallback; recovering Kafka",
            )
        return await super().get_readiness_status()

    async def start_consuming(self) -> None:
        """Subscriptions attach eagerly; keep the protocol's lifetime wait alive."""
        await self.start()
        await self._recovery_stopped.wait()

    async def close(self) -> None:
        """Cancel recovery before closing transports so shutdown cannot reconnect."""
        self._recovery_closed = True
        task, self._recovery_task = self._recovery_task, None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        async with self._transport_lock:
            self._transport_generation += 1
            await self._close_kafka_transport()
            await self._local_close()
            self._attachments.clear()
            self._attachment_topics.clear()
            self._unsubscribes.clear()
            self._restored.clear()
            self._pending_attachments.clear()
            self._kafka_reconnected = False
            self._kafka_joined_any = False
            self._fallback_active = False
            self._broker_lost = False
            self._recovery_stopped.set()


__all__: list[str] = ["EventBusRecovering"]
