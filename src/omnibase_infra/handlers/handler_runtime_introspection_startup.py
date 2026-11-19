# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Startup introspection of the runtime host (OMN-1930, OMN-21590).

After handlers are wired the runtime announces its presence. Two protections
apply: jitter (a random delay so many nodes restarting together do not publish
at once) and throttling (skip when the last introspection was too recent).

The runtime's own introspection service always publishes the runtime's event
first. With announcements, one more introspection event per ``announce`` record
is built from the announcement and the runtime's identity and published on the
introspection topic, so the registry reads the announced nodes.

OMN-24496: the heartbeat is mandatory. A request that demands a heartbeat
(``heartbeat_required``) refuses the startup instead of skipping: no
introspection service, introspection disabled by config, a startup publish
that keeps failing, or a first heartbeat that cannot be published all raise
the typed :class:`RuntimeHeartbeatRequiredError`. Every refusal is announced
on the runtime error topic (best-effort, via :meth:`publish_refusal_event`)
before it propagates, so a runtime whose heartbeat cannot be published is
visibly not healthy, never silently running. A request that does not demand
a heartbeat keeps the previous degrade-quietly behaviour.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, TypeVar, cast
from uuid import UUID, uuid4

from pydantic import BaseModel

from omnibase_core.enums import EnumCoreErrorCode
from omnibase_core.models.events.contract_registration.model_node_announcement import (
    ModelNodeAnnouncement,
)
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import (
    EnumHandlerType,
    EnumHandlerTypeCategory,
    EnumIntrospectionReason,
    EnumRuntimeHeartbeatRefusalReason,
)
from omnibase_infra.errors.error_infra import RuntimeHostError
from omnibase_infra.models.errors.model_infra_error_context import (
    ModelInfraErrorContext,
)
from omnibase_infra.models.health.enum_runtime_error_category import (
    EnumRuntimeErrorCategory,
)
from omnibase_infra.models.health.enum_runtime_error_severity import (
    EnumRuntimeErrorSeverity,
)
from omnibase_infra.models.health.model_runtime_error_event import (
    ModelRuntimeErrorEvent,
)
from omnibase_infra.models.model_runtime_heartbeat_health import (
    ModelRuntimeHeartbeatHealth,
)
from omnibase_infra.models.model_runtime_heartbeat_refusal import (
    ModelRuntimeHeartbeatRefusal,
)
from omnibase_infra.models.model_runtime_identity import ModelRuntimeIdentity
from omnibase_infra.models.model_runtime_introspection_startup_request import (
    ModelRuntimeIntrospectionStartupRequest,
)
from omnibase_infra.models.model_runtime_introspection_startup_result import (
    ModelRuntimeIntrospectionStartupResult,
)
from omnibase_infra.models.registration.model_node_introspection_event import (
    ModelNodeIntrospectionEvent,
)
from omnibase_infra.models.runtime import ModelRuntimeIntrospectionConfig
from omnibase_infra.models.runtime_timeouts import RUNTIME_TIMEOUTS
from omnibase_infra.protocols.protocol_event_bus_like import ProtocolEventBusLike
from omnibase_infra.protocols.protocol_node_introspection import (
    ProtocolNodeIntrospection,
)
from omnibase_infra.topics import SUFFIX_NODE_INTROSPECTION, SUFFIX_RUNTIME_ERROR

if TYPE_CHECKING:
    from omnibase_infra.mixins.mixin_node_introspection import MixinNodeIntrospection

logger = logging.getLogger(__name__)

_ServiceT = TypeVar("_ServiceT")


class RuntimeHeartbeatRequiredError(RuntimeHostError):
    """A runtime refused to start because its heartbeat could not run (OMN-24496).

    Raised at the refusal points of this handler and carries the refusal as a
    ``ModelRuntimeHeartbeatRefusal`` on ``.refusal``. The refusal records how
    many times this process has refused for this reason (``count``), so a
    crash loop that keeps hitting the same wall is visible in the refusal
    stream rather than only in log volume.
    """

    def __init__(
        self,
        message: str,
        refusal: ModelRuntimeHeartbeatRefusal,
        context: ModelInfraErrorContext | None = None,
        **extra_context: object,
    ) -> None:
        self.refusal = refusal
        super().__init__(
            message=message,
            error_code=EnumCoreErrorCode.INVALID_STATE,
            context=context,
            **extra_context,
        )


class HandlerRuntimeIntrospectionStartup:
    """Publish the runtime's startup introspection with jitter and throttling.

    The heartbeat is mandatory (OMN-24496): a runtime that cannot build its
    introspection service or start its heartbeat task refuses to start with a
    typed :class:`RuntimeHeartbeatRequiredError` instead of being announced
    without a heartbeat.
    """

    # Per-reason refusal counters for this process. A runtime that keeps
    # hitting the same wall in a crash loop increments its reason's counter on
    # every refused start, so the refusal stream shows the loop. Tests reset
    # this via :meth:`reset_refusal_counts`.
    _refusal_counts: dict[EnumRuntimeHeartbeatRefusalReason, int] = {}

    @classmethod
    def refusal_count(cls, reason: EnumRuntimeHeartbeatRefusalReason) -> int:
        """Return how many refusals of ``reason`` this process has made."""
        return cls._refusal_counts.get(reason, 0)

    @classmethod
    def reset_refusal_counts(cls) -> None:
        """Clear the per-reason refusal counters (test isolation)."""
        cls._refusal_counts.clear()

    def refusal_error(
        self,
        *,
        reason: EnumRuntimeHeartbeatRefusalReason,
        runtime_name: str,
        detail: str,
        correlation_id: UUID,
    ) -> RuntimeHeartbeatRequiredError:
        """Build the typed refusal for a runtime that cannot heartbeat.

        Increments the per-reason refusal counter, logs the refusal at error
        level, and returns the error the caller raises (``from exc``).

        Args:
            reason: Why the runtime cannot heartbeat
            runtime_name: Name of the runtime being refused
            detail: Human-readable failure detail behind the refusal
            correlation_id: Correlation ID of the refused startup attempt

        Returns:
            The typed refusal error carrying the counted refusal
        """
        count = self._refusal_counts.get(reason, 0) + 1
        self._refusal_counts[reason] = count
        refusal = ModelRuntimeHeartbeatRefusal(
            runtime_label=runtime_name,
            reason=reason,
            detail=detail,
            correlation_id=correlation_id,
            count=count,
        )
        logger.error(
            "Runtime heartbeat is mandatory: refusing to start",
            extra={"refusal": refusal.model_dump(mode="json")},
        )
        return RuntimeHeartbeatRequiredError(
            f"Runtime heartbeat is mandatory: refusing to start runtime "
            f"'{runtime_name}' ({reason.value}): {detail}",
            refusal=refusal,
        )

    async def publish_refusal_event(
        self,
        refusal: ModelRuntimeHeartbeatRefusal,
        event_bus: ProtocolEventBusLike | None,
    ) -> None:
        """Announce one heartbeat refusal as a runtime-error event (best-effort).

        OMN-24496: a runtime whose heartbeat cannot be published must be
        visibly not healthy, never silently running. The refusal lands on the
        runtime error topic so the refusal stream shows it even when the
        refused runtime never starts. Never raises: failing to announce a
        refusal is logged, not turned into a different crash.

        Args:
            refusal: The counted refusal to announce
            event_bus: The bus to publish on; ``None`` logs a warning only
        """
        if event_bus is None:
            logger.warning(
                "No event bus to announce the heartbeat refusal on",
                extra={"refusal": refusal.model_dump(mode="json")},
            )
            return
        try:
            event = ModelRuntimeErrorEvent.create(
                logger_family=__name__,
                log_level="CRITICAL",
                message_template=(f"runtime heartbeat refused: {refusal.reason.value}"),
                raw_message=(
                    f"runtime '{refusal.runtime_label}' heartbeat refused "
                    f"({refusal.reason.value}) count={refusal.count}: "
                    f"{refusal.detail}"
                ),
                error_category=EnumRuntimeErrorCategory.RUNTIME,
                severity=EnumRuntimeErrorSeverity.CRITICAL,
                correlation_id=refusal.correlation_id,
                service_label=refusal.runtime_label,
            )
            await self._publish_event(
                event,
                refusal.correlation_id,
                event_bus,
                topic=SUFFIX_RUNTIME_ERROR,
            )
        except Exception:
            logger.exception(
                "Failed to announce the heartbeat refusal",
                extra={
                    "runtime_label": refusal.runtime_label,
                    "reason": refusal.reason.value,
                    "correlation_id": str(refusal.correlation_id),
                },
            )

    async def build_required_service(
        self,
        *,
        runtime_name: str,
        correlation_id: UUID,
        build: Callable[[], _ServiceT],
        event_bus: ProtocolEventBusLike | None,
    ) -> _ServiceT:
        """Build the introspection service, refusing to start without one.

        Each refusal is announced on the runtime error topic (best-effort)
        before the typed error propagates, so the refusal is visible even
        when the refused runtime never starts.

        Args:
            runtime_name: Name of the runtime the service is built for
            correlation_id: Correlation ID of the startup attempt
            build: Zero-argument callable that builds the service
            event_bus: Bus the refusal event is announced on, if any

        Returns:
            The built introspection service

        Raises:
            RuntimeHeartbeatRequiredError: The runtime name is empty, or the
                service could not be built
        """
        if not runtime_name.strip():
            err = self.refusal_error(
                reason=EnumRuntimeHeartbeatRefusalReason.RUNTIME_NAME_EMPTY,
                runtime_name=runtime_name,
                detail="runtime_name is empty or whitespace",
                correlation_id=correlation_id,
            )
            await self.publish_refusal_event(err.refusal, event_bus)
            raise err
        try:
            return build()
        except Exception as exc:
            err = self.refusal_error(
                reason=(
                    EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_SERVICE_UNBUILDABLE
                ),
                runtime_name=runtime_name,
                detail=f"{type(exc).__name__}: {exc}",
                correlation_id=correlation_id,
            )
            await self.publish_refusal_event(err.refusal, event_bus)
            raise err from exc

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.INFRA_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    async def handle(
        self,
        request: ModelRuntimeIntrospectionStartupRequest,
        *,
        service: ProtocolNodeIntrospection | None,
        config: ModelRuntimeIntrospectionConfig,
        event_bus: ProtocolEventBusLike | None,
        build_events: Callable[
            [tuple[ModelNodeAnnouncement, ...], ModelRuntimeIdentity],
            tuple[ModelNodeIntrospectionEvent, ...],
        ],
    ) -> ModelRuntimeIntrospectionStartupResult:
        """Publish the startup introspection unless disabled or throttled.

        OMN-24496: when the request demands a heartbeat
        (``heartbeat_required``), every refusal on the way is announced on
        the runtime error topic (best-effort) before it propagates.
        """
        try:
            return await self._handle_startup(
                request,
                service=service,
                config=config,
                event_bus=event_bus,
                build_events=build_events,
            )
        except RuntimeHeartbeatRequiredError as exc:
            await self.publish_refusal_event(exc.refusal, event_bus)
            # A refused runtime must not keep a heartbeat task it started.
            with suppress(Exception):
                if service is not None:
                    await service.stop_heartbeat_task()
            raise

    async def _handle_startup(
        self,
        request: ModelRuntimeIntrospectionStartupRequest,
        *,
        service: ProtocolNodeIntrospection | None,
        config: ModelRuntimeIntrospectionConfig,
        event_bus: ProtocolEventBusLike | None,
        build_events: Callable[
            [tuple[ModelNodeAnnouncement, ...], ModelRuntimeIdentity],
            tuple[ModelNodeIntrospectionEvent, ...],
        ],
    ) -> ModelRuntimeIntrospectionStartupResult:
        """Publish the startup introspection unless disabled or throttled.

        When ``request.heartbeat_required`` is set the guards below refuse
        the startup instead of skipping; when it is not set the runtime
        degrades quietly exactly as before OMN-24496.
        """
        correlation_id = request.correlation_id
        required = request.heartbeat_required
        last_time = request.last_introspection_time
        not_published = ModelRuntimeIntrospectionStartupResult(
            published=False, last_introspection_time=last_time
        )

        # Guard: Skip if introspection not configured
        if service is None:
            if required:
                # No service means no heartbeat can ever run: refuse rather
                # than start a silent runtime.
                raise self.refusal_error(
                    reason=(
                        EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_SERVICE_MISSING
                    ),
                    runtime_name="unknown",
                    detail="no introspection service was given to the runtime",
                    correlation_id=correlation_id,
                )
            logger.debug("Introspection service not configured, skipping")
            return not_published

        if not config.enabled:
            if required:
                # Disabled introspection means no heartbeat task either.
                raise self.refusal_error(
                    reason=EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_DISABLED,
                    runtime_name=self._runtime_label(service),
                    detail="introspection is disabled by config",
                    correlation_id=correlation_id,
                )
            logger.debug("Introspection disabled by config, skipping")
            return not_published

        # Throttle check: Skip if last introspection was too recent
        if last_time is not None:
            elapsed = time.monotonic() - last_time
            if elapsed < config.throttle_min_interval_s:
                logger.debug(
                    "Introspection throttled",
                    extra={
                        "elapsed_seconds": elapsed,
                        "throttle_min_interval_s": (config.throttle_min_interval_s),
                    },
                )
                if required:
                    # Throttling suppresses only the startup introspection
                    # publish, never the heartbeat: a throttled startup still
                    # starts and confirms its heartbeat before returning.
                    await self.start_heartbeat(
                        correlation_id, service=service, config=config
                    )
                    await self._confirm_first_heartbeat(correlation_id, service=service)
                return not_published

        # Apply jitter: Random delay to prevent thundering herd
        jitter_ms = random.randint(0, config.jitter_max_ms)
        if jitter_ms > 0:
            logger.debug(
                "Applying introspection jitter",
                extra={"jitter_ms": jitter_ms},
            )
            await asyncio.sleep(jitter_ms / 1000.0)

        # Publish introspection event
        try:
            # OMN-24496: a required heartbeat bounds the startup publish with
            # the configured attempt budget and backoff; without the
            # requirement one failed attempt degrades quietly (as before).
            attempts = (
                RUNTIME_TIMEOUTS.introspection_heartbeat_start_attempts
                if required
                else 1
            )
            for attempt in range(1, attempts + 1):
                try:
                    await service.publish_introspection(
                        reason=EnumIntrospectionReason.STARTUP,
                        correlation_id=correlation_id,
                    )
                    break
                except Exception as retry_exc:
                    if not required or attempt == attempts:
                        raise
                    logger.warning(
                        "Startup introspection publish failed; retrying",
                        extra={
                            "attempt": attempt,
                            "max_attempts": attempts,
                            "error": str(retry_exc),
                            "correlation_id": str(correlation_id),
                        },
                    )
                    await asyncio.sleep(
                        RUNTIME_TIMEOUTS.introspection_heartbeat_retry_backoff_seconds
                    )

            # Record time for throttle tracking immediately after publish
            last_time = time.monotonic()

            logger.info(
                "Startup introspection published",
                extra={
                    "correlation_id": str(correlation_id),
                    "jitter_ms": jitter_ms,
                },
            )

        except Exception as e:
            if required:
                # A required runtime whose startup publish keeps failing
                # refuses to start instead of being announced silently.
                err = self.refusal_error(
                    reason=EnumRuntimeHeartbeatRefusalReason.STARTUP_PUBLISH_FAILED,
                    runtime_name=self._runtime_label(service),
                    detail=f"{type(e).__name__}: {e}",
                    correlation_id=correlation_id,
                )
                raise err from e
            # Log the failure but do not fail startup - introspection is optional
            logger.exception(
                "Failed to publish startup introspection",
                extra={
                    "error": str(e),
                    "correlation_id": str(correlation_id),
                },
            )
            return not_published

        # OMN-24496: The heartbeat is mandatory. It starts right after the
        # runtime's own presence is published and BEFORE any announced node is
        # published, outside the try above so a refusal propagates: no node is
        # announced for a runtime whose heartbeat cannot start.
        await self.start_heartbeat(correlation_id, service=service, config=config)
        if required:
            # A heartbeat task that starts but never lands a publish is still
            # a silent runtime: confirm the first heartbeat right now.
            await self._confirm_first_heartbeat(correlation_id, service=service)

        # Runtime presence has succeeded. Optional announced-node publishing
        # must not erase its throttle timestamp or suppress its heartbeat.
        if request.announcements and event_bus is not None:
            try:
                await self._publish_announced(service, request, event_bus, build_events)
            except Exception as e:
                logger.exception(
                    "Failed to publish announced node introspection",
                    extra={
                        "error": str(e),
                        "correlation_id": str(correlation_id),
                    },
                )
        return ModelRuntimeIntrospectionStartupResult(
            published=True, last_introspection_time=last_time
        )

    async def start_heartbeat(
        self,
        correlation_id: UUID,
        *,
        service: ProtocolNodeIntrospection | None,
        config: ModelRuntimeIntrospectionConfig,
    ) -> None:
        """Start the heartbeat task for periodic liveness announcements.

        The heartbeat is mandatory for kernel runtimes (OMN-24496): a failure
        to start it refuses the runtime's startup. Service ``None`` or
        introspection disabled by config returns silently — those are not
        kernel runtimes.
        """
        if service is None or not config.enabled:
            return
        try:
            await service.start_heartbeat_task()
        except Exception as exc:
            raise self.refusal_error(
                reason=EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_START_FAILED,
                runtime_name=self._runtime_label(service),
                detail=f"{type(exc).__name__}: {exc}",
                correlation_id=correlation_id,
            ) from exc

    def heartbeat_readiness(
        self,
        *,
        service: ProtocolNodeIntrospection | None,
        config: ModelRuntimeIntrospectionConfig,
    ) -> tuple[bool, dict[str, object]]:
        """Return the ``heartbeat`` readiness row of a runtime (OMN-24496).

        The row is ``(ready, detail)`` in the shape of the runtime's other
        supplemental readiness probes: false with the refusal reason while the
        heartbeat cannot run or keeps failing. A service that cannot report
        its heartbeat health is not ready, never assumed healthy.

        Args:
            service: The runtime's introspection service
            config: The runtime's introspection config

        Returns:
            ``(ready, detail)``; ``detail`` carries ``reason`` and ``summary``
            when not ready
        """
        if service is None:
            reason = EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_SERVICE_MISSING
            return False, {
                "reason": reason.value,
                "summary": "no introspection service, so no heartbeat",
            }
        if not config.enabled:
            reason = EnumRuntimeHeartbeatRefusalReason.INTROSPECTION_DISABLED
            return False, {
                "reason": reason.value,
                "summary": "introspection is disabled, so no heartbeat",
            }
        health_of = getattr(service, "heartbeat_health", None)
        health = health_of() if callable(health_of) else None
        if not isinstance(health, ModelRuntimeHeartbeatHealth):
            reason = EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_TASK_STOPPED
            return False, {
                "reason": reason.value,
                "summary": "the introspection service cannot report heartbeat health",
            }
        if health.healthy:
            return True, {"consecutive_failures": health.consecutive_failures}
        return False, {
            "reason": health.reason.value if health.reason else "unknown",
            "consecutive_failures": health.consecutive_failures,
            "summary": f"heartbeat not healthy: {health.detail}",
        }

    async def publish_beat(self, service: MixinNodeIntrospection) -> bool:
        """Publish one heartbeat of ``service``; its account counts the outcome.

        Args:
            service: The introspection service that owns the heartbeat

        Returns:
            True if the heartbeat was published, False if it failed
        """
        return await service._publish_heartbeat()

    def health_of(self, service: MixinNodeIntrospection) -> ModelRuntimeHeartbeatHealth:
        """Return the heartbeat health verdict of ``service`` (OMN-24559).

        Unhealthy while the heartbeat task is not running, once consecutive
        failures reach ``introspection_heartbeat_unhealthy_after_failures``, and
        until the first heartbeat has published; healthy only after a beat.

        Args:
            service: The introspection service whose heartbeat is judged

        Returns:
            The typed verdict, always carrying the failure count
        """
        account = service.heartbeat_account
        task = service._heartbeat_task
        reasons = EnumRuntimeHeartbeatRefusalReason
        reason: EnumRuntimeHeartbeatRefusalReason | None = None
        detail = ""
        if task is None or task.done():
            reason, detail = (
                reasons.HEARTBEAT_TASK_STOPPED,
                "heartbeat task is not running",
            )
        elif (
            account.consecutive_failures
            >= RUNTIME_TIMEOUTS.introspection_heartbeat_unhealthy_after_failures
        ):
            reason, detail = reasons.HEARTBEAT_PUBLISH_FAILING, account.last_error
        elif not account.published_once:
            reason, detail = (
                reasons.HEARTBEAT_PUBLISH_FAILING,
                "no heartbeat published yet",
            )
        return ModelRuntimeHeartbeatHealth(
            healthy=reason is None,
            reason=reason,
            detail=detail,
            consecutive_failures=account.consecutive_failures,
            published_once=account.published_once,
        )

    async def announce_failing_heartbeat(
        self, service: MixinNodeIntrospection, correlation_id: UUID
    ) -> None:
        """Announce a failing heartbeat streak as one refusal event (best-effort).

        Args:
            service: The introspection service whose heartbeats keep failing
            correlation_id: Correlation ID tying the event to this outage
        """
        err = self.refusal_error(
            reason=EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_PUBLISH_FAILING,
            runtime_name=self._runtime_label(service),
            detail=service.heartbeat_account.last_error,
            correlation_id=correlation_id,
        )
        await self.publish_refusal_event(
            err.refusal,
            cast("ProtocolEventBusLike | None", service._introspection_event_bus),
        )

    async def run_heartbeat_loop(
        self, service: MixinNodeIntrospection, interval: float
    ) -> None:
        """Publish heartbeats every ``interval`` seconds until the stop event is set.

        Moved here from ``MixinNodeIntrospection._heartbeat_loop`` (OMN-24559) so
        a failing heartbeat is visible: each failed publish is counted on the
        service's account, and the failure that reaches
        ``introspection_heartbeat_unhealthy_after_failures`` announces one
        refusal event for the outage (``==``, not ``>=``, so one outage is one
        event). The loop itself runs on, so a recovered bus heals the verdict.

        Args:
            service: The introspection service that owns the heartbeat
            interval: Time between heartbeats in seconds
        """
        node_id = service._introspection_node_id
        if service._introspection_stop_event is None:
            service._introspection_stop_event = asyncio.Event()
        stop_event = service._introspection_stop_event
        logger.info(
            f"Starting heartbeat loop for {node_id}",
            extra={"node_id": node_id, "interval_seconds": interval},
        )
        while not stop_event.is_set():
            loop_correlation_id = uuid4()
            published = True
            try:
                published = await self.publish_beat(service)
            except asyncio.CancelledError:
                logger.debug(
                    f"Heartbeat loop cancelled for {node_id}",
                    extra={
                        "node_id": node_id,
                        "correlation_id": str(loop_correlation_id),
                    },
                )
                break
            except Exception as e:
                logger.exception(
                    f"Error in heartbeat loop for {node_id}",
                    extra={
                        "node_id": node_id,
                        "correlation_id": str(loop_correlation_id),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )
            if (
                published is False
                and service.heartbeat_account.consecutive_failures
                == RUNTIME_TIMEOUTS.introspection_heartbeat_unhealthy_after_failures
            ):
                await self.announce_failing_heartbeat(service, loop_correlation_id)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                break
            except TimeoutError:
                pass
        logger.info(f"Heartbeat loop stopped for {node_id}", extra={"node_id": node_id})

    async def publish_capability_change(
        self,
        *,
        service: ProtocolNodeIntrospection | None,
        config: ModelRuntimeIntrospectionConfig,
        announced: set[str],
        node_name: str,
        correlation_id: UUID,
    ) -> None:
        """Publish a CAPABILITY_CHANGE introspection event for a live-materialized handler.

        Moved from ``RuntimeHostProcess._publish_capability_change`` (OMN-1989,
        OMN-24559). Idempotent: skips when ``node_name`` is already in
        ``announced``. Bypasses jitter and throttle: consumers reacting to
        CAPABILITY_CHANGE may send traffic right away. A no-op when there is
        no service or introspection is disabled.

        Args:
            service: The runtime's introspection service
            config: The runtime's introspection config
            announced: Node names already announced during this runtime boot
            node_name: Handler node name that was materialized
            correlation_id: Correlation ID for distributed tracing
        """
        if service is None or not config.enabled:
            return
        if node_name in announced:
            logger.debug(
                "CAPABILITY_CHANGE already announced, skipping",
                extra={"node_name": node_name, "correlation_id": str(correlation_id)},
            )
            return
        try:
            await service.publish_introspection(
                reason=EnumIntrospectionReason.CAPABILITY_CHANGE,
                correlation_id=correlation_id,
            )
            announced.add(node_name)
            logger.info(
                "CAPABILITY_CHANGE introspection published",
                extra={"node_name": node_name, "correlation_id": str(correlation_id)},
            )
        except Exception as e:
            logger.exception(
                "Failed to publish CAPABILITY_CHANGE introspection",
                extra={
                    "node_name": node_name,
                    "error": str(e),
                    "correlation_id": str(correlation_id),
                },
            )

    @staticmethod
    def _runtime_label(service: object | None) -> str:
        """Return the service's runtime label, falling back to ``unknown``.

        Args:
            service: The introspection service, or ``None``

        Returns:
            The service's introspection node name when it has one, else
            ``"unknown"``
        """
        name = getattr(service, "_introspection_node_name", None)
        return name if isinstance(name, str) and name else "unknown"

    async def _confirm_first_heartbeat(
        self,
        correlation_id: UUID,
        *,
        service: ProtocolNodeIntrospection,
    ) -> None:
        """Confirm the first heartbeat can actually be published.

        OMN-24496: a heartbeat task that starts but never lands a publish is
        still a silent runtime. This publishes one heartbeat immediately via
        the service's ``publish_heartbeat_now``, retrying within the
        configured startup attempt budget, and refuses the startup when none
        of the attempts lands.

        Args:
            correlation_id: Correlation ID of the startup attempt
            service: The introspection service whose heartbeat is confirmed

        Raises:
            RuntimeHeartbeatRequiredError: No attempt published a heartbeat
                within the configured attempt budget
        """
        attempts = RUNTIME_TIMEOUTS.introspection_heartbeat_start_attempts
        backoff = RUNTIME_TIMEOUTS.introspection_heartbeat_retry_backoff_seconds
        publish_now = getattr(service, "publish_heartbeat_now", None)
        for attempt in range(1, attempts + 1):
            detail = "service cannot publish a heartbeat on demand"
            if callable(publish_now):
                ok = await publish_now()
                if ok is True:
                    return
                detail = "the heartbeat publish did not succeed"
            logger.warning(
                "First heartbeat publish attempt failed",
                extra={
                    "attempt": attempt,
                    "max_attempts": attempts,
                    "runtime_label": self._runtime_label(service),
                    "detail": detail,
                    "correlation_id": str(correlation_id),
                },
            )
            if attempt < attempts:
                await asyncio.sleep(backoff)
        raise self.refusal_error(
            reason=EnumRuntimeHeartbeatRefusalReason.HEARTBEAT_PUBLISH_FAILED,
            runtime_name=self._runtime_label(service),
            detail=(
                f"the first heartbeat could not be published after {attempts} attempts"
            ),
            correlation_id=correlation_id,
        )

    async def _publish_announced(
        self,
        service: ProtocolNodeIntrospection,
        request: ModelRuntimeIntrospectionStartupRequest,
        event_bus: ProtocolEventBusLike,
        build_events: Callable[
            [tuple[ModelNodeAnnouncement, ...], ModelRuntimeIdentity],
            tuple[ModelNodeIntrospectionEvent, ...],
        ],
    ) -> None:
        """Publish one introspection event per announce record."""
        get_data = getattr(service, "get_introspection_data", None)
        if not callable(get_data):
            raise TypeError("introspection service cannot report its identity")
        data = await get_data()
        if not isinstance(data, ModelNodeIntrospectionEvent):
            raise TypeError("introspection service cannot report its identity")
        identity = ModelRuntimeIdentity.from_introspection_event(data)
        for event in build_events(request.announcements, identity):
            await self._publish_event(
                event, identity.node_id, event_bus, topic=SUFFIX_NODE_INTROSPECTION
            )

    async def _publish_event(
        self,
        payload: BaseModel,
        key_id: UUID,
        event_bus: ProtocolEventBusLike,
        *,
        topic: str,
    ) -> None:
        """Publish one payload on ``topic`` the way the introspection mixin does.

        Prefers ``publish_envelope`` when the bus has it, falling back to
        raw JSON bytes on ``publish``.

        Args:
            payload: The pydantic model to publish
            key_id: UUID the partition key is derived from; also used as the
                envelope correlation ID when the payload carries none
            event_bus: The bus to publish on
            topic: The topic suffix to publish on
        """
        payload_correlation_id = getattr(payload, "correlation_id", None)
        correlation_id = (
            payload_correlation_id
            if isinstance(payload_correlation_id, UUID)
            else key_id
        )
        if hasattr(event_bus, "publish_envelope"):
            envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
                payload=payload,
                correlation_id=correlation_id,
                tenant_id=None,
            )
            await event_bus.publish_envelope(envelope=envelope, topic=topic)
        else:
            await event_bus.publish(
                topic=topic,
                key=str(key_id).encode("utf-8"),
                value=json.dumps(payload.model_dump(mode="json")).encode("utf-8"),
            )


__all__ = ["HandlerRuntimeIntrospectionStartup", "RuntimeHeartbeatRequiredError"]
