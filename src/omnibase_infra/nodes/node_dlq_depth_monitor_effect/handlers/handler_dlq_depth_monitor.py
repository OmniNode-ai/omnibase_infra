# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Read-only DLQ depth/arrival probe (OMN-16769).

Enumerates every DLQ topic on the broker, reads three offsets per partition
(log-start, high-water mark, and the offset as of the window start), folds
them into one observation per topic, and hands the result to the pure
evaluator :class:`HandlerDlqDepthEvaluate`.

Read-only by construction
-------------------------
The only broker surface this handler can reach is
:class:`ProtocolDlqAdminTransport`, whose five methods are all reads. There
is no produce, commit, topic-mutation or consumer-group-mutation path
available to it — the scheduled workflow is dry-run-safe because the type
system says so, not because the caller remembered to pass a flag.

Why the window start comes from the broker, not from stored state
-----------------------------------------------------------------
Arrivals-in-window is measured as ``high_watermark - offset_at(now - window)``
using the broker's own offset-for-timestamp index, rather than by
differencing against a previously persisted snapshot. That choice matters:

* it is exact on the FIRST run, with no prior-state bootstrap;
* a missed, failed, or delayed tick cannot corrupt the next run's delta;
* it needs no database, so the monitor has no dependency on the migration
  state of the lane it is monitoring — which would be a circular
  reliability dependency for an observability surface.

The two sharp edges of that index are handled explicitly below: a ``None``
result (nothing at/after the timestamp) and a window start that retention
has already deleted.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import ModelInfraErrorContext, RuntimeHostError
from omnibase_infra.event_bus.dlq_offset_reader_kafka import (
    AiokafkaDlqOffsetReader,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.handlers.handler_dlq_depth_evaluate import (
    HandlerDlqDepthEvaluate,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_depth_evaluate_request import (
    ModelDlqDepthEvaluateRequest,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_depth_evaluate_result import (
    ModelDlqDepthEvaluateResult,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_threshold_policy import (
    ModelDlqThresholdPolicy,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_topic_observation import (
    ModelDlqTopicObservation,
)
from omnibase_infra.nodes.node_dlq_depth_monitor_effect.models.model_dlq_depth_monitor_request import (
    ModelDlqDepthMonitorRequest,
)
from omnibase_infra.nodes.node_dlq_depth_monitor_effect.models.model_dlq_depth_monitor_result import (
    ModelDlqDepthMonitorResult,
)
from omnibase_infra.protocols.protocol_dlq_admin_transport import (
    ProtocolDlqAdminTransport,
    TopicPartition,
)

logger = logging.getLogger(__name__)

_MILLIS_PER_SECOND = 1000
_BOOTSTRAP_ENV_VAR = "KAFKA_BOOTSTRAP_SERVERS"
_KILL_SWITCH_ENV_VAR = "ONEX_DLQ_MONITOR_DISABLED"


class HandlerDlqDepthMonitor:
    """Probe DLQ topic offsets read-only and evaluate them."""

    def __init__(
        self,
        transport: ProtocolDlqAdminTransport | None = None,
        *,
        evaluator: HandlerDlqDepthEvaluate | None = None,
        kill_switch_disabled: bool | None = None,
    ) -> None:
        """Inject a transport for tests; leave it None to build the live one.

        When ``transport`` is None the handler constructs (and owns the
        lifecycle of) an aiokafka-backed adapter from
        ``KAFKA_BOOTSTRAP_SERVERS``. When one is injected, the CALLER owns
        its lifecycle — the handler will not start or stop it.
        """
        self._transport = transport
        self._owns_transport = transport is None
        self._evaluator = evaluator or HandlerDlqDepthEvaluate()
        self._kill_switch_ctor = kill_switch_disabled

    async def _observe_topics(
        self,
        transport: ProtocolDlqAdminTransport,
        topics: list[str],
        window_start_ms: int,
    ) -> tuple[ModelDlqTopicObservation, ...]:
        """Fold per-partition offsets into one observation per topic."""
        partitions_by_topic: dict[str, list[TopicPartition]] = {}
        all_partitions: list[TopicPartition] = []
        for topic in topics:
            partition_ids = await transport.partitions_for_topic(topic)
            topic_partitions = [(topic, pid) for pid in partition_ids]
            if not topic_partitions:
                # A topic the broker lists but reports no partitions for is a
                # metadata race, not a zero-depth topic. Skipping it is
                # correct; reporting it as 0/0/0 would fabricate a reading.
                logger.warning(
                    "DLQ topic %s reported zero partitions — skipping rather "
                    "than materializing a fabricated zero observation.",
                    topic,
                )
                continue
            partitions_by_topic[topic] = topic_partitions
            all_partitions.extend(topic_partitions)

        if not all_partitions:
            return ()

        log_starts = await transport.beginning_offsets(all_partitions)
        high_watermarks = await transport.end_offsets(all_partitions)
        window_starts = await transport.offsets_for_times(
            dict.fromkeys(all_partitions, window_start_ms)
        )

        observations: list[ModelDlqTopicObservation] = []
        for topic, topic_partitions in partitions_by_topic.items():
            topic_log_start = 0
            topic_high_watermark = 0
            topic_window_start = 0
            for partition in topic_partitions:
                log_start = log_starts[partition]
                high_watermark = high_watermarks[partition]

                # SHARP EDGE 1 — `None` means "no record at or after the window
                # start", i.e. nothing arrived. Normalize to the high-water mark
                # so arrivals compute to 0. Treating None as 0 here would report
                # the topic's ENTIRE lifetime volume as one window's arrivals —
                # on quarantine.v1 that would be a phantom 8.88M-arrival alert
                # on every single run.
                resolved = window_starts.get(partition)
                window_start = high_watermark if resolved is None else resolved

                # SHARP EDGE 2 — retention may already have deleted the record
                # that was at the window start. Clamp forward to the log start
                # so the arrival count covers only records that still exist,
                # rather than including already-deleted ones.
                window_start = max(window_start, log_start)

                topic_log_start += log_start
                topic_high_watermark += high_watermark
                topic_window_start += window_start

            observations.append(
                ModelDlqTopicObservation(
                    topic=topic,
                    partition_count=len(topic_partitions),
                    log_start_offset=topic_log_start,
                    high_watermark=topic_high_watermark,
                    window_start_offset=topic_window_start,
                )
            )
        return tuple(observations)

    async def handle(
        self, request: ModelDlqDepthMonitorRequest
    ) -> ModelDlqDepthMonitorResult:
        """Sweep the broker's DLQ topics and evaluate them. Read-only.

        Owns the transport lifecycle only when it built the transport
        itself; an injected transport is the caller's to start and stop.
        """
        if self._kill_switch_engaged():
            logger.warning(
                "%s is set — DLQ depth monitor disabled, zero I/O performed.",
                _KILL_SWITCH_ENV_VAR,
            )
            return self._empty_result(request)

        if self._owns_transport:
            live_transport = self._build_live_transport()
            async with live_transport:
                return await self._sweep(live_transport, request)

        if self._transport is None:  # pragma: no cover - guarded by _owns_transport
            raise RuntimeHostError(
                "DLQ depth monitor has no transport to probe with.",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="dlq_depth_monitor_configure",
                ),
            )
        return await self._sweep(self._transport, request)

    def _kill_switch_engaged(self) -> bool:
        if self._kill_switch_ctor is not None:
            return self._kill_switch_ctor
        return bool(os.environ.get(_KILL_SWITCH_ENV_VAR, ""))

    @staticmethod
    def _build_live_transport() -> AiokafkaDlqOffsetReader:
        """Build the live adapter, failing loudly on missing configuration.

        No default bootstrap server: a silently-wrong broker would make the
        monitor report a clean bill of health for a lane it never contacted,
        which is a worse failure than not running at all.
        """
        bootstrap = os.environ.get(_BOOTSTRAP_ENV_VAR, "").strip()
        if not bootstrap:
            raise RuntimeHostError(
                f"{_BOOTSTRAP_ENV_VAR} is not set — refusing to probe an "
                "unspecified broker. A monitor that silently contacts the "
                "wrong lane reports a false clean bill of health.",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="dlq_depth_monitor_configure",
                ),
            )
        return AiokafkaDlqOffsetReader(bootstrap)

    def _empty_result(
        self, request: ModelDlqDepthMonitorRequest
    ) -> ModelDlqDepthMonitorResult:
        evaluated_at = datetime.now(tz=UTC)
        return ModelDlqDepthMonitorResult(
            correlation_id=request.correlation_id,
            evaluated_at=evaluated_at,
            window_seconds=request.window_seconds,
            topics_matched=0,
            evaluation=ModelDlqDepthEvaluateResult(
                correlation_id=request.correlation_id,
                evaluated_at=evaluated_at,
                window_seconds=request.window_seconds,
            ),
        )

    async def _sweep(
        self,
        transport: ProtocolDlqAdminTransport,
        request: ModelDlqDepthMonitorRequest,
    ) -> ModelDlqDepthMonitorResult:
        evaluated_at = datetime.now(tz=UTC)
        window_start_ms = int(
            (evaluated_at.timestamp() - request.window_seconds) * _MILLIS_PER_SECOND
        )

        all_topics = await transport.list_topics()
        dlq_topics = sorted(
            topic for topic in all_topics if topic.startswith(request.topic_prefix)
        )
        logger.info(
            "DLQ depth monitor: %d topic(s) match prefix %r; window=%ds.",
            len(dlq_topics),
            request.topic_prefix,
            request.window_seconds,
        )

        observations = await self._observe_topics(
            transport, dlq_topics, window_start_ms
        )

        evaluation = await self._evaluator.handle(
            ModelDlqDepthEvaluateRequest(
                correlation_id=request.correlation_id,
                observations=observations,
                policy=ModelDlqThresholdPolicy(
                    window_seconds=request.window_seconds,
                    default_max_arrivals_per_window=request.default_max_arrivals_per_window,
                    max_retained_depth=request.max_retained_depth,
                ),
                evaluated_at=evaluated_at,
            )
        )

        result = ModelDlqDepthMonitorResult(
            correlation_id=request.correlation_id,
            evaluated_at=evaluated_at,
            window_seconds=request.window_seconds,
            topics_matched=len(dlq_topics),
            evaluation=evaluation,
        )

        if evaluation.alert_triggered and not request.suppress_alert_exit:
            offenders = ", ".join(
                f"{verdict.topic} (+{verdict.arrivals_in_window} in "
                f"{request.window_seconds}s, bound {verdict.max_arrivals_per_window}, "
                f"retained {verdict.retained_depth})"
                for verdict in evaluation.alerting_verdicts
            )
            raise RuntimeHostError(
                f"DLQ arrival alert: {evaluation.topics_alerting} topic(s) "
                f"exceeded their declared bound — {offenders}",
                context=ModelInfraErrorContext.with_correlation(
                    transport_type=EnumInfraTransportType.KAFKA,
                    operation="dlq_depth_monitor",
                    correlation_id=request.correlation_id,
                ),
            )

        return result


__all__ = ["HandlerDlqDepthMonitor"]
