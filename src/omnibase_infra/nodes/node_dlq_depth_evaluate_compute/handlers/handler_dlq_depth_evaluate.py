# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Pure evaluator for DLQ depth and arrival rate (OMN-16769).

Why this is a separate COMPUTE node
-----------------------------------
The judgement ("is this sink taking traffic it should not be?") is a pure
function of broker offsets, declared bounds, and an injected clock. Keeping
it out of the probing EFFECT means every acceptance criterion on OMN-16769
that this node owns is testable with no broker, no network, and no
database — including the AC3 falsification case (16 quarantined commands
must alert), which would otherwise require reproducing a live outage.

Why arrivals, not depth, is the primary signal
----------------------------------------------
Measured on the .201 dev lane 2026-08-27:

    onex.dlq.omnibase-infra.quarantine.v1   log-start 6          HWM 8,878,932
    onex.dlq.omnibase-infra.events.v1       log-start 8,157,557  HWM 8,170,442

Two things follow, and both are load-bearing:

1. ``log_start_offset`` MOVES under retention. The high-water mark is a
   LIFETIME counter, not a depth. Reading ``events.v1`` as 8.17M deep
   would overstate its actual retained backlog (12,885) by ~634x.
2. ``quarantine.v1`` holds ~8.88M retained records right now. Any finite
   depth bound is therefore either already breached — alerting on every
   run forever, which AC4 rejects in as many words ("a depth alert that
   trips permanently on a pre-existing backlog is not an alert") — or set
   above 8.88M and thus never able to fire. Depth cannot gate the run.

So depth is REPORTED on every row (it is the number that makes the
standing backlog visible at a glance) but ARRIVALS gate the alert. A
quiet 8.88M backlog is correctly not an alert; sixteen new arrivals into
that same sink correctly is one.
"""

from __future__ import annotations

from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.enum_dlq_depth_verdict import (
    EnumDlqDepthVerdict,
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
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_topic_verdict import (
    ModelDlqTopicVerdict,
)

_SECONDS_PER_MINUTE = 60.0


class HandlerDlqDepthEvaluate:
    """Evaluate DLQ observations against contract-declared bounds. Pure."""

    @staticmethod
    def _override_reason_for(policy: ModelDlqThresholdPolicy, topic: str) -> str:
        for override in policy.overrides:
            if override.topic == topic:
                return override.reason
        return ""

    @classmethod
    def _evaluate_one(
        cls,
        observation: ModelDlqTopicObservation,
        request: ModelDlqDepthEvaluateRequest,
    ) -> ModelDlqTopicVerdict:
        policy = request.policy
        bound = policy.bound_for(observation.topic)
        arrivals = observation.arrivals_in_window
        depth = observation.retained_depth

        # Arrivals first: it is the primary signal, so it wins the verdict
        # slot when both would fire. An operator triaging a row that is
        # BOTH taking new traffic and deep needs to see the live symptom,
        # not the historical one.
        if arrivals > bound:
            verdict = EnumDlqDepthVerdict.ALERT_ARRIVALS
        elif (
            policy.max_retained_depth is not None and depth > policy.max_retained_depth
        ):
            verdict = EnumDlqDepthVerdict.ALERT_DEPTH
        else:
            verdict = EnumDlqDepthVerdict.OK

        windows_per_minute = policy.window_seconds / _SECONDS_PER_MINUTE

        return ModelDlqTopicVerdict(
            topic=observation.topic,
            partition_count=observation.partition_count,
            log_start_offset=observation.log_start_offset,
            high_watermark=observation.high_watermark,
            window_start_offset=observation.window_start_offset,
            retained_depth=depth,
            arrivals_in_window=arrivals,
            arrivals_per_minute=arrivals / windows_per_minute,
            max_arrivals_per_window=bound,
            override_reason=cls._override_reason_for(policy, observation.topic),
            verdict=verdict,
            window_seconds=policy.window_seconds,
            evaluated_at=request.evaluated_at,
        )

    async def handle(
        self, request: ModelDlqDepthEvaluateRequest
    ) -> ModelDlqDepthEvaluateResult:
        """Evaluate every observation; return the histogram + alert decision."""
        verdicts = tuple(
            sorted(
                (
                    self._evaluate_one(observation, request)
                    for observation in request.observations
                ),
                # Worst first, then alphabetical so the ordering is total and
                # the histogram is byte-stable across runs with equal traffic.
                key=lambda verdict: (-verdict.arrivals_in_window, verdict.topic),
            )
        )

        alerting = tuple(
            verdict
            for verdict in verdicts
            if verdict.verdict is not EnumDlqDepthVerdict.OK
        )

        return ModelDlqDepthEvaluateResult(
            correlation_id=request.correlation_id,
            evaluated_at=request.evaluated_at,
            window_seconds=request.policy.window_seconds,
            verdicts=verdicts,
            topics_observed=len(verdicts),
            topics_alerting=len(alerting),
            total_arrivals_in_window=sum(
                verdict.arrivals_in_window for verdict in verdicts
            ),
            total_retained_depth=sum(verdict.retained_depth for verdict in verdicts),
            alert_triggered=bool(alerting),
        )


__all__ = ["HandlerDlqDepthEvaluate"]
