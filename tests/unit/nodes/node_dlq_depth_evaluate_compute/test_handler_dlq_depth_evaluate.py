# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""RED-first tests for the pure DLQ depth/arrival evaluator (OMN-16769).

The evaluator is a pure function, so every acceptance criterion this node
owns is expressible as a table-driven unit test with no broker, no clock,
and no database. The live-lane numbers baked into these tests are REAL
readings taken from the .201 dev lane on 2026-08-27 via
``rpk topic describe -p`` — not invented fixtures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.handlers.handler_dlq_depth_evaluate import (
    HandlerDlqDepthEvaluate,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.enum_dlq_depth_verdict import (
    EnumDlqDepthVerdict,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_depth_evaluate_request import (
    ModelDlqDepthEvaluateRequest,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_threshold_policy import (
    ModelDlqThresholdPolicy,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_topic_observation import (
    ModelDlqTopicObservation,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_topic_threshold_override import (
    ModelDlqTopicThresholdOverride,
)

pytestmark = pytest.mark.unit

_EVALUATED_AT = datetime(2026, 8, 27, 18, 30, 0, tzinfo=UTC)

# Real .201 dev-lane readings, 2026-08-27 (rpk topic describe -p).
_QUARANTINE = "onex.dlq.omnibase-infra.quarantine.v1"
_EVENTS = "onex.dlq.omnibase-infra.events.v1"


def _observation(
    topic: str,
    *,
    log_start: int,
    hwm: int,
    window_start: int,
    partitions: int = 1,
) -> ModelDlqTopicObservation:
    return ModelDlqTopicObservation(
        topic=topic,
        partition_count=partitions,
        log_start_offset=log_start,
        high_watermark=hwm,
        window_start_offset=window_start,
    )


def _request(
    *observations: ModelDlqTopicObservation,
    policy: ModelDlqThresholdPolicy | None = None,
) -> ModelDlqDepthEvaluateRequest:
    return ModelDlqDepthEvaluateRequest(
        correlation_id=uuid4(),
        observations=observations,
        policy=policy or ModelDlqThresholdPolicy(),
        evaluated_at=_EVALUATED_AT,
    )


class TestOmn16767ReproductionMustAlert:
    """AC3's own falsification test, encoded literally."""

    async def test_sixteen_quarantined_commands_in_one_window_alerts(self) -> None:
        """AC3: 'falsified by ... 16 quarantined commands in one minute with no alert'."""
        request = _request(
            _observation(
                _QUARANTINE,
                log_start=6,
                hwm=8_878_948,
                window_start=8_878_932,  # 16 arrivals during the window
            )
        )

        result = await HandlerDlqDepthEvaluate().handle(request)

        assert result.alert_triggered is True
        assert result.topics_alerting == 1
        verdict = result.verdicts[0]
        assert verdict.arrivals_in_window == 16
        assert verdict.verdict is EnumDlqDepthVerdict.ALERT_ARRIVALS


class TestStandingBacklogDoesNotPermanentlyAlert:
    """AC4: 'a depth alert that trips permanently on a pre-existing backlog is not an alert'."""

    async def test_88m_backlog_with_zero_arrivals_is_ok(self) -> None:
        """The real 8.88M quarantine backlog, quiet, must NOT alert."""
        request = _request(
            _observation(
                _QUARANTINE,
                log_start=6,
                hwm=8_878_932,
                window_start=8_878_932,  # nothing new arrived
            )
        )

        result = await HandlerDlqDepthEvaluate().handle(request)

        assert result.alert_triggered is False
        verdict = result.verdicts[0]
        assert verdict.verdict is EnumDlqDepthVerdict.OK
        assert verdict.arrivals_in_window == 0
        # Depth is still REPORTED — it is context, just not the gate.
        assert verdict.retained_depth == 8_878_926

    async def test_depth_bound_is_disabled_by_default(self) -> None:
        """Default policy must carry no depth bound at all."""
        assert ModelDlqThresholdPolicy().max_retained_depth is None

    async def test_depth_bound_alerts_only_when_operator_opts_in(self) -> None:
        request = _request(
            _observation(
                _QUARANTINE, log_start=6, hwm=8_878_932, window_start=8_878_932
            ),
            policy=ModelDlqThresholdPolicy(max_retained_depth=1_000_000),
        )

        result = await HandlerDlqDepthEvaluate().handle(request)

        assert result.alert_triggered is True
        assert result.verdicts[0].verdict is EnumDlqDepthVerdict.ALERT_DEPTH


class TestRetentionTrimmedTopicIsMeasuredCorrectly:
    """The events.v1 finding: log-start moves, so HWM alone is not depth."""

    async def test_retained_depth_uses_log_start_not_hwm(self) -> None:
        """Real reading: log-start 8,157,557 / HWM 8,170,442 -> depth 12,885, not 8.17M."""
        request = _request(
            _observation(
                _EVENTS,
                log_start=8_157_557,
                hwm=8_170_442,
                window_start=8_170_442,
            )
        )

        result = await HandlerDlqDepthEvaluate().handle(request)

        verdict = result.verdicts[0]
        assert verdict.retained_depth == 12_885
        assert verdict.high_watermark == 8_170_442


class TestPerTopicOverrides:
    """AC3: bounds are contract-declared, and allowances must be justified."""

    async def test_override_raises_the_bound_for_one_topic_only(self) -> None:
        policy = ModelDlqThresholdPolicy(
            overrides=(
                ModelDlqTopicThresholdOverride(
                    topic=_EVENTS,
                    max_arrivals_per_window=500,
                    reason=(
                        "events.v1 carries measured standing traffic on the dev "
                        "lane; bound set above the observed rate so the sink is "
                        "watched for a step change rather than alerting always."
                    ),
                    ratify_by="when the events.v1 producers are attributed and fixed",
                ),
            )
        )
        request = _request(
            _observation(_EVENTS, log_start=0, hwm=400, window_start=0),
            _observation(_QUARANTINE, log_start=0, hwm=400, window_start=0),
            policy=policy,
        )

        result = await HandlerDlqDepthEvaluate().handle(request)

        by_topic = {verdict.topic: verdict for verdict in result.verdicts}
        assert by_topic[_EVENTS].verdict is EnumDlqDepthVerdict.OK
        assert by_topic[_EVENTS].max_arrivals_per_window == 500
        assert by_topic[_EVENTS].override_reason != ""
        # The default-bound topic with identical traffic still alerts.
        assert by_topic[_QUARANTINE].verdict is EnumDlqDepthVerdict.ALERT_ARRIVALS
        assert result.alert_triggered is True

    def test_override_without_a_real_reason_is_rejected(self) -> None:
        """No silent allowances — a stub reason must not parse."""
        with pytest.raises(ValidationError):
            ModelDlqTopicThresholdOverride(
                topic=_EVENTS,
                max_arrivals_per_window=500,
                reason="noisy",
                ratify_by="later",
            )

    def test_duplicate_overrides_are_rejected(self) -> None:
        override = ModelDlqTopicThresholdOverride(
            topic=_EVENTS,
            max_arrivals_per_window=1,
            reason="a sufficiently long and genuine explanation of the traffic",
            ratify_by="when the producer is fixed",
        )
        with pytest.raises(ValidationError):
            ModelDlqThresholdPolicy(overrides=(override, override))


class TestHistogramOrdering:
    """The operator reads this top-down; worst must be first."""

    async def test_verdicts_are_ordered_by_arrivals_descending(self) -> None:
        request = _request(
            _observation(
                "onex.dlq.omnibase-infra.a.v1", log_start=0, hwm=5, window_start=0
            ),
            _observation(
                "onex.dlq.omnibase-infra.b.v1", log_start=0, hwm=90, window_start=0
            ),
            _observation(
                "onex.dlq.omnibase-infra.c.v1", log_start=0, hwm=40, window_start=0
            ),
        )

        result = await HandlerDlqDepthEvaluate().handle(request)

        assert [verdict.arrivals_in_window for verdict in result.verdicts] == [
            90,
            40,
            5,
        ]

    async def test_ties_break_on_topic_name_for_determinism(self) -> None:
        request = _request(
            _observation(
                "onex.dlq.omnibase-infra.z.v1", log_start=0, hwm=7, window_start=0
            ),
            _observation(
                "onex.dlq.omnibase-infra.a.v1", log_start=0, hwm=7, window_start=0
            ),
        )

        result = await HandlerDlqDepthEvaluate().handle(request)

        assert [verdict.topic for verdict in result.verdicts] == [
            "onex.dlq.omnibase-infra.a.v1",
            "onex.dlq.omnibase-infra.z.v1",
        ]


class TestRunLevelAggregates:
    async def test_totals_and_counts(self) -> None:
        request = _request(
            _observation(
                _QUARANTINE, log_start=6, hwm=8_878_948, window_start=8_878_932
            ),
            _observation(
                _EVENTS, log_start=8_157_557, hwm=8_170_442, window_start=8_170_442
            ),
        )

        result = await HandlerDlqDepthEvaluate().handle(request)

        assert result.topics_observed == 2
        assert result.topics_alerting == 1
        assert result.total_arrivals_in_window == 16
        # quarantine: 8,878,948 - 6 = 8,878,942 (this case has the 16 arrivals
        # already appended, so its depth is 16 above the quiet-baseline case).
        assert result.total_retained_depth == 8_878_942 + 12_885
        assert len(result.alerting_verdicts) == 1

    async def test_empty_observation_set_does_not_alert(self) -> None:
        """No topics is not an alert — it is a separate (probe) failure mode."""
        result = await HandlerDlqDepthEvaluate().handle(_request())

        assert result.alert_triggered is False
        assert result.topics_observed == 0

    async def test_arrivals_per_minute_normalizes_to_the_window(self) -> None:
        request = _request(
            _observation(_QUARANTINE, log_start=0, hwm=60, window_start=0),
            policy=ModelDlqThresholdPolicy(window_seconds=1800),
        )

        result = await HandlerDlqDepthEvaluate().handle(request)

        assert result.verdicts[0].arrivals_per_minute == pytest.approx(2.0)


class TestPurity:
    """def-B purity: same request in, same result out. No clock, no I/O."""

    async def test_evaluation_is_deterministic(self) -> None:
        request = _request(
            _observation(
                _QUARANTINE, log_start=6, hwm=8_878_948, window_start=8_878_932
            )
        )

        first = await HandlerDlqDepthEvaluate().handle(request)
        second = await HandlerDlqDepthEvaluate().handle(request)

        assert first == second

    async def test_evaluated_at_is_carried_from_the_request_not_generated(self) -> None:
        request = _request(
            _observation(
                _QUARANTINE, log_start=6, hwm=8_878_932, window_start=8_878_932
            )
        )

        result = await HandlerDlqDepthEvaluate().handle(request)

        assert result.evaluated_at == _EVALUATED_AT
        assert result.verdicts[0].evaluated_at == _EVALUATED_AT

    def test_naive_evaluated_at_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ModelDlqDepthEvaluateRequest(
                correlation_id=uuid4(),
                observations=(),
                evaluated_at=datetime(2026, 8, 27, 18, 30, 0),
            )


class TestImpossibleOffsetTriplesFailClosed:
    """A garbage reading must not be materialized as a confident zero."""

    def test_hwm_behind_log_start_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _observation(_QUARANTINE, log_start=100, hwm=50, window_start=50)

    def test_window_start_ahead_of_hwm_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _observation(_QUARANTINE, log_start=0, hwm=50, window_start=80)

    def test_window_start_behind_log_start_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _observation(_QUARANTINE, log_start=100, hwm=200, window_start=50)

    def test_duplicate_topic_observations_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _request(
                _observation(_QUARANTINE, log_start=0, hwm=1, window_start=0),
                _observation(_QUARANTINE, log_start=0, hwm=2, window_start=0),
            )
