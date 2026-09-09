# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The alerting run must carry its own evidence (OMN-18088).

WHAT THIS EXISTS TO STOP HAPPENING AGAIN
----------------------------------------
OMN-17163 fixed the runner, the address, the transport and the drift guard, so
the monitor finally reached the .201 dev-lane broker. Scheduled run
``34399417506`` then authenticated over SCRAM-SHA-256, matched 67
``onex.dlq.*`` topics, measured every one of them -- and went red carrying
**nothing**::

    "result": {"workflow_result": "failed", "exit_code": 1,
               "terminal_payload": null, "handler_result": null, "error": ""}

No topic named, no depth, no reason. Meanwhile run ``34387859397`` at the same
head, differing only by ``suppress_alert_exit=true``, returned the full
histogram: 67 topics observed, 3 alerting, 250517 arrivals in window, 3280496
retained.

The alerting run BUILT that histogram and threw it away. Three layers
conspired, and only the first is ours to fix:

1. the handler raised ``RuntimeHostError`` with the offender list formatted
   into its message, instead of returning the result it had already built;
2. ``RuntimeLocal`` caught that exception, recorded ``result=failed`` and did
   NOT re-raise -- so ``receipt_mode``'s ``except Exception`` never fired and
   ``runtime_error`` stayed ``""``;
3. ``receipt_mode``'s typed-result branch is gated on ``status.is_success_like``
   (``receipt_mode.py:951-955``), so the failed run took the runtime-summary
   branch, where ``handler_result`` and ``terminal_payload`` were both already
   ``None``.

The message carrying the offender list existed only inside the exception, and
nothing between those layers wrote it down.

THE FIX THESE TESTS PIN
-----------------------
The alert is a **value**, not a control-flow event. ``_sweep`` returns the same
typed result on both paths and states the gating decision once, in
``alert_exit_requested``. The alerting run and the characterization run then
write the IDENTICAL file, and the workflow takes its non-zero exit by reading
that field rather than by an exception surviving two layers of runtime.

Every assertion that a value is present is paired with the shape the defect
produced, so a regression cannot pass by returning an empty-but-well-typed
result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.enum_dlq_depth_verdict import (
    EnumDlqDepthVerdict,
)
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.model_dlq_depth_evaluate_result import (
    ModelDlqDepthEvaluateResult,
)
from omnibase_infra.nodes.node_dlq_depth_monitor_effect.handlers.handler_dlq_depth_monitor import (
    HandlerDlqDepthMonitor,
)
from omnibase_infra.nodes.node_dlq_depth_monitor_effect.models.model_dlq_depth_monitor_request import (
    ModelDlqDepthMonitorRequest,
)
from omnibase_infra.nodes.node_dlq_depth_monitor_effect.models.model_dlq_depth_monitor_result import (
    ModelDlqDepthMonitorResult,
)
from omnibase_infra.protocols.protocol_dlq_admin_transport import TopicPartition

pytestmark = pytest.mark.unit

# The three topics that actually alerted on run 34387859397, with that run's
# own numbers. Using the live shape rather than invented ones keeps the test
# honest about what an alerting sweep looks like.
_EVENTS = "onex.dlq.omnibase-infra.events.v1"
_COMMANDS = "onex.dlq.omnibase-infra.commands.v1"
_QUARANTINE = "onex.dlq.omnibase-infra.quarantine.v1"
_QUIET = "onex.dlq.omnimarket.node-aislop-sweep.v1"


class FakeDlqAdminTransport:
    """Structural stand-in for ``ProtocolDlqAdminTransport``.

    ``topics`` maps topic -> partition -> ``(log_start, high_watermark,
    window_start)``. A ``None`` window start means the broker's
    offset-for-timestamp index found no record at or after the window, which
    the handler normalizes to the high-water mark.
    """

    def __init__(
        self, topics: Mapping[str, Mapping[int, tuple[int, int, int | None]]]
    ) -> None:
        self._topics = topics

    async def list_topics(self) -> Sequence[str]:
        return sorted(self._topics)

    async def partitions_for_topic(self, topic: str) -> Sequence[int]:
        return sorted(self._topics.get(topic, {}))

    async def beginning_offsets(
        self, partitions: Sequence[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        return {tp: self._topics[tp[0]][tp[1]][0] for tp in partitions}

    async def end_offsets(
        self, partitions: Sequence[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        return {tp: self._topics[tp[0]][tp[1]][1] for tp in partitions}

    async def offsets_for_times(
        self, partition_timestamps: Mapping[TopicPartition, int]
    ) -> Mapping[TopicPartition, int | None]:
        return {tp: self._topics[tp[0]][tp[1]][2] for tp in partition_timestamps}


def _alerting_transport() -> FakeDlqAdminTransport:
    """Three breaching sinks and one quiet one, from run 34387859397."""
    return FakeDlqAdminTransport(
        topics={
            _EVENTS: {0: (18_703_776, 21_011_921, 20_761_539)},
            _COMMANDS: {0: (1_000, 2_535, 2_465)},
            _QUARANTINE: {0: (7_908_402, 8_878_948, 8_878_883)},
            # Quiet: nothing arrived in the window, so it must NOT be an
            # offender. Without this row an "all topics are offenders" bug
            # would pass every assertion below.
            _QUIET: {0: (0, 4_000, None)},
        }
    )


def _request(**overrides: object) -> ModelDlqDepthMonitorRequest:
    payload: dict[str, object] = {"correlation_id": uuid4()}
    payload.update(overrides)
    return ModelDlqDepthMonitorRequest(**payload)  # type: ignore[arg-type]


class TestTheAlertingRunReturnsItsEvidence:
    """The defect: the alert path produced no payload at all."""

    async def test_alert_path_returns_a_result_instead_of_raising(self) -> None:
        """RED at the parent sha: this raises ``RuntimeHostError`` instead."""
        result = await HandlerDlqDepthMonitor(_alerting_transport()).handle(
            _request(suppress_alert_exit=False)
        )

        assert isinstance(result, ModelDlqDepthMonitorResult)
        assert result.alert_triggered is True
        assert result.alert_exit_requested is True

    async def test_the_returned_result_names_every_offender(self) -> None:
        result = await HandlerDlqDepthMonitor(_alerting_transport()).handle(
            _request(suppress_alert_exit=False)
        )

        offenders = {v.topic for v in result.evaluation.alerting_verdicts}
        assert offenders == {_EVENTS, _COMMANDS, _QUARANTINE}
        # POSITIVE CONTROL for the exclusion: the quiet topic is observed and
        # reported, it is simply not an offender. An empty offender set and a
        # set containing everything are both wrong, and only checking one
        # direction cannot tell them apart.
        assert _QUIET in {v.topic for v in result.evaluation.verdicts}
        assert _QUIET not in offenders

    async def test_every_offender_carries_arrivals_bound_and_retained_depth(
        self,
    ) -> None:
        """AC1's summary cannot name a depth the payload does not carry."""
        result = await HandlerDlqDepthMonitor(_alerting_transport()).handle(
            _request(suppress_alert_exit=False)
        )

        by_topic = {v.topic: v for v in result.evaluation.alerting_verdicts}
        events = by_topic[_EVENTS]
        assert events.arrivals_in_window == 250_382
        assert events.retained_depth == 2_308_145
        assert events.max_arrivals_per_window == 0
        assert events.verdict is EnumDlqDepthVerdict.ALERT_ARRIVALS

    async def test_the_serialized_payload_is_not_the_null_shape(self) -> None:
        """Pins the exact JSON the red run wrote, as the thing to never emit."""
        result = await HandlerDlqDepthMonitor(_alerting_transport()).handle(
            _request(suppress_alert_exit=False)
        )

        payload = result.model_dump(mode="json")
        assert payload["evaluation"]["verdicts"], (
            "run 34399417506 emitted handler_result=null; a payload with an "
            "empty histogram is the same defect wearing a type."
        )
        assert payload["alert_exit_requested"] is True
        assert payload["evaluation"]["topics_alerting"] == 3

    async def test_the_depth_bound_alert_also_returns_rather_than_raising(
        self,
    ) -> None:
        """The secondary bound took the identical raise, and the same fix."""
        quiet_but_deep = FakeDlqAdminTransport(
            topics={_QUARANTINE: {0: (6, 8_878_932, None)}}
        )

        result = await HandlerDlqDepthMonitor(quiet_but_deep).handle(
            _request(suppress_alert_exit=False, max_retained_depth=1_000_000)
        )

        assert result.alert_exit_requested is True
        assert (
            result.evaluation.alerting_verdicts[0].verdict
            is EnumDlqDepthVerdict.ALERT_DEPTH
        )


class TestBothPathsWriteTheSameFile:
    """AC2 -- the characterization run is the reference shape, unchanged."""

    async def test_suppressed_run_still_returns_the_histogram_and_does_not_gate(
        self,
    ) -> None:
        result = await HandlerDlqDepthMonitor(_alerting_transport()).handle(
            _request(suppress_alert_exit=True)
        )

        assert result.alert_triggered is True
        assert result.alert_exit_requested is False
        assert len(result.evaluation.alerting_verdicts) == 3

    async def test_the_two_paths_agree_on_every_field_but_the_gating_decision(
        self,
    ) -> None:
        """The whole point: one shape, so one renderer reads both runs."""
        alerting = await HandlerDlqDepthMonitor(_alerting_transport()).handle(
            _request(suppress_alert_exit=False)
        )
        suppressed = await HandlerDlqDepthMonitor(_alerting_transport()).handle(
            _request(suppress_alert_exit=True)
        )

        alerting_payload = alerting.model_dump(mode="json")
        suppressed_payload = suppressed.model_dump(mode="json")
        assert set(alerting_payload) == set(suppressed_payload)

        differing = {
            key
            for key in alerting_payload
            if alerting_payload[key] != suppressed_payload[key]
            # Both are per-run values that are expected to differ.
            and key not in {"correlation_id", "evaluated_at", "evaluation"}
        }
        assert differing == {"alert_exit_requested", "suppress_alert_exit"}

        # Same histogram, row for row. `evaluated_at` is stamped per run and
        # is the one field two independent sweeps are expected to differ on.
        def _rows(payload: dict[str, object]) -> list[dict[str, object]]:
            evaluation = payload["evaluation"]
            assert isinstance(evaluation, dict)
            verdicts = evaluation["verdicts"]
            assert isinstance(verdicts, list)
            return [
                {k: v for k, v in row.items() if k != "evaluated_at"}
                for row in verdicts
            ]

        assert _rows(alerting_payload) == _rows(suppressed_payload)


class TestTheGatingDecisionCannotBeStatedWrongly:
    """A field a caller can set independently is a field that can lie."""

    @staticmethod
    def _evaluation(*, alert_triggered: bool) -> ModelDlqDepthEvaluateResult:
        return ModelDlqDepthEvaluateResult(
            correlation_id=uuid4(),
            evaluated_at=datetime.now(tz=UTC),
            window_seconds=1800,
            alert_triggered=alert_triggered,
        )

    def test_claiming_no_gate_on_an_alerting_evaluation_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="disagrees with its own evidence"):
            ModelDlqDepthMonitorResult(
                correlation_id=uuid4(),
                evaluated_at=datetime.now(tz=UTC),
                window_seconds=1800,
                topics_matched=1,
                evaluation=self._evaluation(alert_triggered=True),
                suppress_alert_exit=False,
                alert_exit_requested=False,
            )

    def test_claiming_a_gate_on_a_clean_evaluation_is_refused(self) -> None:
        """The other direction: a false alarm is as wrong as a missed one."""
        with pytest.raises(ValidationError, match="disagrees with its own evidence"):
            ModelDlqDepthMonitorResult(
                correlation_id=uuid4(),
                evaluated_at=datetime.now(tz=UTC),
                window_seconds=1800,
                topics_matched=1,
                evaluation=self._evaluation(alert_triggered=False),
                suppress_alert_exit=False,
                alert_exit_requested=True,
            )

    def test_the_consistent_pair_constructs(self) -> None:
        """POSITIVE CONTROL: the validator refuses lies, not every value."""
        result = ModelDlqDepthMonitorResult(
            correlation_id=uuid4(),
            evaluated_at=datetime.now(tz=UTC),
            window_seconds=1800,
            topics_matched=1,
            evaluation=self._evaluation(alert_triggered=True),
            suppress_alert_exit=True,
            alert_exit_requested=False,
        )

        assert result.alert_triggered is True
        assert result.alert_exit_requested is False
