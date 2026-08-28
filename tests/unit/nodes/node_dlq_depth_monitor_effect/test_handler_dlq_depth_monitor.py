# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Tests for the read-only DLQ depth probe against a fake admin transport.

The fake mirrors the real ``aiokafka`` shapes, INCLUDING the two edges that
would silently corrupt the measurement if mishandled:

  * ``offsets_for_times`` returns ``None`` when nothing landed in the window;
  * retention can delete the record that was at the window start.

Live readings used as fixtures were taken from the .201 dev lane on
2026-08-27 via ``rpk topic describe -p``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from uuid import uuid4

import pytest

from omnibase_infra.errors import RuntimeHostError
from omnibase_infra.nodes.node_dlq_depth_evaluate_compute.models.enum_dlq_depth_verdict import (
    EnumDlqDepthVerdict,
)
from omnibase_infra.nodes.node_dlq_depth_monitor_effect.handlers.handler_dlq_depth_monitor import (
    HandlerDlqDepthMonitor,
)
from omnibase_infra.nodes.node_dlq_depth_monitor_effect.models.model_dlq_depth_monitor_request import (
    ModelDlqDepthMonitorRequest,
)
from omnibase_infra.protocols.protocol_dlq_admin_transport import (
    TopicPartition,
)

pytestmark = pytest.mark.unit

_QUARANTINE = "onex.dlq.omnibase-infra.quarantine.v1"
_EVENTS = "onex.dlq.omnibase-infra.events.v1"
_NOT_A_DLQ = "onex.evt.omnibase-infra.delegation-completed.v1"


class FakeDlqAdminTransport:
    """In-memory ``ProtocolDlqAdminTransport``. Records every call it serves."""

    def __init__(
        self,
        *,
        topics: Mapping[str, Mapping[int, tuple[int, int, int | None]]],
    ) -> None:
        """``topics[name][partition] = (log_start, high_watermark, window_start)``.

        ``window_start`` of ``None`` models the broker returning no offset for
        the requested timestamp.
        """
        self._topics = topics
        self.list_topics_calls = 0
        # Proof-of-read-only: the fake exposes no write surface at all, so a
        # handler that tried to produce/commit could not even compile against it.

    async def list_topics(self) -> Sequence[str]:
        self.list_topics_calls += 1
        return list(self._topics)

    async def partitions_for_topic(self, topic: str) -> Sequence[int]:
        return list(self._topics[topic])

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


class MissingOffsetTransport(FakeDlqAdminTransport):
    async def beginning_offsets(
        self, partitions: Sequence[TopicPartition]
    ) -> Mapping[TopicPartition, int]:
        return {
            tp: self._topics[tp[0]][tp[1]][0]
            for tp in partitions
            if tp != (_QUARANTINE, 1)
        }


def _request(**overrides: object) -> ModelDlqDepthMonitorRequest:
    payload: dict[str, object] = {
        "correlation_id": uuid4(),
        "suppress_alert_exit": True,
    }
    payload.update(overrides)
    return ModelDlqDepthMonitorRequest(**payload)  # type: ignore[arg-type]


class TestNoneWindowStartIsNormalizedToHighWaterMark:
    """SHARP EDGE 1 — the phantom-8.88M-arrival bug this test exists to prevent."""

    async def test_none_window_start_yields_zero_arrivals_not_lifetime_volume(
        self,
    ) -> None:
        transport = FakeDlqAdminTransport(
            topics={_QUARANTINE: {0: (6, 8_878_932, None)}}
        )

        result = await HandlerDlqDepthMonitor(transport).handle(_request())

        verdict = result.evaluation.verdicts[0]
        assert verdict.arrivals_in_window == 0, (
            "A None offset-for-time means nothing arrived. Treating it as 0 "
            "would report the topic's whole 8.88M lifetime as one window."
        )
        assert verdict.window_start_offset == 8_878_932
        assert verdict.retained_depth == 8_878_926
        assert result.evaluation.alert_triggered is False


class TestRetentionTrimmedWindowStartIsClamped:
    """SHARP EDGE 2 — never count records retention already deleted."""

    async def test_window_start_behind_log_start_clamps_forward(self) -> None:
        # Broker answers with an offset that retention has since trimmed away.
        transport = FakeDlqAdminTransport(
            topics={_EVENTS: {0: (8_157_557, 8_170_442, 8_000_000)}}
        )

        result = await HandlerDlqDepthMonitor(transport).handle(_request())

        verdict = result.evaluation.verdicts[0]
        assert verdict.window_start_offset == 8_157_557
        assert verdict.arrivals_in_window == 12_885
        assert verdict.retained_depth == 12_885


class TestTopicEnumeration:
    async def test_only_dlq_prefixed_topics_are_probed(self) -> None:
        transport = FakeDlqAdminTransport(
            topics={
                _QUARANTINE: {0: (0, 10, 10)},
                _NOT_A_DLQ: {0: (0, 99_999, 0)},
            }
        )

        result = await HandlerDlqDepthMonitor(transport).handle(_request())

        assert result.topics_matched == 1
        assert [v.topic for v in result.evaluation.verdicts] == [_QUARANTINE]

    async def test_custom_prefix_narrows_the_sweep(self) -> None:
        transport = FakeDlqAdminTransport(
            topics={
                _QUARANTINE: {0: (0, 10, 10)},
                "onex.dlq.omnimarket.pr-merge.v1": {0: (0, 5, 5)},
            }
        )

        result = await HandlerDlqDepthMonitor(transport).handle(
            _request(topic_prefix="onex.dlq.omnimarket.")
        )

        assert result.topics_matched == 1
        assert result.evaluation.verdicts[0].topic == "onex.dlq.omnimarket.pr-merge.v1"

    async def test_topic_with_zero_partitions_is_skipped_not_fabricated(self) -> None:
        """A metadata race must not become a confident zero-depth row."""
        transport = FakeDlqAdminTransport(
            topics={_QUARANTINE: {}, _EVENTS: {0: (0, 3, 0)}}
        )

        result = await HandlerDlqDepthMonitor(transport).handle(_request())

        observed = [v.topic for v in result.evaluation.verdicts]
        assert _QUARANTINE not in observed
        assert observed == [_EVENTS]

    async def test_topic_with_missing_partition_offset_is_skipped_not_partial(
        self,
    ) -> None:
        transport = MissingOffsetTransport(
            topics={_QUARANTINE: {0: (0, 10, 9), 1: (0, 100, 90)}}
        )

        result = await HandlerDlqDepthMonitor(transport).handle(_request())

        assert result.evaluation.verdicts == ()
        assert result.evaluation.topics_observed == 0

    async def test_empty_broker_yields_empty_non_alerting_result(self) -> None:
        result = await HandlerDlqDepthMonitor(FakeDlqAdminTransport(topics={})).handle(
            _request()
        )

        assert result.topics_matched == 0
        assert result.evaluation.topics_observed == 0
        assert result.evaluation.alert_triggered is False


class TestMultiPartitionFolding:
    async def test_offsets_are_summed_across_partitions(self) -> None:
        transport = FakeDlqAdminTransport(
            topics={_QUARANTINE: {0: (1, 100, 90), 1: (2, 200, 150), 2: (0, 50, 50)}}
        )

        result = await HandlerDlqDepthMonitor(transport).handle(_request())

        verdict = result.evaluation.verdicts[0]
        assert verdict.partition_count == 3
        assert verdict.log_start_offset == 3
        assert verdict.high_watermark == 350
        assert verdict.window_start_offset == 290
        assert verdict.arrivals_in_window == 60
        assert verdict.retained_depth == 347

    async def test_mixed_none_and_real_window_starts_across_partitions(self) -> None:
        """One quiet partition, one active — only the active one contributes."""
        transport = FakeDlqAdminTransport(
            topics={_QUARANTINE: {0: (0, 100, None), 1: (0, 100, 80)}}
        )

        result = await HandlerDlqDepthMonitor(transport).handle(_request())

        assert result.evaluation.verdicts[0].arrivals_in_window == 20


class TestAlertGating:
    """The red-run alert surface."""

    async def test_alert_raises_by_default_so_the_workflow_goes_red(self) -> None:
        transport = FakeDlqAdminTransport(
            topics={_QUARANTINE: {0: (6, 8_878_948, 8_878_932)}}
        )

        with pytest.raises(RuntimeHostError) as excinfo:
            await HandlerDlqDepthMonitor(transport).handle(
                _request(suppress_alert_exit=False)
            )

        message = str(excinfo.value)
        assert _QUARANTINE in message
        assert "+16" in message

    async def test_suppress_alert_exit_returns_the_histogram_instead(self) -> None:
        transport = FakeDlqAdminTransport(
            topics={_QUARANTINE: {0: (6, 8_878_948, 8_878_932)}}
        )

        result = await HandlerDlqDepthMonitor(transport).handle(
            _request(suppress_alert_exit=True)
        )

        assert result.alert_triggered is True
        assert (
            result.evaluation.verdicts[0].verdict is EnumDlqDepthVerdict.ALERT_ARRIVALS
        )

    async def test_quiet_standing_backlog_does_not_raise(self) -> None:
        """The real 8.88M backlog, quiet: must NOT gate the run (AC4)."""
        transport = FakeDlqAdminTransport(
            topics={_QUARANTINE: {0: (6, 8_878_932, None)}}
        )

        result = await HandlerDlqDepthMonitor(transport).handle(
            _request(suppress_alert_exit=False)
        )

        assert result.alert_triggered is False
        assert result.evaluation.verdicts[0].retained_depth == 8_878_926

    async def test_opt_in_depth_bound_gates_the_run(self) -> None:
        transport = FakeDlqAdminTransport(
            topics={_QUARANTINE: {0: (6, 8_878_932, None)}}
        )

        with pytest.raises(RuntimeHostError):
            await HandlerDlqDepthMonitor(transport).handle(
                _request(suppress_alert_exit=False, max_retained_depth=1_000_000)
            )


class TestKillSwitch:
    async def test_false_string_does_not_disable_monitor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_DLQ_MONITOR_DISABLED", "false")
        transport = FakeDlqAdminTransport(topics={_QUARANTINE: {0: (0, 10, 10)}})

        result = await HandlerDlqDepthMonitor(transport).handle(_request())

        assert transport.list_topics_calls == 1
        assert result.topics_matched == 1

    async def test_truthy_string_disables_monitor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_DLQ_MONITOR_DISABLED", "true")
        transport = FakeDlqAdminTransport(topics={_QUARANTINE: {0: (0, 10, 10)}})

        result = await HandlerDlqDepthMonitor(transport).handle(_request())

        assert transport.list_topics_calls == 0
        assert result.topics_matched == 0


class TestWindowPlumbing:
    async def test_window_seconds_reaches_the_evaluation(self) -> None:
        transport = FakeDlqAdminTransport(topics={_QUARANTINE: {0: (0, 10, 10)}})

        result = await HandlerDlqDepthMonitor(transport).handle(
            _request(window_seconds=3600)
        )

        assert result.window_seconds == 3600
        assert result.evaluation.window_seconds == 3600
        assert result.evaluation.verdicts[0].window_seconds == 3600

    async def test_broker_is_enumerated_exactly_once_per_run(self) -> None:
        transport = FakeDlqAdminTransport(topics={_QUARANTINE: {0: (0, 10, 10)}})

        await HandlerDlqDepthMonitor(transport).handle(_request())

        assert transport.list_topics_calls == 1


class TestKillSwitchAndConfiguration:
    """Halt and fail-closed behavior, matching the sibling sweeps' precedent."""

    async def test_kill_switch_performs_zero_io(self) -> None:
        transport = FakeDlqAdminTransport(
            topics={_QUARANTINE: {0: (6, 8_878_948, 8_878_932)}}
        )

        result = await HandlerDlqDepthMonitor(
            transport, kill_switch_disabled=True
        ).handle(_request(suppress_alert_exit=False))

        assert transport.list_topics_calls == 0
        assert result.topics_matched == 0
        assert result.alert_triggered is False

    async def test_kill_switch_honors_the_env_var(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_DLQ_MONITOR_DISABLED", "1")
        transport = FakeDlqAdminTransport(topics={_QUARANTINE: {0: (0, 10, 0)}})

        result = await HandlerDlqDepthMonitor(transport).handle(
            _request(suppress_alert_exit=False)
        )

        assert transport.list_topics_calls == 0
        assert result.alert_triggered is False

    async def test_missing_bootstrap_fails_closed_rather_than_defaulting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A monitor that silently probes the wrong lane is worse than none."""
        monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
        monkeypatch.delenv("ONEX_DLQ_MONITOR_DISABLED", raising=False)

        with pytest.raises(RuntimeHostError, match="KAFKA_BOOTSTRAP_SERVERS"):
            await HandlerDlqDepthMonitor().handle(_request())
