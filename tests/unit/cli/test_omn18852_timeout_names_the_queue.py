# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-18852 AC4: a delegation refusal must name the queue it was behind.

The hard-timeout backstop reported *that* the run did not terminalize and for
how long. Both are true and neither answers the caller's question. Measured on
the ``.201`` dev lane 2026-09-19: three callers timed out at ~306 s, every one
of them got a correct answer published after it had exited, and a control run
spent 179 s of its 181 s wall clock queued behind an inference that took
1.559 s. "Slow" and "behind" need completely different responses and the
refusal could not tell them apart.

The number must be an OBSERVATION. These tests pin three things: it is read
from the broker's own committed/log-end offsets, an unresolvable depth is
reported as unresolved rather than as zero, and the two states cannot both be
set on one report.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.cli.delegate_queue_depth import observe_delegate_queue_depth
from omnibase_infra.cli.model_delegate_queue_depth import ModelDelegateQueueDepth

pytestmark = pytest.mark.unit


class TestQueueDepthReport:
    """The tri-state that keeps "unknown" from reading as "empty"."""

    def test_an_unresolved_depth_is_not_zero(self) -> None:
        """The whole point: absent must not render as an idle queue."""
        report = ModelDelegateQueueDepth(unresolved_reason="the broker refused")
        assert report.records_ahead is None
        assert "0" not in report.describe()
        assert "could not be resolved" in report.describe()
        assert "the broker refused" in report.describe()

    def test_an_observed_zero_is_distinguishable_from_unresolved(self) -> None:
        """A genuinely empty queue is a real finding and must read as one."""
        report = ModelDelegateQueueDepth(
            consumer_group="g", backlog_records=1, records_ahead=0
        )
        assert report.records_ahead == 0
        assert "not queueing" in report.describe()
        assert "could not be resolved" not in report.describe()

    def test_an_observed_depth_names_the_group_and_the_backlog(self) -> None:
        report = ModelDelegateQueueDepth(
            consumer_group="local.omnimarket.delegate.consume.v1",
            backlog_records=8,
            records_ahead=7,
        )
        text = report.describe()
        assert "7 record(s) were ahead" in text
        assert "local.omnimarket.delegate.consume.v1" in text
        assert "backlog 8" in text

    def test_a_depth_and_a_reason_cannot_both_be_set(self) -> None:
        """A number beside a reason lets a reader take the untrustworthy one."""
        with pytest.raises(ValidationError, match="exactly one of"):
            ModelDelegateQueueDepth(
                backlog_records=3, records_ahead=2, unresolved_reason="also broke"
            )

    def test_neither_set_is_refused(self) -> None:
        """An empty report is a silent unknown, which is what this replaces."""
        with pytest.raises(ValidationError, match="exactly one of"):
            ModelDelegateQueueDepth()

    def test_an_observed_depth_must_carry_its_backlog(self) -> None:
        """``records_ahead`` is derived; the figure it came from is reported."""
        with pytest.raises(ValidationError, match="backlog_records"):
            ModelDelegateQueueDepth(consumer_group="g", records_ahead=4)


class TestQueueDepthResolution:
    """Every path that cannot observe a depth must say so, and name why."""

    def test_in_process_bus_has_no_queue_and_says_so(self) -> None:
        report = observe_delegate_queue_depth(
            bus="inmemory",
            broker="",
            command_topic="onex.cmd.omnimarket.delegate-skill.v1",
            consumer_groups=("g",),
        )
        assert report.records_ahead is None
        assert "in-process" in report.unresolved_reason

    def test_no_broker_is_a_named_reason(self) -> None:
        report = observe_delegate_queue_depth(
            bus="kafka",
            broker="",
            command_topic="onex.cmd.omnimarket.delegate-skill.v1",
            consumer_groups=("g",),
        )
        assert report.records_ahead is None
        assert "broker" in report.unresolved_reason

    def test_no_known_consumer_group_is_a_named_reason(self) -> None:
        """No group means no committed offset, so no backlog to measure."""
        report = observe_delegate_queue_depth(
            bus="kafka",
            broker="localhost:9092",
            command_topic="onex.cmd.omnimarket.delegate-skill.v1",
            consumer_groups=(),
        )
        assert report.records_ahead is None
        assert "consumer group" in report.unresolved_reason
        assert "onex.cmd.omnimarket.delegate-skill.v1" in report.unresolved_reason

    def test_an_unreachable_broker_refuses_rather_than_hangs_or_guesses(self) -> None:
        """The caller has already blown its deadline; the probe is bounded.

        An unroutable address is used rather than a closed port, so the
        failure is a connect timeout -- the shape that would hang if the probe
        were unbounded.
        """
        report = observe_delegate_queue_depth(
            bus="kafka",
            broker="192.0.2.1:9092",  # onex-allow-internal-ip
            command_topic="onex.cmd.omnimarket.delegate-skill.v1",
            consumer_groups=("local.omnimarket.delegate.consume.v1",),
            probe_seconds=1.0,
        )
        assert report.records_ahead is None
        assert report.unresolved_reason
        assert report.consumer_group == "local.omnimarket.delegate.consume.v1"
