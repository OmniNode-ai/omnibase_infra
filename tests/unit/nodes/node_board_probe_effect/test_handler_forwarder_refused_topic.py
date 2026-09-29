# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19930: the ``forwarder_refused_topic`` board check grades three states.

Plan task S5. The check reads a lab lane's gateway forwarder and grades it:

* PASS: the process is up, has been up for at least one retry interval, its
  cloud leg is the broker the lane declares, and no inbound topic is refused;
* FAIL: a topic is refused, or the process is down;
* INDETERMINATE (which the lab proof grades as a failure): the state could not
  be read, the lane declares no cloud broker, the forwarder's cloud leg is
  another broker, or the process has not yet run one retry interval with
  nothing refused.

The refused recording is the forwarder's own log on the .201 lab on
2026-09-28, where the dev cloud broker refused the webhook-delivery topic.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from omnibase_infra.lab_proof.enum_lab_proof_check import EnumLabProofCheck
from omnibase_infra.nodes.node_board_probe_effect.handlers.handler_forwarder_refused_topic import (
    CHECK_ID,
    HandlerForwarderRefusedTopic,
    grade_forwarder_refused_topic,
)
from omnibase_infra.nodes.node_board_probe_effect.models import (
    EnumBoardCheckSurfaceClass,
    EnumBoardProbeOutcome,
    ModelForwarderRefusedTopicRequest,
    ModelForwarderStateObservation,
)

pytestmark = [pytest.mark.unit]

DECLARED = "b-1.dev-cloud.example:9098,b-2.dev-cloud.example:9098"
OTHER = "b-1.other-cloud.example:9098"
NOW = datetime(2026, 9, 28, 16, 10, tzinfo=UTC)
WEBHOOK = "tenant-beta-gateway-canary-79afa7263852.onex.cmd.github.webhook-delivery.v1"


def _request(declared: str = DECLARED) -> ModelForwarderRefusedTopicRequest:
    return ModelForwarderRefusedTopicRequest(
        subject_lane="dev",
        forwarder_container="omninode-gateway-forwarder",
        declared_cloud_broker=declared,
        cloud_broker_ref="gateway.cloud.kafka.broker",
        retry_interval_seconds=300,
    )


def _observation(**overrides: object) -> ModelForwarderStateObservation:
    fields: dict[str, object] = {
        "read_ok": True,
        "running": True,
        "started_at": NOW - timedelta(hours=2),
        "observed_at": NOW,
        "window_seconds": 360,
        "refused_topics": (),
        "observed_cloud_broker": DECLARED,
    }
    fields.update(overrides)
    return ModelForwarderStateObservation.model_validate(fields)


# (case, request declared broker, observation overrides, outcome)
CASES: list[tuple[str, str, dict[str, object], EnumBoardProbeOutcome]] = [
    ("up_nothing_refused", DECLARED, {}, EnumBoardProbeOutcome.PASS),
    (
        "webhook_topic_refused",
        DECLARED,
        {"refused_topics": (WEBHOOK,)},
        EnumBoardProbeOutcome.FAIL,
    ),
    ("process_down", DECLARED, {"running": False}, EnumBoardProbeOutcome.FAIL),
    (
        "process_down_with_unreadable_broker",
        DECLARED,
        {"running": False, "observed_cloud_broker": ""},
        EnumBoardProbeOutcome.FAIL,
    ),
    (
        "cloud_leg_is_another_broker",
        DECLARED,
        {"observed_cloud_broker": OTHER},
        EnumBoardProbeOutcome.INDETERMINATE,
    ),
    (
        "refused_on_another_broker",
        DECLARED,
        {"observed_cloud_broker": OTHER, "refused_topics": (WEBHOOK,)},
        EnumBoardProbeOutcome.INDETERMINATE,
    ),
    ("lane_declares_no_broker", "", {}, EnumBoardProbeOutcome.INDETERMINATE),
    (
        "state_unreadable",
        DECLARED,
        {"read_ok": False, "read_error": "docker inspect: no such container"},
        EnumBoardProbeOutcome.INDETERMINATE,
    ),
    (
        "younger_than_one_retry_interval",
        DECLARED,
        {"started_at": NOW - timedelta(seconds=60)},
        EnumBoardProbeOutcome.INDETERMINATE,
    ),
    (
        "refused_while_younger_than_one_retry_interval",
        DECLARED,
        {"started_at": NOW - timedelta(seconds=60), "refused_topics": (WEBHOOK,)},
        EnumBoardProbeOutcome.FAIL,
    ),
    (
        "log_window_shorter_than_retry_interval",
        DECLARED,
        {"window_seconds": 120},
        EnumBoardProbeOutcome.INDETERMINATE,
    ),
]


@pytest.mark.parametrize(
    ("declared", "overrides", "expected"),
    [(declared, overrides, expected) for _, declared, overrides, expected in CASES],
    ids=[case for case, *_ in CASES],
)
def test_the_states_grade_as_the_plan_states(
    declared: str, overrides: dict[str, object], expected: EnumBoardProbeOutcome
) -> None:
    result = grade_forwarder_refused_topic(
        _request(declared), _observation(**overrides)
    )
    assert result.outcome is expected, result.reasons
    assert result.check_id == CHECK_ID == "forwarder_refused_topic"
    assert result.surface_class is EnumBoardCheckSurfaceClass.LAB_HARDWARE
    assert result.subject == "dev"
    assert result.reasons, "every outcome names its reason"


def test_a_refused_topic_is_named_in_the_evidence() -> None:
    result = grade_forwarder_refused_topic(
        _request(), _observation(refused_topics=(WEBHOOK,))
    )
    assert result.evidence_items == (WEBHOOK,)
    assert any(WEBHOOK in reason for reason in result.reasons)


@pytest.mark.parametrize(
    ("overrides", "declared", "passed"),
    [
        ({}, DECLARED, True),
        ({"refused_topics": (WEBHOOK,)}, DECLARED, False),
        ({}, "", False),  # INDETERMINATE is a lab-proof failure, never a pass
    ],
)
def test_the_lab_proof_check_result_fails_everything_but_pass(
    overrides: dict[str, object], declared: str, passed: bool
) -> None:
    result = grade_forwarder_refused_topic(
        _request(declared), _observation(**overrides)
    )
    check = result.as_lab_proof_check_result()
    assert check.check is EnumLabProofCheck.FORWARDER_REFUSED_TOPIC
    assert check.passed is passed
    assert check.detail
    # onex node exits non-zero unless the status is success.
    assert result.status == ("success" if passed else "failure")


class _Reader:
    def __init__(self, observation: ModelForwarderStateObservation) -> None:
        self.observation = observation
        self.requests: list[ModelForwarderRefusedTopicRequest] = []

    async def observe(
        self, request: ModelForwarderRefusedTopicRequest
    ) -> ModelForwarderStateObservation:
        self.requests.append(request)
        return self.observation


def test_the_handler_observes_through_its_injected_reader_and_grades() -> None:
    reader = _Reader(_observation(refused_topics=(WEBHOOK,)))
    handler = HandlerForwarderRefusedTopic(reader=reader)
    result = asyncio.run(handler.handle(_request()))
    assert reader.requests == [_request()]
    assert result.outcome is EnumBoardProbeOutcome.FAIL


def test_the_lab_proof_check_vocabulary_names_the_check() -> None:
    assert EnumLabProofCheck("forwarder_refused_topic") is (
        EnumLabProofCheck.FORWARDER_REFUSED_TOPIC
    )
