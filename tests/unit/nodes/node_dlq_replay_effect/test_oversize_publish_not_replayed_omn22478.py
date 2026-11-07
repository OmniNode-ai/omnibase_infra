# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A dead letter for an oversize publish is not replayed (OMN-22478).

On 2026-11-06 one work-ledger answer of 73 MB was refused by the broker
(max_request_size 1,048,588), dead-lettered under the publish error's class
name, and the replay consumer redelivered the command; each delivery answered
the same size again, and the query consumer stayed jammed from 09:34Z to about
10:31Z. The size of the record is a function of the request, so a replay cannot
succeed.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.enums import EnumNonRetryableErrorCategory
from omnibase_infra.event_bus.topic_constants import get_dlq_topic_for_original
from omnibase_infra.handlers.handler_work_ledger_query import (
    load_work_ledger_query_client_config,
)
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

pytestmark = pytest.mark.unit

_COMMAND_TOPIC = load_work_ledger_query_client_config().command_topic
_DLQ_TOPIC = get_dlq_topic_for_original(_COMMAND_TOPIC)


def _config() -> ModelDlqReplayEngineConfig:
    return ModelDlqReplayEngineConfig(
        bootstrap_servers="localhost:9092", dlq_topic=_DLQ_TOPIC
    )


def _dead_letter(error_type: str, failure_reason: str = "") -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=_COMMAND_TOPIC,
        original_key=None,
        original_value='{"query": "newest-per-lane"}',
        correlation_id=uuid4(),
        retry_count=0,
        error_type=error_type,
        failure_reason=failure_reason,
        dlq_offset=1,
        dlq_partition=0,
    )


@pytest.mark.parametrize(
    "error_type", ["EventPayloadTooLargeError", "MessageSizeTooLargeError"]
)
def test_golden_an_oversize_publish_is_not_eligible_for_replay(
    error_type: str,
) -> None:
    eligible, reason = should_replay(_dead_letter(error_type), _config())
    assert eligible is False
    assert error_type in reason


def test_golden_the_class_name_inside_the_failure_reason_is_enough() -> None:
    """The boundary DLQs under its wrapper's class, so the cause is only text."""
    eligible, _ = should_replay(
        _dead_letter(
            "BoundaryApplyPublishError",
            "EventPayloadTooLargeError: payload is 73182575 bytes",
        ),
        _config(),
    )
    assert eligible is False


def test_error_a_transient_publish_failure_is_still_replayed() -> None:
    """The anti-over-refusal control: a broker outage is worth a second look."""
    eligible, reason = should_replay(_dead_letter("InfraConnectionError"), _config())
    assert eligible is True, reason


def test_error_both_class_names_are_in_the_shared_non_retryable_set() -> None:
    values = EnumNonRetryableErrorCategory.get_all_values()
    assert {"EventPayloadTooLargeError", "MessageSizeTooLargeError"} <= values
