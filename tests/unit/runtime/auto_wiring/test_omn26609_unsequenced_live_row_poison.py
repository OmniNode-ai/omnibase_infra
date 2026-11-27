# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26609 -- the dead letter of an unsequenced live ledger row is never replayed.

Measured on the dev lane's ``omninode-runtime`` (started 2026-11-24T16:08:48Z) from its log and
the ``onex.evt.omnimarket.work-ledger-status.v1`` topic: 206 ``adopt-merged`` STATUS rows with no
``ledger_seq`` (source ``onex-ledger``) were refused by omnimarket's work-ledger projection writer
(``UnsequencedLiveRowError``, OMN-23488, a POISON refusal there). The runtime dead-lettered each
record after 6 redeliveries as ``ProjectionWedgeExhaustedError`` (bound 5). The DLQ replay node
then replayed each dead letter back onto the topic, up to five replays per row, because its error
type is the wrapper's name and the class was not in the non-retryable enum; each replay was refused
6 more times. The one partition's consumer spent about 7 s per record, so STATUS rows from the
canonical ledger reached the projection 1902 s to 2154 s late (row 2026-11-24T16:29:27Z projected
17:01:09Z; row 2026-11-24T16:38:35Z read at 17:14:29Z).

The row below is the real refused record at offset 271060 (its DLQ record is at offset 1434803 of
``onex.dlq.omnibase-infra.intents.v1``).

The classifier half (dead-letter at once, no 6 redeliveries) is not in this change: naming the
omnimarket node module in ``POISON_EVENT_IDENTITY_REFUSALS`` is a cross-repo edge that the
node-boundary import gate refuses and its baseline may not grow.
"""

from __future__ import annotations

from uuid import UUID

import pytest

from omnibase_infra.enums import EnumNonRetryableErrorCategory
from omnibase_infra.nodes.node_dlq_replay_effect.engine_dlq_replay import (
    ModelDlqReplayEngineConfig,
    should_replay,
)
from omnibase_infra.nodes.node_dlq_replay_effect.models.model_dlq_message import (
    ModelDlqMessage,
)

_TOPIC = "onex.evt.omnimarket.work-ledger-status.v1"
_ROW_ID = "be19e77f16e6008e3dcedc1c3aa3227256f3c8f3d6a4e496a2084619c02f431b"
# The failure text of the real dead letter, as the wedge guard flattened it.
_FAILURE_REASON = (
    "ProjectionWedgeExhaustedError: [ONEX_CORE_081_OPERATION_FAILED] projection withhold "
    f"exhausted after 6 consecutive identical refusals of {_TOPIC} partition 0 offset 271060 "
    "(bound 5): [ONEX_CORE_081_OPERATION_FAILED] projection handler "
    f"WorkLedgerProjectionWriter consumed an event from {_TOPIC} and wrote no row because its "
    "write path failed (UnsequencedLiveRowError: unsequenced-live-row: source 'onex-ledger' "
    f"requires a ledger_seq; row_id {_ROW_ID})"
)
_CONFIG = ModelDlqReplayEngineConfig(
    bootstrap_servers="localhost:9092",
    dlq_topic="onex.dlq.omnibase-infra.intents.v1",
)


def _dead_letter(retry_count: int) -> ModelDlqMessage:
    return ModelDlqMessage(
        original_topic=_TOPIC,
        original_value="{}",
        failure_reason=_FAILURE_REASON,
        correlation_id=UUID("05f94053-160e-47e3-9863-0f09edaf9abd"),
        retry_count=retry_count,
        error_type="ProjectionWedgeExhaustedError",
        dlq_offset=1434803,
        dlq_partition=0,
    )


@pytest.mark.unit
def test_the_class_is_a_named_non_retryable_category() -> None:
    assert EnumNonRetryableErrorCategory.is_non_retryable("UnsequencedLiveRowError")


@pytest.mark.unit
@pytest.mark.parametrize("retry_count", [0, 1, 2])
def test_the_dead_letter_of_an_unsequenced_live_row_is_not_replayed(
    retry_count: int,
) -> None:
    eligible, why = should_replay(_dead_letter(retry_count), _CONFIG)
    assert eligible is False, why
    assert "non-retryable" in why.lower()


@pytest.mark.unit
def test_a_transient_write_path_failure_is_still_replayed() -> None:
    """Positive control: the fix names one class; it does not turn replay off."""
    message = ModelDlqMessage(
        original_topic=_TOPIC,
        original_value="{}",
        failure_reason="ProjectionWedgeExhaustedError: write path failed (InfraConnectionError: broker unreachable)",
        correlation_id=UUID("05f94053-160e-47e3-9863-0f09edaf9abd"),
        retry_count=0,
        error_type="ProjectionWedgeExhaustedError",
        dlq_offset=1,
        dlq_partition=0,
    )
    eligible, why = should_replay(message, _CONFIG)
    assert eligible is True, why
