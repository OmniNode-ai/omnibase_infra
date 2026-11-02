# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21920: the bare ``from_group_id`` of the agent-actions handover keeps its exact spelling.

A dumped and revalidated handover names the same group, and a trailing newline
or a space is refused, not trimmed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from omnibase_infra.models.event_bus.model_consumer_group_handover import (
    ModelConsumerGroupHandover,
)

pytestmark = pytest.mark.unit


def test_the_bare_group_round_trips_through_a_dumped_model() -> None:
    handover = ModelConsumerGroupHandover(
        from_group_id="agent-observability-postgres",
        from_auto_offset_reset="earliest",
    )
    restored = ModelConsumerGroupHandover.model_validate(handover.model_dump())
    assert restored.source_group("prod") == "agent-observability-postgres"


@pytest.mark.parametrize(
    "bad",
    ["agent-observability-postgres\n", "agent observability", "", ".", "..", "..."],
)
def test_a_group_name_with_whitespace_or_only_dots_is_refused(bad: str) -> None:
    with pytest.raises(ValidationError):
        ModelConsumerGroupHandover(from_group_id=bad, from_auto_offset_reset="earliest")
