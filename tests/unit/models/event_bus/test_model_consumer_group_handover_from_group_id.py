# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``from_group_id`` on ``ModelConsumerGroupHandover`` (OMN-21932, under OMN-21288).

A folded consumer whose retired group is not ``{env}.<service>.<node>.consume.
<version>`` names it by its literal id.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from omnibase_infra.models.event_bus.model_consumer_group_handover import (
    ModelConsumerGroupHandover,
)

pytestmark = pytest.mark.unit

LITERAL_GROUP = "agent-observability-postgres"

FOUR_FIELDS: dict[str, Any] = {
    "from_service": "omnimarket-projections",
    "from_node": "live-events-writer",
    "from_version": "v1",
    "from_auto_offset_reset": "earliest",
}


def test_from_group_id_is_returned_verbatim_in_every_environment() -> None:
    handover = ModelConsumerGroupHandover(**FOUR_FIELDS, from_group_id=LITERAL_GROUP)

    assert handover.from_group_id == LITERAL_GROUP
    assert handover.source_group("local") == LITERAL_GROUP
    assert handover.source_group("other-lane") == LITERAL_GROUP


def test_unset_from_group_id_keeps_the_derived_group() -> None:
    handover = ModelConsumerGroupHandover(**FOUR_FIELDS)

    assert handover.from_group_id is None
    assert handover.source_group("local") == (
        "local.omnimarket-projections.live-events-writer.consume.v1"
    )


@pytest.mark.parametrize(
    "value",
    ["", "   ", " agent-observability-postgres", "agent-observability-postgres\n"],
    ids=["blank", "whitespace-only", "padded-leading", "padded-trailing"],
)
def test_a_blank_or_whitespace_padded_from_group_id_is_refused(value: str) -> None:
    with pytest.raises(ValidationError, match="from_group_id"):
        ModelConsumerGroupHandover(**FOUR_FIELDS, from_group_id=value)


@pytest.mark.parametrize("value", [7, 1.5, True, b"group", ["group"]])
def test_a_non_string_from_group_id_is_refused(value: object) -> None:
    with pytest.raises(ValidationError, match="from_group_id"):
        ModelConsumerGroupHandover.model_validate(
            {**FOUR_FIELDS, "from_group_id": value}
        )
