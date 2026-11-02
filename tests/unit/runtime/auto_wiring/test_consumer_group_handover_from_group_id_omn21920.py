# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21920: a handover can name a retired consumer group that is a bare name.

The agent-actions consumer's group, ``agent-observability-postgres``, is not
built from service, node and version and carries no environment prefix, so
``compute_consumer_group_id`` cannot produce it. ``from_group_id`` names such a
group as it is. The service, node and version form stays valid and is the only
form that derives a group.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.handlers.handler_contract_subscribe_preflight import (
    load_consumer_group_handover,
)
from omnibase_infra.models.event_bus.model_consumer_group_handover import (
    ModelConsumerGroupHandover,
)

pytestmark = pytest.mark.unit

_OLD_GROUP = "agent-observability-postgres"


def test_the_bare_group_name_is_the_source_group_in_every_environment() -> None:
    handover = ModelConsumerGroupHandover(
        from_group_id=_OLD_GROUP, from_auto_offset_reset="earliest"
    )
    assert handover.source_group("dev") == _OLD_GROUP
    assert handover.source_group("prod") == _OLD_GROUP


def test_the_four_field_form_still_derives_its_group() -> None:
    handover = ModelConsumerGroupHandover(
        from_service="omnimarket-projections",
        from_node="live-events-writer",
        from_version="v1",
        from_auto_offset_reset="earliest",
    )
    assert handover.from_group_id is None
    assert (
        handover.source_group("dev")
        == "dev.omnimarket-projections.live-events-writer.consume.v1"
    )


def test_the_contract_loader_reads_the_group_id_form(tmp_path: Path) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "consumer_group_handover:\n"
        f"  from_group_id: {_OLD_GROUP}\n"
        "  from_auto_offset_reset: earliest\n",
        encoding="utf-8",
    )
    handover = load_consumer_group_handover(contract)
    assert handover is not None
    assert handover.source_group("dev") == _OLD_GROUP
    assert handover.from_auto_offset_reset == "earliest"


@pytest.mark.parametrize(
    "fields",
    [
        {"from_group_id": _OLD_GROUP, "from_service": "svc"},
        {
            "from_group_id": _OLD_GROUP,
            "from_service": "svc",
            "from_node": "node",
            "from_version": "v1",
        },
    ],
)
def test_a_group_id_beside_derivation_fields_is_refused(
    fields: dict[str, str],
) -> None:
    with pytest.raises(ValidationError, match="from_group_id"):
        ModelConsumerGroupHandover(from_auto_offset_reset="earliest", **fields)


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"from_service": "svc"},
        {"from_service": "svc", "from_node": "node"},
        {"from_node": "node", "from_version": "v1"},
    ],
)
def test_neither_form_complete_is_refused(fields: dict[str, str]) -> None:
    with pytest.raises(ValidationError, match="from_service, from_node"):
        ModelConsumerGroupHandover(from_auto_offset_reset="earliest", **fields)


@pytest.mark.parametrize(
    "group_id",
    ["", " ", "has space", "tab\tname", "a\nb", f"{_OLD_GROUP}\n", f"{_OLD_GROUP}\r\n"],
)
def test_a_group_id_that_is_not_a_kafka_group_name_is_refused(group_id: str) -> None:
    with pytest.raises(ValidationError):
        ModelConsumerGroupHandover(
            from_group_id=group_id, from_auto_offset_reset="earliest"
        )


def test_an_unknown_field_is_still_refused() -> None:
    with pytest.raises(ValidationError):
        ModelConsumerGroupHandover.model_validate(
            {
                "from_group_id": _OLD_GROUP,
                "from_auto_offset_reset": "earliest",
                "from_topic": "x",
            }
        )


def test_unvalidated_incomplete_model_raises_an_explicit_error() -> None:
    handover = ModelConsumerGroupHandover.model_construct(
        from_auto_offset_reset="earliest"
    )
    with pytest.raises(ValueError, match="a handover declares either"):
        handover.source_group("dev")
