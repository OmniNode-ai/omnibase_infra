# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Shared registration intent payloads must remain outside node internals."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import BaseModel

from omnibase_infra.models.ledger.model_payload_ledger_append import (
    ModelPayloadLedgerAppend,
)
from omnibase_infra.models.registration.model_node_registration_record import (
    ModelNodeRegistrationRecord,
)
from omnibase_infra.models.registration.model_payload_postgres_update_registration import (
    ModelPayloadPostgresUpdateRegistration,
)
from omnibase_infra.models.registration.model_payload_postgres_upsert_registration import (
    ModelPayloadPostgresUpsertRegistration,
)
from omnibase_infra.models.registration.model_registration_ack_update import (
    ModelRegistrationAckUpdate,
)
from omnibase_infra.models.registration.model_registration_heartbeat_update import (
    ModelRegistrationHeartbeatUpdate,
)

CORRELATION_ID = "00000000-0000-0000-0000-000000000001"
ENTITY_ID = "00000000-0000-0000-0000-000000000002"
TIMESTAMP = "2026-10-07T12:00:00Z"
DEADLINE = "2026-10-07T12:01:00Z"
ACK_UPDATE = {
    "current_state": "active",
    "liveness_deadline": DEADLINE,
    "updated_at": TIMESTAMP,
}
HEARTBEAT_UPDATE = {
    "last_heartbeat_at": TIMESTAMP,
    "liveness_deadline": DEADLINE,
    "updated_at": TIMESTAMP,
}


@pytest.mark.parametrize(
    ("model_class", "fixture", "intent_type"),
    [
        pytest.param(
            ModelPayloadLedgerAppend,
            {
                "topic": "registration-events",
                "partition": 1,
                "kafka_offset": 42,
                "event_key": "bm9kZQ==",
                "event_value": "e30=",
                "onex_headers": {"schema_version": 1},
                "correlation_id": CORRELATION_ID,
                "envelope_id": ENTITY_ID,
                "event_type": "node_registered",
                "source": "registration",
                "event_timestamp": TIMESTAMP,
            },
            "ledger.append",
            id="ledger-append",
        ),
        pytest.param(
            ModelPayloadPostgresUpsertRegistration,
            {
                "correlation_id": CORRELATION_ID,
                "record": ModelNodeRegistrationRecord.model_validate(
                    {
                        "node_id": ENTITY_ID,
                        "node_type": "effect",
                        "registered_at": TIMESTAMP,
                        "updated_at": TIMESTAMP,
                    }
                ),
            },
            "postgres.upsert_registration",
            id="postgres-upsert",
        ),
        *[
            pytest.param(
                ModelPayloadPostgresUpdateRegistration,
                {
                    "correlation_id": CORRELATION_ID,
                    "entity_id": ENTITY_ID,
                    "updates": updates,
                },
                "postgres.update_registration",
                id=f"postgres-update-{name}",
            )
            for name, updates in (("ack", ACK_UPDATE), ("heartbeat", HEARTBEAT_UPDATE))
        ],
        pytest.param(ModelRegistrationAckUpdate, ACK_UPDATE, None, id="ack-update"),
        pytest.param(
            ModelRegistrationHeartbeatUpdate,
            HEARTBEAT_UPDATE,
            None,
            id="heartbeat-update",
        ),
    ],
)
def test_registration_intent_payload_seam(
    model_class: type[BaseModel], fixture: dict[str, object], intent_type: str | None
) -> None:
    """Preserve JSON data, routing literals, and shared model ownership."""
    assert not model_class.__module__.startswith("omnibase_infra.nodes.")
    assert model_class.__module__ != "omnibase_infra.nodes"

    model = model_class.model_validate(fixture)
    serialized = model.model_dump(mode="json")
    round_trip_fixture = serialized.copy()
    if model_class is ModelPayloadPostgresUpsertRegistration:
        # The existing SerializeAsAny[BaseModel] field needs the consumer's
        # concrete record schema to restore its subclass fields from JSON.
        round_trip_fixture["record"] = ModelNodeRegistrationRecord.model_validate(
            serialized["record"]
        )
    restored = model_class.model_validate(round_trip_fixture)
    assert restored.model_dump(mode="json") == serialized
    assert restored == model

    if intent_type is not None:
        assert serialized["intent_type"] == intent_type
        assert get_args(model_class.model_fields["intent_type"].annotation) == (
            intent_type,
        )
