# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Frozen wire contracts shared by registration intent producers and consumers.

Schema changes require a contract bump and fixture regeneration in the same PR
as the corresponding producer and consumer changes. JSON fixtures intentionally
have no SPDX comments, matching the other files under tests/fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import get_args
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

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
from omnibase_infra.nodes.node_registration_orchestrator.models.model_projection_record import (
    ModelProjectionRecord,
)
from omnibase_infra.nodes.node_registration_reducer import RegistrationReducer
from omnibase_infra.nodes.node_registration_reducer.models import ModelRegistrationState
from omnibase_infra.runtime.intent_effects.intent_effect_postgres_update import (
    IntentEffectPostgresUpdate,
)
from omnibase_infra.runtime.intent_effects.intent_effect_postgres_upsert import (
    IntentEffectPostgresUpsert,
)
from tests.helpers import create_introspection_event
from tests.unit.nodes.node_ledger_write_effect.handlers.test_handler_ledger_append import (
    make_db_result,
    make_handler_with_mock_db,
)
from tests.unit.runtime.test_intent_effect_postgres_update import _make_mock_pool

pytestmark = pytest.mark.unit

FIXTURE_DIR = (
    Path(__file__).resolve().parents[3] / "fixtures/seams/registration_intent_payloads"
)
CONTRACTS: tuple[tuple[str, type[BaseModel]], ...] = (
    ("model_payload_ledger_append", ModelPayloadLedgerAppend),
    (
        "model_payload_postgres_upsert_registration",
        ModelPayloadPostgresUpsertRegistration,
    ),
    (
        "model_payload_postgres_update_registration",
        ModelPayloadPostgresUpdateRegistration,
    ),
    ("model_registration_ack_update", ModelRegistrationAckUpdate),
    ("model_registration_heartbeat_update", ModelRegistrationHeartbeatUpdate),
)
CORRELATION_ID = UUID("00000000-0000-0000-0000-000000000001")
ENTITY_ID = UUID("00000000-0000-0000-0000-000000000002")


def _fixture_json(stem: str, kind: str = "valid") -> str:
    return (FIXTURE_DIR / f"{stem}.{kind}.json").read_text(encoding="utf-8")


def _validate_wire_json(
    model_class: type[BaseModel],
    wire_json: str,
    *,
    record_class: type[BaseModel] | None = None,
) -> BaseModel:
    """Validate the outer DTO and restore an upsert's concrete record type.

    SerializeAsAny[BaseModel] describes an open record schema, not a JSON
    discriminator for reconstructing subclasses. The effect accepts a typed
    payload; its caller must decode the concrete record before execute(). Do
    that explicitly with production models rather than a test-only record DTO.
    """
    payload = model_class.model_validate_json(wire_json)
    if isinstance(payload, ModelPayloadPostgresUpsertRegistration):
        assert record_class is not None
        record = record_class.model_validate_json(
            json.dumps(json.loads(wire_json)["record"])
        )
        payload = ModelPayloadPostgresUpsertRegistration(
            intent_type=payload.intent_type,
            correlation_id=payload.correlation_id,
            record=record,
        )
    return payload


@pytest.mark.parametrize(
    ("stem", "model_class"), CONTRACTS, ids=[s for s, _ in CONTRACTS]
)
def test_registration_intent_payload_seam(
    stem: str, model_class: type[BaseModel]
) -> None:
    """Freeze schema, routing literals, and ownership outside node internals."""
    assert not model_class.__module__.startswith("omnibase_infra.nodes.")
    assert model_class.__module__ != "omnibase_infra.nodes"

    live_schema = json.dumps(model_class.model_json_schema(), sort_keys=True, indent=2)
    assert live_schema + "\n" == _fixture_json(stem, "schema"), (
        f"{model_class.__name__} wire contract drifted: bump the contract and "
        "regenerate fixtures in the same PR as both producer and consumer changes."
    )
    if "intent_type" in model_class.model_fields:
        intent_field = model_class.model_fields["intent_type"]
        assert get_args(intent_field.annotation) == (intent_field.default,)


@pytest.mark.parametrize(
    ("stem", "model_class"), CONTRACTS, ids=[s for s, _ in CONTRACTS]
)
def test_valid_fixtures(stem: str, model_class: type[BaseModel]) -> None:
    """Committed wire examples satisfy the frozen schema and live parser."""
    paths = sorted(FIXTURE_DIR.glob(f"{stem}*.valid.json"))
    assert paths, f"Missing valid wire fixture for {stem}"
    validator = Draft202012Validator(json.loads(_fixture_json(stem, "schema")))
    for path in paths:
        wire_json = path.read_text(encoding="utf-8")
        wire_data = json.loads(wire_json)
        validator.validate(wire_data)
        payload = _validate_wire_json(
            model_class, wire_json, record_class=ModelProjectionRecord
        )
        assert isinstance(payload, model_class)
        assert payload.model_dump(mode="json") == wire_data, path.name


@pytest.mark.parametrize(
    ("stem", "model_class"), CONTRACTS, ids=[s for s, _ in CONTRACTS]
)
def test_invalid_fixtures(stem: str, model_class: type[BaseModel]) -> None:
    """Reject known-bad wire data, including extra keys and bad routing literals."""
    paths = sorted(FIXTURE_DIR.glob(f"{stem}*.invalid.json"))
    assert paths, f"Missing invalid wire fixture for {stem}"
    for path in paths:
        with pytest.raises(ValidationError):
            model_class.model_validate_json(path.read_text(encoding="utf-8"))


def test_registration_reducer_produces_conforming_payloads() -> None:
    """Exercise the real idle -> pending producer transition across the seam.

    RegistrationReducer emits postgres upserts only; ledger append intents are
    produced by the separate audit ledger reducer.
    """
    event = create_introspection_event(node_id=ENTITY_ID, correlation_id=CORRELATION_ID)
    output = RegistrationReducer().reduce(ModelRegistrationState(), event)
    assert output.result.status == "pending"
    assert output.result.node_id == event.node_id
    assert output.intents, "Transition must emit an intent to exercise the seam"
    contracts_by_intent = {
        model.model_fields["intent_type"].default: (stem, model)
        for stem, model in CONTRACTS
        if "intent_type" in model.model_fields
    }
    assert any(
        intent.intent_type == "postgres.upsert_registration"
        for intent in output.intents
    )
    for intent in output.intents:
        stem, model_class = contracts_by_intent[intent.intent_type]
        assert isinstance(intent.payload, model_class)
        wire_data = intent.payload.model_dump(mode="json")
        Draft202012Validator(json.loads(_fixture_json(stem, "schema"))).validate(
            wire_data
        )
        restored = _validate_wire_json(
            model_class, json.dumps(wire_data), record_class=ModelNodeRegistrationRecord
        )
        assert isinstance(restored, model_class)
        assert restored.model_dump(mode="json") == wire_data
        assert wire_data["correlation_id"] == str(CORRELATION_ID)


@pytest.mark.asyncio
async def test_postgres_upsert_consumer_accepts_fixture_json() -> None:
    """Decode committed JSON and execute the real adapter against its DB fake."""
    payload = _validate_wire_json(
        ModelPayloadPostgresUpsertRegistration,
        _fixture_json("model_payload_postgres_upsert_registration"),
        record_class=ModelProjectionRecord,
    )
    assert isinstance(payload, ModelPayloadPostgresUpsertRegistration)
    assert isinstance(payload.record, ModelProjectionRecord)
    projector = MagicMock()
    projector.upsert_partial = AsyncMock(return_value=True)

    await IntentEffectPostgresUpsert(projector=projector).execute(payload)

    projector.upsert_partial.assert_awaited_once()
    kwargs = projector.upsert_partial.call_args.kwargs
    assert kwargs["aggregate_id"] == ENTITY_ID
    assert kwargs["correlation_id"] == CORRELATION_ID
    assert kwargs["conflict_columns"] == ["entity_id", "domain"]
    expected_values = payload.record.model_dump(mode="json")
    expected_values.update(expected_values.pop("data"))
    # Timestamp and UUID columns are normalized for asyncpg by the consumer.
    actual_values = kwargs["values"]
    assert set(actual_values) == set(expected_values)
    for key, value in expected_values.items():
        actual = actual_values[key]
        if key in {"registered_at", "updated_at"}:
            assert actual.isoformat().replace("+00:00", "Z") == value
        else:
            assert str(actual) == str(value)


@pytest.mark.parametrize(
    ("stem", "updates_class"),
    [
        ("model_payload_postgres_update_registration", ModelRegistrationAckUpdate),
        (
            "model_payload_postgres_update_registration.heartbeat",
            ModelRegistrationHeartbeatUpdate,
        ),
    ],
    ids=["ack", "heartbeat"],
)
@pytest.mark.asyncio
async def test_postgres_update_consumer_accepts_fixture_json(
    stem: str, updates_class: type[BaseModel]
) -> None:
    """Both update variants survive JSON decoding and reach parameterized SQL."""
    payload = ModelPayloadPostgresUpdateRegistration.model_validate_json(
        _fixture_json(stem)
    )
    assert isinstance(payload.updates, updates_class)
    pool = _make_mock_pool()

    await IntentEffectPostgresUpdate(pool=pool).execute(payload)

    pool._mock_conn.execute.assert_awaited_once()
    call = pool._mock_conn.execute.call_args
    sql, *parameters = call.args
    assert sql.startswith('UPDATE "registration_projections" SET ')
    updates = payload.updates.model_dump()
    assert parameters[: len(updates)] == list(updates.values())
    assert parameters[len(updates) : len(updates) + 2] == [ENTITY_ID, "registration"]
    for column in updates:
        assert f'"{column}" = $' in sql
    if isinstance(payload.updates, ModelRegistrationHeartbeatUpdate):
        assert '"last_heartbeat_at" IS NULL OR "last_heartbeat_at" <' in sql
        assert parameters[-1] == payload.updates.last_heartbeat_at
        assert len(parameters) == len(updates) + 3
    else:
        assert "last_heartbeat_at" not in sql.split("WHERE")[1]
        assert len(parameters) == len(updates) + 2
    assert call.kwargs == {"timeout": 30.0}


@pytest.mark.parametrize("entry_point", ["execute", "handle"])
@pytest.mark.asyncio
async def test_ledger_consumer_parses_fixture_json(entry_point: str) -> None:
    """Use real envelope parsers and existing DB doubles, including BYTEA decode."""
    handler, db_handler = make_handler_with_mock_db()
    db_handler.execute.return_value = make_db_result(
        rows=[{"ledger_entry_id": str(ENTITY_ID)}]
    )
    envelope = {
        "operation": "ledger.append",
        "correlation_id": str(CORRELATION_ID),
        "payload": json.loads(_fixture_json("model_payload_ledger_append")),
    }

    output = await getattr(handler, entry_point)(envelope)

    assert output.result is not None
    assert output.result.success
    assert output.result.duplicate is False
    assert output.result.ledger_entry_id == ENTITY_ID
    assert (
        output.result.topic,
        output.result.partition,
        output.result.kafka_offset,
    ) == (
        "registration-events",
        1,
        42,
    )
    db_handler.execute.assert_awaited_once()
    db_envelope = db_handler.execute.call_args.args[0]
    assert db_envelope["operation"] == "db.query"
    assert db_envelope["correlation_id"] == str(CORRELATION_ID)
    parameters = db_envelope["payload"]["parameters"]
    assert parameters[:5] == ["registration-events", 1, 42, b"node", b"{}"]
    assert json.loads(parameters[5]) == {"schema_version": 1}
    assert parameters[6:10] == [
        str(ENTITY_ID),
        str(CORRELATION_ID),
        "node_registered",
        "registration",
    ]
    assert parameters[10].isoformat().replace("+00:00", "Z") == "2026-10-07T12:00:00Z"
