# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration test for projection handler DB injection wiring [OMN-8684].

Verifies that the auto-wiring engine correctly routes events to projection
handlers (handlers with db_io.db_tables in contract) by injecting a DB adapter
and _event_type, rather than passing a raw ModelEventEnvelope.

Uses in-memory fakes only — no real DB or Kafka required.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_core.models.contracts.subcontracts.model_db_ownership_subcontract import (
    ModelDbOwnershipSubcontract,
)
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_projection_db_adapter,
    _make_projection_dispatch_callback,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from tests.helpers.application_db_topology import projection_database_target


@pytest.fixture(autouse=True)
def _configured_projection_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIDASH_ANALYTICS_DB_URL", "postgresql://fixture")
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", "postgresql://fixture")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_contract(
    tmp_path: Path, *, declare_db_io: bool = False
) -> ModelDiscoveredContract:
    """Write a minimal contract.yaml and return a ModelDiscoveredContract."""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        "name: projection_registration\n"
        "node_type: reducer\n"
        "contract_version: {major: 1, minor: 0, patch: 0}\n"
        "event_bus:\n"
        "  subscribe_topics:\n"
        "    - onex.evt.platform.node-heartbeat.v1\n"
    )
    return ModelDiscoveredContract(
        name="projection_registration",
        node_type="reducer",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name="projection_registration",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.platform.node-heartbeat.v1",),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerProjectionRegistration",
                        module="omnimarket.nodes.node_projection_registration.handlers.handler_projection_registration",
                    ),
                ),
            ),
        ),
        db_io=(
            ModelDbOwnershipSubcontract(
                db_tables=[
                    ModelDbTableDeclaration(
                        name="node_service_registry",
                        database_ref="application",
                        schema="tenant",
                        migration="tests/node_service_registry.sql",
                        access="read_write",
                        role="service_registry",
                    )
                ]
            )
            if declare_db_io
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_projection_callback_end_to_end_with_fake_db(tmp_path: Path) -> None:
    """Full projection dispatch path: contract → db_tables read → adapter injected → handler called."""
    upserted_rows: list[dict] = []

    class FakeProjectionHandler:
        def handle(self, input_data: dict) -> dict:
            db = input_data.pop("_db")
            event_type = input_data.pop("_event_type")
            db.upsert(
                "node_service_registry",
                "service_name",
                {**input_data, "event_type": event_type},
            )
            return {"rows_upserted": 1}

    class FakeDb:
        def upsert(self, table: str, key: str, row: dict) -> bool:
            upserted_rows.append({"table": table, "key": key, "row": row})
            return True

        def query(self, table: str, filters: dict | None = None) -> list:
            return []

    db_tables = projection_database_target("node_service_registry")
    handler = FakeProjectionHandler()
    callback = _make_projection_dispatch_callback(
        handler, db_tables, ("onex.evt.platform.node-heartbeat.v1",)
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    envelope.payload = {"service_name": "test-svc", "health_status": "healthy"}

    fake_db = FakeDb()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter",
        return_value=fake_db,
    ):
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get",
            return_value="postgresql://user:pass@host:5432/omnidash_analytics",
        ):
            asyncio.run(callback(envelope))

    assert len(upserted_rows) == 1
    assert upserted_rows[0]["table"] == "node_service_registry"
    assert upserted_rows[0]["row"]["service_name"] == "test-svc"
    assert upserted_rows[0]["row"]["event_type"] == "heartbeat"


@pytest.mark.integration
def test_projection_callback_uses_sole_subscribed_topic_when_envelope_has_no_topic(
    tmp_path: Path,
) -> None:
    """Runtime-dispatched envelopes do not always carry topic metadata."""
    upserted_rows: list[dict] = []

    class FakeProjectionHandler:
        def handle(self, input_data: dict) -> dict:
            db = input_data.pop("_db")
            event_type = input_data.pop("_event_type")
            db.upsert(
                "node_service_registry",
                "service_name",
                {**input_data, "event_type": event_type},
            )
            return {"rows_upserted": 1}

    class FakeDb:
        def upsert(self, table: str, key: str, row: dict) -> bool:
            upserted_rows.append({"table": table, "key": key, "row": row})
            return True

        def query(self, table: str, filters: dict | None = None) -> list:
            return []

    callback = _make_projection_dispatch_callback(
        FakeProjectionHandler(),
        projection_database_target("node_service_registry"),
        ("onex.evt.platform.node-heartbeat.v1",),
    )

    envelope = MagicMock()
    envelope.topic = ""
    envelope.payload = {"service_name": "runtime-host", "health_status": "healthy"}

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter",
        return_value=FakeDb(),
    ):
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get",
            return_value="postgresql://user:pass@host:5432/omnidash_analytics",
        ):
            asyncio.run(callback(envelope))

    assert len(upserted_rows) == 1
    assert upserted_rows[0]["row"]["service_name"] == "runtime-host"
    assert upserted_rows[0]["row"]["event_type"] == "heartbeat"


@pytest.mark.integration
def test_projection_callback_uses_event_type_when_multitopic_envelope_has_no_topic(
    tmp_path: Path,
) -> None:
    """Runtime dispatch preserves event_type even when envelopes omit topic metadata."""
    upserted_rows: list[dict] = []

    class FakeProjectionHandler:
        def handle(self, input_data: dict) -> dict:
            db = input_data.pop("_db")
            event_type = input_data.pop("_event_type")
            db.upsert(
                "node_service_registry",
                "service_name",
                {**input_data, "event_type": event_type},
            )
            return {"rows_upserted": 1}

    class FakeDb:
        def upsert(self, table: str, key: str, row: dict) -> bool:
            upserted_rows.append({"table": table, "key": key, "row": row})
            return True

        def query(self, table: str, filters: dict | None = None) -> list:
            return []

    callback = _make_projection_dispatch_callback(
        FakeProjectionHandler(),
        projection_database_target("node_service_registry"),
        (
            "onex.evt.platform.node-introspection.v1",
            "onex.evt.platform.node-heartbeat.v1",
        ),
    )

    envelope = MagicMock()
    envelope.topic = ""
    envelope.event_type = "platform.node-heartbeat"
    envelope.payload = {"service_name": "runtime-host", "health_status": "healthy"}

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter",
        return_value=FakeDb(),
    ):
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get",
            return_value="postgresql://user:pass@host:5432/omnidash_analytics",
        ):
            asyncio.run(callback(envelope))

    assert len(upserted_rows) == 1
    assert upserted_rows[0]["row"]["service_name"] == "runtime-host"
    assert upserted_rows[0]["row"]["event_type"] == "heartbeat"


@pytest.mark.integration
def test_projection_callback_uses_materialized_dispatch_trace_topic(
    tmp_path: Path,
) -> None:
    """Dispatch engine passes projection callbacks materialized dispatch dicts."""
    upserted_rows: list[dict] = []

    class FakeProjectionHandler:
        def handle(self, input_data: dict) -> dict:
            db = input_data.pop("_db")
            event_type = input_data.pop("_event_type")
            db.upsert(
                "node_service_registry",
                "service_name",
                {**input_data, "event_type": event_type},
            )
            return {"rows_upserted": 1}

    class FakeDb:
        def upsert(self, table: str, key: str, row: dict) -> bool:
            upserted_rows.append({"table": table, "key": key, "row": row})
            return True

        def query(self, table: str, filters: dict | None = None) -> list:
            return []

    callback = _make_projection_dispatch_callback(
        FakeProjectionHandler(),
        projection_database_target("node_service_registry"),
        (
            "onex.evt.platform.node-introspection.v1",
            "onex.evt.platform.node-heartbeat.v1",
        ),
    )

    materialized_dispatch = {
        "payload": {"service_name": "runtime-host", "health_status": "healthy"},
        "__bindings": {},
        "__debug_trace": {
            "event_type": None,
            "topic": "onex.evt.platform.node-heartbeat.v1",
        },
    }

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter",
        return_value=FakeDb(),
    ):
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get",
            return_value="postgresql://user:pass@host:5432/omnidash_analytics",
        ):
            asyncio.run(callback(materialized_dispatch))

    assert len(upserted_rows) == 1
    assert upserted_rows[0]["row"]["service_name"] == "runtime-host"
    assert upserted_rows[0]["row"]["event_type"] == "heartbeat"


@pytest.mark.integration
def test_projection_callback_maps_node_state_change_topic(
    tmp_path: Path,
) -> None:
    """Projection event aliases include all node registration subscribed topics."""
    upserted_rows: list[dict] = []

    class FakeProjectionHandler:
        def handle(self, input_data: dict) -> dict:
            db = input_data.pop("_db")
            event_type = input_data.pop("_event_type")
            db.upsert(
                "node_service_registry",
                "service_name",
                {**input_data, "event_type": event_type},
            )
            return {"rows_upserted": 1}

    class FakeDb:
        def upsert(self, table: str, key: str, row: dict) -> bool:
            upserted_rows.append({"table": table, "key": key, "row": row})
            return True

        def query(self, table: str, filters: dict | None = None) -> list:
            return []

    callback = _make_projection_dispatch_callback(
        FakeProjectionHandler(),
        projection_database_target("node_service_registry"),
        (
            "onex.evt.platform.node-introspection.v1",
            "onex.evt.platform.node-heartbeat.v1",
            "onex.evt.platform.node-state-change.v1",
        ),
    )

    materialized_dispatch = {
        "payload": {"service_name": "runtime-host", "new_state": "active"},
        "__bindings": {},
        "__debug_trace": {
            "event_type": None,
            "topic": "onex.evt.platform.node-state-change.v1",
        },
    }

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter",
        return_value=FakeDb(),
    ):
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get",
            return_value="postgresql://user:pass@host:5432/omnidash_analytics",
        ):
            asyncio.run(callback(materialized_dispatch))

    assert len(upserted_rows) == 1
    assert upserted_rows[0]["row"]["service_name"] == "runtime-host"
    assert upserted_rows[0]["row"]["event_type"] == "state_change"


@pytest.mark.integration
def test_wire_handler_entry_uses_projection_path_when_db_io_declared(
    tmp_path: Path,
) -> None:
    """_wire_handler_entry selects projection callback (not standard) when contract has db_io."""
    contract = _make_contract(tmp_path, declare_db_io=True)

    assert contract.db_io is not None
    assert contract.db_io.db_tables[0].database_ref == "application"
    assert contract.db_io.db_tables[0].schema == "tenant"


@pytest.mark.integration
def test_wire_handler_entry_uses_standard_path_when_no_db_io(tmp_path: Path) -> None:
    """_wire_handler_entry uses standard envelope path when contract has no db_io."""
    contract = _make_contract(tmp_path)
    assert contract.db_io is None


@pytest.mark.integration
def test_projection_callback_rejects_missing_db_url_at_wiring(tmp_path: Path) -> None:
    """Projection wiring fails before dispatch when a required DSN is unset."""
    call_count = [0]

    class CountingHandler:
        def handle(self, input_data: dict) -> dict:
            call_count[0] += 1
            return {}

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get",
        return_value="",
    ):
        with pytest.raises(ValueError, match="tenant_projection"):
            _make_projection_dispatch_callback(
                CountingHandler(),
                projection_database_target("node_service_registry"),
                (),
            )

    assert call_count[0] == 0


@pytest.mark.integration
def test_sync_psycopg2_adapter_preserves_text_array_lists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync adapter must not JSON-wrap Postgres text[] values."""

    captured_execute: dict[str, object] = {}

    class FakeJson:
        def __init__(self, value: object) -> None:
            self.value = value

    class FakeCursor:
        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: object | None = None) -> None:
            captured_execute["sql"] = sql
            captured_execute["params"] = params

        def fetchone(self) -> tuple[str, str]:
            return ("omninode_runtime", "omnidash_analytics")

    class FakeConnection:
        closed = False
        autocommit = False

        def cursor(self, *args: object, **kwargs: object) -> FakeCursor:
            return FakeCursor()

        # OMN-15301: the adapter now runs each statement inside an explicit
        # tenant-scoped transaction, so the double must model the transaction
        # control every real psycopg2 connection has.
        def commit(self) -> None:
            captured_execute["committed"] = True

        def rollback(self) -> None:
            captured_execute["rolled_back"] = True

        def close(self) -> None:
            self.closed = True

    fake_extras = types.SimpleNamespace(
        Json=FakeJson,
        RealDictCursor=object,
        register_uuid=lambda: None,
    )
    fake_psycopg2 = types.SimpleNamespace(
        connect=lambda dsn: FakeConnection(),
        extras=fake_extras,
    )

    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_extras)

    target = projection_database_target("swarm_runs", schema="omninode_internal")
    adapter = _build_projection_db_adapter(
        {"omninode_runtime_service": "postgresql://example"},
        target,
        None,
        None,
    )
    result = adapter.upsert(
        "swarm_runs",
        "run_id",
        {
            "run_id": "run-1",
            "models_used": ["qwen3", "gpt-5"],
            "machines_used": ["worker-a", "worker-b"],
            "metadata": {"source": "integration-test"},
        },
    )

    assert result is True
    params = captured_execute["params"]
    assert isinstance(params, dict)
    assert params["models_used"] == ["qwen3", "gpt-5"]
    assert params["machines_used"] == ["worker-a", "worker-b"]
    assert isinstance(params["metadata"], FakeJson)
    assert params["metadata"].value == {"source": "integration-test"}
    # OMN-15421: this is an internal-domain operation, so it uses ordinary
    # autocommit and never enters the tenant transaction helper.
    assert "set_config" not in str(captured_execute["sql"])
    assert "committed" not in captured_execute
    assert "rolled_back" not in captured_execute
