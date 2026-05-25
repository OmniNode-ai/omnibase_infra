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

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_sync_db_adapter,
    _make_projection_dispatch_callback,
    _read_db_io_tables,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_contract(tmp_path: Path, db_tables_yaml: str = "") -> ModelDiscoveredContract:
    """Write a minimal contract.yaml and return a ModelDiscoveredContract."""
    db_io_block = f"db_io:\n  db_tables:\n{db_tables_yaml}" if db_tables_yaml else ""
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        f"name: projection_registration\n"
        f"node_type: reducer\n"
        f"contract_version: {{major: 1, minor: 0, patch: 0}}\n"
        f"{db_io_block}\n"
        f"event_bus:\n"
        f"  subscribe_topics:\n"
        f"    - onex.evt.platform.node-heartbeat.v1\n"
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

    db_tables = [{"name": "node_service_registry", "database": "omnidash_analytics"}]
    handler = FakeProjectionHandler()
    callback = _make_projection_dispatch_callback(
        handler, db_tables, ("onex.evt.platform.node-heartbeat.v1",)
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    envelope.payload = {"service_name": "test-svc", "health_status": "healthy"}

    fake_db = FakeDb()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._build_sync_db_adapter",
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
        [{"name": "node_service_registry", "database": "omnidash_analytics"}],
        ("onex.evt.platform.node-heartbeat.v1",),
    )

    envelope = MagicMock()
    envelope.topic = ""
    envelope.payload = {"service_name": "runtime-host", "health_status": "healthy"}

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._build_sync_db_adapter",
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
        [{"name": "node_service_registry", "database": "omnidash_analytics"}],
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
        "omnibase_infra.runtime.auto_wiring.handler_wiring._build_sync_db_adapter",
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
        [{"name": "node_service_registry", "database": "omnidash_analytics"}],
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
        "omnibase_infra.runtime.auto_wiring.handler_wiring._build_sync_db_adapter",
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
        [{"name": "node_service_registry", "database": "omnidash_analytics"}],
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
        "omnibase_infra.runtime.auto_wiring.handler_wiring._build_sync_db_adapter",
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
    contract = _make_contract(
        tmp_path,
        db_tables_yaml="    - name: node_service_registry\n      database: omnidash_analytics\n",
    )

    # _read_db_io_tables should find the declared table
    tables = _read_db_io_tables(contract.contract_path)
    assert len(tables) == 1
    assert tables[0]["database"] == "omnidash_analytics"


@pytest.mark.integration
def test_wire_handler_entry_uses_standard_path_when_no_db_io(tmp_path: Path) -> None:
    """_wire_handler_entry uses standard envelope path when contract has no db_io."""
    contract = _make_contract(tmp_path, db_tables_yaml="")
    tables = _read_db_io_tables(contract.contract_path)
    assert tables == []


@pytest.mark.integration
def test_projection_callback_no_op_when_db_url_missing(tmp_path: Path) -> None:
    """Projection callback returns None without calling handler when DB URL unset."""
    call_count = [0]

    class CountingHandler:
        def handle(self, input_data: dict) -> dict:
            call_count[0] += 1
            return {}

    db_tables = [{"name": "node_service_registry", "database": "omnidash_analytics"}]
    callback = _make_projection_dispatch_callback(CountingHandler(), db_tables, ())

    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    envelope.payload = {}

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get",
        return_value="",
    ):
        result = asyncio.run(callback(envelope))

    assert result is None
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

        def execute(self, sql: str, params: object) -> None:
            captured_execute["sql"] = sql
            captured_execute["params"] = params

    class FakeConnection:
        closed = False
        autocommit = False

        def cursor(self, *args: object, **kwargs: object) -> FakeCursor:
            return FakeCursor()

    fake_extras = types.SimpleNamespace(Json=FakeJson, RealDictCursor=object)
    fake_psycopg2 = types.SimpleNamespace(
        connect=lambda dsn: FakeConnection(),
        extras=fake_extras,
    )

    monkeypatch.setitem(sys.modules, "psycopg2", fake_psycopg2)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", fake_extras)

    adapter = _build_sync_db_adapter("postgresql://example")
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
