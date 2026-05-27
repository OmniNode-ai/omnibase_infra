# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for projection handler DB injection in auto-wiring [OMN-8684]."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_sync_db_adapter,
    _make_dispatch_callback,
    _make_projection_dispatch_callback,
    _read_db_io_tables,
)

_PATCH_BUILD_ADAPTER = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._build_sync_db_adapter"
)
_PATCH_ENVIRON_GET = "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get"


# ---------------------------------------------------------------------------
# Tests: _read_db_io_tables
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_db_io_tables_returns_tables(tmp_path: Path) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "name: test\n"
        "db_io:\n"
        "  db_tables:\n"
        "    - name: node_service_registry\n"
        "      database: omnidash_analytics\n"
    )
    tables = _read_db_io_tables(contract)
    assert len(tables) == 1
    assert tables[0]["name"] == "node_service_registry"
    assert tables[0]["database"] == "omnidash_analytics"


@pytest.mark.unit
def test_read_db_io_tables_returns_empty_when_absent(tmp_path: Path) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text("name: test\nnode_type: EFFECT_GENERIC\n")
    assert _read_db_io_tables(contract) == []


@pytest.mark.unit
def test_read_db_io_tables_returns_empty_on_missing_file() -> None:
    assert _read_db_io_tables(Path("/nonexistent/contract.yaml")) == []


# ---------------------------------------------------------------------------
# Tests: _make_dispatch_callback (standard, non-projection path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_standard_callback_calls_async_handle() -> None:
    received: list[object] = []

    class FakeHandler:
        async def handle(self, envelope: object) -> None:
            received.append(envelope)

    handler = FakeHandler()
    callback = _make_dispatch_callback(handler)
    envelope = MagicMock()
    asyncio.run(callback(envelope))
    assert len(received) == 1
    assert received[0] is envelope


# ---------------------------------------------------------------------------
# Tests: _make_projection_dispatch_callback (db injection path)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_projection_callback_injects_db_and_event_type() -> None:
    """Handler receives a dict with _db and _event_type injected."""
    received: list[dict] = []

    class FakeHandler:
        def handle(self, input_data: dict) -> dict:
            received.append(dict(input_data))
            return {"rows_upserted": 1}

    db_tables = [{"name": "node_service_registry", "database": "omnidash_analytics"}]
    handler = FakeHandler()
    callback = _make_projection_dispatch_callback(
        handler, db_tables, ("onex.evt.platform.node-heartbeat.v1",)
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    envelope.payload = {"service_name": "svc-a", "health_status": "healthy"}

    fake_adapter = MagicMock()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
            asyncio.run(callback(envelope))

    assert len(received) == 1
    assert received[0]["_event_type"] == "heartbeat"
    assert received[0]["_db"] is fake_adapter


@pytest.mark.unit
def test_projection_callback_runs_sync_handler_outside_active_loop() -> None:
    """Sync projection handlers may legitimately own their own event loop."""

    class LoopOwningHandler:
        def handle(self, input_data: dict) -> dict:
            async def _project() -> dict:
                return {"event_type": input_data["_event_type"]}

            return asyncio.run(_project())

    db_tables = [{"name": "delegation_events", "database": "omnidash_analytics"}]
    callback = _make_projection_dispatch_callback(
        LoopOwningHandler(),
        db_tables,
        ("onex.evt.omniclaude.task-delegated.v1",),
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.omniclaude.task-delegated.v1"
    envelope.payload = {"correlation_id": "corr-1", "task_type": "release-proof"}

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            result = asyncio.run(callback(envelope))

    assert result is None


@pytest.mark.unit
def test_projection_callback_connects_runner_db_before_handle() -> None:
    """BaseProjectionRunner-style handlers need their async DB pool initialized."""

    class FakeRunnerDb:
        def __init__(self) -> None:
            self._pool = None
            self.connected = False

        async def connect(self) -> None:
            await asyncio.sleep(0)
            self._pool = object()
            self.connected = True

    class DelegationProjectionRunner:
        def __init__(self) -> None:
            self.db = FakeRunnerDb()
            self.handled = False

        def handle(self, input_data: dict) -> dict:
            assert self.db.connected is True
            self.handled = True
            return {"projected": True}

    handler = DelegationProjectionRunner()
    db_tables = [{"name": "delegation_events", "database": "omnidash_analytics"}]
    callback = _make_projection_dispatch_callback(
        handler,
        db_tables,
        ("onex.evt.omniclaude.task-delegated.v1",),
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.omniclaude.task-delegated.v1"
    envelope.payload = {"correlation_id": "corr-1", "task_type": "release-proof"}

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=MagicMock()):
            result = asyncio.run(callback(envelope))

    assert result is None
    assert handler.handled is True
    assert handler.db.connected is True


@pytest.mark.unit
def test_projection_callback_skips_standalone_projection_runner() -> None:
    """Standalone Kafka runners are not safe as direct DB-injection callbacks."""

    class DelegationProjectionRunner:
        topics = ["onex.evt.omniclaude.task-delegated.v1"]

        def __init__(self) -> None:
            self.db = object()
            self.called = False

        async def project_event(self) -> None:
            self.called = True

        def handle(self, input_data: dict) -> dict:
            self.called = True
            return {"projected": True}

    handler = DelegationProjectionRunner()
    db_tables = [{"name": "delegation_events", "database": "omnidash_analytics"}]
    callback = _make_projection_dispatch_callback(
        handler,
        db_tables,
        ("onex.evt.omniclaude.task-delegated.v1",),
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.omniclaude.task-delegated.v1"
    envelope.payload = {"correlation_id": "corr-1", "task_type": "release-proof"}

    result = asyncio.run(callback(envelope))

    assert result is None
    assert handler.called is False


@pytest.mark.unit
def test_projection_callback_maps_introspection_event_type() -> None:
    """node-introspection topic maps to _event_type='introspection'."""
    received: list[dict] = []

    class FakeHandler:
        def handle(self, input_data: dict) -> dict:
            received.append(dict(input_data))
            return {}

    db_tables = [{"name": "node_service_registry", "database": "omnidash_analytics"}]
    handler = FakeHandler()
    callback = _make_projection_dispatch_callback(
        handler, db_tables, ("onex.evt.platform.node-introspection.v1",)
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-introspection.v1"
    envelope.payload = {"service_name": "svc-b"}
    fake_adapter = MagicMock()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
            asyncio.run(callback(envelope))

    assert len(received) == 1
    assert received[0]["_event_type"] == "introspection"


@pytest.mark.unit
def test_projection_callback_awaits_async_handle() -> None:
    """Async projection handlers are awaited after DB and event-type injection."""
    received: list[dict] = []

    class FakeHandler:
        async def handle(self, input_data: dict) -> dict:
            await asyncio.sleep(0)
            received.append(dict(input_data))
            return {"rows_upserted": 1}

    db_tables = [{"name": "node_service_registry", "database": "omnidash_analytics"}]
    handler = FakeHandler()
    callback = _make_projection_dispatch_callback(
        handler, db_tables, ("onex.evt.platform.node-heartbeat.v1",)
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    envelope.payload = {"service_name": "svc-a"}
    fake_adapter = MagicMock()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
            asyncio.run(callback(envelope))

    assert len(received) == 1
    assert received[0]["_event_type"] == "heartbeat"
    assert received[0]["_db"] is fake_adapter


@pytest.mark.unit
def test_projection_callback_skips_when_db_url_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When DB URL env var is absent, optional projections are skipped cleanly."""
    call_count = [0]

    class FakeHandler:
        def handle(self, input_data: dict) -> dict:
            call_count[0] += 1
            return {}

    db_tables = [{"name": "node_service_registry", "database": "omnidash_analytics"}]
    handler = FakeHandler()
    callback = _make_projection_dispatch_callback(handler, db_tables, ())

    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    envelope.payload = {}

    with caplog.at_level(
        logging.INFO, logger="omnibase_infra.runtime.auto_wiring.handler_wiring"
    ):
        with patch(_PATCH_ENVIRON_GET, return_value=""):
            result = asyncio.run(callback(envelope))

    assert result is None
    assert call_count[0] == 0
    assert any(
        r.levelno == logging.INFO and "Projection handler inactive" in r.message
        for r in caplog.records
    )


@pytest.mark.unit
def test_projection_callback_logs_type_error_not_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """TypeError from handler is logged (not propagated), and log entry is emitted."""

    class BrokenHandler:
        def handle(self, input_data: dict) -> dict:
            raise TypeError("missing _db")

    db_tables = [{"name": "node_service_registry", "database": "omnidash_analytics"}]
    handler = BrokenHandler()
    callback = _make_projection_dispatch_callback(handler, db_tables, ())

    envelope = MagicMock()
    envelope.topic = "onex.evt.platform.node-heartbeat.v1"
    envelope.payload = {}
    fake_adapter = MagicMock()

    with caplog.at_level(
        logging.ERROR, logger="omnibase_infra.runtime.auto_wiring.handler_wiring"
    ):
        with patch(
            _PATCH_ENVIRON_GET,
            return_value="postgresql://user:pass@host:5432/omnidash_analytics",
        ):
            with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
                result = asyncio.run(callback(envelope))

    assert result is None
    assert any("TypeError" in r.message for r in caplog.records)


@pytest.mark.unit
def test_sync_db_adapter_accepts_multi_column_conflict_key() -> None:
    """Projection DB adapter preserves protocol support for composite UPSERT keys."""
    cursor = MagicMock()
    cursor_context = MagicMock()
    cursor_context.__enter__.return_value = cursor
    conn = MagicMock()
    conn.closed = False
    conn.cursor.return_value = cursor_context

    with patch("psycopg2.connect", return_value=conn):
        adapter = _build_sync_db_adapter("postgresql://user:pass@host/db")
        result = adapter.upsert(
            "savings_estimates",
            "session_id,event_timestamp,model_local,model_cloud_baseline",
            {
                "session_id": "sess-1",
                "event_timestamp": "2026-05-20T20:00:00+00:00",
                "model_local": "local-model",
                "model_cloud_baseline": "cloud-model",
                "savings_usd": "0.001",
            },
        )

    assert result is True
    sql = cursor.execute.call_args.args[0]
    assert (
        'ON CONFLICT ("session_id", "event_timestamp", '
        '"model_local", "model_cloud_baseline") DO UPDATE SET'
    ) in sql
    assert '"savings_usd" = EXCLUDED."savings_usd"' in sql


@pytest.mark.unit
def test_sync_db_adapter_json_adapts_list_values() -> None:
    """JSONB list fields must not be sent to Postgres as text arrays."""
    import psycopg2.extras

    cursor = MagicMock()
    cursor_context = MagicMock()
    cursor_context.__enter__.return_value = cursor
    conn = MagicMock()
    conn.closed = False
    conn.cursor.return_value = cursor_context

    with patch("psycopg2.connect", return_value=conn):
        adapter = _build_sync_db_adapter("postgresql://user:pass@host/db")
        result = adapter.upsert(
            "delegation_events",
            "correlation_id",
            {
                "correlation_id": "corr-1",
                "quality_gates_checked": 1,
                "quality_gates_checked_jsonb": ["delegate-skill-terminal"],
            },
        )

    assert result is True
    params = cursor.execute.call_args.args[1]
    assert isinstance(params["quality_gates_checked_jsonb"], psycopg2.extras.Json)


# ---------------------------------------------------------------------------
# Tests: terminal event emission (OMN-11187)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_projection_callback_emits_terminal_event_on_success() -> None:
    """After a successful projection, terminal event is published to event_bus."""
    import json
    import uuid

    published: list[tuple] = []
    test_correlation_id = uuid.uuid4()

    class FakeHandler:
        def handle(self, input_data: dict) -> dict:
            return {"rows_upserted": 1}

    class FakeEventBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    db_tables = [{"name": "delegation_events", "database": "omnidash_analytics"}]
    handler = FakeHandler()
    fake_bus = FakeEventBus()
    terminal_topic = "onex.evt.omnimarket.projection-delegation-applied.v1"
    callback = _make_projection_dispatch_callback(
        handler,
        db_tables,
        ("onex.evt.omniclaude.task-delegated.v1",),
        event_bus=fake_bus,
        terminal_event=terminal_topic,
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.omniclaude.task-delegated.v1"
    envelope.payload = {"task_type": "code-review"}
    envelope.correlation_id = test_correlation_id
    fake_adapter = MagicMock()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
            asyncio.run(callback(envelope))

    assert len(published) == 1
    topic_published, _, raw_bytes = published[0]
    assert topic_published == terminal_topic
    parsed = json.loads(raw_bytes.decode("utf-8"))
    assert parsed["event_type"] == terminal_topic
    assert parsed["correlation_id"] == str(test_correlation_id)


@pytest.mark.unit
def test_projection_callback_emits_terminal_event_from_materialized_dict() -> None:
    """Terminal event correlation is extracted from materialized dispatch dicts."""
    import json
    import uuid

    published: list[tuple] = []
    test_correlation_id = uuid.uuid4()

    class FakeHandler:
        def handle(self, input_data: dict) -> dict:
            return {"rows_upserted": 1}

    class FakeEventBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    db_tables = [{"name": "delegation_events", "database": "omnidash_analytics"}]
    terminal_topic = "onex.evt.omnimarket.projection-delegation-applied.v1"
    callback = _make_projection_dispatch_callback(
        FakeHandler(),
        db_tables,
        ("onex.evt.omniclaude.task-delegated.v1",),
        event_bus=FakeEventBus(),
        terminal_event=terminal_topic,
    )
    envelope = {
        "payload": {"task_type": "code-review"},
        "__debug_trace": {
            "topic": "onex.evt.omniclaude.task-delegated.v1",
            "correlation_id": str(test_correlation_id),
        },
    }
    fake_adapter = MagicMock()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
            asyncio.run(callback(envelope))  # type: ignore[arg-type]

    assert len(published) == 1
    topic_published, _, raw_bytes = published[0]
    assert topic_published == terminal_topic
    parsed = json.loads(raw_bytes.decode("utf-8"))
    assert parsed["event_type"] == terminal_topic
    assert parsed["correlation_id"] == str(test_correlation_id)


@pytest.mark.unit
def test_projection_callback_does_not_emit_terminal_event_on_handler_error() -> None:
    """When handler raises, no terminal event is emitted."""
    published: list[tuple] = []

    class FailingHandler:
        def handle(self, input_data: dict) -> dict:
            raise RuntimeError("db failure")

    class FakeEventBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            published.append((topic, key, value))

    db_tables = [{"name": "delegation_events", "database": "omnidash_analytics"}]
    handler = FailingHandler()
    fake_bus = FakeEventBus()
    terminal_topic = "onex.evt.omnimarket.projection-delegation-applied.v1"
    callback = _make_projection_dispatch_callback(
        handler,
        db_tables,
        (),
        event_bus=fake_bus,
        terminal_event=terminal_topic,
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.omniclaude.task-delegated.v1"
    envelope.payload = {}
    envelope.correlation_id = "test-corr-id"
    fake_adapter = MagicMock()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
            asyncio.run(callback(envelope))

    assert len(published) == 0


@pytest.mark.unit
def test_projection_callback_no_terminal_event_when_bus_is_none() -> None:
    """When event_bus is None, no terminal event is emitted (no error)."""

    class FakeHandler:
        def handle(self, input_data: dict) -> dict:
            return {"rows_upserted": 1}

    db_tables = [{"name": "delegation_events", "database": "omnidash_analytics"}]
    handler = FakeHandler()
    callback = _make_projection_dispatch_callback(
        handler,
        db_tables,
        (),
        event_bus=None,
        terminal_event="onex.evt.omnimarket.projection-delegation-applied.v1",
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.omniclaude.task-delegated.v1"
    envelope.payload = {}
    envelope.correlation_id = "test-corr-id"
    fake_adapter = MagicMock()

    with patch(
        _PATCH_ENVIRON_GET,
        return_value="postgresql://user:pass@host:5432/omnidash_analytics",
    ):
        with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
            result = asyncio.run(callback(envelope))

    assert result is None


@pytest.mark.unit
def test_projection_callback_terminal_event_publish_failure_does_not_propagate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If terminal event publish fails, error is logged but not raised."""
    import uuid

    class FakeHandler:
        def handle(self, input_data: dict) -> dict:
            return {"rows_upserted": 1}

    class BrokenEventBus:
        async def publish(self, topic: str, key: object, value: bytes) -> None:
            raise OSError("kafka unavailable")

    db_tables = [{"name": "delegation_events", "database": "omnidash_analytics"}]
    handler = FakeHandler()
    callback = _make_projection_dispatch_callback(
        handler,
        db_tables,
        (),
        event_bus=BrokenEventBus(),
        terminal_event="onex.evt.omnimarket.projection-delegation-applied.v1",
    )

    envelope = MagicMock()
    envelope.topic = "onex.evt.omniclaude.task-delegated.v1"
    envelope.payload = {}
    envelope.correlation_id = uuid.uuid4()
    fake_adapter = MagicMock()

    with caplog.at_level(
        logging.ERROR, logger="omnibase_infra.runtime.auto_wiring.handler_wiring"
    ):
        with patch(
            _PATCH_ENVIRON_GET,
            return_value="postgresql://user:pass@host:5432/omnidash_analytics",
        ):
            with patch(_PATCH_BUILD_ADAPTER, return_value=fake_adapter):
                result = asyncio.run(callback(envelope))

    assert result is None
    assert any("projection terminal event" in r.message.lower() for r in caplog.records)
