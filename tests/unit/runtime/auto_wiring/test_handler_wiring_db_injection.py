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
def test_projection_callback_skips_when_db_url_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When DB URL env var is absent, handler is NOT called and error is logged."""
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
        logging.ERROR, logger="omnibase_infra.runtime.auto_wiring.handler_wiring"
    ):
        with patch(_PATCH_ENVIRON_GET, return_value=""):
            result = asyncio.run(callback(envelope))

    assert result is None
    assert call_count[0] == 0
    assert any("not set" in r.message for r in caplog.records)


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
