# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Tests for the opt-in state_io runtime dispatch seam (OMN-14208).

Covers the wiring-selection disjointness matrix (G4): a contract declaring
``db_io`` only selects the projection callback, ``state_io`` only selects the
stateful callback, both is a wiring-time error, and neither falls back to the
plain dispatch callback. Also covers that ``payload_type_matcher`` scoping
(OMN-12416) survives the state_io wrapper, and that a missing
``OMNIBASE_INFRA_DB_URL`` fails closed at wiring time (not lazily per-dispatch,
unlike the optional db_io projection path).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from omnibase_core.models.errors import ModelOnexError
from omnibase_core.services.service_handler_resolver import ServiceHandlerResolver
from omnibase_core.services.service_local_handler_ownership_query import (
    ServiceLocalHandlerOwnershipQuery,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _prepare_handler_wiring,
    _read_state_io,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.state_io.state_store_adapter import StateIoUnconfiguredError

_THIS_MODULE = "tests.unit.runtime.test_handler_wiring_state_io"
_PATCH_IMPORT_HANDLER_CLASS = (
    "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class"
)


class ModelStateIoPayload(BaseModel):
    """Real importable event model used as the handler entry's event_model."""

    correlation_id: str


class _FakeHandler:
    async def handle(self, payload: object) -> None:
        return None


def _write_contract(
    tmp_path: Path,
    *,
    db_io: bool = False,
    state_io: bool = False,
    codec: str = "  codec:\n    module: tests.unit.runtime.test_handler_wiring_state_io\n    name: _FakeHandler\n",
) -> Path:
    contract_path = tmp_path / "contract.yaml"
    body = "name: node_local\n"
    if db_io:
        body += "db_io:\n  db_tables:\n    - name: some_table\n      database: omnidash_analytics\n"
    if state_io:
        body += (
            "state_io:\n"
            "  database: omnibase_infra\n"
            "  table: delegation_workflow_state\n"
            "  key: correlation_id\n" + codec
        )
    contract_path.write_text(body)
    return contract_path


def _contract(
    contract_path: Path, entry: ModelHandlerRoutingEntry
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_local",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name="node_local",
        package_name="test-pkg",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.cmd.test-service.shared-command.v1",),
            publish_topics=(),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="operation_match",
            handlers=(entry,),
        ),
    )


def _entry(with_event_model: bool = False) -> ModelHandlerRoutingEntry:
    return ModelHandlerRoutingEntry(
        handler=ModelHandlerRef(name="HandlerLocal", module=_THIS_MODULE),
        event_model=(
            ModelHandlerRef(name="ModelStateIoPayload", module=_THIS_MODULE)
            if with_event_model
            else None
        ),
        operation="local_op",
    )


def _prepare(contract_path: Path, entry: ModelHandlerRoutingEntry) -> object:
    contract = _contract(contract_path, entry)
    ownership = ServiceLocalHandlerOwnershipQuery(
        local_node_names=frozenset({contract.name})
    )
    resolver = ServiceHandlerResolver()
    with patch(_PATCH_IMPORT_HANDLER_CLASS, return_value=_FakeHandler):
        return _prepare_handler_wiring(
            contract=contract,
            entry=entry,
            dispatch_engine=None,
            resolver=resolver,
            ownership_query=ownership,
            event_bus=None,
            container=None,
        )


# ---------------------------------------------------------------------------
# Tests: _read_state_io
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_state_io_returns_block(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path, state_io=True)
    state_io = _read_state_io(contract_path)
    assert state_io["database"] == "omnibase_infra"
    assert state_io["table"] == "delegation_workflow_state"


@pytest.mark.unit
def test_read_state_io_empty_when_absent(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path)
    assert _read_state_io(contract_path) == {}


@pytest.mark.unit
def test_read_state_io_empty_on_missing_file() -> None:
    assert _read_state_io(Path("/nonexistent/contract.yaml")) == {}


# ---------------------------------------------------------------------------
# Tests: G4 disjointness matrix
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_db_io_only_selects_projection_callback(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path, db_io=True)
    prepared = _prepare(contract_path, _entry())
    assert prepared.dispatcher.__qualname__.startswith(  # type: ignore[attr-defined]
        "_make_projection_dispatch_callback"
    )


@pytest.mark.unit
def test_state_io_only_selects_stateful_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL", "postgresql://user:pass@host:5432/omnibase_infra"
    )
    contract_path = _write_contract(tmp_path, state_io=True)
    prepared = _prepare(contract_path, _entry())
    assert prepared.dispatcher.__qualname__.startswith(  # type: ignore[attr-defined]
        "_make_stateful_dispatch_callback"
    )


@pytest.mark.unit
def test_both_db_io_and_state_io_raises_wiring_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL", "postgresql://user:pass@host:5432/omnibase_infra"
    )
    contract_path = _write_contract(tmp_path, db_io=True, state_io=True)
    with pytest.raises(ModelOnexError, match="disjoint wiring arms"):
        _prepare(contract_path, _entry())


@pytest.mark.unit
def test_neither_selects_plain_dispatch_callback(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path)
    prepared = _prepare(contract_path, _entry())
    assert prepared.dispatcher.__qualname__.startswith(  # type: ignore[attr-defined]
        "_make_dispatch_callback"
    )


# ---------------------------------------------------------------------------
# Tests: payload_type_matcher survives state_io wrapping (OMN-12416)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_payload_type_matcher_survives_state_io_wrapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL", "postgresql://user:pass@host:5432/omnibase_infra"
    )
    contract_path = _write_contract(tmp_path, state_io=True)
    prepared = _prepare(contract_path, _entry(with_event_model=True))
    matcher = prepared.payload_type_matcher  # type: ignore[attr-defined]
    assert matcher is not None
    assert matcher(ModelStateIoPayload(correlation_id="corr-1")) is True
    assert matcher({"correlation_id": "corr-1"}) is True


@pytest.mark.unit
def test_no_event_model_produces_no_matcher_under_state_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL", "postgresql://user:pass@host:5432/omnibase_infra"
    )
    contract_path = _write_contract(tmp_path, state_io=True)
    prepared = _prepare(contract_path, _entry(with_event_model=False))
    assert prepared.payload_type_matcher is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests: fail-closed at wiring time
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_state_io_missing_db_url_raises_at_wiring_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike optional db_io (logs + returns None per dispatch), state_io is a
    REQUIRED durability seam: a missing DSN must fail the wiring call itself,
    not degrade silently at dispatch time."""
    monkeypatch.delenv("OMNIBASE_INFRA_DB_URL", raising=False)
    contract_path = _write_contract(tmp_path, state_io=True)
    with pytest.raises(StateIoUnconfiguredError, match="OMNIBASE_INFRA_DB_URL"):
        _prepare(contract_path, _entry())


@pytest.mark.unit
def test_state_io_missing_codec_raises_at_wiring_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL", "postgresql://user:pass@host:5432/omnibase_infra"
    )
    contract_path = _write_contract(tmp_path, state_io=True, codec="")
    with pytest.raises(ModelOnexError, match=r"state_io\.codec"):
        _prepare(contract_path, _entry())


@pytest.mark.unit
def test_state_io_unknown_database_raises_at_wiring_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "OMNIBASE_INFRA_DB_URL", "postgresql://user:pass@host:5432/omnibase_infra"
    )
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        "name: node_local\n"
        "state_io:\n"
        "  database: not_a_real_database\n"
        "  table: delegation_workflow_state\n"
        "  key: correlation_id\n"
        "  codec:\n"
        "    module: tests.unit.runtime.test_handler_wiring_state_io\n"
        "    name: _FakeHandler\n"
    )
    with pytest.raises(ModelOnexError, match=r"state_io\.database"):
        _prepare(contract_path, _entry())
