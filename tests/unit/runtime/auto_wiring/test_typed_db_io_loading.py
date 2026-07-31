# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed ``db_io`` runtime loading and topology resolution (OMN-15418)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.runtime.auto_wiring.discovery import _parse_contract
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _build_projection_db_adapter,
    _make_projection_dispatch_callback,
    _resolve_projection_database_target,
)
from omnibase_infra.runtime.auto_wiring.models import ModelDiscoveredContract

pytestmark = pytest.mark.unit

_FIXTURE_ROOT = (
    Path(__file__).parents[3] / "fixtures" / "application_relation_ownership"
)


@pytest.fixture(autouse=True)
def _configured_projection_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMNIDASH_ANALYTICS_DB_URL", "postgresql://fixture")
    monkeypatch.setenv("OMNINODE_INTERNAL_DB_URL", "postgresql://fixture")


def _parse(path: Path) -> ModelDiscoveredContract:
    return _parse_contract(
        contract_path=path,
        entry_point_name="typed_projection",
        package_name="test-package",
        package_version="1.0.0",
    )


def test_discovery_instantiates_typed_db_ownership_subcontract(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        """\
name: typed_projection
node_type: EFFECT_GENERIC
contract_version: {major: 1, minor: 0, patch: 0}
db_io:
  db_tables:
    - name: delegation_events
      database_ref: application
      schema: tenant
      migration: nodes/node_projection_delegation/0001.sql
      access: read_write
      role: events
""",
        encoding="utf-8",
    )

    contract = _parse(contract_path)

    assert contract.db_io is not None
    assert isinstance(contract.db_io.db_tables[0], ModelDbTableDeclaration)
    assert contract.db_io.db_tables[0].database_ref == "application"
    assert contract.db_io.db_tables[0].schema == "tenant"


@pytest.mark.parametrize(
    "invalid_table",
    [
        """\
    - name: delegation_events
      schema: tenant
      migration: 0001.sql
      role: events
""",
        """\
    - name: delegation_events
      database_ref: application
      migration: 0001.sql
      role: events
""",
        """\
    - name: delegation_events
      database: omnidash_analytics
      database_ref: application
      schema: tenant
      migration: 0001.sql
      role: events
""",
    ],
)
def test_discovery_fails_closed_on_missing_or_parallel_location_fields(
    tmp_path: Path,
    invalid_table: str,
) -> None:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(
        "name: invalid_projection\n"
        "node_type: EFFECT_GENERIC\n"
        "db_io:\n"
        "  db_tables:\n"
        f"{invalid_table}",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        _parse(contract_path)


def test_projection_target_exposes_typed_database_schema_and_domain() -> None:
    topology = ModelDeploymentTopology.from_yaml(_FIXTURE_ROOT / "topology.yaml")
    table = ModelDbTableDeclaration(
        name="delegation_events",
        database_ref="application",
        schema="tenant",
        migration="nodes/node_projection_delegation/0001.sql",
        access="read_write",
        role="events",
    )

    target = _resolve_projection_database_target((table,), topology)

    assert target.database_refs == ("application",)
    assert target.physical_database == "omnidash_analytics"
    assert target.schemas == ("tenant",)
    assert target.domains == (EnumDatabaseSchemaDomain.TENANT,)
    assert target.dsn_envs == ("OMNIDASH_ANALYTICS_DB_URL",)
    assert [binding.binding_ref for binding in target.bindings] == ["tenant_projection"]


def test_projection_target_rejects_unknown_schema() -> None:
    topology = ModelDeploymentTopology.from_yaml(_FIXTURE_ROOT / "topology.yaml")
    table = ModelDbTableDeclaration(
        name="delegation_events",
        database_ref="application",
        schema="unknown_schema",
        migration="0001.sql",
        role="events",
    )

    with pytest.raises(ValueError, match="Unknown schema"):
        _resolve_projection_database_target((table,), topology)


def test_projection_target_preserves_multiple_schemas_in_one_database() -> None:
    topology = ModelDeploymentTopology.from_yaml(_FIXTURE_ROOT / "topology.yaml")
    tables = (
        ModelDbTableDeclaration(
            name="delegation_events",
            database_ref="application",
            schema="tenant",
            migration="0001.sql",
            role="events",
        ),
        ModelDbTableDeclaration(
            name="generation_events",
            database_ref="application",
            schema="omninode_internal",
            migration="0002.sql",
            role="generation_events",
        ),
    )

    target = _resolve_projection_database_target(tables, topology)

    assert target.physical_database == "omnidash_analytics"
    assert target.schemas == ("omninode_internal", "tenant")
    assert target.domains == (
        EnumDatabaseSchemaDomain.OMNINODE_INTERNAL,
        EnumDatabaseSchemaDomain.TENANT,
    )
    assert [table_target.table.name for table_target in target.table_targets] == [
        "delegation_events",
        "generation_events",
    ]


def test_non_tenant_target_selects_explicit_internal_operation() -> None:
    topology = ModelDeploymentTopology.from_yaml(_FIXTURE_ROOT / "topology.yaml")
    table = ModelDbTableDeclaration(
        name="generation_events",
        database_ref="application",
        schema="omninode_internal",
        migration="0002.sql",
        role="events",
    )
    target = _resolve_projection_database_target((table,), topology)

    adapter = _build_projection_db_adapter(
        {"omninode_runtime_service": "postgresql://fixture"},
        target,
        None,
        None,
    )

    assert callable(adapter.upsert)


def test_projection_adapter_selection_receives_resolved_domain_target() -> None:
    topology = ModelDeploymentTopology.from_yaml(_FIXTURE_ROOT / "topology.yaml")
    table = ModelDbTableDeclaration(
        name="delegation_events",
        database_ref="application",
        schema="tenant",
        migration="0001.sql",
        role="events",
    )
    target = _resolve_projection_database_target((table,), topology)
    received: list[dict[str, object]] = []

    class Handler:
        def handle(self, payload: dict[str, object]) -> dict[str, int]:
            received.append(payload)
            return {"rows_upserted": 1}

    callback = _make_projection_dispatch_callback(
        Handler(), target, ("onex.evt.omniclaude.task-delegated.v1",)
    )
    envelope = MagicMock(
        topic="onex.evt.omniclaude.task-delegated.v1",
        payload={"task_type": "verification"},
    )
    adapter = MagicMock()

    with (
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring.os.environ.get",
            return_value="postgresql://fixture",
        ),
        patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._build_projection_db_adapter",
            return_value=adapter,
        ) as build_adapter,
    ):

        async def invoke_callback() -> None:
            await callback(envelope)

        asyncio.run(invoke_callback())

    build_adapter.assert_called_once_with(
        {"tenant_projection": "postgresql://fixture"},
        target,
        None,
        envelope,
    )
    assert received[0]["_db"] is adapter
