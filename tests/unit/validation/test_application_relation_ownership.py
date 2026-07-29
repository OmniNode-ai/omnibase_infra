# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Global application-relation ownership validation (OMN-15418)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.validation.application_relation_ownership import (
    EnumApplicationRelationKind,
    EnumApplicationRelationPurpose,
    EnumApplicationRelationViolation,
    _table_purpose,
    assert_application_relation_ownership,
    load_application_relation_inventory,
    load_service_ownership_manifest,
    validate_application_relation_ownership,
)

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parents[2] / "fixtures" / "application_relation_ownership"


def _topology() -> ModelDeploymentTopology:
    return ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml")


def test_node_and_service_sources_cover_every_relation_once() -> None:
    report = validate_application_relation_ownership(
        topology=_topology(),
        node_contract_paths=(_FIXTURES / "node-owner.yaml",),
        service_manifest_paths=(_FIXTURES / "service-owner.yaml",),
        inventory=load_application_relation_inventory(_FIXTURES / "inventory.yaml"),
    )

    assert report.is_valid
    assert not report.violations
    assert {declaration.kind for declaration in report.declarations} == {
        EnumApplicationRelationKind.TABLE,
        EnumApplicationRelationKind.VIEW,
        EnumApplicationRelationKind.MATERIALIZED_VIEW,
        EnumApplicationRelationKind.FUNCTION,
    }
    assert any(
        declaration.purpose is EnumApplicationRelationPurpose.MIGRATION_LEDGER
        for declaration in report.declarations
    )


def test_read_only_node_declaration_is_an_explicit_reader_not_second_owner() -> None:
    report = validate_application_relation_ownership(
        topology=_topology(),
        node_contract_paths=(
            _FIXTURES / "node-owner.yaml",
            _FIXTURES / "node-reader.yaml",
        ),
        service_manifest_paths=(_FIXTURES / "service-owner.yaml",),
        inventory=load_application_relation_inventory(_FIXTURES / "inventory.yaml"),
    )

    assert report.is_valid
    delegation = next(
        declaration
        for declaration in report.declarations
        if declaration.name == "delegation_events"
        and declaration.owner_declaration is not None
    )
    assert delegation.owner_declaration == "node:node_projection_delegation"
    assert "node:node_delegation_reader" in report.readers_for(delegation.identity)


@pytest.mark.parametrize(
    ("inventory_name", "node_names", "service_names", "expected"),
    [
        (
            "inventory-with-missing.yaml",
            ("node-owner.yaml",),
            ("service-owner.yaml",),
            EnumApplicationRelationViolation.MISSING_OWNER,
        ),
        (
            "inventory.yaml",
            ("node-owner.yaml",),
            ("service-owner.yaml", "duplicate-owner.yaml"),
            EnumApplicationRelationViolation.DUPLICATE_OWNER,
        ),
        (
            "inventory.yaml",
            ("node-owner.yaml", "conflicting-location.yaml"),
            ("service-owner.yaml",),
            EnumApplicationRelationViolation.CONFLICTING_LOCATION,
        ),
        (
            "inventory-without-delegation.yaml",
            ("node-owner.yaml",),
            ("service-owner.yaml",),
            EnumApplicationRelationViolation.UNKNOWN_RELATION,
        ),
        (
            "inventory.yaml",
            ("node-owner.yaml",),
            ("blocked-service.yaml",),
            EnumApplicationRelationViolation.BLOCKED_RELATION,
        ),
        (
            "inventory.yaml",
            ("node-owner.yaml",),
            ("incomplete-census.yaml",),
            EnumApplicationRelationViolation.INCOMPLETE_CENSUS,
        ),
    ],
)
def test_seeded_ownership_defects_fail_closed(
    inventory_name: str,
    node_names: tuple[str, ...],
    service_names: tuple[str, ...],
    expected: EnumApplicationRelationViolation,
) -> None:
    report = validate_application_relation_ownership(
        topology=_topology(),
        node_contract_paths=tuple(_FIXTURES / name for name in node_names),
        service_manifest_paths=tuple(_FIXTURES / name for name in service_names),
        inventory=load_application_relation_inventory(_FIXTURES / inventory_name),
    )

    assert not report.is_valid
    assert expected in {violation.code for violation in report.violations}
    with pytest.raises(ModelOnexError, match=expected.value):
        assert_application_relation_ownership(report)


def test_service_loader_reuses_strict_typed_table_declaration() -> None:
    with pytest.raises(ValidationError, match="database"):
        load_service_ownership_manifest(_FIXTURES / "parallel-location.yaml")


def test_unknown_topology_schema_fails_closed() -> None:
    report = validate_application_relation_ownership(
        topology=_topology(),
        node_contract_paths=(_FIXTURES / "unknown-schema.yaml",),
        service_manifest_paths=(),
        inventory=load_application_relation_inventory(
            _FIXTURES / "inventory-unknown-schema.yaml"
        ),
    )

    assert not report.is_valid
    assert EnumApplicationRelationViolation.UNKNOWN_SCHEMA in {
        violation.code for violation in report.violations
    }


def test_rich_omn15423_inventory_projects_without_dropping_blockers() -> None:
    inventory = load_application_relation_inventory(_FIXTURES / "rich-inventory.yaml")

    assert inventory.source_relation_count == 5
    assert [relation.name for relation in inventory.relations] == [
        "delegation_events",
        "delegation_summary",
        "resolve_delegation",
    ]
    assert inventory.excluded_database_objects == (
        "delegation_events_id_seq",
        "pgcrypto",
    )
    assert inventory.relations[0].database_ref == "application"
    assert inventory.relations[0].schema == "tenant"

    report = validate_application_relation_ownership(
        topology=_topology(),
        node_contract_paths=(),
        service_manifest_paths=(),
        inventory=inventory,
    )

    codes = {violation.code for violation in report.violations}
    assert EnumApplicationRelationViolation.BLOCKED_RELATION in codes
    assert EnumApplicationRelationViolation.INCOMPLETE_CENSUS in codes


def test_only_explicit_migration_ledger_roles_have_ledger_purpose() -> None:
    webhook_table = ModelDbTableDeclaration(
        name="stripe_webhook_events",
        database_ref="application",
        schema="omninode_internal",
        migration="20251215_m4_stripe_billing.sql",
        role="stripe_webhook_ledger",
    )
    migration_table = ModelDbTableDeclaration(
        name="schema_migrations",
        database_ref="application",
        schema="omninode_internal",
        migration="00000000_migrations_tracking.sql",
        role="canonical_migration_ledger",
    )

    assert _table_purpose(webhook_table) is EnumApplicationRelationPurpose.DATA
    assert (
        _table_purpose(migration_table)
        is EnumApplicationRelationPurpose.MIGRATION_LEDGER
    )
