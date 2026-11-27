# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Regression tests for pruned relation grants (OMN-26444).

These tests catch the following failure modes:
- ``savings_priced_models`` gets no DELETE grant.
- ``floor_integration_rows`` or ``gate_store_reservations`` lose DELETE.
- an unlisted vendored table gains DELETE.
- the node contract or its entry point drifts from ``HandlerTableGrantDerivation``.
"""

import importlib
import importlib.resources
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_infra.handlers.handler_table_grant_derivation.handler import (
    PRUNED_RELATIONS,
    HandlerTableGrantDerivation,
)
from omnibase_infra.models.model_table_grant_prune_query import (
    ModelTableGrantPruneQuery,
)
from omnibase_infra.topology import load_topology_profile
from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    derive_topology_table_grants,
)

pytestmark = pytest.mark.unit


def _delete_grants(table: str) -> list:
    selection = [
        declaration
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name == table
    ]
    assert selection, f"no legacy declaration for {table!r}"

    derived = derive_topology_table_grants(load_topology_profile("local"), selection)
    grants: list = []
    for principal_grants in derived.per_database["application"].grants.values():
        for grant in principal_grants:
            if (
                table in grant.objects
                and EnumDatabasePrivilege.DELETE in grant.privileges
            ):
                grants.append(grant)
    return grants


def test_savings_priced_models_gets_a_delete_select_grant() -> None:
    grants = _delete_grants("savings_priced_models")

    assert len(grants) == 1
    grant = grants[0]
    assert grant.schema == "public"
    assert {privilege.value for privilege in grant.privileges} == {"DELETE", "SELECT"}


@pytest.mark.parametrize(
    "table",
    ["floor_integration_rows", "gate_store_reservations"],
)
def test_existing_pruned_relations_keep_delete(table: str) -> None:
    grants = _delete_grants(table)

    assert len(grants) == 1
    assert {privilege.value for privilege in grants[0].privileges} == {
        "DELETE",
        "SELECT",
    }


def test_an_unlisted_vendored_table_gets_no_delete() -> None:
    pruned_tables = {
        "gate_store_reservations",
        "floor_integration_rows",
        "savings_priced_models",
    }
    candidate = next(
        declaration
        for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        if declaration.table.name not in pruned_tables
        and declaration.table.schema == "omninode_internal"
    )

    assert _delete_grants(candidate.table.name) == []


def test_handler_answers_from_the_contract() -> None:
    handler = HandlerTableGrantDerivation()

    pruned = handler.handle(
        ModelTableGrantPruneQuery(
            namespace="public",
            table="savings_priced_models",
        )
    )
    assert {privilege.value for privilege in pruned.privileges} == {
        "DELETE",
        "SELECT",
    }

    unpruned = handler.handle(
        ModelTableGrantPruneQuery(
            namespace="omninode_internal",
            table="savings_priced_models",
        )
    )
    assert unpruned.privileges == ()


def test_the_node_is_registered_and_routes_the_handler() -> None:
    pyproject_path = Path(__file__).resolve().parents[4] / "pyproject.toml"
    with pyproject_path.open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file)

    assert (
        project["project"]["entry-points"]["onex.nodes"]["node_table_grant_derivation"]
        == "omnibase_infra.nodes.node_table_grant_derivation"
    )

    module_name = "omnibase_infra.nodes.node_table_grant_derivation"
    contract = yaml.safe_load(
        importlib.resources.files(module_name).joinpath("contract.yaml").read_text()
    )
    handlers_raw = contract["handler_routing"]["handlers"]
    handler_routing = SimpleNamespace(
        handlers=[
            SimpleNamespace(handler=SimpleNamespace(**entry.get("handler", entry)))
            for entry in handlers_raw
        ]
    )
    handler_ref = handlers_raw[0].get("handler", handlers_raw[0])

    assert handler_routing.handlers[0].handler.name == "HandlerTableGrantDerivation"
    assert (
        getattr(
            importlib.import_module(handler_ref["module"]),
            handler_ref["name"],
        )
        is HandlerTableGrantDerivation
    )


def test_the_contract_declares_exactly_the_handlers_pruned_relations() -> None:
    contract = yaml.safe_load(
        importlib.resources.files("omnibase_infra.nodes.node_table_grant_derivation")
        .joinpath("contract.yaml")
        .read_text(encoding="utf-8")
    )
    declared = {
        (entry["schema"], entry["table"])
        for entry in contract["config"]["pruned_relations"]
    }
    assert declared == set(PRUNED_RELATIONS)
    assert all(entry["reason"] for entry in contract["config"]["pruned_relations"])
