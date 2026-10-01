# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The delegation eval-run supplemental bridges were retired (OMN-19793 / OMN-18863).

Infra vendored the ``delegation_eval_item_verdicts`` and
``delegation_eval_results`` migrations before omnimarket#3127 landed the
declaring node contract, so two hand-authored
``LEGACY_MIGRATION_TABLE_DECLARATIONS`` entries carried them in the interim.
The omnimarket contract pin advancing to 49ee50d5f1b0 carries omnimarket#3127,
so the bridges are redundant and were deleted from ``table_grant_derivation.py``
and ``_INTERIM_ENTRIES``.

This module asserts that the deletion stuck and did not drop what it bridged:

1. no supplemental bridge for either relation remains;
2. the vendored migration lineage and the shipped topology instances still
   grant both relations to ``tenant_projection_writer``; and
3. when the pinned omnimarket checkout is present, the pinned contracts alone
   declare both relations ``read_write`` in ``application.public``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from omnibase_infra.topology.table_grant_derivation import (
    LEGACY_MIGRATION_TABLE_DECLARATIONS,
    load_contract_declarations,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PROFILES = ("local", "onex-dev", "onex-prod")
_VENDOR = Path("docker/migrations/forward/nodes/node_projection_delegation_eval")
_RELATIONS = {
    "delegation_eval_item_verdicts": "0003_create_delegation_eval_item_verdicts.sql",
    "delegation_eval_results": "0004_create_delegation_eval_results.sql",
}
_PROOF_DEPENDENCIES = _REPO_ROOT / ".proof-dependencies"
_CONTRACTS_SUFFIX = ("src", "omnimarket", "nodes")


def _pinned_contracts_root() -> Path:
    mirror = _PROOF_DEPENDENCIES.joinpath("omnimarket-pin", *_CONTRACTS_SUFFIX)
    if mirror.is_dir():
        return mirror
    return _PROOF_DEPENDENCIES.joinpath("omnimarket", *_CONTRACTS_SUFFIX)


class TestTheDelegationEvalRunBridgesWereRetired:
    def test_no_supplemental_bridge_remains(self) -> None:
        carried = {
            declaration.table.name
            for declaration in LEGACY_MIGRATION_TABLE_DECLARATIONS
        }
        lingering = sorted(set(_RELATIONS) & carried)
        assert not lingering, (
            f"{lingering} still have supplemental LEGACY_MIGRATION_TABLE_"
            "DECLARATIONS entries, but the pinned omnimarket contracts "
            "(49ee50d5f1b0, omnimarket#3127) declare them."
        )

    @pytest.mark.parametrize("filename", sorted(_RELATIONS.values()))
    def test_the_migration_lineage_that_created_it_is_still_in_the_tree(
        self, filename: str
    ) -> None:
        assert (_REPO_ROOT / _VENDOR / filename).is_file()

    @pytest.mark.parametrize("relation", sorted(_RELATIONS))
    @pytest.mark.parametrize("profile", _PROFILES)
    def test_the_shipped_instance_still_grants_it(
        self, profile: str, relation: str
    ) -> None:
        instance = yaml.safe_load(
            (
                _REPO_ROOT
                / "src"
                / "omnibase_infra"
                / "topology"
                / "instances"
                / f"{profile}.yaml"
            ).read_text(encoding="utf-8")
        )
        shipped = [
            entry
            for entry in instance["databases"]["application"]["principals"][
                "tenant_projection_writer"
            ]["grants"]
            if entry["object_type"] == "TABLE"
            and entry["schema"] == "public"
            and relation in entry["objects"]
        ]
        assert len(shipped) == 1, (
            f"{profile} no longer grants {relation} to tenant_projection_writer; "
            "retiring the bridge must leave the shipped grant byte-identical"
        )
        assert set(shipped[0]["privileges"]) == {"SELECT", "INSERT", "UPDATE"}

    @pytest.mark.skipif(
        not _pinned_contracts_root().is_dir(),
        reason="requires the pinned omnimarket checkout under .proof-dependencies",
    )
    @pytest.mark.parametrize("relation", sorted(_RELATIONS))
    def test_the_pinned_contracts_declare_it(self, relation: str) -> None:
        declared = [
            declaration.table
            for declaration in load_contract_declarations(_pinned_contracts_root())
            if declaration.table.name == relation
        ]
        assert declared, f"the pinned omnimarket contracts do not declare {relation}"
        for table in declared:
            assert table.database_ref == "application"
            assert table.schema == "public"
            assert table.access == "read_write"
