# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Immutable mapping regressions for application-database evidence models."""

from __future__ import annotations

import json
from uuid import UUID

import pytest
import yaml

from omnibase_infra.validation.models.model_application_database_relation_state import (
    ModelApplicationDatabaseRelationState,
)
from omnibase_infra.validation.models.model_application_database_tenant_isolation_evidence import (
    ModelApplicationDatabaseTenantIsolationEvidence,
)

pytestmark = pytest.mark.unit

_TENANT_A = UUID("11111111-1111-1111-1111-111111111111")
_TENANT_B = UUID("22222222-2222-2222-2222-222222222222")


def _relation_state() -> ModelApplicationDatabaseRelationState:
    return ModelApplicationDatabaseRelationState.model_validate(
        {
            "declaration": {
                "name": "events",
                "database_ref": "application",
                "schema": "tenant",
                "kind": "table",
                "purpose": "data",
                "domain": "TENANT",
                "owner_declaration": "node:fixture_owner",
                "access": "write",
                "role": "projection_state",
                "source_path": "tests/fixtures/OMN-15361.yaml",
            },
            "declared_restrictive_policy_names": ("suspended_tenant_deny",),
            "restrictive_policy_proofs": {
                "suspended_tenant_deny": "pytest:tenant-suspension-isolation"
            },
        }
    )


def _tenant_evidence() -> ModelApplicationDatabaseTenantIsolationEvidence:
    return ModelApplicationDatabaseTenantIsolationEvidence(
        expected_rows_by_tenant={_TENANT_A: 2, _TENANT_B: 1},
        observed_rows_by_tenant={_TENANT_A: 2, _TENANT_B: 1},
        unset_context_rows=0,
        malformed_context_denied=True,
    )


def test_restrictive_policy_proofs_reject_in_place_mutation() -> None:
    state = _relation_state()

    with pytest.raises(TypeError, match="does not support item assignment"):
        state.restrictive_policy_proofs["suspended_tenant_deny"] = "forged"  # type: ignore[index]


@pytest.mark.parametrize(
    "field_name",
    ["expected_rows_by_tenant", "observed_rows_by_tenant"],
)
def test_tenant_row_count_mappings_reject_in_place_mutation(
    field_name: str,
) -> None:
    evidence = _tenant_evidence()
    row_counts = getattr(evidence, field_name)

    with pytest.raises(TypeError, match="does not support item assignment"):
        row_counts[_TENANT_A] = 99


def test_immutable_evidence_mappings_preserve_json_and_yaml_wire_shapes() -> None:
    relation = _relation_state()
    evidence = _tenant_evidence()
    expected_relation_proofs = {
        "suspended_tenant_deny": "pytest:tenant-suspension-isolation"
    }
    expected_row_counts = {
        str(_TENANT_A): 2,
        str(_TENANT_B): 1,
    }

    assert relation.model_dump(mode="json")["restrictive_policy_proofs"] == (
        expected_relation_proofs
    )
    assert json.loads(relation.model_dump_json())["restrictive_policy_proofs"] == (
        expected_relation_proofs
    )
    assert (
        yaml.safe_load(yaml.safe_dump(relation.model_dump(mode="json")))[
            "restrictive_policy_proofs"
        ]
        == expected_relation_proofs
    )

    evidence_json = evidence.model_dump(mode="json")
    assert evidence_json["expected_rows_by_tenant"] == expected_row_counts
    assert evidence_json["observed_rows_by_tenant"] == expected_row_counts
    assert json.loads(evidence.model_dump_json()) == evidence_json
    assert yaml.safe_load(yaml.safe_dump(evidence_json)) == evidence_json
