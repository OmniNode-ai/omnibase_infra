# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Static contract for the pinned generated application ACL candidate."""

from __future__ import annotations

from pathlib import Path

import yaml

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_infra.validation.application_database_acl import (
    PUBLIC_PRINCIPAL,
    validate_application_database_acl_matrix,
)
from omnibase_infra.validation.models.model_application_database_acl_matrix import (
    ModelApplicationDatabaseAclMatrix,
    ModelApplicationDatabaseAclSource,
)

_ROOT = Path(__file__).parents[2]
_PROOF = _ROOT / "docker" / "application-acl-proof"


def _candidate() -> ModelApplicationDatabaseAclMatrix:
    return ModelApplicationDatabaseAclMatrix.model_validate(
        yaml.safe_load(
            (_PROOF / "generated" / "candidate-matrix.yaml").read_text(encoding="utf-8")
        )
    )


def test_candidate_is_a_complete_fail_closed_projection_of_locked_sources() -> None:
    lock = yaml.safe_load((_PROOF / "source-lock.yaml").read_text(encoding="utf-8"))
    locked_sources = tuple(
        ModelApplicationDatabaseAclSource.model_validate(source)
        for source in lock["sources"]
    )
    matrix = _candidate()

    assert set(matrix.sources) == set(locked_sources)
    assert matrix.status == "BLOCKED"
    assert matrix.scaffold_status == "BLOCKED"
    assert matrix.scaffold_blockers
    assert len(matrix.objects) == 112
    assert len(matrix.rows) == 5 * (1 + 3 + len(matrix.objects))
    assert len(matrix.default_privileges) == 3 * 4 * 5
    assert matrix.declared_principals == {
        "application": (
            "app_dashboard",
            "omninode_runtime",
            "onex_api",
            "tenant_projection_writer",
        )
    }
    assert matrix.observed_principals == {"application": ()}
    assert matrix.absent_principals == {"application": ()}
    assert set(matrix.required_connect_databases) == {
        "keycloak",
        "omnibase_infra",
        "omnidash_analytics",
        "omninode_cloud",
        "omniclaude",
        "omniintelligence",
        "omnimemory",
        "umami",
    }
    assert not matrix.allowed_connect_principals
    assert not matrix.observed_connect_principals
    assert not matrix.absent_connect_principals
    assert not matrix.observed_connect_database_owners
    violations = validate_application_database_acl_matrix(matrix)
    assert violations
    assert any("cross-domain" in violation for violation in violations)
    assert any("not materialized" in violation for violation in violations)
    assert all(
        f"ACL policy violation: {violation}" in matrix.blockers
        for violation in violations
    )
    assert not matrix.database_owners
    assert not matrix.allowed_memberships
    assert all(not row.privileges for row in matrix.default_privileges)
    assert all(
        not row.privileges for row in matrix.rows if row.principal == PUBLIC_PRINCIPAL
    )


def test_candidate_retains_real_blockers_and_never_emits_blocked_sql() -> None:
    matrix = _candidate()
    blocker_text = "\n".join(matrix.blockers)

    assert "full_day_datname_usename_activity='blocked'" in blocker_text
    assert "live_catalog_parity='blocked'" in blocker_text
    assert "schema 'unresolved'" in blocker_text
    assert "no authoritative repository DDL" in blocker_text
    assert "principal_inventory" in blocker_text
    assert "acl_policy" in blocker_text
    assert "relation_counts.type is not inventoried" in blocker_text
    assert "relation_counts.procedure is not inventoried" in blocker_text
    assert "requires exactly one CONNECT policy" in blocker_text
    assert "full object ACL rendering is gated" in blocker_text
    assert "explicit function_signature" in blocker_text
    assert not (_PROOF / "generated" / "application-acl.sql").exists()


def test_matrix_spans_every_required_object_and_default_privilege_kind() -> None:
    matrix = _candidate()

    assert {obj.object_type for obj in matrix.objects} == {
        EnumDatabaseGrantObjectType.TABLE,
        EnumDatabaseGrantObjectType.SEQUENCE,
        EnumDatabaseGrantObjectType.FUNCTION,
    }
    # No source head declares a type object, but future TYPE privileges are still
    # deny-by-default for every actual owner and workload principal.
    assert {row.object_type for row in matrix.default_privileges} == {
        EnumDatabaseGrantObjectType.TABLE,
        EnumDatabaseGrantObjectType.SEQUENCE,
        EnumDatabaseGrantObjectType.FUNCTION,
        EnumDatabaseGrantObjectType.TYPE,
    }
