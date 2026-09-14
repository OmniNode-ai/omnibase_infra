# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Incident replay for the live application-database ACL apply gate (OMN-15355).

The artifact is the real generated ACL matrix as it was committed to
``omnibase_infra`` on 2026-07-31 by omnibase_infra#2548 -- captured with
``git cat-file`` from the object, not retyped -- and it has never once been
appliable. It carries ``status: BLOCKED``, 332 policy blockers and 34 scaffold
blockers, an empty ``verified_evidence_source_keys``, and empty
``allowed_connect_principals`` and ``observed_connect_principals`` maps.

The false_green being replaced is the ABSENCE of any gate. Before this change
the renderer had exactly one consumer outside the unit tests, an ephemeral
PostgreSQL 16 proof container, and it called the renderer with
``allow_synthetic_proof=True``. Nothing anywhere read a matrix's status before
putting privilege on a live database, because nothing anywhere put privilege on
a live database: the only remaining route was hand-run ``psql``, which is
ungated by construction and leaves no snapshot to roll back to. Applying this
particular matrix by that route would have written its 332 blockers into live
ACLs -- ``app_dashboard`` and ``onex_api`` reaching ``tenant`` and
``platform_catalog``, ``omninode_runtime`` reaching ``omninode_internal`` --
which is the precise inverse of the separation the ticket exists to enforce.

The discriminator is load-bearing rather than a formality: a gate that refused
every matrix would satisfy the rejection trivially and would block the change
window permanently, so the same function is driven over a READY,
deployment-scoped matrix built by the same shared builder the real generator
calls, and is required to accept. The accept control cannot be made from the
captured bytes, and the test that measures why is part of this case.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_core.models.core.model_deployment_topology import (
    ModelDeploymentTopology,
)
from omnibase_infra.validation.application_database_acl import (
    build_application_database_acl_matrix,
)
from omnibase_infra.validation.application_database_acl_apply import (
    AclApplyRefusalError,
    assert_matrix_appliable,
)
from omnibase_infra.validation.enums.enum_application_database_acl_authorization_scope import (
    EnumApplicationDatabaseAclAuthorizationScope,
)
from omnibase_infra.validation.models.model_application_database_acl_matrix import (
    ModelApplicationDatabaseAclMatrix,
    ModelApplicationDatabaseAclSource,
)
from omnibase_infra.validation.models.model_application_database_acl_policy import (
    ModelApplicationDatabaseAclPolicy,
)
from omnibase_infra.validation.models.model_application_database_principal_inventory import (
    ModelApplicationDatabasePrincipalInventory,
)
from omnibase_infra.validation.models.model_application_relation_evidence_inventory import (
    ModelApplicationRelationEvidenceInventory,
)

pytestmark = pytest.mark.ci

_FIXTURE = (
    Path(__file__).parents[1]
    / "fixtures"
    / "omn15355"
    / "candidate-matrix.yaml.captured"
)
_FIXTURE_SHA256 = "8c61d33acdbe02bff8ade90465e99f571fe331543f534d273f7129515b685650"


_PROOF_FIXTURES = Path(__file__).parents[1] / "fixtures" / "application_database_acl"


def _proof_source(source_id: str, purpose: str) -> ModelApplicationDatabaseAclSource:
    return ModelApplicationDatabaseAclSource.model_validate(
        {
            "source_key": source_id,
            "repository": "synthetic/proof",
            "revision": "a" * 40,
            "path": f"proof/{source_id}.yaml",
            "sha256": "b" * 64,
            "purpose": purpose,
        }
    )


def _ready_deployment_matrix() -> ModelApplicationDatabaseAclMatrix:
    """Build a READY matrix through the real builder, scoped to deployment."""
    synthetic = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_PROOF_FIXTURES / "topology.yaml"),
        sources=(
            _proof_source("topology", "topology"),
            _proof_source("inventory", "relation_inventory"),
            _proof_source("principal_inventory", "principal_inventory"),
            _proof_source("acl_policy", "acl_policy"),
        ),
        relation_inventories={
            "inventory": ModelApplicationRelationEvidenceInventory.model_validate(
                yaml.safe_load(
                    (_PROOF_FIXTURES / "inventory.yaml").read_text(encoding="utf-8")
                )
            )
        },
        service_manifests={},
        principal_inventories={
            "principal_inventory": (
                ModelApplicationDatabasePrincipalInventory.model_validate(
                    yaml.safe_load(
                        (_PROOF_FIXTURES / "principal-inventory.yaml").read_text(
                            encoding="utf-8"
                        )
                    )
                )
            )
        },
        acl_policies={
            "acl_policy": ModelApplicationDatabaseAclPolicy.model_validate(
                yaml.safe_load(
                    (_PROOF_FIXTURES / "acl-policy.yaml").read_text(encoding="utf-8")
                )
            )
        },
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )
    payload = synthetic.model_dump(mode="json")
    payload["authorization_scope"] = (
        EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT.value
    )
    payload["sources"] = [
        *payload["sources"],
        _proof_source("catalog_result", "catalog_result_evidence").model_dump(
            mode="json"
        ),
        _proof_source("activity_result", "activity_result_evidence").model_dump(
            mode="json"
        ),
    ]
    payload["verified_evidence_source_keys"] = ["catalog_result", "activity_result"]
    return ModelApplicationDatabaseAclMatrix.model_validate(payload)


def _captured_bytes() -> bytes:
    return _FIXTURE.read_bytes()


def _captured_payload() -> dict[str, object]:
    payload = yaml.safe_load(_captured_bytes().decode("utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_the_fixture_is_the_captured_object_and_not_a_reconstruction() -> None:
    """R1: the bytes under test are the bytes the registry records."""
    digest = hashlib.sha256(_captured_bytes()).hexdigest()

    assert digest == _FIXTURE_SHA256


def test_the_real_gate_refuses_the_matrix_that_has_never_been_appliable() -> None:
    """The captured matrix must not reach a live database, and must say why."""
    matrix = ModelApplicationDatabaseAclMatrix.model_validate(_captured_payload())

    assert (
        matrix.authorization_scope
        is EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT
    )
    assert matrix.status == "BLOCKED"

    with pytest.raises(AclApplyRefusalError) as refusal:
        assert_matrix_appliable(matrix)

    message = str(refusal.value)
    assert "BLOCKED" in message
    # The refusal names the cross-domain reach an ungated apply would have
    # written into live ACLs, rather than refusing anonymously.
    assert "cross-domain privileges on tenant" in message
    assert "omninode_internal" in message


def test_the_captured_matrix_really_does_carry_the_blockers_this_case_claims() -> None:
    """A positive control on the artifact itself, not on the guard."""
    payload = _captured_payload()

    assert len(payload["blockers"]) == 332  # type: ignore[arg-type]
    assert len(payload["scaffold_blockers"]) == 34  # type: ignore[arg-type]
    assert payload["verified_evidence_source_keys"] == []
    # Empty CONNECT maps are the reason the change window cannot proceed on
    # this matrix: nothing has observed which principals actually connect.
    assert payload["allowed_connect_principals"] == {}
    assert payload["observed_connect_principals"] == {}


def test_the_captured_bytes_cannot_even_be_flipped_to_ready_by_hand() -> None:
    """A second, independent line of defence, measured rather than assumed.

    Editing ``status`` to READY on the captured payload does not produce a
    READY matrix. The model's own validator refuses it, and the reason it
    gives is the P0 prerequisite itself: a READY deployment matrix requires
    every locked catalog and activity result source to be semantically
    verified, and the captured artifact locks neither. The apply gate is
    therefore not the only thing standing between this artifact and a live
    database -- which is worth pinning, because a reader of the gate alone
    would reasonably assume a hand-edited YAML could walk past it.
    """
    payload = _captured_payload()
    payload["status"] = "READY"
    payload["blockers"] = []
    payload["scaffold_status"] = "READY"
    payload["scaffold_blockers"] = []

    with pytest.raises(
        ValidationError, match="requires every locked catalog/activity result source"
    ):
        ModelApplicationDatabaseAclMatrix.model_validate(payload)


def test_the_same_gate_accepts_a_ready_deployment_matrix() -> None:
    """Discriminator: a gate that refuses everything enforces nothing.

    The accept control is built by the SAME shared builder the real generator
    calls, from the committed proof fixtures, and is re-scoped to deployment
    with both locked evidence sources present -- the shape the change window
    will actually hand the gate once the catalog and full-day activity results
    exist. The captured artifact cannot supply this control itself, for the
    reason the test above measures.
    """
    ready = _ready_deployment_matrix()

    assert ready.status == "READY"
    assert (
        ready.authorization_scope
        is EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT
    )

    assert_matrix_appliable(ready)
