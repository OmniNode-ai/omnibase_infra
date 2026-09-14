# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Gates and sequencing of the live application-database ACL apply path.

OMN-15355. These tests pin the refusals, not the SQL: the renderer and its
PostgreSQL 16 proof already own the SQL. What has never existed is a sanctioned
way to put that SQL onto a *live* database, so what is asserted here is every
condition under which the apply path must refuse, and the one ordering
constraint that makes a rollback possible at all -- the pre-change snapshot is
durable on disk before the first mutating statement runs.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.validation.application_database_acl import (
    build_application_database_acl_matrix,
)
from omnibase_infra.validation.application_database_acl_apply import (
    AclApplyRefusalError,
    apply_application_database_acl,
    assert_matrix_appliable,
    plan_connection_probes,
    resolve_consent_citation,
)
from omnibase_infra.validation.enums.enum_acl_connection_probe_kind import (
    EnumAclConnectionProbeKind,
)
from omnibase_infra.validation.enums.enum_application_database_acl_authorization_scope import (
    EnumApplicationDatabaseAclAuthorizationScope,
)
from omnibase_infra.validation.models.model_acl_apply_consent import (
    ModelAclApplyConsent,
)
from omnibase_infra.validation.models.model_acl_connection_probe import (
    ModelAclConnectionProbe,
    resolve_role_dsn_env_name,
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

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parents[2] / "fixtures" / "application_database_acl"

_CONSENT_ROW = (
    "2026-09-14T12:11:31Z | OPERATOR-CONSENT | lane=omn15355-change-window | "
    'approved_by=operator | "authorized" | APPROVED SCOPE: the OMN-15355 '
    "change window: live GRANT/REVOKE of PostgreSQL role and object privileges "
    "per the generated ACL matrix on the dev lane then onex-dev | OUT OF SCOPE: "
    "onex-prod; the public cluster | This row is the durable authorization "
    "evidence"
)


def _source(source_id: str, purpose: str) -> ModelApplicationDatabaseAclSource:
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


def _ready_matrix() -> ModelApplicationDatabaseAclMatrix:
    """A READY matrix in SYNTHETIC_PROOF scope, straight from the shared builder."""
    inventory = ModelApplicationRelationEvidenceInventory.model_validate(
        yaml.safe_load((_FIXTURES / "inventory.yaml").read_text(encoding="utf-8"))
    )
    principal_inventory = ModelApplicationDatabasePrincipalInventory.model_validate(
        yaml.safe_load(
            (_FIXTURES / "principal-inventory.yaml").read_text(encoding="utf-8")
        )
    )
    policy = ModelApplicationDatabaseAclPolicy.model_validate(
        yaml.safe_load((_FIXTURES / "acl-policy.yaml").read_text(encoding="utf-8"))
    )
    return build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=(
            _source("topology", "topology"),
            _source("inventory", "relation_inventory"),
            _source("principal_inventory", "principal_inventory"),
            _source("acl_policy", "acl_policy"),
        ),
        relation_inventories={"inventory": inventory},
        service_manifests={},
        principal_inventories={"principal_inventory": principal_inventory},
        acl_policies={"acl_policy": policy},
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )


def _blocked_matrix() -> ModelApplicationDatabaseAclMatrix:
    payload = _ready_matrix().model_dump(mode="json")
    payload["status"] = "BLOCKED"
    payload["blockers"] = ["ACL policy violation: seeded control"]
    return ModelApplicationDatabaseAclMatrix.model_validate(payload)


def _deployment_matrix() -> ModelApplicationDatabaseAclMatrix:
    payload = _ready_matrix().model_dump(mode="json")
    payload["authorization_scope"] = (
        EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT.value
    )
    payload["status"] = "BLOCKED"
    payload["scaffold_status"] = "BLOCKED"
    payload["blockers"] = ["ACL policy violation: seeded control"]
    payload["scaffold_blockers"] = ["ACL scaffold policy violation: seeded control"]
    return ModelApplicationDatabaseAclMatrix.model_validate(payload)


def _ledger(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "LEDGER.md"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Consent citation
# ---------------------------------------------------------------------------


def test_consent_citation_resolves_a_well_formed_operator_consent_row(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path, "header", _CONSENT_ROW)

    consent = resolve_consent_citation(
        f"{ledger}:2",
        ticket="OMN-15355",
        ledger_root=tmp_path,
    )

    assert isinstance(consent, ModelAclApplyConsent)
    assert consent.approved_by == "operator"
    assert consent.lane == "omn15355-change-window"
    assert consent.line_number == 2


def test_consent_citation_refuses_a_row_that_is_not_an_operator_consent_row(
    tmp_path: Path,
) -> None:
    ledger = _ledger(
        tmp_path, "2026-09-14T00:00:00Z | CLAIM | lane=x | ticket=OMN-15355"
    )

    with pytest.raises(AclApplyRefusalError, match="not an OPERATOR-CONSENT row"):
        resolve_consent_citation(
            f"{ledger}:1", ticket="OMN-15355", ledger_root=tmp_path
        )


def test_consent_citation_refuses_an_approver_outside_the_named_pair(
    tmp_path: Path,
) -> None:
    ledger = _ledger(
        tmp_path, _CONSENT_ROW.replace("approved_by=operator", "approved_by=lane")
    )

    with pytest.raises(AclApplyRefusalError, match="approver"):
        resolve_consent_citation(
            f"{ledger}:1", ticket="OMN-15355", ledger_root=tmp_path
        )


def test_consent_citation_refuses_a_row_missing_the_out_of_scope_half(
    tmp_path: Path,
) -> None:
    truncated = _CONSENT_ROW.split("| OUT OF SCOPE:")[0]
    ledger = _ledger(tmp_path, truncated)

    with pytest.raises(AclApplyRefusalError, match="OUT OF SCOPE"):
        resolve_consent_citation(
            f"{ledger}:1", ticket="OMN-15355", ledger_root=tmp_path
        )


def test_consent_citation_refuses_a_scope_that_does_not_name_the_ticket(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path, _CONSENT_ROW)

    with pytest.raises(AclApplyRefusalError, match="OMN-99999"):
        resolve_consent_citation(
            f"{ledger}:1", ticket="OMN-99999", ledger_root=tmp_path
        )


def test_consent_citation_refuses_a_line_number_past_the_end_of_the_ledger(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path, _CONSENT_ROW)

    with pytest.raises(AclApplyRefusalError, match="line 9"):
        resolve_consent_citation(
            f"{ledger}:9", ticket="OMN-15355", ledger_root=tmp_path
        )


def test_consent_citation_refuses_a_malformed_citation(tmp_path: Path) -> None:
    with pytest.raises(AclApplyRefusalError, match="citation"):
        resolve_consent_citation(
            "no-line-number.md", ticket="OMN-15355", ledger_root=tmp_path
        )


# ---------------------------------------------------------------------------
# Matrix eligibility
# ---------------------------------------------------------------------------


def test_appliable_matrix_must_carry_deployment_authorization_scope() -> None:
    with pytest.raises(AclApplyRefusalError, match="synthetic_proof"):
        assert_matrix_appliable(_ready_matrix())


def test_appliable_matrix_refuses_a_blocked_status_and_names_the_blockers() -> None:
    with pytest.raises(AclApplyRefusalError, match="seeded control"):
        assert_matrix_appliable(_deployment_matrix())


def test_appliable_matrix_refuses_a_blocked_synthetic_matrix_on_scope_first() -> None:
    with pytest.raises(AclApplyRefusalError, match="synthetic_proof"):
        assert_matrix_appliable(_blocked_matrix())


# ---------------------------------------------------------------------------
# Connection probes
# ---------------------------------------------------------------------------


def test_probe_plan_pairs_every_allowed_principal_positively() -> None:
    probes = plan_connection_probes(_ready_matrix())

    positive = {
        (probe.database, probe.principal)
        for probe in probes
        if probe.kind is EnumAclConnectionProbeKind.POSITIVE
    }
    assert positive == {
        ("omnidash_analytics", "app_dashboard"),
        ("omnidash_analytics", "omninode_runtime"),
        ("omnidash_analytics", "onex_api"),
        ("omnidash_analytics", "tenant_projection_writer"),
    }


def test_probe_plan_always_carries_a_public_connect_negative() -> None:
    probes = plan_connection_probes(_ready_matrix())

    public_negatives = [
        probe
        for probe in probes
        if probe.kind is EnumAclConnectionProbeKind.NEGATIVE_PUBLIC
    ]
    assert {probe.database for probe in public_negatives} == {"omnidash_analytics"}


def test_probe_plan_negates_a_declared_principal_absent_from_a_database() -> None:
    payload = _ready_matrix().model_dump(mode="json")
    payload["allowed_connect_principals"] = {"omnidash_analytics": ["app_dashboard"]}
    matrix = ModelApplicationDatabaseAclMatrix.model_validate(payload)

    negatives = {
        (probe.database, probe.principal)
        for probe in plan_connection_probes(matrix)
        if probe.kind is EnumAclConnectionProbeKind.NEGATIVE
    }
    assert ("omnidash_analytics", "omninode_runtime") in negatives
    assert ("omnidash_analytics", "app_dashboard") not in negatives


def test_probe_dsn_is_read_from_a_named_environment_variable_never_a_literal() -> None:
    assert resolve_role_dsn_env_name("app_dashboard") == "ACL_ROLE_DSN_APP_DASHBOARD"


def test_probe_renders_without_ever_carrying_a_credential() -> None:
    probe = ModelAclConnectionProbe(
        database="omnidash_analytics",
        principal="app_dashboard",
        kind=EnumAclConnectionProbeKind.POSITIVE,
    )

    assert "ACL_ROLE_DSN_APP_DASHBOARD" in probe.describe()
    assert "password" not in probe.describe().lower()


# ---------------------------------------------------------------------------
# Ordering: the snapshot is durable before the first mutation
# ---------------------------------------------------------------------------


def test_apply_writes_the_rollback_snapshot_before_it_mutates(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, _CONSENT_ROW)
    payload = _ready_matrix().model_dump(mode="json")
    payload["authorization_scope"] = (
        EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT.value
    )
    payload["verified_evidence_source_keys"] = []
    payload["sources"] = [
        *payload["sources"],
        _source("catalog_result", "catalog_result_evidence").model_dump(mode="json"),
        _source("activity_result", "activity_result_evidence").model_dump(mode="json"),
    ]
    payload["verified_evidence_source_keys"] = ["catalog_result", "activity_result"]
    matrix = ModelApplicationDatabaseAclMatrix.model_validate(payload)
    order: list[str] = []
    snapshot_path = tmp_path / "prechange.json"

    def capture(_: ModelApplicationDatabaseAclMatrix) -> dict[str, object]:
        order.append("capture")
        return {"database_acl": []}

    def execute(_: str) -> None:
        order.append("execute")

    def probe(_: ModelAclConnectionProbe) -> bool:
        order.append("probe")
        return True

    apply_application_database_acl(
        matrix,
        consent_citation=f"{ledger}:1",
        ticket="OMN-15355",
        ledger_root=tmp_path,
        snapshot_path=snapshot_path,
        capture_snapshot=capture,
        execute_sql=execute,
        run_probe=probe,
        execute=True,
    )

    assert order[0] == "capture"
    assert order[1] == "execute"
    assert snapshot_path.is_file()
    assert order.index("execute") < order.index("probe")


def test_apply_refuses_to_mutate_when_the_snapshot_cannot_be_written(
    tmp_path: Path,
) -> None:
    ledger = _ledger(tmp_path, _CONSENT_ROW)
    payload = _ready_matrix().model_dump(mode="json")
    payload["authorization_scope"] = (
        EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT.value
    )
    payload["sources"] = [
        *payload["sources"],
        _source("catalog_result", "catalog_result_evidence").model_dump(mode="json"),
        _source("activity_result", "activity_result_evidence").model_dump(mode="json"),
    ]
    payload["verified_evidence_source_keys"] = ["catalog_result", "activity_result"]
    matrix = ModelApplicationDatabaseAclMatrix.model_validate(payload)
    executed: list[str] = []

    with pytest.raises(AclApplyRefusalError, match="snapshot"):
        apply_application_database_acl(
            matrix,
            consent_citation=f"{ledger}:1",
            ticket="OMN-15355",
            ledger_root=tmp_path,
            snapshot_path=tmp_path / "missing-directory" / "prechange.json",
            capture_snapshot=lambda _: {"database_acl": []},
            execute_sql=lambda sql: executed.append(sql),
            run_probe=lambda _: True,
            execute=True,
        )

    assert executed == []


def test_apply_without_execute_is_a_dry_run_that_never_mutates(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, _CONSENT_ROW)
    payload = _ready_matrix().model_dump(mode="json")
    payload["authorization_scope"] = (
        EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT.value
    )
    payload["sources"] = [
        *payload["sources"],
        _source("catalog_result", "catalog_result_evidence").model_dump(mode="json"),
        _source("activity_result", "activity_result_evidence").model_dump(mode="json"),
    ]
    payload["verified_evidence_source_keys"] = ["catalog_result", "activity_result"]
    matrix = ModelApplicationDatabaseAclMatrix.model_validate(payload)
    executed: list[str] = []

    report = apply_application_database_acl(
        matrix,
        consent_citation=f"{ledger}:1",
        ticket="OMN-15355",
        ledger_root=tmp_path,
        snapshot_path=tmp_path / "prechange.json",
        capture_snapshot=lambda _: {"database_acl": []},
        execute_sql=lambda sql: executed.append(sql),
        run_probe=lambda _: True,
        execute=False,
    )

    assert executed == []
    assert report.mutated is False
    assert report.probes_run == 0


def test_apply_restores_the_snapshot_when_a_probe_fails(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, _CONSENT_ROW)
    payload = _ready_matrix().model_dump(mode="json")
    payload["authorization_scope"] = (
        EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT.value
    )
    payload["sources"] = [
        *payload["sources"],
        _source("catalog_result", "catalog_result_evidence").model_dump(mode="json"),
        _source("activity_result", "activity_result_evidence").model_dump(mode="json"),
    ]
    payload["verified_evidence_source_keys"] = ["catalog_result", "activity_result"]
    matrix = ModelApplicationDatabaseAclMatrix.model_validate(payload)
    restored: list[Path] = []

    with pytest.raises(AclApplyRefusalError, match="probe"):
        apply_application_database_acl(
            matrix,
            consent_citation=f"{ledger}:1",
            ticket="OMN-15355",
            ledger_root=tmp_path,
            snapshot_path=tmp_path / "prechange.json",
            capture_snapshot=lambda _: {"database_acl": []},
            execute_sql=lambda _: None,
            run_probe=lambda _: False,
            restore_snapshot=restored.append,
            execute=True,
        )

    assert restored == [tmp_path / "prechange.json"]


def test_apply_fails_closed_when_a_probe_cannot_be_run_at_all(tmp_path: Path) -> None:
    """A probe with no credential is NOT a pass; an unverified principal fails."""
    ledger = _ledger(tmp_path, _CONSENT_ROW)
    payload = _ready_matrix().model_dump(mode="json")
    payload["authorization_scope"] = (
        EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT.value
    )
    payload["sources"] = [
        *payload["sources"],
        _source("catalog_result", "catalog_result_evidence").model_dump(mode="json"),
        _source("activity_result", "activity_result_evidence").model_dump(mode="json"),
    ]
    payload["verified_evidence_source_keys"] = ["catalog_result", "activity_result"]
    matrix = ModelApplicationDatabaseAclMatrix.model_validate(payload)

    def unavailable(_: ModelAclConnectionProbe) -> bool:
        raise LookupError("ACL_ROLE_DSN_APP_DASHBOARD is not set")

    with pytest.raises(AclApplyRefusalError, match="could not be run"):
        apply_application_database_acl(
            matrix,
            consent_citation=f"{ledger}:1",
            ticket="OMN-15355",
            ledger_root=tmp_path,
            snapshot_path=tmp_path / "prechange.json",
            capture_snapshot=lambda _: {"database_acl": []},
            execute_sql=lambda _: None,
            run_probe=unavailable,
            restore_snapshot=lambda _: None,
            execute=True,
        )


def test_consent_row_timestamp_is_carried_into_the_report(tmp_path: Path) -> None:
    ledger = _ledger(tmp_path, _CONSENT_ROW)

    consent = resolve_consent_citation(
        f"{ledger}:1", ticket="OMN-15355", ledger_root=tmp_path
    )

    assert consent.recorded_at == datetime(2026, 9, 14, 12, 11, 31, tzinfo=UTC)
    assert consent.recorded_at < datetime.now(UTC) + timedelta(days=1)
