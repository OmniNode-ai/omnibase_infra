# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Generated application-database ACL matrix tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.validation.application_database_acl import (
    PUBLIC_PRINCIPAL,
    build_application_database_acl_matrix,
    render_application_database_acl_sql,
    validate_application_database_acl_matrix,
    validate_application_database_acl_scaffold,
    validate_application_database_principal_evidence,
)
from omnibase_infra.validation.enums.enum_application_database_acl_authorization_scope import (
    EnumApplicationDatabaseAclAuthorizationScope,
)
from omnibase_infra.validation.enums.enum_application_database_acl_render_phase import (
    EnumApplicationDatabaseAclRenderPhase,
)
from omnibase_infra.validation.models.model_application_database_acl_matrix import (
    ModelApplicationDatabaseAclMatrix,
    ModelApplicationDatabaseAclSource,
)
from omnibase_infra.validation.models.model_application_database_acl_object import (
    ModelApplicationDatabaseAclObject,
)
from omnibase_infra.validation.models.model_application_database_acl_policy import (
    ModelApplicationDatabaseAclPolicy,
)
from omnibase_infra.validation.models.model_application_database_acl_row import (
    ModelApplicationDatabaseAclRow,
)
from omnibase_infra.validation.models.model_application_database_activity_result_evidence import (
    ModelApplicationDatabaseActivityResultEvidence,
)
from omnibase_infra.validation.models.model_application_database_catalog_result_evidence import (
    ModelApplicationDatabaseCatalogResultEvidence,
)
from omnibase_infra.validation.models.model_application_database_principal_inventory import (
    ModelApplicationDatabasePrincipalInventory,
)
from omnibase_infra.validation.models.model_application_relation_evidence_inventory import (
    ModelApplicationRelationEvidenceInventory,
)
from omnibase_infra.validation.models.model_migration_ownership_manifest import (
    ModelMigrationOwnershipManifest,
)

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).parents[2] / "fixtures" / "application_database_acl"


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


def _inventory() -> ModelApplicationRelationEvidenceInventory:
    return ModelApplicationRelationEvidenceInventory.model_validate(
        yaml.safe_load((_FIXTURES / "inventory.yaml").read_text(encoding="utf-8"))
    )


def _principal_inventory() -> ModelApplicationDatabasePrincipalInventory:
    return ModelApplicationDatabasePrincipalInventory.model_validate(
        yaml.safe_load(
            (_FIXTURES / "principal-inventory.yaml").read_text(encoding="utf-8")
        )
    )


def _acl_policy() -> ModelApplicationDatabaseAclPolicy:
    return ModelApplicationDatabaseAclPolicy.model_validate(
        yaml.safe_load((_FIXTURES / "acl-policy.yaml").read_text(encoding="utf-8"))
    )


def _authorized_inventory(
    *,
    active_principals: tuple[str, ...] = ("app_dashboard",),
    observation_count: int = 1,
) -> ModelApplicationDatabasePrincipalInventory:
    payload = _principal_inventory().model_dump(mode="json")
    started = datetime(2026, 7, 28, tzinfo=UTC)
    payload.update(
        {
            "source_kind": "authorized_catalog",
            "live_database_read": True,
            "catalog_query_sha256": "1" * 64,
            "catalog_result_sha256": "2" * 64,
            "catalog_query_source_key": "catalog_query",
            "catalog_result_source_key": "catalog_result",
            "activity_principal_refs": list(active_principals),
            "activity_evidence": {
                "window_started_at": started.isoformat(),
                "window_ended_at": (started + timedelta(hours=24)).isoformat(),
                "query_sha256": "3" * 64,
                "result_sha256": "4" * 64,
                "query_source_key": "activity_query",
                "result_source_key": "activity_result",
                "observation_count": observation_count,
            },
        }
    )
    return ModelApplicationDatabasePrincipalInventory.model_validate(payload)


def _complete_sources() -> tuple[ModelApplicationDatabaseAclSource, ...]:
    return (
        _source("topology", "topology"),
        _source("inventory", "relation_inventory"),
        _source("principal_inventory", "principal_inventory"),
        _source("acl_policy", "acl_policy"),
    )


def _matrix() -> ModelApplicationDatabaseAclMatrix:
    return build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )


def test_matrix_covers_every_principal_object_and_future_acl_cell() -> None:
    matrix = _matrix()

    assert matrix.status == "READY"
    assert not matrix.blockers
    assert len(matrix.objects) == 21
    # PUBLIC plus the exact seven-principal synthetic census across all targets.
    assert len(matrix.rows) == 8 * (1 + 3 + 21)
    # Three owners/schemas x four kinds x the complete eight-grantee universe.
    assert len(matrix.default_privileges) == 3 * 4 * 8
    assert matrix.required_connect_databases == ("omnidash_analytics",)
    assert matrix.allowed_connect_principals == {
        "omnidash_analytics": (
            "app_dashboard",
            "omninode_runtime",
            "onex_api",
            "tenant_projection_writer",
        )
    }
    partitioned = next(
        obj for obj in matrix.objects if obj.object_ref == "partitioned_events"
    )
    assert partitioned.catalog_kind == "table"
    assert not validate_application_database_acl_matrix(matrix)


def test_renderer_emits_exact_objects_and_deny_by_default_future_acls() -> None:
    sql = render_application_database_acl_sql(
        _matrix(),
        allow_synthetic_proof=True,
    )

    assert sql.splitlines()[sql.splitlines().index("\\set ON_ERROR_STOP on") + 1] == (
        "BEGIN;"
    )
    assert sql.rstrip().endswith("COMMIT;")
    assert "LOCK TABLE pg_catalog.pg_depend IN SHARE MODE;" in sql
    assert 'REVOKE ALL PRIVILEGES ON DATABASE "omnidash_analytics" FROM PUBLIC' in sql
    assert (
        'ALTER DATABASE "omnidash_analytics" OWNER TO "owner_platform_catalog";' in sql
    )
    database_revoke = next(
        line
        for line in sql.splitlines()
        if line.startswith("REVOKE ALL PRIVILEGES ON DATABASE")
    )
    assert '"untrusted_login"' in database_revoke
    assert 'CREATE ROLE "untrusted_login"' not in sql
    assert 'ALTER ROLE "untrusted_login"' not in sql
    assert 'ALTER ROLE "rls_admin"' not in sql
    assert (
        'ALTER TABLE "tenant"."delegation_events" OWNER TO "owner_onex_tenant";' in sql
    )
    assert (
        'ALTER TABLE "tenant"."partitioned_events" OWNER TO "owner_onex_tenant";' in sql
    )
    assert sql.index(
        'ALTER TABLE "tenant"."delegation_events" OWNER TO "owner_onex_tenant";'
    ) < sql.index(
        'ALTER SEQUENCE "tenant"."delegation_events_id_seq" OWNER TO '
        '"owner_onex_tenant";'
    )
    assert (
        'GRANT SELECT ON TABLE "tenant"."delegation_events" TO "app_dashboard";' in sql
    )
    assert 'ALTER VIEW "tenant"."tenant_account_names" OWNER TO' in sql
    assert (
        'GRANT SELECT ON TABLE "tenant"."tenant_account_names" TO "app_dashboard";'
    ) in sql
    assert (
        'GRANT "owner_onex_tenant" TO "db_migrator" WITH ADMIN FALSE, '
        "INHERIT FALSE, SET TRUE;"
    ) in sql
    assert "ALL TABLES IN SCHEMA" not in sql
    assert "ALL SEQUENCES IN SCHEMA" not in sql
    assert (
        'ALTER DEFAULT PRIVILEGES FOR ROLE "owner_onex_tenant" '
        "REVOKE ALL PRIVILEGES ON FUNCTIONS FROM PUBLIC CASCADE;"
    ) in sql
    assert (
        'ALTER DEFAULT PRIVILEGES FOR ROLE "owner_onex_tenant" IN SCHEMA '
        '"tenant" REVOKE ALL PRIVILEGES ON FUNCTIONS FROM PUBLIC CASCADE;'
    ) in sql
    assert (
        "'(\"arg''name%\" integer)'" in sql
        and '"hostile_signature"("arg\'name%" integer)' in sql
        and '"hostile_signature"("arg\'\'name%%" integer)' in sql
    )
    assert "defaults.defaclnamespace = 0" in sql
    assert (
        "OR namespace.nspname IN ('omninode_internal', 'platform_catalog', 'tenant')"
        in sql
    )


def test_scaffold_renders_while_nonmaterialized_full_phase_stays_blocked() -> None:
    ready = _matrix()
    staged = ready.model_copy(
        update={
            "status": "BLOCKED",
            "blockers": ("target objects are not materialized",),
            "objects": tuple(
                obj.model_copy(update={"target_materialized": False})
                for obj in ready.objects
            ),
        }
    )

    assert staged.scaffold_status == "READY"
    assert not validate_application_database_acl_scaffold(staged)
    scaffold = render_application_database_acl_sql(
        staged,
        allow_synthetic_proof=True,
        phase=EnumApplicationDatabaseAclRenderPhase.SCAFFOLD,
    )
    assert "-- Render phase: scaffold" in scaffold
    assert "current_database()" in scaffold
    assert "locked schema census is stale: expected present schema tenant" in scaffold
    assert 'ALTER SCHEMA "tenant"' not in scaffold
    assert 'REVOKE ALL PRIVILEGES ON SCHEMA "public"' not in scaffold
    assert 'REVOKE ALL PRIVILEGES ON DATABASE "omnidash_analytics"' not in scaffold
    assert "ALTER DEFAULT PRIVILEGES" not in scaffold
    assert "ALTER TABLE " not in scaffold
    assert "GRANT SELECT ON TABLE " not in scaffold
    with pytest.raises(ValueError, match="Cannot render blocked ACL matrix"):
        render_application_database_acl_sql(staged, allow_synthetic_proof=True)


def test_scaffold_creates_policy_declared_role_with_exact_absence_evidence() -> None:
    payload = _principal_inventory().model_dump(mode="json")
    payload["principal_refs"].remove("omninode_runtime")
    payload["absent_principal_refs"] = ["omninode_runtime"]
    payload["observed_role_states"] = [
        state
        for state in payload["observed_role_states"]
        if state["role"] != "omninode_runtime"
    ]
    for obj in payload["observed_objects"]:
        if obj["owner"] == "omninode_runtime":
            obj["owner"] = "app_dashboard"
    inventory = ModelApplicationDatabasePrincipalInventory.model_validate(payload)
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": inventory},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "READY", matrix.blockers
    assert matrix.absent_principals == {"application": ("omninode_runtime",)}
    rendered = render_application_database_acl_sql(
        matrix,
        allow_synthetic_proof=True,
        phase=EnumApplicationDatabaseAclRenderPhase.SCAFFOLD,
    )
    assert 'CREATE ROLE "omninode_runtime" LOGIN' in rendered


@pytest.mark.parametrize("unsafe_role", ["app_dashboard", "owner_onex_tenant"])
def test_unsafe_existing_roles_block_only_additive_scaffold(
    unsafe_role: str,
) -> None:
    payload = _principal_inventory().model_dump(mode="json")
    for state in payload["observed_role_states"]:
        if state["role"] == unsafe_role:
            state["create_role"] = True
            break
    inventory = ModelApplicationDatabasePrincipalInventory.model_validate(payload)

    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": inventory},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )

    assert matrix.status == "READY", matrix.blockers
    assert matrix.scaffold_status == "BLOCKED"
    assert any(unsafe_role in blocker for blocker in matrix.scaffold_blockers)
    full_sql = render_application_database_acl_sql(
        matrix,
        allow_synthetic_proof=True,
    )
    assert f'ALTER ROLE "{unsafe_role}"' in full_sql
    with pytest.raises(ValueError, match="blocked ACL scaffold"):
        render_application_database_acl_sql(
            matrix,
            allow_synthetic_proof=True,
            phase=EnumApplicationDatabaseAclRenderPhase.SCAFFOLD,
        )


def test_cluster_global_role_presence_cannot_conflict_with_absence_evidence() -> None:
    payload = _matrix().model_dump(mode="json")
    payload["absent_principals"] = {"application": ["service_only_role"]}
    payload["observed_connect_principals"]["omnidash_analytics"].append(
        "service_only_role"
    )

    with pytest.raises(ValidationError, match="cluster-global role presence/absence"):
        ModelApplicationDatabaseAclMatrix.model_validate(payload)


@pytest.mark.parametrize(
    "control_id",
    [
        "public-connect",
        "public-execute",
        "runtime-owner",
        "runtime-ddl",
        "unsafe-default-privilege",
    ],
)
def test_seeded_acl_red_control_fails_closed(control_id: str) -> None:
    matrix = _matrix()
    rows = list(matrix.rows)
    objects = list(matrix.objects)
    defaults = list(matrix.default_privileges)
    expected: str
    if control_id == "public-connect":
        target = next(
            row
            for row in rows
            if row.principal == PUBLIC_PRINCIPAL
            and row.object_type is EnumDatabaseGrantObjectType.DATABASE
        )
        rows[rows.index(target)] = target.model_copy(
            update={"privileges": (EnumDatabasePrivilege.CONNECT,)}
        )
        expected = "PUBLIC has"
    elif control_id == "public-execute":
        target = next(
            row
            for row in rows
            if row.principal == PUBLIC_PRINCIPAL
            and row.object_type is EnumDatabaseGrantObjectType.FUNCTION
        )
        rows[rows.index(target)] = target.model_copy(
            update={"privileges": (EnumDatabasePrivilege.EXECUTE,)}
        )
        expected = "PUBLIC has"
    elif control_id == "runtime-owner":
        objects[0] = objects[0].model_copy(update={"owner": "app_dashboard"})
        expected = "owns"
    elif control_id == "runtime-ddl":
        target = next(
            row
            for row in rows
            if row.principal == "app_dashboard"
            and row.object_type is EnumDatabaseGrantObjectType.SCHEMA
            and row.schema_ref == "tenant"
        )
        rows[rows.index(target)] = target.model_copy(
            update={
                "privileges": (
                    EnumDatabasePrivilege.USAGE,
                    EnumDatabasePrivilege.CREATE,
                )
            }
        )
        expected = "DDL privilege"
    else:
        target = next(
            row
            for row in defaults
            if row.grantee == "app_dashboard"
            and row.object_type is EnumDatabaseGrantObjectType.TABLE
        )
        defaults[defaults.index(target)] = target.model_copy(
            update={"privileges": (EnumDatabasePrivilege.SELECT,)}
        )
        expected = "broad future"
    red = matrix.model_copy(
        update={
            "objects": tuple(objects),
            "rows": tuple(rows),
            "default_privileges": tuple(defaults),
        }
    )

    violations = validate_application_database_acl_matrix(red)

    assert any(expected in violation for violation in violations)


def test_red_control_runtime_bypassrls() -> None:
    payload = _principal_inventory().model_dump(mode="json")
    role_state = next(
        state
        for state in payload["observed_role_states"]
        if state["role"] == "app_dashboard"
    )
    role_state["bypass_rls"] = True
    inventory = ModelApplicationDatabasePrincipalInventory.model_validate(payload)

    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": inventory},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )

    assert matrix.scaffold_status == "BLOCKED"
    assert any("app_dashboard" in blocker for blocker in matrix.scaffold_blockers)


def test_red_control_cross_domain_grant() -> None:
    topology = ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml")
    database = topology.databases["application"]
    runtime = database.principals["omninode_runtime"]
    dashboard = database.principals["app_dashboard"]
    tenant_usage = next(
        grant
        for grant in dashboard.grants
        if grant.object_type is EnumDatabaseGrantObjectType.SCHEMA
        and grant.schema == "tenant"
    )
    tenant_read = next(
        grant
        for grant in dashboard.grants
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and "delegation_events" in grant.objects
    )
    principals = dict(database.principals)
    principals["omninode_runtime"] = runtime.model_copy(
        update={"grants": (*runtime.grants, tenant_usage, tenant_read)}
    )
    topology = topology.model_copy(
        update={
            "databases": {
                "application": database.model_copy(update={"principals": principals})
            }
        }
    )

    matrix = build_application_database_acl_matrix(
        topology=topology,
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any("cross-domain" in blocker for blocker in matrix.blockers)
    assert any(
        "cross-domain" in violation
        for violation in validate_application_database_acl_matrix(matrix)
    )
    with pytest.raises(ValueError, match="Cannot render blocked ACL matrix"):
        render_application_database_acl_sql(matrix)


def test_authorized_principal_inventory_requires_full_day_activity() -> None:
    principal_inventory = _principal_inventory().model_copy(
        update={
            "source_kind": "authorized_catalog",
            "live_database_read": True,
            "activity_evidence": None,
            "catalog_query_sha256": None,
            "catalog_result_sha256": None,
        }
    )
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": principal_inventory},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT,
    )

    assert matrix.status == "BLOCKED"
    assert any("full-day" in blocker for blocker in matrix.blockers)


def test_incomplete_inventory_blocks_sql_without_fabricating_activity() -> None:
    inventory = _inventory().model_copy(
        update={"completion_status": "blocked_pending_live_catalog_and_activity"}
    )
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": inventory},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any("completion_status" in blocker for blocker in matrix.blockers)
    with pytest.raises(ValueError, match="Cannot render blocked ACL matrix"):
        render_application_database_acl_sql(matrix)


def test_missing_type_count_is_incomplete_not_silently_zero() -> None:
    payload = yaml.safe_load((_FIXTURES / "inventory.yaml").read_text(encoding="utf-8"))
    del payload["relation_counts"]["type"]
    inventory = ModelApplicationRelationEvidenceInventory.model_validate(payload)
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": inventory},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert inventory.relation_counts.type is None
    assert matrix.status == "BLOCKED"
    assert any(
        "relation_counts.type is not inventoried" in item for item in matrix.blockers
    )


def test_missing_procedure_count_is_incomplete_not_silently_zero() -> None:
    payload = yaml.safe_load((_FIXTURES / "inventory.yaml").read_text(encoding="utf-8"))
    del payload["relation_counts"]["procedure"]
    inventory = ModelApplicationRelationEvidenceInventory.model_validate(payload)
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": inventory},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert inventory.relation_counts.procedure is None
    assert matrix.status == "BLOCKED"
    assert any(
        "relation_counts.procedure is not inventoried" in item
        for item in matrix.blockers
    )


def test_relation_inventory_requires_explicit_ready_retained_census_status() -> None:
    payload = yaml.safe_load((_FIXTURES / "inventory.yaml").read_text(encoding="utf-8"))
    payload["retained_live_census"]["parity_status"] = None
    inventory = ModelApplicationRelationEvidenceInventory.model_validate(payload)
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": inventory},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )

    assert matrix.status == "BLOCKED"
    assert any("retained_live_census=None" in blocker for blocker in matrix.blockers)


def test_ready_relation_inventory_blocks_retained_count_mismatch() -> None:
    payload = yaml.safe_load((_FIXTURES / "inventory.yaml").read_text(encoding="utf-8"))
    payload["retained_live_census"]["observed_base_tables"] += 1
    inventory = ModelApplicationRelationEvidenceInventory.model_validate(payload)
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": inventory},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )

    assert matrix.status == "BLOCKED"
    assert any(
        "retained_live_census.observed_base_tables=6 "
        "does not match exact typed rows=5" in blocker
        for blocker in matrix.blockers
    )


def test_topology_object_grant_without_typed_ownership_is_a_blocker() -> None:
    topology = ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml")
    database = topology.databases["application"]
    principal = database.principals["app_dashboard"]
    grant = principal.grants[-1].model_copy(update={"objects": ("phantom_type",)})
    principals = dict(database.principals)
    principals["app_dashboard"] = principal.model_copy(
        update={"grants": (*principal.grants, grant)}
    )
    topology = topology.model_copy(
        update={
            "databases": {
                "application": database.model_copy(update={"principals": principals})
            }
        }
    )

    matrix = build_application_database_acl_matrix(
        topology=topology,
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any("targets no typed ownership object" in item for item in matrix.blockers)


def test_missing_ownership_sources_and_objects_block_sql() -> None:
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=(_source("topology", "topology"),),
        relation_inventories={},
        service_manifests={},
        principal_inventories={},
        acl_policies={},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any("typed ownership evidence source" in item for item in matrix.blockers)
    assert any("zero database objects" in item for item in matrix.blockers)
    assert any("no typed ownership objects" in item for item in matrix.blockers)
    with pytest.raises(ValueError, match="Cannot render blocked ACL matrix"):
        render_application_database_acl_sql(matrix)


def test_function_without_signature_blocks_even_without_execute_grant() -> None:
    topology = ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml")
    database = topology.databases["application"]
    dashboard = database.principals["app_dashboard"]
    principals = dict(database.principals)
    principals["app_dashboard"] = dashboard.model_copy(
        update={
            "grants": tuple(
                grant
                for grant in dashboard.grants
                if grant.object_type is not EnumDatabaseGrantObjectType.FUNCTION
            )
        }
    )
    topology = topology.model_copy(
        update={
            "databases": {
                "application": database.model_copy(update={"principals": principals})
            }
        }
    )
    inventory = _inventory()
    relations = tuple(
        relation.model_copy(update={"function_signature": None})
        if relation.kind.value == "function"
        else relation
        for relation in inventory.relations
    )
    matrix = build_application_database_acl_matrix(
        topology=topology,
        sources=_complete_sources(),
        relation_inventories={
            "inventory": inventory.model_copy(update={"relations": relations})
        },
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any(
        "require an explicit function_signature" in item for item in matrix.blockers
    )
    with pytest.raises(ValueError, match="Cannot render blocked ACL matrix"):
        render_application_database_acl_sql(matrix)


def test_overloaded_functions_keep_exact_identities_and_block_ambiguous_grant() -> None:
    inventory = _inventory()
    function = next(
        relation
        for relation in inventory.relations
        if relation.kind.value == "function"
    )
    overloaded = function.model_copy(update={"function_signature": "(uuid)"})
    relation_counts = inventory.relation_counts.model_copy(update={"function": 2})
    inventory = inventory.model_copy(
        update={
            "relations": (*inventory.relations, overloaded),
            "relation_counts": relation_counts,
        }
    )

    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": inventory},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    overloads = [
        obj for obj in matrix.objects if obj.object_ref == "delegation_event_count"
    ]
    assert {obj.function_signature for obj in overloads} == {"()", "(uuid)"}
    assert matrix.status == "BLOCKED"
    assert any("overload-specific targeting" in item for item in matrix.blockers)
    assert not any(
        row.privileges
        for row in matrix.rows
        if row.object_ref == "delegation_event_count"
    )


def test_duplicate_exact_authoritative_evidence_is_blocked() -> None:
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=(
            *_complete_sources(),
            _source("inventory_duplicate", "relation_inventory"),
        ),
        relation_inventories={
            "inventory": _inventory(),
            "inventory_duplicate": _inventory(),
        },
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any(
        "duplicate authoritative ownership declaration" in blocker
        for blocker in matrix.blockers
    )


def test_function_signature_rejects_executable_sql_tokens() -> None:
    function = next(
        obj
        for obj in _matrix().objects
        if obj.object_type is EnumDatabaseGrantObjectType.FUNCTION
    )

    with pytest.raises(ValidationError, match="function_signature"):
        ModelApplicationDatabaseAclObject.model_validate(
            {
                **function.model_dump(),
                "function_signature": "(); DROP TABLE tenant.delegation_events; --",
            }
        )


@pytest.mark.parametrize(
    ("object_type", "catalog_kind"),
    [
        (EnumDatabaseGrantObjectType.TABLE, "function"),
        (EnumDatabaseGrantObjectType.FUNCTION, "type"),
        (EnumDatabaseGrantObjectType.TYPE, "table"),
    ],
)
def test_acl_object_rejects_mismatched_catalog_target_shapes(
    object_type: EnumDatabaseGrantObjectType,
    catalog_kind: str,
) -> None:
    table = next(
        obj
        for obj in _matrix().objects
        if obj.catalog_kind == "table" and obj.function_signature is None
    )
    payload = table.model_dump()
    payload.update(
        {
            "object_type": object_type,
            "catalog_kind": catalog_kind,
            "function_signature": "()" if catalog_kind == "function" else None,
        }
    )

    with pytest.raises(ValidationError, match="requires object_type"):
        ModelApplicationDatabaseAclObject.model_validate(payload)


def test_nonroutine_acl_object_rejects_function_signature() -> None:
    table = next(obj for obj in _matrix().objects if obj.catalog_kind == "table")
    with pytest.raises(ValidationError, match="only valid"):
        ModelApplicationDatabaseAclObject.model_validate(
            {**table.model_dump(), "function_signature": "()"}
        )


@pytest.mark.parametrize(
    "signature",
    [
        '(), "other_function"()',
        '(uuid), "other_function"()',
        '(uuid)) , "other_function"((',
    ],
)
def test_function_signature_rejects_target_list_escape(signature: str) -> None:
    matrix = _matrix()
    function = next(
        obj
        for obj in matrix.objects
        if obj.object_type is EnumDatabaseGrantObjectType.FUNCTION
    )
    function_row = next(
        row
        for row in matrix.rows
        if row.object_type is EnumDatabaseGrantObjectType.FUNCTION
    )

    with pytest.raises(ValidationError, match="function_signature"):
        ModelApplicationDatabaseAclObject.model_validate(
            {**function.model_dump(), "function_signature": signature}
        )
    with pytest.raises(ValidationError, match="function_signature"):
        ModelApplicationDatabaseAclRow.model_validate(
            {**function_row.model_dump(), "function_signature": signature}
        )


def test_current_source_schema_cannot_be_rendered_as_materialized_target() -> None:
    inventory = _inventory()
    relation = inventory.relations[0].model_copy(update={"current_schema": ("public",)})
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={
            "inventory": inventory.model_copy(
                update={"relations": (relation, *inventory.relations[1:])}
            )
        },
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any(
        "full object ACL rendering is gated" in blocker for blocker in matrix.blockers
    )
    with pytest.raises(ValueError, match="Cannot render blocked ACL matrix"):
        render_application_database_acl_sql(matrix, allow_synthetic_proof=True)


def test_additive_source_and_target_schema_coexistence_is_materialized() -> None:
    inventory = _inventory()
    relation = inventory.relations[0].model_copy(
        update={"current_schema": ("public", "tenant")}
    )
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={
            "inventory": inventory.model_copy(
                update={"relations": (relation, *inventory.relations[1:])}
            )
        },
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "READY"
    target = next(obj for obj in matrix.objects if obj.object_ref == relation.name)
    assert target.target_materialized


def test_full_matrix_rejects_an_object_on_the_wrong_physical_database() -> None:
    matrix = _matrix()
    objects = list(matrix.objects)
    objects[0] = objects[0].model_copy(update={"physical_database": "wrong_database"})
    malformed = matrix.model_copy(update={"objects": tuple(objects)})

    violations = validate_application_database_acl_scaffold(
        malformed,
        require_safe_existing_roles=False,
    )
    assert any("physical database target disagrees" in item for item in violations)
    with pytest.raises(ValueError, match="physical database target disagrees"):
        render_application_database_acl_sql(
            malformed,
            allow_synthetic_proof=True,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_key", "topology\nDROP_TABLE"),
        ("repository", "synthetic/proof\n--"),
        ("path", "proof/topology.yaml\n--"),
    ],
)
def test_source_metadata_rejects_sql_comment_line_breaks(
    field: str,
    value: str,
) -> None:
    payload = _source("topology", "topology").model_dump()
    payload[field] = value

    with pytest.raises(ValidationError):
        ModelApplicationDatabaseAclSource.model_validate(payload)


def test_synthetic_matrix_requires_explicit_proof_render_authorization() -> None:
    with pytest.raises(ValueError, match="explicit allow_synthetic_proof=true"):
        render_application_database_acl_sql(_matrix())

    deployment = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT,
    )
    assert deployment.status == "BLOCKED"
    assert any("deployment authorization" in item for item in deployment.blockers)
    assert any(
        "explicit eight-database CONNECT inventory" in item
        for item in deployment.blockers
    )


def test_deployment_cannot_narrow_connect_scope_to_topology_database() -> None:
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT,
        required_connect_databases=("omnidash_analytics",),
    )

    assert matrix.status == "BLOCKED"
    assert any(
        "approved eight-database scope" in blocker for blocker in matrix.blockers
    )
    with pytest.raises(ValueError, match="Cannot render blocked ACL matrix"):
        render_application_database_acl_sql(matrix)


def test_authorized_catalog_provenance_requires_durable_complete_day() -> None:
    payload = _principal_inventory().model_dump(mode="json")
    payload.update(
        {
            "source_kind": "authorized_catalog",
            "live_database_read": True,
            "catalog_query_sha256": "1" * 64,
            "catalog_result_sha256": "2" * 64,
            "catalog_query_source_key": "catalog_query",
            "catalog_result_source_key": "catalog_result",
            "activity_principal_refs": ["app_dashboard"],
        }
    )
    started = datetime(2026, 7, 28, tzinfo=UTC)
    payload["activity_evidence"] = {
        "window_started_at": started.isoformat(),
        "window_ended_at": (started + timedelta(hours=23, minutes=59)).isoformat(),
        "query_sha256": "3" * 64,
        "result_sha256": "4" * 64,
        "query_source_key": "activity_query",
        "result_source_key": "activity_result",
        "observation_count": 1,
    }

    with pytest.raises(ValidationError, match="at least 24 hours"):
        ModelApplicationDatabasePrincipalInventory.model_validate(payload)

    payload["activity_evidence"]["window_ended_at"] = (
        started + timedelta(hours=24)
    ).isoformat()
    inventory = ModelApplicationDatabasePrincipalInventory.model_validate(payload)
    assert inventory.activity_evidence is not None


def test_authorized_catalog_accepts_a_complete_zero_activity_window() -> None:
    inventory = _authorized_inventory(
        active_principals=(),
        observation_count=0,
    )
    assert inventory.activity_principal_refs == ()
    assert inventory.activity_evidence is not None
    assert inventory.activity_evidence.observation_count == 0

    activity = ModelApplicationDatabaseActivityResultEvidence.model_validate(
        {
            "database_ref": inventory.database_ref,
            "physical_database": inventory.physical_database,
            "window_started_at": inventory.activity_evidence.window_started_at,
            "window_ended_at": inventory.activity_evidence.window_ended_at,
            "activity_query_sha256": inventory.activity_evidence.query_sha256,
            "observation_count": 0,
            "active_principals": [],
        }
    )
    assert activity.active_principals == ()


def test_parsed_catalog_and_activity_results_must_semantically_match_inventory() -> (
    None
):
    inventory = _authorized_inventory()
    assert inventory.activity_evidence is not None
    catalog_payload = {
        "database_ref": inventory.database_ref,
        "physical_database": inventory.physical_database,
        "completion_status": inventory.completion_status,
        "catalog_parity_status": inventory.catalog_parity_status,
        "catalog_query_sha256": inventory.catalog_query_sha256,
        "database_owner_role": inventory.database_owner_role,
        "principal_refs": inventory.principal_refs,
        "absent_principal_refs": inventory.absent_principal_refs,
        "owner_refs": inventory.owner_refs,
        "absent_owner_refs": inventory.absent_owner_refs,
        "observed_role_states": inventory.observed_role_states,
        "observed_schema_owners": inventory.observed_schema_owners,
        "absent_schema_refs": inventory.absent_schema_refs,
        "observed_objects": inventory.observed_objects,
    }
    activity_payload = {
        "database_ref": inventory.database_ref,
        "physical_database": inventory.physical_database,
        "window_started_at": inventory.activity_evidence.window_started_at,
        "window_ended_at": inventory.activity_evidence.window_ended_at,
        "activity_query_sha256": inventory.activity_evidence.query_sha256,
        "observation_count": 1,
        "active_principals": [{"principal": "app_dashboard", "observation_count": 1}],
    }
    catalog = ModelApplicationDatabaseCatalogResultEvidence.model_validate(
        catalog_payload
    )
    activity = ModelApplicationDatabaseActivityResultEvidence.model_validate(
        activity_payload
    )
    assert not validate_application_database_principal_evidence(
        inventory,
        catalog,
        activity,
    )

    catalog_payload["physical_database"] = "wrong_database"
    activity_payload["active_principals"] = [
        {"principal": "onex_api", "observation_count": 1}
    ]
    violations = validate_application_database_principal_evidence(
        inventory,
        ModelApplicationDatabaseCatalogResultEvidence.model_validate(catalog_payload),
        ModelApplicationDatabaseActivityResultEvidence.model_validate(activity_payload),
    )
    assert "catalog result physical_database disagrees with inventory" in violations
    assert "activity result active principals disagree with inventory" in violations


@pytest.mark.parametrize("owner_surface", ["schema", "object"])
def test_catalog_evidence_rejects_unclassified_present_owners(
    owner_surface: str,
) -> None:
    payload = _principal_inventory().model_dump(mode="json")
    if owner_surface == "schema":
        payload["observed_schema_owners"]["tenant"] = "unclassified_owner"
    else:
        payload["observed_objects"][0]["owner"] = "unclassified_owner"

    with pytest.raises(ValidationError, match="classified in the present role census"):
        ModelApplicationDatabasePrincipalInventory.model_validate(payload)


def test_deployment_catalog_hashes_require_locked_evidence_blobs() -> None:
    payload = _principal_inventory().model_dump(mode="json")
    started = datetime(2026, 7, 28, tzinfo=UTC)
    payload.update(
        {
            "source_kind": "authorized_catalog",
            "live_database_read": True,
            "catalog_query_sha256": "1" * 64,
            "catalog_result_sha256": "2" * 64,
            "catalog_query_source_key": "catalog_query",
            "catalog_result_source_key": "catalog_result",
            "activity_principal_refs": ["app_dashboard"],
            "activity_evidence": {
                "window_started_at": started.isoformat(),
                "window_ended_at": (started + timedelta(hours=24)).isoformat(),
                "query_sha256": "3" * 64,
                "result_sha256": "4" * 64,
                "query_source_key": "activity_query",
                "result_source_key": "activity_result",
                "observation_count": 1,
            },
        }
    )
    inventory = ModelApplicationDatabasePrincipalInventory.model_validate(payload)

    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": inventory},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT,
        required_connect_databases=("omnidash_analytics",),
    )

    assert matrix.scaffold_status == "BLOCKED"
    assert any(
        "evidence source 'catalog_query' is absent from the immutable source lock"
        in blocker
        for blocker in matrix.scaffold_blockers
    )


def test_full_ready_deployment_cannot_bypass_semantic_result_verification() -> None:
    payload = _matrix().model_dump(mode="json")
    payload.update(
        {
            "authorization_scope": "deployment",
            "scaffold_status": "BLOCKED",
            "scaffold_blockers": ["unsafe existing role requires FULL hardening"],
            "status": "READY",
            "blockers": [],
            "sources": [
                *payload["sources"],
                _source(
                    "catalog_result",
                    "catalog_result_evidence",
                ).model_dump(mode="json"),
            ],
            "verified_evidence_source_keys": [],
        }
    )

    with pytest.raises(ValidationError, match="READY deployment phase"):
        ModelApplicationDatabaseAclMatrix.model_validate(payload)


def test_schema_only_cross_domain_grant_is_blocked() -> None:
    topology = ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml")
    database = topology.databases["application"]
    dashboard = database.principals["app_dashboard"]
    internal_schema_usage = next(
        grant
        for grant in database.principals["omninode_runtime"].grants
        if grant.object_type is EnumDatabaseGrantObjectType.SCHEMA
    )
    principals = dict(database.principals)
    principals["app_dashboard"] = dashboard.model_copy(
        update={"grants": (*dashboard.grants, internal_schema_usage)}
    )
    topology = topology.model_copy(
        update={
            "databases": {
                "application": database.model_copy(update={"principals": principals})
            }
        }
    )

    matrix = build_application_database_acl_matrix(
        topology=topology,
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any("cross-domain" in item for item in matrix.blockers)


def test_runtime_trigger_privilege_is_blocked() -> None:
    topology = ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml")
    database = topology.databases["application"]
    api = database.principals["onex_api"]
    tenant_accounts = next(
        grant
        for grant in api.grants
        if grant.object_type is EnumDatabaseGrantObjectType.TABLE
        and "tenant_accounts" in grant.objects
    )
    trigger = tenant_accounts.model_copy(
        update={"privileges": (EnumDatabasePrivilege.TRIGGER,)}
    )
    principals = dict(database.principals)
    principals["onex_api"] = api.model_copy(update={"grants": (*api.grants, trigger)})
    topology = topology.model_copy(
        update={
            "databases": {
                "application": database.model_copy(update={"principals": principals})
            }
        }
    )

    matrix = build_application_database_acl_matrix(
        topology=topology,
        sources=_complete_sources(),
        relation_inventories={"inventory": _inventory()},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any("DDL privilege" in item for item in matrix.blockers)


def test_service_db_io_table_requires_exact_bidirectional_evidence() -> None:
    fixture = (
        Path(__file__).parents[2]
        / "fixtures"
        / "application_relation_ownership"
        / "service-owner.yaml"
    )
    payload = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    payload.update(
        {
            "completion_status": "verified",
            "retained_live_census": {
                "observed_base_tables": 1,
                "observed_views_and_materialized_views": 2,
                "parity_status": "verified",
            },
            "runtime_evidence": {
                "full_day_datname_usename_activity": {
                    "status": "verified",
                    "reason": "synthetic fixture",
                },
                "live_catalog_parity": {
                    "status": "verified",
                    "reason": "synthetic fixture",
                },
            },
        }
    )
    payload["relation_evidence"] = [
        row
        for row in payload["relation_evidence"]
        if row["name"] != "schema_migrations"
    ]
    manifest = ModelMigrationOwnershipManifest.model_validate(payload)

    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=(
            *_complete_sources(),
            _source("service_ownership", "service_ownership"),
        ),
        relation_inventories={"inventory": _inventory()},
        service_manifests={"service_ownership": manifest},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any(
        "db_io table application.omninode_internal.schema_migrations lacks exact"
        in item
        for item in matrix.blockers
    )
    assert not any(
        "projection_delegation_summary" in item
        and "requires exact database_objects" in item
        for item in matrix.blockers
    )


def test_service_function_relation_requires_signature_capable_object_evidence() -> None:
    fixture = (
        Path(__file__).parents[2]
        / "fixtures"
        / "application_relation_ownership"
        / "service-owner.yaml"
    )
    payload = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    payload.update(
        {
            "completion_status": "verified",
            "materialized_physical_databases": ["omnidash_analytics"],
            "retained_live_census": {
                "observed_base_tables": 1,
                "observed_views_and_materialized_views": 2,
                "parity_status": "verified",
            },
            "runtime_evidence": {
                "full_day_datname_usename_activity": {
                    "status": "verified",
                    "reason": "synthetic fixture",
                },
                "live_catalog_parity": {
                    "status": "verified",
                    "reason": "synthetic fixture",
                },
            },
        }
    )
    payload["relation_evidence"][1]["kind"] = "function"
    manifest = ModelMigrationOwnershipManifest.model_validate(payload)

    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=(
            *_complete_sources(),
            _source("service_ownership", "service_ownership"),
        ),
        relation_inventories={"inventory": _inventory()},
        service_manifests={"service_ownership": manifest},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF,
    )

    assert matrix.status == "BLOCKED"
    assert any(
        "function relation_evidence 'projection_delegation_summary' requires exact "
        "database_objects evidence" in blocker
        for blocker in matrix.blockers
    )


def test_service_inventory_requires_explicit_ready_retained_census_status() -> None:
    fixture = (
        Path(__file__).parents[2]
        / "fixtures"
        / "application_relation_ownership"
        / "service-owner.yaml"
    )
    payload = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    payload.update(
        {
            "completion_status": "verified",
            "retained_live_census": {
                "observed_base_tables": 1,
                "observed_views_and_materialized_views": 2,
                "observed_sequences": 0,
                "observed_functions": 0,
                "observed_procedures": 0,
                "observed_types": 0,
                "observed_extensions": 0,
                "parity_status": None,
            },
            "runtime_evidence": {
                "full_day_datname_usename_activity": {
                    "status": "verified",
                    "reason": "synthetic fixture",
                },
                "live_catalog_parity": {
                    "status": "verified",
                    "reason": "synthetic fixture",
                },
            },
        }
    )
    manifest = ModelMigrationOwnershipManifest.model_validate(payload)
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=(
            *_complete_sources(),
            _source("service_ownership", "service_ownership"),
        ),
        relation_inventories={"inventory": _inventory()},
        service_manifests={"service_ownership": manifest},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )

    assert matrix.status == "BLOCKED"
    assert any(
        "service_ownership: retained_live_census=None" in blocker
        for blocker in matrix.blockers
    )


@pytest.mark.parametrize(
    ("kind", "object_type", "signature"),
    [
        ("foreign_table", EnumDatabaseGrantObjectType.TABLE, None),
        ("aggregate", EnumDatabaseGrantObjectType.FUNCTION, "(integer)"),
        ("window_function", EnumDatabaseGrantObjectType.FUNCTION, "(integer)"),
        ("base_type", EnumDatabaseGrantObjectType.TYPE, None),
        ("range_type", EnumDatabaseGrantObjectType.TYPE, None),
        ("multirange_type", EnumDatabaseGrantObjectType.TYPE, None),
    ],
)
def test_supporting_inventory_kinds_are_acl_governed_not_excluded(
    kind: str,
    object_type: EnumDatabaseGrantObjectType,
    signature: str | None,
) -> None:
    payload = _inventory().model_dump(mode="json")
    supporting = dict(payload["relations"][0])
    supporting.update(
        {
            "name": f"acl_{kind}",
            "kind": kind,
            "owner_declaration": "service:onex_api",
            "function_signature": signature,
        }
    )
    payload["relations"].append(supporting)
    inventory = ModelApplicationRelationEvidenceInventory.model_validate(payload)

    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": inventory},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )

    governed = [obj for obj in matrix.objects if obj.object_ref == f"acl_{kind}"]
    assert len(governed) == 1
    assert governed[0].catalog_kind == kind
    assert governed[0].object_type is object_type
    assert governed[0].function_signature == signature
    assert not any(f":{kind}" in item for item in matrix.excluded_objects)
    assert any(
        "live catalog object identities differ" in blocker
        for blocker in matrix.blockers
    )
    public_row = next(
        row
        for row in matrix.rows
        if row.principal == PUBLIC_PRINCIPAL and row.object_ref == f"acl_{kind}"
    )
    assert not public_row.privileges


def test_extension_inventory_is_an_explicit_acl_blocker_not_an_exclusion() -> None:
    payload = _inventory().model_dump(mode="json")
    extension = dict(payload["relations"][0])
    extension.update(
        {
            "name": "pgcrypto",
            "kind": "extension",
            "owner_declaration": "service:onex_api",
            "function_signature": None,
        }
    )
    payload["relations"].append(extension)
    payload["relation_counts"]["extension"] += 1
    payload["retained_live_census"]["observed_extensions"] += 1
    inventory = ModelApplicationRelationEvidenceInventory.model_validate(payload)

    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=_complete_sources(),
        relation_inventories={"inventory": inventory},
        service_manifests={},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )

    assert matrix.status == "BLOCKED"
    assert any(
        "extension" in blocker and "no PostgreSQL object ACL" in blocker
        for blocker in matrix.blockers
    )
    assert not any(":extension" in item for item in matrix.excluded_objects)


def _supporting_service_manifest() -> ModelMigrationOwnershipManifest:
    return ModelMigrationOwnershipManifest.model_validate(
        {
            "schema_version": "1.0",
            "service": "acl_supporting_objects",
            "current_physical_database": "omnidash_analytics",
            "materialized_physical_databases": ["omnidash_analytics"],
            "target_database_ref": "application",
            "db_io": {"db_tables": []},
            "relation_evidence": [
                {
                    "name": "acl_foreign_table",
                    "kind": "foreign_table",
                    "database_ref": "application",
                    "schema": "tenant",
                    "current_schemas": ["tenant"],
                    "domain": "TENANT",
                    "owner_declaration": "service:acl_supporting_objects",
                }
            ],
            "database_objects": [
                {
                    "name": "acl_aggregate",
                    "kind": "aggregate",
                    "database_ref": "application",
                    "schema": "tenant",
                    "current_schemas": ["tenant"],
                    "domain": "TENANT",
                    "owner_declaration": "service:acl_supporting_objects",
                    "function_signature": "(integer)",
                },
                {
                    "name": "acl_window_function",
                    "kind": "window_function",
                    "database_ref": "application",
                    "schema": "tenant",
                    "current_schemas": ["tenant"],
                    "domain": "TENANT",
                    "owner_declaration": "service:acl_supporting_objects",
                    "function_signature": "(integer)",
                },
                *(
                    {
                        "name": f"acl_{kind}",
                        "kind": kind,
                        "database_ref": "application",
                        "schema": "tenant",
                        "current_schemas": ["tenant"],
                        "domain": "TENANT",
                        "owner_declaration": "service:acl_supporting_objects",
                    }
                    for kind in ("base_type", "range_type", "multirange_type")
                ),
            ],
            "blocked_relations": [],
            "completion_status": "verified",
            "retained_live_census": {
                "observed_base_tables": 0,
                "observed_views_and_materialized_views": 0,
                "observed_sequences": 0,
                "observed_functions": 2,
                "observed_procedures": 0,
                "observed_types": 3,
                "observed_extensions": 0,
                "parity_status": "verified",
                "reason": "synthetic exact supporting-object census",
            },
            "runtime_evidence": {
                "full_day_datname_usename_activity": {
                    "status": "verified",
                    "reason": "synthetic fixture",
                },
                "live_catalog_parity": {
                    "status": "verified",
                    "reason": "synthetic fixture",
                },
            },
        }
    )


def test_service_supporting_kinds_render_exact_owner_acl_and_catalog_guards() -> None:
    principal_payload = _principal_inventory().model_dump(mode="json")
    principal_payload["observed_objects"].extend(
        [
            {
                "schema_ref": "tenant",
                "catalog_kind": "foreign_table",
                "object_ref": "acl_foreign_table",
                "owner": "onex_api",
            },
            {
                "schema_ref": "tenant",
                "catalog_kind": "aggregate",
                "object_ref": "acl_aggregate",
                "function_signature": "(integer)",
                "owner": "onex_api",
            },
            {
                "schema_ref": "tenant",
                "catalog_kind": "window_function",
                "object_ref": "acl_window_function",
                "function_signature": "(integer)",
                "owner": "onex_api",
            },
            *(
                {
                    "schema_ref": "tenant",
                    "catalog_kind": kind,
                    "object_ref": f"acl_{kind}",
                    "owner": "onex_api",
                }
                for kind in ("base_type", "range_type", "multirange_type")
            ),
        ]
    )
    principal_inventory = ModelApplicationDatabasePrincipalInventory.model_validate(
        principal_payload
    )
    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=(
            *_complete_sources(),
            _source("service_ownership", "service_ownership"),
        ),
        relation_inventories={"inventory": _inventory()},
        service_manifests={"service_ownership": _supporting_service_manifest()},
        principal_inventories={"principal_inventory": principal_inventory},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )

    assert matrix.status == "READY", matrix.blockers
    assert not matrix.excluded_objects
    assert all(
        not row.privileges
        for row in matrix.rows
        if row.principal == PUBLIC_PRINCIPAL
        and row.object_ref is not None
        and row.object_ref.startswith("acl_")
    )
    sql = render_application_database_acl_sql(matrix, allow_synthetic_proof=True)
    assert (
        'ALTER FOREIGN TABLE "tenant"."acl_foreign_table" OWNER TO '
        '"owner_onex_tenant";' in sql
    )
    assert (
        'REVOKE ALL PRIVILEGES ON TABLE "tenant"."acl_foreign_table" FROM PUBLIC' in sql
    )
    assert (
        'ALTER AGGREGATE "tenant"."acl_aggregate"(integer) OWNER TO '
        '"owner_onex_tenant";' in sql
    )
    assert (
        'REVOKE ALL PRIVILEGES ON FUNCTION "tenant"."acl_aggregate"(integer) '
        "FROM PUBLIC" in sql
    )
    assert (
        'ALTER FUNCTION "tenant"."acl_window_function"(integer) OWNER TO '
        '"owner_onex_tenant";' in sql
    )
    assert 'ALTER TYPE "tenant"."acl_range_type" OWNER TO "owner_onex_tenant";' in sql
    assert "WHEN 'f' THEN 'foreign_table'" in sql
    assert "WHEN 'a' THEN 'aggregate'" in sql
    assert "WHEN 'w' THEN 'window_function'" in sql
    assert "WHEN 'b' THEN 'base_type'" in sql
    assert "WHEN 'r' THEN 'range_type'" in sql
    assert "WHEN 'm' THEN 'multirange_type'" in sql
    assert "object.prokind = 'a'" in sql
    assert "object.prokind = 'w'" in sql
    assert "object.typtype = 'b'" in sql
    assert "object.typtype = 'r'" in sql
    assert "object.typtype = 'm'" in sql


def test_service_extension_is_an_explicit_acl_blocker_not_an_exclusion() -> None:
    payload = _supporting_service_manifest().model_dump(mode="json")
    payload["relation_evidence"] = []
    payload["database_objects"] = [
        {
            "name": "pgcrypto",
            "kind": "extension",
            "database_ref": "application",
            "schema": "tenant",
            "current_schemas": ["tenant"],
            "domain": "TENANT",
            "owner_declaration": "service:acl_supporting_objects",
        }
    ]
    payload["retained_live_census"].update(
        {
            "observed_functions": 0,
            "observed_types": 0,
            "observed_extensions": 1,
        }
    )
    manifest = ModelMigrationOwnershipManifest.model_validate(payload)

    matrix = build_application_database_acl_matrix(
        topology=ModelDeploymentTopology.from_yaml(_FIXTURES / "topology.yaml"),
        sources=(
            *_complete_sources(),
            _source("service_ownership", "service_ownership"),
        ),
        relation_inventories={"inventory": _inventory()},
        service_manifests={"service_ownership": manifest},
        principal_inventories={"principal_inventory": _principal_inventory()},
        acl_policies={"acl_policy": _acl_policy()},
        authorization_scope=(
            EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        ),
    )

    assert matrix.status == "BLOCKED"
    assert any(
        "extension" in blocker and "no PostgreSQL object ACL" in blocker
        for blocker in matrix.blockers
    )
    assert not any(":extension" in item for item in matrix.excluded_objects)
