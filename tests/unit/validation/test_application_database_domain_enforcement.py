# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed application database domain enforcement (OMN-15361)."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.topology.physical_schema_mapping import (
    INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
    physical_grant_schema_for_table,
)
from omnibase_infra.validation.application_database_domain_enforcement import (
    CANONICAL_TENANT_PREDICATE,
    application_database_created_catalog_identities,
    application_database_function_definition_sha256,
    application_database_sql_target_requirements,
    lint_application_database_sql,
    load_application_database_ownership_identities,
    validate_application_database_catalog_census,
    validate_application_database_pool_identities,
    validate_application_database_relation_states,
)
from omnibase_infra.validation.enums.enum_application_database_identity_root import (
    EnumApplicationDatabaseIdentityRoot,
)
from omnibase_infra.validation.enums.enum_application_database_identity_root_operation import (
    EnumApplicationDatabaseIdentityRootOperation,
)
from omnibase_infra.validation.enums.enum_application_inventory_object_kind import (
    EnumApplicationInventoryObjectKind,
)
from omnibase_infra.validation.enums.enum_application_relation_kind import (
    EnumApplicationRelationKind,
)
from omnibase_infra.validation.enums.enum_application_relation_purpose import (
    EnumApplicationRelationPurpose,
)
from omnibase_infra.validation.models.model_application_database_catalog_identity import (
    ModelApplicationDatabaseCatalogIdentity,
)
from omnibase_infra.validation.models.model_application_database_column_state import (
    ModelApplicationDatabaseColumnState,
)
from omnibase_infra.validation.models.model_application_database_function_state import (
    ModelApplicationDatabaseFunctionState,
)
from omnibase_infra.validation.models.model_application_database_identity_root_control_state import (
    ModelApplicationDatabaseIdentityRootControlState,
)
from omnibase_infra.validation.models.model_application_database_policy_state import (
    ModelApplicationDatabasePolicyState,
)
from omnibase_infra.validation.models.model_application_database_pool_identity import (
    ModelApplicationDatabasePoolIdentity,
)
from omnibase_infra.validation.models.model_application_database_relation_state import (
    ModelApplicationDatabaseRelationState,
)
from omnibase_infra.validation.models.model_application_database_tenant_isolation_evidence import (
    ModelApplicationDatabaseTenantIsolationEvidence,
)
from omnibase_infra.validation.models.model_application_relation_declaration import (
    ModelApplicationRelationDeclaration,
)

pytestmark = pytest.mark.unit

_TOPOLOGY = load_topology_profile("local")
_RUNTIME_PRINCIPALS = tuple(
    sorted(
        binding.principal
        for binding in _TOPOLOGY.databases["application"].bindings.values()
    )
)
_TENANT_A = UUID("11111111-1111-1111-1111-111111111111")
_TENANT_B = UUID("22222222-2222-2222-2222-222222222222")


def _evidence() -> ModelApplicationDatabaseTenantIsolationEvidence:
    return ModelApplicationDatabaseTenantIsolationEvidence(
        expected_rows_by_tenant={_TENANT_A: 2, _TENANT_B: 1},
        observed_rows_by_tenant={_TENANT_A: 2, _TENANT_B: 1},
        unset_context_rows=0,
        malformed_context_denied=True,
    )


def _declaration(
    *,
    domain: EnumDatabaseSchemaDomain,
    kind: EnumApplicationRelationKind = EnumApplicationRelationKind.TABLE,
    name: str = "events",
    schema: str | None = None,
) -> ModelApplicationRelationDeclaration:
    resolved_schema = (
        schema
        or {
            EnumDatabaseSchemaDomain.TENANT: "public",
            EnumDatabaseSchemaDomain.OMNINODE_INTERNAL: "omninode_internal",
            EnumDatabaseSchemaDomain.PLATFORM_CATALOG: "platform_catalog",
        }[domain]
    )
    return ModelApplicationRelationDeclaration(
        name=name,
        database_ref="application",
        schema=resolved_schema,
        kind=kind,
        purpose=EnumApplicationRelationPurpose.DATA,
        domain=domain,
        owner_declaration="node:fixture_owner",
        access="write",
        role="projection_state",
        source_path="tests/fixtures/OMN-15361.yaml",
    )


def _tenant_table() -> ModelApplicationDatabaseRelationState:
    return ModelApplicationDatabaseRelationState(
        declaration=_declaration(domain=EnumDatabaseSchemaDomain.TENANT),
        columns=(
            ModelApplicationDatabaseColumnState(
                name="event_id",
                data_type="uuid",
                nullable=False,
                default_expression=None,
            ),
            ModelApplicationDatabaseColumnState(
                name="tenant_id",
                data_type="uuid",
                nullable=False,
                default_expression=None,
            ),
        ),
        primary_key_columns=("event_id",),
        rls_enabled=True,
        rls_forced=True,
        policies=(
            ModelApplicationDatabasePolicyState(
                name="tenant_isolation",
                permissive=True,
                command="ALL",
                roles=("PUBLIC",),
                using_expression=CANONICAL_TENANT_PREDICATE,
                with_check_expression=CANONICAL_TENANT_PREDICATE,
            ),
        ),
        tenant_identity_column="tenant_id",
        canonical_policy_name="tenant_isolation",
    )


def _identity_root_table() -> ModelApplicationDatabaseRelationState:
    identity_column = ModelApplicationDatabaseColumnState(
        name="id",
        data_type="uuid",
        nullable=False,
        default_expression=None,
    )
    predicate = "id = current_setting('app.tenant_id', true)::uuid"
    policy = ModelApplicationDatabasePolicyState(
        name="tenant_identity_isolation",
        permissive=True,
        command="ALL",
        roles=("PUBLIC",),
        using_expression=predicate,
        with_check_expression=predicate,
    )
    operations = tuple(EnumApplicationDatabaseIdentityRootOperation)
    control = ModelApplicationDatabaseIdentityRootControlState(
        role="tenant_control_admin",
        role_can_login=False,
        role_superuser=False,
        role_bypass_rls=True,
        runtime_membership_principals=(),
        runtime_set_role_denied_principals=_RUNTIME_PRINCIPALS,
        declared_operations=operations,
        observed_operations=operations,
        behavioral_proof_ids=(
            "pytest:identity-root-tenant-create",
            "pytest:identity-root-cross-tenant-enumeration",
        ),
    )
    return _tenant_table().model_copy(
        update={
            "declaration": _declaration(
                domain=EnumDatabaseSchemaDomain.TENANT,
                name="tenants",
            ),
            "columns": (identity_column,),
            "primary_key_columns": ("id",),
            "tenant_identity_column": "id",
            "identity_root_contract": (
                EnumApplicationDatabaseIdentityRoot.CANONICAL_TENANT
            ),
            "identity_root_control_state": control,
            "canonical_policy_name": "tenant_identity_isolation",
            "policies": (policy,),
        }
    )


def _non_tenant_table(
    domain: EnumDatabaseSchemaDomain,
) -> ModelApplicationDatabaseRelationState:
    return ModelApplicationDatabaseRelationState(
        declaration=_declaration(domain=domain),
        columns=(
            ModelApplicationDatabaseColumnState(
                name="event_id",
                data_type="uuid",
                nullable=False,
                default_expression=None,
            ),
            ModelApplicationDatabaseColumnState(
                name="source_tenant_id",
                data_type="uuid",
                nullable=True,
                default_expression=None,
            ),
        ),
        primary_key_columns=("event_id",),
        source_tenant_provenance_contract="non_authoritative_provenance",
    )


def _tenant_view() -> ModelApplicationDatabaseRelationState:
    return ModelApplicationDatabaseRelationState(
        declaration=_declaration(
            domain=EnumDatabaseSchemaDomain.TENANT,
            kind=EnumApplicationRelationKind.VIEW,
            name="events_view",
        ),
        security_invoker=True,
        view_tenant_isolation_evidence=_evidence(),
    )


def _security_definer_function() -> ModelApplicationDatabaseRelationState:
    return ModelApplicationDatabaseRelationState(
        declaration=_declaration(
            domain=EnumDatabaseSchemaDomain.TENANT,
            kind=EnumApplicationRelationKind.FUNCTION,
            name="tenant_report",
        ),
        function_state=ModelApplicationDatabaseFunctionState(
            owner="owner_onex_tenant",
            security_definer=True,
            search_path=("pg_catalog", "public", "pg_temp"),
            public_execute=False,
            audit_id=f"OMN-15361:tenant-report:{'a' * 64}",
            definition_sha256="a" * 64,
            audited_definition_sha256="a" * 64,
            tenant_isolation_evidence=_evidence(),
        ),
    )


def _pool_identities() -> tuple[ModelApplicationDatabasePoolIdentity, ...]:
    database = _TOPOLOGY.databases["application"]
    return tuple(
        ModelApplicationDatabasePoolIdentity(
            pool=pool,
            current_database=database.physical_name,
            current_user=binding.principal,
        )
        for pool, binding in database.bindings.items()
    )


def _violations(*states: ModelApplicationDatabaseRelationState) -> str:
    return "\n".join(validate_application_database_relation_states(states, _TOPOLOGY))


def test_green_relation_set_enforces_tenant_internal_and_catalog_domains() -> None:
    assert not validate_application_database_relation_states(
        (
            _tenant_table(),
            _non_tenant_table(EnumDatabaseSchemaDomain.OMNINODE_INTERNAL),
            _non_tenant_table(EnumDatabaseSchemaDomain.PLATFORM_CATALOG),
        ),
        _TOPOLOGY,
    )


def test_red_control_empty_authoritative_relation_set() -> None:
    assert "cannot be empty" in _violations()


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        pytest.param(
            {"columns": ()}, "identity column", id="missing-tenant-identity-column"
        ),
        pytest.param(
            {
                "columns": (
                    ModelApplicationDatabaseColumnState(
                        name="tenant_id",
                        data_type="text",
                        nullable=False,
                        default_expression=None,
                    ),
                )
            },
            "UUID",
            id="tenant-text-key",
        ),
        pytest.param(
            {
                "columns": (
                    ModelApplicationDatabaseColumnState(
                        name="tenant_id",
                        data_type="uuid",
                        nullable=True,
                        default_expression=None,
                    ),
                )
            },
            "NOT NULL",
            id="tenant-nullable",
        ),
        pytest.param(
            {
                "columns": (
                    ModelApplicationDatabaseColumnState(
                        name="tenant_id",
                        data_type="uuid",
                        nullable=False,
                        default_expression="'00000000-0000-0000-0000-000000000000'::uuid",
                    ),
                )
            },
            "default",
            id="tenant-default",
        ),
        pytest.param(
            {"rls_enabled": False},
            "ENABLE ROW LEVEL SECURITY",
            id="missing-enable-rls",
        ),
        pytest.param(
            {"rls_forced": False},
            "FORCE ROW LEVEL SECURITY",
            id="missing-force-rls",
        ),
        pytest.param({"policies": ()}, "canonical policy", id="missing-policy"),
    ],
)
def test_seeded_tenant_shape_defects_fail_closed(
    update: dict[str, object], expected: str
) -> None:
    state = _tenant_table().model_copy(update=update)
    assert expected in _violations(state)


@pytest.mark.parametrize(
    "field",
    [
        pytest.param("using_expression", id="using-drift"),
        pytest.param("with_check_expression", id="with-check-drift"),
    ],
)
def test_seeded_policy_predicate_drift_fails_closed(field: str) -> None:
    policy = (
        _tenant_table()
        .policies[0]
        .model_copy(
            update={field: "tenant_id = current_setting('app.tenant_id', true)::text"}
        )
    )
    state = _tenant_table().model_copy(update={"policies": (policy,)})
    expected = field.replace("_expression", "").replace("_", " ").upper()
    assert expected in _violations(state)


def test_red_control_canonical_policy_unrelated_role() -> None:
    policy = (
        _tenant_table().policies[0].model_copy(update={"roles": ("omninode_runtime",)})
    )
    state = _tenant_table().model_copy(update={"policies": (policy,)})

    assert "role scope" in _violations(state)


def test_policy_role_evidence_is_required_unique_and_immutable() -> None:
    policy = _tenant_table().policies[0]

    with pytest.raises(ValidationError, match="frozen"):
        policy.roles = ("omninode_runtime",)  # type: ignore[misc]
    with pytest.raises(ValidationError, match="roles must be unique"):
        ModelApplicationDatabasePolicyState(
            name="tenant_isolation",
            permissive=True,
            command="ALL",
            roles=("PUBLIC", "PUBLIC"),
            using_expression=CANONICAL_TENANT_PREDICATE,
            with_check_expression=CANONICAL_TENANT_PREDICATE,
        )
    with pytest.raises(ValidationError, match="roles"):
        ModelApplicationDatabasePolicyState.model_validate(
            {
                "name": "tenant_isolation",
                "permissive": True,
                "command": "ALL",
                "using_expression": CANONICAL_TENANT_PREDICATE,
                "with_check_expression": CANONICAL_TENANT_PREDICATE,
            }
        )


def test_tenant_identity_root_requires_closed_contract_relation_and_primary_key() -> (
    None
):
    root = _identity_root_table()
    control = root.identity_root_control_state
    assert control is not None
    assert not validate_application_database_relation_states((root,), _TOPOLOGY)
    assert "reserved" in _violations(
        root.model_copy(
            update={
                "declaration": _declaration(
                    domain=EnumDatabaseSchemaDomain.TENANT,
                    name="events",
                )
            }
        )
    )
    assert "exact primary key" in _violations(
        root.model_copy(update={"primary_key_columns": ()})
    )
    assert "live control-operation evidence" in _violations(
        root.model_copy(update={"identity_root_control_state": None})
    )


def test_red_control_identity_root_runtime_login() -> None:
    root = _identity_root_table()
    control = root.identity_root_control_state
    assert control is not None

    assert "NOLOGIN" in _violations(
        root.model_copy(
            update={
                "identity_root_control_state": control.model_copy(
                    update={"role_can_login": True}
                )
            }
        )
    )


def test_red_control_identity_root_unproven_enumeration() -> None:
    root = _identity_root_table()
    control = root.identity_root_control_state
    assert control is not None

    violations = _violations(
        root.model_copy(
            update={
                "identity_root_control_state": control.model_copy(
                    update={
                        "observed_operations": (
                            EnumApplicationDatabaseIdentityRootOperation.TENANT_CREATION,
                        ),
                        "behavioral_proof_ids": ("pytest:identity-root-tenant-create",),
                    }
                )
            }
        )
    )

    assert "differ from the declared operation set" in violations


def test_red_control_identity_root_runtime_membership() -> None:
    root = _identity_root_table()
    control = root.identity_root_control_state
    assert control is not None

    violations = _violations(
        root.model_copy(
            update={
                "identity_root_control_state": control.model_copy(
                    update={"runtime_membership_principals": ("onex_api",)}
                )
            }
        )
    )

    assert "membership path" in violations


def test_red_control_identity_root_runtime_set_role() -> None:
    root = _identity_root_table()
    control = root.identity_root_control_state
    assert control is not None

    violations = _violations(
        root.model_copy(
            update={
                "identity_root_control_state": control.model_copy(
                    update={
                        "runtime_set_role_denied_principals": _RUNTIME_PRINCIPALS[:-1]
                    }
                )
            }
        )
    )

    assert "SET ROLE denial" in violations


def test_red_control_uncontracted_identity_root() -> None:
    alternate_identity = ModelApplicationDatabaseColumnState(
        name="event_id",
        data_type="uuid",
        nullable=False,
        default_expression=None,
    )
    predicate = "event_id = current_setting('app.tenant_id', true)::uuid"
    policy = ModelApplicationDatabasePolicyState(
        name="tenant_isolation",
        permissive=True,
        command="ALL",
        roles=("PUBLIC",),
        using_expression=predicate,
        with_check_expression=predicate,
    )
    state = _tenant_table().model_copy(
        update={
            "columns": (alternate_identity,),
            "tenant_identity_column": "event_id",
            "policies": (policy,),
        }
    )
    assert "identity-root contract" in _violations(state)


def test_tenant_table_requires_explicit_manifest_identity_and_policy_names() -> None:
    state = _tenant_table().model_copy(
        update={"tenant_identity_column": None, "canonical_policy_name": None}
    )

    text = _violations(state)
    assert "explicit tenant_identity_column" in text
    assert "explicit canonical_policy_name" in text


def test_red_control_widening_permissive_policy() -> None:
    widening = ModelApplicationDatabasePolicyState(
        name="widening_read",
        permissive=True,
        command="SELECT",
        roles=("PUBLIC",),
        using_expression="true",
        with_check_expression=None,
    )
    state = _tenant_table().model_copy(
        update={"policies": (*_tenant_table().policies, widening)}
    )
    assert "permissive policy" in _violations(state)


def test_declared_restrictive_policy_requires_behavioral_proof() -> None:
    restrictive = ModelApplicationDatabasePolicyState(
        name="suspended_tenant_deny",
        permissive=False,
        command="ALL",
        roles=("PUBLIC",),
        using_expression="not tenant_suspended(tenant_id)",
        with_check_expression="not tenant_suspended(tenant_id)",
    )
    state = _tenant_table().model_copy(
        update={
            "policies": (*_tenant_table().policies, restrictive),
            "declared_restrictive_policy_names": ("suspended_tenant_deny",),
        }
    )
    assert "behavioral proof" in _violations(state)
    proven = state.model_copy(
        update={
            "restrictive_policy_proofs": {
                "suspended_tenant_deny": "pytest:tenant-suspension-isolation"
            }
        }
    )
    assert not validate_application_database_relation_states((proven,), _TOPOLOGY)


def test_red_control_owner_security_view() -> None:
    view = _tenant_view().model_copy(update={"security_invoker": False})

    assert "security_invoker" in _violations(view)


def test_red_control_unproven_security_view() -> None:
    view = _tenant_view().model_copy(update={"view_tenant_isolation_evidence": None})

    assert "behavioral evidence" in _violations(view)


def test_security_definer_requires_pg_temp_last_and_audit_bound_definition() -> None:
    function = ModelApplicationDatabaseRelationState(
        declaration=_declaration(
            domain=EnumDatabaseSchemaDomain.TENANT,
            kind=EnumApplicationRelationKind.FUNCTION,
            name="tenant_report",
        ),
        function_state=ModelApplicationDatabaseFunctionState(
            owner="owner_onex_tenant",
            security_definer=True,
            search_path=("pg_catalog", "public"),
            public_execute=False,
            audit_id="OMN-15361:tenant-report",
            tenant_isolation_evidence=_evidence(),
        ).model_copy(update={"definition_sha256": None}),
    )

    text = _violations(function)
    assert "pg_temp" in text
    assert "definition hash" in text


def test_tenant_materialized_view_is_denied() -> None:
    materialized = ModelApplicationDatabaseRelationState(
        declaration=_declaration(
            domain=EnumDatabaseSchemaDomain.TENANT,
            kind=EnumApplicationRelationKind.MATERIALIZED_VIEW,
            name="events_materialized",
        )
    )
    assert "materialized view" in _violations(materialized)


def test_tenant_foreign_table_is_denied() -> None:
    foreign = ModelApplicationDatabaseRelationState(
        declaration=_declaration(
            domain=EnumDatabaseSchemaDomain.TENANT,
            kind=EnumApplicationRelationKind.FOREIGN_TABLE,
            name="remote_events",
        )
    )

    assert "foreign table" in _violations(foreign)


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        ({"observed_rows_by_tenant": {_TENANT_A: 2, _TENANT_B: 2}}, "differ"),
        ({"expected_rows_by_tenant": {_TENANT_A: 1, _TENANT_B: 1}}, "discriminate"),
        ({"unset_context_rows": 1}, "unset"),
        ({"malformed_context_denied": False}, "malformed"),
    ],
)
def test_tenant_behavioral_evidence_fails_closed(
    update: dict[str, object], expected: str
) -> None:
    view = ModelApplicationDatabaseRelationState(
        declaration=_declaration(
            domain=EnumDatabaseSchemaDomain.TENANT,
            kind=EnumApplicationRelationKind.VIEW,
            name="events_view",
        ),
        security_invoker=True,
        view_tenant_isolation_evidence=_evidence().model_copy(update=update),
    )
    assert expected in _violations(view)


def test_red_control_unsafe_security_definer() -> None:
    function = ModelApplicationDatabaseRelationState(
        declaration=_declaration(
            domain=EnumDatabaseSchemaDomain.TENANT,
            kind=EnumApplicationRelationKind.FUNCTION,
            name="tenant_report",
        ),
        function_state=ModelApplicationDatabaseFunctionState(
            owner="app_dashboard",
            security_definer=True,
            search_path=("public", "pg_temp"),
            public_execute=True,
            audit_id=None,
            tenant_isolation_evidence=None,
        ),
    )
    text = _violations(function)
    assert "topology schema owner" in text
    assert "search_path" in text
    assert "PUBLIC EXECUTE" in text
    assert "audit" in text
    assert "behavioral evidence" in text

    green = _security_definer_function()
    assert not validate_application_database_relation_states((green,), _TOPOLOGY)
    green_function = green.function_state
    assert green_function is not None
    mismatched_definition = green.model_copy(
        update={
            "function_state": green_function.model_copy(
                update={"definition_sha256": "b" * 64}
            )
        }
    )
    assert "does not match the audited definition hash" in _violations(
        mismatched_definition
    )


def test_red_control_unproven_security_definer() -> None:
    green = _security_definer_function()
    function_state = green.function_state
    assert function_state is not None
    unproven = green.model_copy(
        update={
            "function_state": function_state.model_copy(
                update={"tenant_isolation_evidence": None}
            )
        }
    )

    assert "behavioral evidence" in _violations(unproven)


def test_red_control_security_definer_volatility_drift() -> None:
    green = _security_definer_function()
    function_state = green.function_state
    assert function_state is not None
    drifted_hash = application_database_function_definition_sha256(
        schema="public",
        name="tenant_report",
        signature="()",
        language="sql",
        source_body="SELECT count(*)::integer FROM public.events",
        parsed_sql_body=None,
        security_definer=True,
        leakproof=False,
        volatility="s",
        parallel="u",
        config=("search_path=pg_catalog, public, pg_temp",),
        kind="f",
        strict=False,
        returns_set=False,
        result_type="integer",
    )
    drifted = green.model_copy(
        update={
            "function_state": function_state.model_copy(
                update={"definition_sha256": drifted_hash}
            )
        }
    )

    assert "does not match the audited definition hash" in _violations(drifted)


@pytest.mark.parametrize(
    ("domain", "defect", "expected"),
    [
        pytest.param(
            EnumDatabaseSchemaDomain.OMNINODE_INTERNAL,
            "tenant-column",
            "tenant_id",
            id="internal-tenant-id",
        ),
        pytest.param(
            EnumDatabaseSchemaDomain.PLATFORM_CATALOG,
            "tenant-column",
            "tenant_id",
            id="catalog-tenant-id",
        ),
        pytest.param(
            EnumDatabaseSchemaDomain.OMNINODE_INTERNAL,
            "tenant-policy",
            "tenant policy",
            id="internal-tenant-policy",
        ),
        pytest.param(
            EnumDatabaseSchemaDomain.PLATFORM_CATALOG,
            "rls",
            "RLS",
            id="catalog-rls",
        ),
    ],
)
def test_non_tenant_domain_red_control_fails_closed(
    domain: EnumDatabaseSchemaDomain,
    defect: str,
    expected: str,
) -> None:
    state = _non_tenant_table(domain)
    if defect == "tenant-column":
        state = state.model_copy(
            update={
                "columns": (
                    *state.columns,
                    ModelApplicationDatabaseColumnState(
                        name="tenant_id",
                        data_type="uuid",
                        nullable=False,
                        default_expression=None,
                    ),
                )
            }
        )
    elif defect == "tenant-policy":
        state = state.model_copy(update={"policies": _tenant_table().policies})
    else:
        state = state.model_copy(update={"rls_enabled": True, "rls_forced": True})

    assert expected in _violations(state)


@pytest.mark.parametrize(
    ("update", "expected"),
    [
        pytest.param(
            {"source_tenant_provenance_contract": None},
            "provenance contract",
            id="uncontracted-source-tenant",
        ),
        pytest.param(
            {
                "columns": (
                    ModelApplicationDatabaseColumnState(
                        name="source_tenant_id",
                        data_type="text",
                        nullable=True,
                        default_expression=None,
                    ),
                )
            },
            "UUID",
            id="source-tenant-non-uuid",
        ),
        pytest.param(
            {
                "columns": (
                    ModelApplicationDatabaseColumnState(
                        name="source_tenant_id",
                        data_type="uuid",
                        nullable=False,
                        default_expression=None,
                    ),
                )
            },
            "nullable",
            id="source-tenant-non-null",
        ),
        pytest.param(
            {"primary_key_columns": ("source_tenant_id",)},
            "primary-key",
            id="source-tenant-primary-key",
        ),
    ],
)
def test_source_tenant_id_is_typed_non_authoritative_provenance(
    update: dict[str, object], expected: str
) -> None:
    state = _non_tenant_table(EnumDatabaseSchemaDomain.OMNINODE_INTERNAL).model_copy(
        update=update
    )
    assert expected in _violations(state)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("unique_index_column_sets", "uniqueness"),
        ("foreign_key_column_sets", "foreign key"),
        ("partition_key_columns", "partition"),
        ("deduplication_key_columns", "deduplication"),
        ("authorization_dependency_columns", "authorization"),
        ("write_eligibility_dependency_columns", "write eligibility"),
    ],
)
def test_source_tenant_id_cannot_drive_authoritative_semantics(
    field: str, expected: str
) -> None:
    value: object = (
        (("source_tenant_id",),)
        if field in {"unique_index_column_sets", "foreign_key_column_sets"}
        else ("source_tenant_id",)
    )
    state = _non_tenant_table(EnumDatabaseSchemaDomain.OMNINODE_INTERNAL).model_copy(
        update={field: value}
    )

    assert expected in _violations(state)


def test_red_control_source_tenant_generated_unique_alias() -> None:
    state = _non_tenant_table(EnumDatabaseSchemaDomain.OMNINODE_INTERNAL).model_copy(
        update={
            "columns": (
                *_non_tenant_table(EnumDatabaseSchemaDomain.OMNINODE_INTERNAL).columns,
                ModelApplicationDatabaseColumnState(
                    name="source_tenant_copy",
                    data_type="uuid",
                    nullable=True,
                    generated_expression="source_tenant_id",
                ),
            ),
            "unique_index_column_sets": (("source_tenant_copy",),),
        }
    )

    assert "uniqueness" in _violations(state)


def test_topology_schema_domain_drift_fails_closed() -> None:
    state = _tenant_table().model_copy(
        update={
            "declaration": _declaration(
                domain=EnumDatabaseSchemaDomain.TENANT,
                schema="omninode_internal",
            )
        }
    )
    assert "typed topology domain" in _violations(state)


def test_red_control_public_catalog_leak() -> None:
    states = (
        _tenant_table(),
        _non_tenant_table(EnumDatabaseSchemaDomain.OMNINODE_INTERNAL),
        _non_tenant_table(EnumDatabaseSchemaDomain.PLATFORM_CATALOG),
    )
    observed = tuple(
        ModelApplicationDatabaseCatalogIdentity(
            schema=state.declaration.schema,
            name=state.declaration.name,
            kind=EnumApplicationInventoryObjectKind(state.declaration.kind.value),
        )
        for state in states
    )
    assert not validate_application_database_catalog_census(states, observed, _TOPOLOGY)
    public_leak = ModelApplicationDatabaseCatalogIdentity(
        schema="public",
        name="shadow_events",
        kind=EnumApplicationInventoryObjectKind.TABLE,
    )
    assert "undeclared" in "\n".join(
        validate_application_database_catalog_census(
            states, (*observed, public_leak), _TOPOLOGY
        )
    )
    assert "missing" in "\n".join(
        validate_application_database_catalog_census(states, observed[1:], _TOPOLOGY)
    )


def test_catalog_census_expected_set_is_manifest_authoritative_and_signature_exact() -> (
    None
):
    states = (_tenant_table(),)
    relation_identity = ModelApplicationDatabaseCatalogIdentity(
        schema="public",
        name="events",
        kind=EnumApplicationInventoryObjectKind.TABLE,
    )
    overload_a = ModelApplicationDatabaseCatalogIdentity(
        schema="public",
        name="safe_report",
        kind=EnumApplicationInventoryObjectKind.FUNCTION,
        function_signature="()",
    )
    overload_b = overload_a.model_copy(update={"function_signature": "(uuid)"})
    authoritative = (relation_identity, overload_a, overload_b)

    assert not validate_application_database_catalog_census(
        states,
        authoritative,
        _TOPOLOGY,
        authoritative_identities=authoritative,
    )
    assert "missing" in "\n".join(
        validate_application_database_catalog_census(
            states,
            authoritative[:-1],
            _TOPOLOGY,
            authoritative_identities=authoritative,
        )
    )
    signature_drift = overload_b.model_copy(update={"function_signature": None})
    drift_violations = validate_application_database_catalog_census(
        states,
        (relation_identity, overload_a, signature_drift),
        _TOPOLOGY,
        authoritative_identities=authoritative,
    )
    assert any("missing" in violation for violation in drift_violations)
    assert any("undeclared" in violation for violation in drift_violations)


def test_red_control_public_application_table() -> None:
    """OMN-17887: `public` is the TENANT domain's schema, and only TENANT's.

    The blanket "prohibited in public" refusal is gone because a TENANT relation
    now lives in `public` (test_green_relation_set_... proves that passes). What
    stays refused is any NON-TENANT relation placed there: it contradicts the
    typed topology's declared domain for `public`. A `public` CREATE is also
    still projected into the created-catalog census, so the SQL gate holds it to
    exactly one ownership declaration.
    """
    for domain in (
        EnumDatabaseSchemaDomain.OMNINODE_INTERNAL,
        EnumDatabaseSchemaDomain.PLATFORM_CATALOG,
    ):
        misplaced = _non_tenant_table(domain).model_copy(
            update={"declaration": _declaration(domain=domain, schema="public")}
        )
        assert "typed topology domain" in _violations(misplaced), domain

    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(
            "CREATE TABLE public.events (id uuid);"
        )
    ) == (("public", "events", "table"),)


def test_red_control_unqualified_application_table() -> None:
    assert "schema-qualified" in "\n".join(
        lint_application_database_sql("CREATE TABLE events (id uuid);", _TOPOLOGY)
    )
    assert "schema-qualified" in "\n".join(
        lint_application_database_sql(
            'CREATE TABLE "omninode_internal.events" (id uuid);', _TOPOLOGY
        )
    )
    assert not lint_application_database_sql(
        "CREATE TABLE omninode_internal.events (id uuid PRIMARY KEY);", _TOPOLOGY
    )


def test_red_control_unqualified_application_mutation_target() -> None:
    violations = lint_application_database_sql(
        "UPDATE events SET payload = '{}'::jsonb;", _TOPOLOGY
    )

    assert "schema-qualified" in "\n".join(violations)


def test_deployed_unqualified_tenant_relation_reference_is_accepted() -> None:
    """OMN-17887: node_projection_delegation/0046 as vendored on dev.

    Tenant-domain relations live in `public`, and this deployed, append-only
    migration names one without a schema. It was accepted while the relation was
    bridged and must stay accepted now that the bridge is gone.
    """
    assert not lint_application_database_sql(
        "ALTER TABLE delegation_events\n"
        "    ADD COLUMN IF NOT EXISTS cohort_key JSONB,\n"
        "    ADD COLUMN IF NOT EXISTS cohort_key_sha256 TEXT,\n"
        "    ADD COLUMN IF NOT EXISTS cohort_key_refusal TEXT;",
        _TOPOLOGY,
    )


def test_unqualified_allowance_does_not_extend_to_other_public_relations() -> None:
    """The allowance is the 24 formerly-bridged names, not every `public` table."""
    for statement in (
        "ALTER TABLE brand_new_tenant_table ADD COLUMN payload jsonb;",
        # Granted in `public` by the topology, but never bridged: still refused.
        "ALTER TABLE dispatch_eval_results ADD COLUMN payload jsonb;",
    ):
        assert "schema-qualified" in "\n".join(
            lint_application_database_sql(statement, _TOPOLOGY)
        ), statement


def test_unqualified_tenant_allowance_equals_the_retired_bridge() -> None:
    """Pinned so the allowance can only shrink, by an explicit edit here."""
    from omnibase_infra.validation import application_database_domain_enforcement

    assert (
        frozenset(
            {
                "agent_routing_decisions",
                "capability_scores",
                "context_roi_scores",
                "delegation_budget_state",
                "delegation_events",
                "delegation_judge_verdict_events",
                "delegation_routing_tenant_overlay",
                "delegation_shadow_comparisons",
                "dep_health_findings",
                "hook_events",
                "instruction_eval_aggregate_snapshots",
                "llm_cost_aggregates",
                "pattern_learning_artifacts",
                "projection_cost_savings_overview",
                "projection_delegation_inference_response_text",
                "projection_delegation_model_routing",
                "projection_delegation_quality_gate",
                "projection_delegation_savings",
                "projection_delegation_savings_series",
                "projection_delegation_summary",
                "projection_delegation_token_usage",
                "savings_estimates",
                "skill_execution_snapshots",
                "tenant_inference_credentials",
            }
        )
        == application_database_domain_enforcement._TENANT_RELATIONS_REFERENCED_UNQUALIFIED
    )


def test_red_control_unknown_topology_schema() -> None:
    violations = lint_application_database_sql(
        "CREATE TABLE mystery.events (id uuid);", _TOPOLOGY
    )

    assert "unknown topology schema" in "\n".join(violations)


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE TEMPORARY TABLE events (id uuid);",
        "CREATE DOMAIN tenant_slug AS text;",
        "CREATE EXTENSION hstore;",
        "ALTER TABLE events ADD COLUMN payload jsonb;",
        "DROP VIEW events_view;",
        "INSERT INTO events (id) VALUES ('00000000-0000-0000-0000-000000000001');",
        "UPDATE events SET payload = '{}'::jsonb;",
        "DELETE FROM events;",
        "TRUNCATE TABLE events;",
        "SELECT * FROM events;",
        "MERGE INTO events USING omninode_internal.incoming ON false WHEN NOT MATCHED THEN DO NOTHING;",
        "COPY events TO STDOUT;",
        "GRANT SELECT ON TABLE events TO app_dashboard;",
        "CREATE INDEX events_payload_idx ON events (payload);",
        "CREATE TABLE omninode_internal.children (parent_id uuid REFERENCES parents(id));",
    ],
)
def test_unqualified_application_relation_targets_are_rejected(
    statement: str,
) -> None:
    assert "schema-qualified" in "\n".join(
        lint_application_database_sql(statement, _TOPOLOGY)
    )


@pytest.mark.parametrize(
    "statement",
    [
        "CREATE MATERIALIZED VIEW omninode_internal.event_rollup AS SELECT 1;",
        "CREATE SEQUENCE omninode_internal.event_seq;",
        "CREATE DOMAIN omninode_internal.tenant_slug AS text;",
        "CREATE EXTENSION hstore WITH SCHEMA omninode_internal;",
        "ALTER TABLE omninode_internal.events ADD COLUMN payload jsonb;",
        "INSERT INTO omninode_internal.events (id) VALUES ('00000000-0000-0000-0000-000000000001');",
        "UPDATE omninode_internal.events SET payload = '{}'::jsonb;",
        "DELETE FROM omninode_internal.events;",
        "TRUNCATE TABLE omninode_internal.events;",
        "MERGE INTO omninode_internal.events USING omninode_internal.incoming ON false WHEN NOT MATCHED THEN DO NOTHING;",
        "CREATE TABLE omninode_internal.children (parent_id uuid REFERENCES omninode_internal.parents(id));",
    ],
)
def test_qualified_application_relation_targets_are_accepted(statement: str) -> None:
    assert not lint_application_database_sql(statement, _TOPOLOGY)


def test_public_unlogged_application_table_is_rejected() -> None:
    """OMN-17887: UNLOGGED cannot hide a CREATE from either check.

    In the retired `tenant` schema it is refused as an unknown topology schema;
    in `public` (the TENANT domain's schema) it is still a created identity the
    SQL gate holds to exactly one ownership declaration.
    """
    assert "'tenant.events' uses unknown topology schema" in "\n".join(
        lint_application_database_sql(
            "CREATE UNLOGGED TABLE tenant.events (id uuid);", _TOPOLOGY
        )
    )
    assert tuple(
        (identity.schema, identity.name, identity.kind.value)
        for identity in application_database_created_catalog_identities(
            "CREATE UNLOGGED TABLE public.events (id uuid);"
        )
    ) == (("public", "events", "table"),)


def test_qualified_ctes_table_functions_and_system_catalog_reads_are_accepted() -> None:
    assert not lint_application_database_sql(
        "WITH recent AS (SELECT * FROM omninode_internal.events) SELECT * FROM recent;",
        _TOPOLOGY,
    )
    assert not lint_application_database_sql(
        "SELECT * FROM pg_catalog.generate_series(1, 2);", _TOPOLOGY
    )


@pytest.mark.parametrize(
    "statement",
    [
        "WITH x AS (SELECT 1) UPDATE events SET payload = 'x';",
        "WITH x AS (SELECT 1) INSERT INTO events (payload) SELECT 1 FROM x;",
        "WITH x AS (SELECT 1) MERGE INTO events USING x ON false WHEN NOT MATCHED THEN DO NOTHING;",
        "SELECT * FROM omninode_internal.events, unqualified;",
        "DROP TABLE omninode_internal.events, unqualified;",
        "CREATE POLICY tenant_policy ON events USING (true);",
        "CREATE TRIGGER tenant_trigger BEFORE INSERT ON events FOR EACH ROW EXECUTE FUNCTION omninode_internal.audit();",
        "REFRESH MATERIALIZED VIEW events;",
        "CREATE TABLE omninode_internal.child PARTITION OF parent FOR VALUES IN (1);",
        "WITH changed AS (UPDATE events SET payload = 'x' RETURNING *) SELECT * FROM changed;",
        "WITH events AS (SELECT * FROM events) SELECT * FROM events;",
        "DROP FUNCTION omninode_internal.safe_report(), rogue();",
        "DROP PROCEDURE omninode_internal.refresh_cache(), rogue();",
        "DROP AGGREGATE omninode_internal.total(integer), rogue(integer);",
        "ALTER FOREIGN TABLE rogue ADD COLUMN payload text;",
        "DROP FOREIGN TABLE omninode_internal.remote_events, rogue;",
        "CREATE TABLE omninode_internal.child (id uuid) INHERITS (omninode_internal.parent, rogue);",
    ],
)
def test_adversarial_unqualified_sql_forms_fail_closed(statement: str) -> None:
    assert "schema-qualified" in "\n".join(
        lint_application_database_sql(statement, _TOPOLOGY)
    )


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        pytest.param(
            "EXPLAIN ANALYZE UPDATE events SET payload = 'x';",
            "schema-qualified",
            id="explain-unqualified-mutation",
        ),
        # OMN-17887: these three rows used `public.<name>` as the refused
        # source. `public` is now the TENANT domain's declared schema (held to
        # ownership, see test_alternate_target_forms_reading_public_are_held_to_
        # ownership), so they refuse the retired `tenant` schema instead.
        pytest.param(
            "CREATE VIEW omninode_internal.events_copy AS TABLE tenant.events;",
            "'tenant.events' uses unknown topology schema",
            id="as-table-retired-schema-read",
        ),
        pytest.param(
            "CREATE TABLE omninode_internal.events_copy (LIKE tenant.events INCLUDING ALL);",
            "'tenant.events' uses unknown topology schema",
            id="like-retired-schema-relation",
        ),
        pytest.param(
            "ALTER TABLE omninode_internal.events INHERIT tenant.parent_events;",
            "'tenant.parent_events' uses unknown topology schema",
            id="inherit-retired-schema-relation",
        ),
        pytest.param(
            "CREATE TRIGGER audit BEFORE INSERT ON omninode_internal.events "
            "FOR EACH ROW EXECUTE FUNCTION audit_event();",
            "schema-qualified",
            id="unqualified-trigger-function",
        ),
        pytest.param(
            "CALL refresh_tenants();",
            "schema-qualified",
            id="unqualified-procedure-call",
        ),
        pytest.param(
            "DO $$ BEGIN EXECUTE 'DROP TABLE public.events'; END $$;",
            "dynamic SQL",
            id="dynamic-sql-target",
        ),
        pytest.param(
            "CREATE TYPE omninode_internal.event_span AS RANGE (SUBTYPE = timestamptz);",
            "MULTIRANGE_TYPE_NAME",
            id="implicit-multirange-identity",
        ),
    ],
)
def test_valid_postgres_alternate_target_forms_fail_closed(
    statement: str,
    expected: str,
) -> None:
    assert expected in "\n".join(lint_application_database_sql(statement, _TOPOLOGY))


@pytest.mark.parametrize(
    ("statement", "location"),
    [
        pytest.param(
            "CREATE VIEW omninode_internal.events_copy AS TABLE public.events;",
            ("public", "events"),
            id="as-table-public-read",
        ),
        pytest.param(
            "CREATE TABLE omninode_internal.events_copy (LIKE public.events INCLUDING ALL);",
            ("public", "events"),
            id="like-public-relation",
        ),
        pytest.param(
            "ALTER TABLE omninode_internal.events INHERIT public.parent_events;",
            ("public", "parent_events"),
            id="inherit-public-relation",
        ),
    ],
)
def test_alternate_target_forms_reading_public_are_held_to_ownership(
    statement: str,
    location: tuple[str, str],
) -> None:
    """OMN-17887: a `public` source in an alternate form is still a real target.

    The lint no longer refuses `public.<name>`, so each form must emit the exact
    ownership requirement the SQL gate refuses when no single owner answers.
    """
    assert location in {
        requirement.location
        for requirement in application_database_sql_target_requirements(
            statement, _TOPOLOGY
        )
    }


@pytest.mark.parametrize(
    "statement",
    [
        "SELECT 'from users'::text;",
        "INSERT INTO omninode_internal.events (payload) VALUES ('join public.events');",
        "SELECT $$update events set payload = 'x'$$::text;",
    ],
)
def test_sql_keywords_inside_literals_are_not_treated_as_relation_targets(
    statement: str,
) -> None:
    assert not lint_application_database_sql(statement, _TOPOLOGY)


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        (
            "CREATE FOREIGN TABLE omninode_internal.remote_events (id uuid) SERVER remote;",
            (("omninode_internal", "remote_events", "foreign_table", None),),
        ),
        (
            "CREATE AGGREGATE omninode_internal.total(integer) (SFUNC = int4pl, STYPE = integer);",
            (("omninode_internal", "total", "aggregate", "(integer)"),),
        ),
        (
            "CREATE FUNCTION omninode_internal.rank_state(integer) RETURNS integer LANGUAGE internal WINDOW AS 'window_rank';",
            (("omninode_internal", "rank_state", "window_function", "(integer)"),),
        ),
        (
            "CREATE TYPE omninode_internal.iso_code (INPUT = textin, OUTPUT = textout);",
            (("omninode_internal", "iso_code", "base_type", None),),
        ),
        (
            "CREATE TYPE omninode_internal.event_span AS RANGE (SUBTYPE = timestamptz, MULTIRANGE_TYPE_NAME = omninode_internal.event_spans);",
            (
                ("omninode_internal", "event_span", "range_type", None),
                ("omninode_internal", "event_spans", "multirange_type", None),
            ),
        ),
        (
            "CREATE EXTENSION hstore WITH SCHEMA omninode_internal;",
            (("omninode_internal", "hstore", "extension", None),),
        ),
        (
            "CREATE DOMAIN omninode_internal.tenant_slug AS text CHECK (VALUE <> '');",
            (("omninode_internal", "tenant_slug", "type", None),),
        ),
    ],
)
def test_created_catalog_classifier_preserves_exact_supporting_object_kind(
    statement: str,
    expected: tuple[tuple[str, str, str, str | None], ...],
) -> None:
    observed = application_database_created_catalog_identities(statement)
    assert (
        tuple(
            (
                identity.schema,
                identity.name,
                identity.kind.value,
                identity.function_signature,
            )
            for identity in observed
        )
        == expected
    )


def test_pool_identity_gate_accepts_exact_topology_bindings() -> None:
    identities = _pool_identities()

    assert not validate_application_database_pool_identities(identities, _TOPOLOGY)


def test_red_control_old_application_database() -> None:
    identities = _pool_identities()
    wrong_database = identities[0].model_copy(
        update={"current_database": "omninode_cloud"}
    )

    assert "one physical database" in "\n".join(
        validate_application_database_pool_identities(
            (wrong_database, *identities[1:]), _TOPOLOGY
        )
    )


def test_red_control_duplicate_pool_user() -> None:
    identities = _pool_identities()
    duplicate_user = identities[1].model_copy(
        update={"current_user": identities[0].current_user}
    )

    assert "distinct" in "\n".join(
        validate_application_database_pool_identities(
            (identities[0], duplicate_user, *identities[2:]), _TOPOLOGY
        )
    )


def test_red_control_wrong_pool_user() -> None:
    identities = _pool_identities()
    wrong_user = identities[0].model_copy(update={"current_user": "wrong_runtime_user"})

    assert "expected current_user" in "\n".join(
        validate_application_database_pool_identities(
            (wrong_user, *identities[1:]), _TOPOLOGY
        )
    )


def test_red_control_missing_pool_binding() -> None:
    identities = _pool_identities()

    assert "exact typed topology binding set" in "\n".join(
        validate_application_database_pool_identities(identities[1:], _TOPOLOGY)
    )


def test_function_definition_fingerprint_covers_security_relevant_catalog_state() -> (
    None
):
    definition: dict[str, object] = {
        "schema": "public",
        "name": "safe_report",
        "signature": "()",
        "language": "sql",
        "source_body": "SELECT count(*)::integer FROM public.events",
        "parsed_sql_body": None,
        "security_definer": True,
        "leakproof": False,
        "volatility": "v",
        "parallel": "u",
        "config": ("search_path=pg_catalog, public, pg_temp",),
        "kind": "f",
        "strict": False,
        "returns_set": False,
        "result_type": "integer",
    }
    expected = application_database_function_definition_sha256(**definition)  # type: ignore[arg-type]
    drifts = {
        "language": "plpgsql",
        "source_body": "SELECT 0",
        "parsed_sql_body": "{QUERY :commandType 1}",
        "security_definer": False,
        "leakproof": True,
        "volatility": "s",
        "parallel": "s",
        "config": ("search_path=public",),
        "kind": "w",
        "strict": True,
        "returns_set": True,
        "result_type": "bigint",
    }
    for field, value in drifts.items():
        changed = {**definition, field: value}
        assert (
            application_database_function_definition_sha256(**changed)  # type: ignore[arg-type]
            != expected
        ), field


def _physically_public_bridge_manifest(
    tmp_path: Path, *, schema: str, name: str
) -> Path:
    """Write a minimal service manifest declaring one relation at ``schema``."""
    manifest = tmp_path / f"{schema}-{name}-ownership.yaml"
    manifest.write_text(
        "schema_version: '1.0'\n"
        "service: omn16993_bridge_fixture\n"
        "target_database_ref: application\n"
        "db_io:\n"
        "  db_tables:\n"
        f"    - name: {name}\n"
        "      database_ref: application\n"
        f"      schema: {schema}\n"
        "      migration: fixture.sql\n"
        "      access: read_write\n"
        "      role: omn16993_bridge_fixture_table\n",
        encoding="utf-8",
    )
    return manifest


# OMN-17887: a relation that was in the deleted tenant bridge set until the
# operator ruling retired the `tenant` schema. Named literally because the set
# it came from no longer exists to be imported.
_FORMER_TENANT_BRIDGE_RELATION = "delegation_events"


def _located_identities(manifest: Path, name: str) -> list[tuple[str, str]]:
    return [
        (identity.schema, identity.name)
        for identity in load_application_database_ownership_identities((manifest,))
        if identity.name == name
    ]


def test_physically_public_bridge_covers_internal_family_and_tenant_is_public(
    tmp_path: Path,
) -> None:
    """OMN-16993 / OMN-17887: every owned relation resolves where SQL names it.

    Deployable SQL must target the PHYSICAL relation. For
    ``INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359`` that is still bare
    ``public`` until the governed OMN-15359 cutover, so a relation declared at
    its LOGICAL ``omninode_internal`` schema must bridge to ``public``. Without
    that, NEW deployable SQL touching one of those relations resolves to ZERO
    ownership declarations and cannot be made to pass.

    The tenant family no longer needs a bridge: OMN-17887 made ``public`` the
    TENANT domain's schema, so a tenant relation is declared ``schema: public``
    directly and must resolve at ``public`` exactly once.
    """
    internal_name = sorted(INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359)[0]
    internal_manifest = _physically_public_bridge_manifest(
        tmp_path, schema="omninode_internal", name=internal_name
    )
    internal_located = set(_located_identities(internal_manifest, internal_name))
    assert ("omninode_internal", internal_name) in internal_located, internal_located
    assert ("public", internal_name) in internal_located, (
        f"omninode_internal.{internal_name} did not bridge to its physical public "
        "identity; deployable SQL targeting public."
        f"{internal_name} would resolve to zero ownership declarations"
    )

    tenant_name = _FORMER_TENANT_BRIDGE_RELATION
    tenant_manifest = _physically_public_bridge_manifest(
        tmp_path, schema="public", name=tenant_name
    )
    assert _located_identities(tenant_manifest, tenant_name) == [
        ("public", tenant_name)
    ]


def test_tenant_schema_is_retired_so_nothing_bridges_or_resolves_through_it(
    tmp_path: Path,
) -> None:
    """OMN-17887: the `tenant` schema is retired, not silently re-aliased.

    This replaces the OMN-16993 disjointness check between the tenant and
    internal bridge families: the tenant family is gone, so the risk it guarded
    -- one name bridging from two logical schemas into two ``public`` twins --
    now reduces to the ``tenant`` half never bridging at all. A relation still
    declared at ``tenant`` must NOT gain a ``public`` twin (that would let a
    stale declaration own a real public relation), the physical grant mapping
    must not route ``tenant`` to ``public``, and the typed topology must refuse
    ``tenant`` as an unknown schema in SQL.
    """
    name = _FORMER_TENANT_BRIDGE_RELATION
    assert name not in INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359

    stale_manifest = _physically_public_bridge_manifest(
        tmp_path, schema="tenant", name=name
    )
    located = _located_identities(stale_manifest, name)
    assert ("public", name) not in located, located
    assert located == [("tenant", name)]

    assert physical_grant_schema_for_table("tenant", name) == "tenant"
    assert "tenant" not in {
        schema_name
        for database in _TOPOLOGY.databases.values()
        for schema_name in database.schemas
    }
    assert "unknown topology schema" in "\n".join(
        lint_application_database_sql(
            f"ALTER TABLE tenant.{name} ADD COLUMN payload text;", _TOPOLOGY
        )
    )


def test_relation_outside_the_internal_family_does_not_bridge_to_public(
    tmp_path: Path,
) -> None:
    """OMN-16993: the bridge is scoped, not a blanket public twin for everything.

    A relation whose physical home really is its logical schema must NOT gain a
    spurious ``public`` declaration -- that would let SQL targeting
    ``public.<name>`` resolve an owner for a relation that does not exist there.
    """
    name = "omn16993_not_physically_public_fixture"
    assert name not in INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359

    manifest = _physically_public_bridge_manifest(
        tmp_path, schema="omninode_internal", name=name
    )
    identities = load_application_database_ownership_identities((manifest,))
    located = {
        (identity.schema, identity.name)
        for identity in identities
        if identity.name == name
    }
    assert ("omninode_internal", name) in located
    assert ("public", name) not in located


# ---------------------------------------------------------------------------
# OMN-17486: a routine signature written across several lines is still one
# identity. Until migration 107 no migration in the repo declared a multi-line
# CREATE FUNCTION, so this path had never been exercised: the extractor kept
# the newlines, and ApplicationDatabaseFunctionSignature refuses a newline by
# design, so the catalog could not be built at all. Normalizing at the two
# extraction sites is what makes the signature catalogable, and both sites have
# to agree or a multi-line CREATE and its multi-line GRANT resolve to two
# different identities.
# ---------------------------------------------------------------------------

_MULTILINE_ROUTINE_SQL = """
CREATE SCHEMA app_domain_probe;

CREATE FUNCTION app_domain_probe.probe_routine(
    p_first TEXT,
    p_second BOOLEAN,
    p_third TIMESTAMPTZ
)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $probe$
BEGIN
    RETURN TRUE;
END;
$probe$;
"""

_SINGLE_LINE_ROUTINE_SQL = """
CREATE SCHEMA app_domain_probe;

CREATE FUNCTION app_domain_probe.probe_routine(p_first TEXT, p_second BOOLEAN, p_third TIMESTAMPTZ)
RETURNS BOOLEAN
LANGUAGE plpgsql
AS $probe$
BEGIN
    RETURN TRUE;
END;
$probe$;
"""


@pytest.mark.unit
def test_multiline_routine_signature_is_catalogued_as_one_identity() -> None:
    identities = application_database_created_catalog_identities(_MULTILINE_ROUTINE_SQL)
    routines = [identity for identity in identities if identity.name == "probe_routine"]
    assert len(routines) == 1
    signature = routines[0].function_signature
    assert signature is not None
    assert "\n" not in signature
    assert signature == "(TEXT, BOOLEAN, TIMESTAMPTZ)"


@pytest.mark.unit
def test_multiline_and_single_line_routine_signatures_are_the_same_identity() -> None:
    """Layout may not change what a routine is called in the catalog."""
    multiline = application_database_created_catalog_identities(_MULTILINE_ROUTINE_SQL)
    single = application_database_created_catalog_identities(_SINGLE_LINE_ROUTINE_SQL)
    assert [
        (identity.schema, identity.name, identity.kind, identity.function_signature)
        for identity in multiline
    ] == [
        (identity.schema, identity.name, identity.kind, identity.function_signature)
        for identity in single
    ]
