# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed application database domain enforcement (OMN-15361)."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_infra.topology.application_database import load_topology_profile
from omnibase_infra.validation.application_database_domain_enforcement import (
    CANONICAL_TENANT_PREDICATE,
    application_database_created_catalog_identities,
    application_database_function_definition_sha256,
    lint_application_database_sql,
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
            EnumDatabaseSchemaDomain.TENANT: "tenant",
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
            search_path=("pg_catalog", "tenant", "pg_temp"),
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
            search_path=("pg_catalog", "tenant"),
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
        schema="tenant",
        name="tenant_report",
        signature="()",
        language="sql",
        source_body="SELECT count(*)::integer FROM tenant.events",
        parsed_sql_body=None,
        security_definer=True,
        leakproof=False,
        volatility="s",
        parallel="u",
        config=("search_path=pg_catalog, tenant, pg_temp",),
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
        schema="tenant",
        name="events",
        kind=EnumApplicationInventoryObjectKind.TABLE,
    )
    overload_a = ModelApplicationDatabaseCatalogIdentity(
        schema="tenant",
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
    assert "public" in "\n".join(
        lint_application_database_sql(
            "CREATE TABLE public.events (id uuid);", _TOPOLOGY
        )
    )


def test_red_control_unqualified_application_table() -> None:
    assert "schema-qualified" in "\n".join(
        lint_application_database_sql("CREATE TABLE events (id uuid);", _TOPOLOGY)
    )
    assert "schema-qualified" in "\n".join(
        lint_application_database_sql(
            'CREATE TABLE "tenant.events" (id uuid);', _TOPOLOGY
        )
    )
    assert not lint_application_database_sql(
        "CREATE TABLE tenant.events (id uuid PRIMARY KEY);", _TOPOLOGY
    )


def test_red_control_unqualified_application_mutation_target() -> None:
    violations = lint_application_database_sql(
        "UPDATE events SET payload = '{}'::jsonb;", _TOPOLOGY
    )

    assert "schema-qualified" in "\n".join(violations)


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
        "MERGE INTO events USING tenant.incoming ON false WHEN NOT MATCHED THEN DO NOTHING;",
        "COPY events TO STDOUT;",
        "GRANT SELECT ON TABLE events TO app_dashboard;",
        "CREATE INDEX events_payload_idx ON events (payload);",
        "CREATE TABLE tenant.children (parent_id uuid REFERENCES parents(id));",
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
        "CREATE MATERIALIZED VIEW tenant.event_rollup AS SELECT 1;",
        "CREATE SEQUENCE tenant.event_seq;",
        "CREATE DOMAIN tenant.tenant_slug AS text;",
        "CREATE EXTENSION hstore WITH SCHEMA tenant;",
        "ALTER TABLE tenant.events ADD COLUMN payload jsonb;",
        "INSERT INTO tenant.events (id) VALUES ('00000000-0000-0000-0000-000000000001');",
        "UPDATE tenant.events SET payload = '{}'::jsonb;",
        "DELETE FROM tenant.events;",
        "TRUNCATE TABLE tenant.events;",
        "MERGE INTO tenant.events USING tenant.incoming ON false WHEN NOT MATCHED THEN DO NOTHING;",
        "CREATE TABLE tenant.children (parent_id uuid REFERENCES tenant.parents(id));",
    ],
)
def test_qualified_application_relation_targets_are_accepted(statement: str) -> None:
    assert not lint_application_database_sql(statement, _TOPOLOGY)


def test_public_unlogged_application_table_is_rejected() -> None:
    assert "public" in "\n".join(
        lint_application_database_sql(
            "CREATE UNLOGGED TABLE public.events (id uuid);", _TOPOLOGY
        )
    )


def test_qualified_ctes_table_functions_and_system_catalog_reads_are_accepted() -> None:
    assert not lint_application_database_sql(
        "WITH recent AS (SELECT * FROM tenant.events) SELECT * FROM recent;",
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
        "SELECT * FROM tenant.events, unqualified;",
        "DROP TABLE tenant.events, unqualified;",
        "CREATE POLICY tenant_policy ON events USING (true);",
        "CREATE TRIGGER tenant_trigger BEFORE INSERT ON events FOR EACH ROW EXECUTE FUNCTION tenant.audit();",
        "REFRESH MATERIALIZED VIEW events;",
        "CREATE TABLE tenant.child PARTITION OF parent FOR VALUES IN (1);",
        "WITH changed AS (UPDATE events SET payload = 'x' RETURNING *) SELECT * FROM changed;",
        "WITH events AS (SELECT * FROM events) SELECT * FROM events;",
        "DROP FUNCTION tenant.safe_report(), rogue();",
        "DROP PROCEDURE tenant.refresh_cache(), rogue();",
        "DROP AGGREGATE tenant.total(integer), rogue(integer);",
        "ALTER FOREIGN TABLE rogue ADD COLUMN payload text;",
        "DROP FOREIGN TABLE tenant.remote_events, rogue;",
        "CREATE TABLE tenant.child (id uuid) INHERITS (tenant.parent, rogue);",
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
        pytest.param(
            "CREATE VIEW tenant.events_copy AS TABLE public.events;",
            "public",
            id="as-table-public-read",
        ),
        pytest.param(
            "CREATE TABLE tenant.events_copy (LIKE public.events INCLUDING ALL);",
            "public",
            id="like-public-relation",
        ),
        pytest.param(
            "ALTER TABLE tenant.events INHERIT public.parent_events;",
            "public",
            id="inherit-public-relation",
        ),
        pytest.param(
            "CREATE TRIGGER audit BEFORE INSERT ON tenant.events "
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
            "CREATE TYPE tenant.event_span AS RANGE (SUBTYPE = timestamptz);",
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
    "statement",
    [
        "SELECT 'from users'::text;",
        "INSERT INTO tenant.events (payload) VALUES ('join public.events');",
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
            "CREATE FOREIGN TABLE tenant.remote_events (id uuid) SERVER remote;",
            (("tenant", "remote_events", "foreign_table", None),),
        ),
        (
            "CREATE AGGREGATE tenant.total(integer) (SFUNC = int4pl, STYPE = integer);",
            (("tenant", "total", "aggregate", "(integer)"),),
        ),
        (
            "CREATE FUNCTION tenant.rank_state(integer) RETURNS integer LANGUAGE internal WINDOW AS 'window_rank';",
            (("tenant", "rank_state", "window_function", "(integer)"),),
        ),
        (
            "CREATE TYPE tenant.iso_code (INPUT = textin, OUTPUT = textout);",
            (("tenant", "iso_code", "base_type", None),),
        ),
        (
            "CREATE TYPE tenant.event_span AS RANGE (SUBTYPE = timestamptz, MULTIRANGE_TYPE_NAME = tenant.event_spans);",
            (
                ("tenant", "event_span", "range_type", None),
                ("tenant", "event_spans", "multirange_type", None),
            ),
        ),
        (
            "CREATE EXTENSION hstore WITH SCHEMA tenant;",
            (("tenant", "hstore", "extension", None),),
        ),
        (
            "CREATE DOMAIN tenant.tenant_slug AS text CHECK (VALUE <> '');",
            (("tenant", "tenant_slug", "type", None),),
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
        "schema": "tenant",
        "name": "safe_report",
        "signature": "()",
        "language": "sql",
        "source_body": "SELECT count(*)::integer FROM tenant.events",
        "parsed_sql_body": None,
        "security_definer": True,
        "leakproof": False,
        "volatility": "v",
        "parallel": "u",
        "config": ("search_path=pg_catalog, tenant, pg_temp",),
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
