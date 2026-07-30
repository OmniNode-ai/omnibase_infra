# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Generate and validate explicit application-database ACL matrices.

The matrix is a projection.  Deployment topology owns role/database/schema and
explicit object grants; typed relation inventories and service manifests own the
object set and access evidence.  No table name or PostgreSQL role allowlist is
maintained here.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from omnibase_core.enums.enum_database_grant_object_type import (
    EnumDatabaseGrantObjectType,
)
from omnibase_core.enums.enum_database_privilege import EnumDatabasePrivilege
from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_infra.validation.enums.enum_application_database_acl_authorization_scope import (
    EnumApplicationDatabaseAclAuthorizationScope,
)
from omnibase_infra.validation.enums.enum_application_database_acl_policy_source_kind import (
    EnumApplicationDatabaseAclPolicySourceKind,
)
from omnibase_infra.validation.enums.enum_application_database_acl_render_phase import (
    EnumApplicationDatabaseAclRenderPhase,
)
from omnibase_infra.validation.enums.enum_application_database_object_kind import (
    EnumApplicationDatabaseObjectKind,
)
from omnibase_infra.validation.enums.enum_application_database_principal_inventory_source_kind import (
    EnumApplicationDatabasePrincipalInventorySourceKind,
)
from omnibase_infra.validation.enums.enum_application_inventory_object_kind import (
    EnumApplicationInventoryObjectKind,
)
from omnibase_infra.validation.enums.enum_application_relation_kind import (
    EnumApplicationRelationKind,
)
from omnibase_infra.validation.models.model_application_database_acl_matrix import (
    ModelApplicationDatabaseAclMatrix,
    ModelApplicationDatabaseAclObject,
    ModelApplicationDatabaseAclRow,
    ModelApplicationDatabaseAclSource,
    ModelApplicationDatabaseDefaultAclRow,
)
from omnibase_infra.validation.models.model_application_database_acl_policy import (
    ModelApplicationDatabaseAclPolicy,
)
from omnibase_infra.validation.models.model_application_database_activity_result_evidence import (
    ModelApplicationDatabaseActivityResultEvidence,
)
from omnibase_infra.validation.models.model_application_database_catalog_object_evidence import (
    ModelApplicationDatabaseCatalogObjectEvidence,
)
from omnibase_infra.validation.models.model_application_database_catalog_result_evidence import (
    ModelApplicationDatabaseCatalogResultEvidence,
)
from omnibase_infra.validation.models.model_application_database_connection_policy import (
    ModelApplicationDatabaseConnectionPolicy,
)
from omnibase_infra.validation.models.model_application_database_observed_role_state import (
    ModelApplicationDatabaseObservedRoleState,
)
from omnibase_infra.validation.models.model_application_database_principal_inventory import (
    ModelApplicationDatabasePrincipalInventory,
)
from omnibase_infra.validation.models.model_application_database_role_membership import (
    ModelApplicationDatabaseRoleMembership,
)
from omnibase_infra.validation.models.model_application_database_role_state import (
    ModelApplicationDatabaseRoleState,
)
from omnibase_infra.validation.models.model_application_relation_evidence_inventory import (
    ModelApplicationRelationEvidenceInventory,
)
from omnibase_infra.validation.models.model_migration_ownership_manifest import (
    ModelMigrationOwnershipManifest,
)

PUBLIC_PRINCIPAL = "PUBLIC"
_SQL_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
_READY_STATUSES = frozenset({"complete", "pass", "passed", "verified"})
_OBJECT_TYPES = (
    EnumDatabaseGrantObjectType.TABLE,
    EnumDatabaseGrantObjectType.SEQUENCE,
    EnumDatabaseGrantObjectType.FUNCTION,
    EnumDatabaseGrantObjectType.TYPE,
)
_DEPLOYMENT_CONNECT_DATABASES = frozenset(
    {
        "keycloak",
        "omnibase_infra",
        "omnidash_analytics",
        "omninode_cloud",
        "omniclaude",
        "omniintelligence",
        "omnimemory",
        "umami",
    }
)


@dataclass
class ObjectEvidence:
    obj: ModelApplicationDatabaseAclObject


class ProtocolApplicationDatabaseRoleAttributeState(Protocol):
    """Structural shape shared by observed and governed PostgreSQL role state."""

    @property
    def login(self) -> bool: ...

    @property
    def superuser(self) -> bool: ...

    @property
    def bypass_rls(self) -> bool: ...

    @property
    def create_database(self) -> bool: ...

    @property
    def create_role(self) -> bool: ...

    @property
    def replication(self) -> bool: ...

    @property
    def inherit(self) -> bool: ...


def _status_is_ready(status: str | None) -> bool:
    return status is not None and status.strip().lower() in _READY_STATUSES


def _role_attribute_values(
    state: ProtocolApplicationDatabaseRoleAttributeState,
) -> tuple[bool, ...]:
    """Return the security-relevant PostgreSQL role attributes in stable order."""
    return (
        state.login,
        state.superuser,
        state.bypass_rls,
        state.create_database,
        state.create_role,
        state.replication,
        state.inherit,
    )


def validate_application_database_principal_evidence(
    inventory: ModelApplicationDatabasePrincipalInventory,
    catalog_result: ModelApplicationDatabaseCatalogResultEvidence,
    activity_result: ModelApplicationDatabaseActivityResultEvidence,
) -> tuple[str, ...]:
    """Bind inventory authorization fields to parsed immutable result content."""
    violations: list[str] = []
    for field_name in (
        "database_ref",
        "physical_database",
        "completion_status",
        "catalog_parity_status",
        "catalog_query_sha256",
        "database_owner_role",
    ):
        inventory_field = (
            inventory.catalog_query_sha256
            if field_name == "catalog_query_sha256"
            else getattr(inventory, field_name)
        )
        if getattr(catalog_result, field_name) != inventory_field:
            violations.append(f"catalog result {field_name} disagrees with inventory")
    for field_name in (
        "principal_refs",
        "absent_principal_refs",
        "owner_refs",
        "absent_owner_refs",
        "absent_schema_refs",
    ):
        if set(getattr(catalog_result, field_name)) != set(
            getattr(inventory, field_name)
        ):
            violations.append(f"catalog result {field_name} disagrees with inventory")
    inventory_role_states = {
        state.role: state.model_dump(mode="json")
        for state in inventory.observed_role_states
    }
    result_role_states = {
        state.role: state.model_dump(mode="json")
        for state in catalog_result.observed_role_states
    }
    if result_role_states != inventory_role_states:
        violations.append(
            "catalog result observed_role_states disagrees with inventory"
        )
    if catalog_result.observed_schema_owners != inventory.observed_schema_owners:
        violations.append(
            "catalog result observed_schema_owners disagrees with inventory"
        )
    inventory_objects = {
        obj.identity: obj.model_dump(mode="json") for obj in inventory.observed_objects
    }
    result_objects = {
        obj.identity: obj.model_dump(mode="json")
        for obj in catalog_result.observed_objects
    }
    if result_objects != inventory_objects:
        violations.append("catalog result observed_objects disagrees with inventory")
    activity = inventory.activity_evidence
    if activity is None:
        violations.append("inventory lacks activity evidence")
    else:
        for field_name, expected in (
            ("database_ref", inventory.database_ref),
            ("physical_database", inventory.physical_database),
            ("window_started_at", activity.window_started_at),
            ("window_ended_at", activity.window_ended_at),
            ("activity_query_sha256", activity.query_sha256),
            ("observation_count", activity.observation_count),
        ):
            if getattr(activity_result, field_name) != expected:
                violations.append(
                    f"activity result {field_name} disagrees with inventory"
                )
        result_activity_principals = {
            row.principal for row in activity_result.active_principals
        }
        if result_activity_principals != set(inventory.activity_principal_refs):
            violations.append(
                "activity result active principals disagree with inventory"
            )
    return tuple(sorted(set(violations)))


def _schema_for_domain(
    *,
    topology: ModelDeploymentTopology,
    database_ref: str,
    domain: EnumDatabaseSchemaDomain,
) -> str:
    database = topology.databases[database_ref]
    matches = [
        name for name, schema in database.schemas.items() if schema.domain is domain
    ]
    if len(matches) != 1:
        raise ValueError(
            f"database_ref {database_ref!r} must resolve domain {domain.value!r} "
            f"to exactly one schema, got {sorted(matches)!r}"
        )
    return matches[0]


def _object_type_from_inventory(
    kind: EnumApplicationInventoryObjectKind,
) -> EnumDatabaseGrantObjectType | None:
    if kind in {
        EnumApplicationInventoryObjectKind.TABLE,
        EnumApplicationInventoryObjectKind.VIEW,
        EnumApplicationInventoryObjectKind.MATERIALIZED_VIEW,
    }:
        return EnumDatabaseGrantObjectType.TABLE
    if kind is EnumApplicationInventoryObjectKind.SEQUENCE:
        return EnumDatabaseGrantObjectType.SEQUENCE
    if kind in {
        EnumApplicationInventoryObjectKind.FUNCTION,
        EnumApplicationInventoryObjectKind.PROCEDURE,
    }:
        return EnumDatabaseGrantObjectType.FUNCTION
    if kind is EnumApplicationInventoryObjectKind.TYPE:
        return EnumDatabaseGrantObjectType.TYPE
    return None


def _catalog_kind_from_inventory(
    kind: EnumApplicationInventoryObjectKind,
) -> str:
    return kind.value


def _object_type_from_service(
    kind: EnumApplicationDatabaseObjectKind,
) -> EnumDatabaseGrantObjectType | None:
    if kind is EnumApplicationDatabaseObjectKind.SEQUENCE:
        return EnumDatabaseGrantObjectType.SEQUENCE
    if kind in {
        EnumApplicationDatabaseObjectKind.FUNCTION,
        EnumApplicationDatabaseObjectKind.PROCEDURE,
    }:
        return EnumDatabaseGrantObjectType.FUNCTION
    if kind is EnumApplicationDatabaseObjectKind.TYPE:
        return EnumDatabaseGrantObjectType.TYPE
    return None


def _source_status_blockers(
    source_id: str,
    inventory: ModelApplicationRelationEvidenceInventory,
) -> list[str]:
    blockers: list[str] = []
    if inventory.relation_counts.type is None:
        blockers.append(
            f"{source_id}: relation_counts.type is not inventoried; "
            "zero user-defined types cannot be inferred"
        )
    if inventory.relation_counts.procedure is None:
        blockers.append(
            f"{source_id}: relation_counts.procedure is not inventoried; "
            "zero procedures cannot be inferred"
        )
    if not _status_is_ready(inventory.completion_status):
        blockers.append(
            f"{source_id}: completion_status={inventory.completion_status!r}"
        )
    census = inventory.retained_live_census
    typed_kind_counts = Counter(relation.kind.value for relation in inventory.relations)
    expected_census_counts = {
        "observed_base_tables": typed_kind_counts.get("table", 0),
        "observed_views_and_materialized_views": (
            typed_kind_counts.get("view", 0)
            + typed_kind_counts.get("materialized_view", 0)
        ),
        "observed_sequences": typed_kind_counts.get("sequence", 0),
        "observed_functions": typed_kind_counts.get("function", 0),
        "observed_procedures": typed_kind_counts.get("procedure", 0),
        "observed_types": typed_kind_counts.get("type", 0),
        "observed_extensions": typed_kind_counts.get("extension", 0),
    }
    for field_name in (
        "observed_base_tables",
        "observed_views_and_materialized_views",
        "observed_sequences",
        "observed_functions",
        "observed_procedures",
        "observed_types",
        "observed_extensions",
    ):
        live_count = getattr(census, field_name)
        if live_count is None:
            blockers.append(
                f"{source_id}: retained_live_census.{field_name} is not inventoried"
            )
        elif live_count != expected_census_counts[field_name]:
            blockers.append(
                f"{source_id}: retained_live_census.{field_name}={live_count} "
                "does not match exact typed rows="
                f"{expected_census_counts[field_name]}"
            )
    if not _status_is_ready(census.parity_status):
        blockers.append(
            f"{source_id}: retained_live_census={census.parity_status!r}: "
            f"{census.reason or 'no reason supplied'}"
        )
    for evidence_name in (
        "full_day_datname_usename_activity",
        "live_catalog_parity",
    ):
        status = getattr(inventory.runtime_evidence, evidence_name)
        if not _status_is_ready(status.status):
            blockers.append(
                f"{source_id}: runtime_evidence.{evidence_name}="
                f"{status.status.value!r}: "
                f"{status.reason or 'no reason supplied'}"
            )
    blockers.extend(
        f"{source_id}: blocked {blocked.kind} {blocked.name!r}: {blocked.reason}"
        for blocked in inventory.blocked_relations
    )
    return blockers


def _service_status_blockers(
    source_id: str,
    manifest: ModelMigrationOwnershipManifest,
) -> list[str]:
    blockers: list[str] = []
    if not manifest.completion_status:
        blockers.append(f"{source_id}: completion_status is missing")
    elif not _status_is_ready(manifest.completion_status):
        blockers.append(
            f"{source_id}: completion_status={manifest.completion_status!r}"
        )
    if manifest.retained_live_census is None:
        blockers.append(f"{source_id}: retained_live_census is missing")
    else:
        census = manifest.retained_live_census
        relation_counts = Counter(
            relation.kind.value for relation in manifest.relation_evidence
        )
        object_counts = Counter(
            database_object.kind.value for database_object in manifest.database_objects
        )
        expected_counts = {
            "observed_base_tables": relation_counts.get("table", 0),
            "observed_views_and_materialized_views": (
                relation_counts.get("view", 0)
                + relation_counts.get("materialized_view", 0)
            ),
            "observed_sequences": object_counts.get("sequence", 0),
            "observed_functions": object_counts.get("function", 0),
            "observed_procedures": object_counts.get("procedure", 0),
            "observed_types": object_counts.get("type", 0),
            "observed_extensions": object_counts.get("extension", 0),
        }
        for field_name, typed_count in expected_counts.items():
            live_count = getattr(census, field_name)
            if live_count is None:
                blockers.append(
                    f"{source_id}: retained_live_census.{field_name} is missing"
                )
            elif live_count != typed_count:
                blockers.append(
                    f"{source_id}: retained_live_census.{field_name}={live_count} "
                    f"does not match exact typed rows={typed_count}"
                )
        census_status = census.parity_status
        if not _status_is_ready(census_status):
            blockers.append(
                f"{source_id}: retained_live_census={census_status!r}: "
                f"{census.reason or 'no reason supplied'}"
            )
    required_runtime_evidence = {
        "full_day_datname_usename_activity",
        "live_catalog_parity",
    }
    missing_runtime_evidence = sorted(
        required_runtime_evidence - manifest.runtime_evidence.keys()
    )
    if missing_runtime_evidence:
        blockers.append(
            f"{source_id}: required runtime_evidence is missing "
            f"{missing_runtime_evidence!r}"
        )
    for evidence_name, runtime_status in sorted(manifest.runtime_evidence.items()):
        if not _status_is_ready(runtime_status.status):
            blockers.append(
                f"{source_id}: runtime_evidence.{evidence_name}="
                f"{runtime_status.status.value!r}: "
                f"{runtime_status.reason or 'no reason supplied'}"
            )
    blockers.extend(
        f"{source_id}: blocked {blocked.kind} {blocked.name!r}: {blocked.reason}"
        for blocked in manifest.blocked_relations
    )
    return blockers


def _merge_object(
    objects: dict[
        tuple[str, str, EnumDatabaseGrantObjectType, str, str | None],
        ObjectEvidence,
    ],
    evidence: ObjectEvidence,
    blockers: list[str],
) -> None:
    identity = evidence.obj.identity
    existing = objects.get(identity)
    if existing is None:
        objects[identity] = evidence
        return
    blockers.append(
        f"duplicate authoritative ownership declaration for {identity!r}: "
        f"{existing.obj.owner_declaration!r} from {existing.obj.source_keys!r} vs "
        f"{evidence.obj.owner_declaration!r} from {evidence.obj.source_keys!r}"
    )


def _append_inventory_objects(
    *,
    topology: ModelDeploymentTopology,
    source_id: str,
    inventory: ModelApplicationRelationEvidenceInventory,
    objects: dict[
        tuple[str, str, EnumDatabaseGrantObjectType, str, str | None],
        ObjectEvidence,
    ],
    blockers: list[str],
    excluded: list[str],
) -> None:
    database = topology.databases.get(inventory.database_ref)
    if database is None:
        blockers.append(f"{source_id}: unknown database_ref {inventory.database_ref!r}")
        return
    if database.physical_name != inventory.physical_seed_database:
        blockers.append(
            f"{source_id}: physical database drift: topology="
            f"{database.physical_name!r}, inventory="
            f"{inventory.physical_seed_database!r}"
        )
    for relation in inventory.relations:
        object_type = _object_type_from_inventory(relation.kind)
        if object_type is None:
            excluded.append(
                f"{source_id}:{relation.target_schema}.{relation.name}:"
                f"{relation.kind.value}"
            )
            continue
        if relation.target_schema not in database.schemas:
            blockers.append(
                f"{source_id}: object {relation.name!r} uses unknown schema "
                f"{relation.target_schema!r}"
            )
            continue
        domain = database.schemas[relation.target_schema].domain
        if relation.domain is None or relation.domain is not domain:
            blockers.append(
                f"{source_id}: object {relation.name!r} domain does not match "
                f"topology schema {relation.target_schema!r}"
            )
        if relation.classification_status.strip().lower() != "classified":
            blockers.append(
                f"{source_id}: object {relation.name!r} classification_status="
                f"{relation.classification_status!r}"
            )
        if relation.target_schema not in relation.current_schema:
            blockers.append(
                f"{source_id}: object {relation.name!r} is currently in schemas "
                f"{sorted(relation.current_schema)!r}, not the target schema "
                f"{relation.target_schema!r}; full object ACL rendering is gated "
                "until the additive target object is materialized and inventoried"
            )
        for reason in relation.blocked_reasons:
            blockers.append(f"{source_id}: object {relation.name!r} blocked: {reason}")
        if relation.owner_declaration is None:
            blockers.append(
                f"{source_id}: object {relation.name!r} lacks an exact "
                "owner_declaration"
            )
            continue
        owner = database.schemas[relation.target_schema].owner
        _merge_object(
            objects,
            ObjectEvidence(
                obj=ModelApplicationDatabaseAclObject(
                    database_ref=inventory.database_ref,
                    physical_database=database.physical_name,
                    schema_ref=relation.target_schema,
                    domain=domain,
                    object_type=object_type,
                    object_ref=relation.name,
                    catalog_kind=_catalog_kind_from_inventory(relation.kind),
                    owner=owner,
                    owner_declaration=relation.owner_declaration,
                    target_materialized=(
                        inventory.physical_seed_database == database.physical_name
                        and relation.target_schema in relation.current_schema
                    ),
                    function_signature=relation.function_signature,
                    source_keys=(source_id,),
                ),
            ),
            blockers,
        )


def _append_service_objects(
    *,
    topology: ModelDeploymentTopology,
    source_id: str,
    manifest: ModelMigrationOwnershipManifest,
    objects: dict[
        tuple[str, str, EnumDatabaseGrantObjectType, str, str | None],
        ObjectEvidence,
    ],
    blockers: list[str],
    excluded: list[str],
) -> None:
    database = topology.databases.get(manifest.target_database_ref)
    if database is None:
        blockers.append(
            f"{source_id}: unknown database_ref {manifest.target_database_ref!r}"
        )
        return
    if manifest.current_physical_database is None:
        blockers.append(f"{source_id}: current_physical_database is not inventoried")
    if database.physical_name not in manifest.materialized_physical_databases:
        blockers.append(
            f"{source_id}: target physical database {database.physical_name!r} is "
            "absent from materialized_physical_databases; full object ACL rendering "
            "is gated until the additive target is inventoried"
        )
    tables_by_name: dict[str, list[ModelDbTableDeclaration]] = defaultdict(list)
    for table in manifest.db_io.db_tables:
        tables_by_name[table.name].append(table)
    covered_table_identities: set[tuple[str, str, str]] = set()
    for relation in manifest.relation_evidence:
        if relation.kind is EnumApplicationRelationKind.FUNCTION:
            blockers.append(
                f"{source_id}: function relation_evidence {relation.name!r} requires "
                "exact database_objects evidence including a routine signature"
            )
            continue
        schema_name: str | None
        database_ref: str
        if relation.kind is EnumApplicationRelationKind.TABLE:
            matching_tables = tables_by_name.get(relation.name, [])
            if relation.database_ref is not None:
                matching_tables = [
                    table
                    for table in matching_tables
                    if table.database_ref == relation.database_ref
                ]
            if relation.schema is not None:
                matching_tables = [
                    table
                    for table in matching_tables
                    if table.schema == relation.schema
                ]
            if len(matching_tables) != 1:
                blockers.append(
                    f"{source_id}: table relation_evidence {relation.name!r} must "
                    "match exactly one db_io.db_tables declaration, got "
                    f"{[(table.database_ref, table.schema, table.name) for table in matching_tables]!r}"
                )
                continue
            table = matching_tables[0]
            schema_name = relation.schema or table.schema
            database_ref = relation.database_ref or table.database_ref
            covered_table_identities.add((table.database_ref, table.schema, table.name))
        else:
            schema_name = relation.schema
            database_ref = relation.database_ref or manifest.target_database_ref
        if schema_name is None:
            try:
                schema_name = _schema_for_domain(
                    topology=topology,
                    database_ref=database_ref,
                    domain=relation.domain,
                )
            except (KeyError, ValueError) as exc:
                blockers.append(f"{source_id}: {relation.name!r}: {exc}")
                continue
        if database_ref != manifest.target_database_ref:
            blockers.append(
                f"{source_id}: object {relation.name!r} database_ref "
                f"{database_ref!r} conflicts with manifest target"
            )
            continue
        if schema_name not in database.schemas:
            blockers.append(
                f"{source_id}: object {relation.name!r} uses unknown schema "
                f"{schema_name!r}"
            )
            continue
        if schema_name not in relation.current_schemas:
            blockers.append(
                f"{source_id}: object {relation.name!r} current_schemas="
                f"{relation.current_schemas!r} do not prove target schema "
                f"{schema_name!r}; full object ACL rendering is gated"
            )
        topology_domain = database.schemas[schema_name].domain
        if relation.domain is not topology_domain:
            blockers.append(
                f"{source_id}: object {relation.name!r} domain does not match "
                f"topology schema {schema_name!r}"
            )
        _merge_object(
            objects,
            ObjectEvidence(
                obj=ModelApplicationDatabaseAclObject(
                    database_ref=database_ref,
                    physical_database=database.physical_name,
                    schema_ref=schema_name,
                    domain=topology_domain,
                    object_type=EnumDatabaseGrantObjectType.TABLE,
                    object_ref=relation.name,
                    catalog_kind=relation.kind.value,
                    owner=database.schemas[schema_name].owner,
                    owner_declaration=relation.owner_declaration,
                    target_materialized=(
                        database.physical_name
                        in manifest.materialized_physical_databases
                        and schema_name in relation.current_schemas
                    ),
                    source_keys=(source_id,),
                ),
            ),
            blockers,
        )

    for table in manifest.db_io.db_tables:
        table_identity = (table.database_ref, table.schema, table.name)
        if table_identity not in covered_table_identities:
            blockers.append(
                f"{source_id}: db_io table {table.database_ref}."
                f"{table.schema}.{table.name} lacks exact relation_evidence"
            )

    for database_object in manifest.database_objects:
        object_type = _object_type_from_service(database_object.kind)
        if object_type is None:
            excluded.append(
                f"{source_id}:{database_object.name}:{database_object.kind.value}"
            )
            continue
        database_ref = database_object.database_ref or manifest.target_database_ref
        try:
            schema_name = database_object.schema or _schema_for_domain(
                topology=topology,
                database_ref=database_ref,
                domain=database_object.domain,
            )
        except (KeyError, ValueError) as exc:
            blockers.append(f"{source_id}: {database_object.name!r}: {exc}")
            continue
        if database_ref != manifest.target_database_ref:
            blockers.append(
                f"{source_id}: object {database_object.name!r} database_ref "
                f"{database_ref!r} conflicts with manifest target"
            )
            continue
        topology_domain = database.schemas[schema_name].domain
        if database_object.domain is not topology_domain:
            blockers.append(
                f"{source_id}: object {database_object.name!r} domain does not "
                f"match topology schema {schema_name!r}"
            )
        if schema_name not in database_object.current_schemas:
            blockers.append(
                f"{source_id}: object {database_object.name!r} current_schemas="
                f"{database_object.current_schemas!r} do not prove target schema "
                f"{schema_name!r}; full object ACL rendering is gated"
            )
        _merge_object(
            objects,
            ObjectEvidence(
                obj=ModelApplicationDatabaseAclObject(
                    database_ref=database_ref,
                    physical_database=database.physical_name,
                    schema_ref=schema_name,
                    domain=topology_domain,
                    object_type=object_type,
                    object_ref=database_object.name,
                    catalog_kind=database_object.kind.value,
                    owner=database.schemas[schema_name].owner,
                    owner_declaration=database_object.owner_declaration,
                    target_materialized=(
                        database.physical_name
                        in manifest.materialized_physical_databases
                        and schema_name in database_object.current_schemas
                    ),
                    function_signature=database_object.function_signature,
                    source_keys=(source_id,),
                ),
            ),
            blockers,
        )


def _topology_privileges(
    topology: ModelDeploymentTopology,
) -> dict[
    tuple[
        str,
        str,
        EnumDatabaseGrantObjectType,
        str | None,
        str | None,
    ],
    set[EnumDatabasePrivilege],
]:
    result: dict[
        tuple[
            str,
            str,
            EnumDatabaseGrantObjectType,
            str | None,
            str | None,
        ],
        set[EnumDatabasePrivilege],
    ] = defaultdict(set)
    for database_ref, database in topology.databases.items():
        for principal_name, principal in database.principals.items():
            for grant in principal.grants:
                if grant.object_type in {
                    EnumDatabaseGrantObjectType.DATABASE,
                    EnumDatabaseGrantObjectType.SCHEMA,
                }:
                    scope_key: tuple[
                        str,
                        str,
                        EnumDatabaseGrantObjectType,
                        str | None,
                        str | None,
                    ] = (
                        principal_name,
                        database_ref,
                        grant.object_type,
                        grant.schema,
                        None,
                    )
                    result[scope_key].update(grant.privileges)
                    continue
                for object_name in grant.objects:
                    object_key: tuple[
                        str,
                        str,
                        EnumDatabaseGrantObjectType,
                        str | None,
                        str | None,
                    ] = (
                        principal_name,
                        database_ref,
                        grant.object_type,
                        grant.schema,
                        object_name,
                    )
                    result[object_key].update(grant.privileges)
    return result


def build_application_database_acl_matrix(
    *,
    topology: ModelDeploymentTopology,
    sources: Sequence[ModelApplicationDatabaseAclSource],
    relation_inventories: Mapping[str, ModelApplicationRelationEvidenceInventory],
    service_manifests: Mapping[str, ModelMigrationOwnershipManifest],
    principal_inventories: Mapping[str, ModelApplicationDatabasePrincipalInventory],
    acl_policies: Mapping[str, ModelApplicationDatabaseAclPolicy],
    authorization_scope: EnumApplicationDatabaseAclAuthorizationScope,
    required_connect_databases: Sequence[str] | None = None,
    catalog_results: Mapping[
        str,
        ModelApplicationDatabaseCatalogResultEvidence,
    ]
    | None = None,
    activity_results: Mapping[
        str,
        ModelApplicationDatabaseActivityResultEvidence,
    ]
    | None = None,
) -> ModelApplicationDatabaseAclMatrix:
    """Build a complete cell matrix without inventing missing grants."""
    source_ids = {source.source_key for source in sources}
    if len(source_ids) != len(sources):
        raise ValueError("Matrix source keys must be unique")
    source_by_key = {source.source_key: source for source in sources}
    parsed_catalog_results = dict(catalog_results or {})
    parsed_activity_results = dict(activity_results or {})
    unknown_result_sources = sorted(
        set(parsed_catalog_results).union(parsed_activity_results) - source_ids
    )
    if unknown_result_sources:
        raise ValueError(
            "Parsed evidence keys are absent from source records: "
            f"{unknown_result_sources!r}"
        )
    invalid_catalog_sources = sorted(
        source_key
        for source_key in parsed_catalog_results
        if source_by_key[source_key].purpose != "catalog_result_evidence"
    )
    invalid_activity_sources = sorted(
        source_key
        for source_key in parsed_activity_results
        if source_by_key[source_key].purpose != "activity_result_evidence"
    )
    if invalid_catalog_sources or invalid_activity_sources:
        raise ValueError(
            "Parsed evidence source purposes disagree: "
            f"catalog={invalid_catalog_sources!r} activity={invalid_activity_sources!r}"
        )
    verified_evidence: set[str] = set()
    missing_source_ids = sorted(
        (
            set(relation_inventories)
            | set(service_manifests)
            | set(principal_inventories)
            | set(acl_policies)
        )
        - source_ids
    )
    if missing_source_ids:
        raise ValueError(f"Matrix inputs lack source records: {missing_source_ids!r}")

    blockers: list[str] = []
    object_blockers: list[str] = []
    if required_connect_databases is None:
        if (
            authorization_scope
            is EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT
        ):
            blockers.append(
                "deployment authorization requires the explicit eight-database "
                "CONNECT inventory"
            )
            required_connect_values: Sequence[str] = ()
        else:
            required_connect_values = tuple(
                database.physical_name for database in topology.databases.values()
            )
    else:
        required_connect_values = required_connect_databases
    required_connect = tuple(sorted(required_connect_values))
    if (
        authorization_scope is EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT
        and set(required_connect) != _DEPLOYMENT_CONNECT_DATABASES
    ):
        blockers.append(
            "deployment CONNECT database universe disagrees with the approved "
            f"eight-database scope: missing="
            f"{sorted(_DEPLOYMENT_CONNECT_DATABASES - set(required_connect))!r} "
            f"extra={sorted(set(required_connect) - _DEPLOYMENT_CONNECT_DATABASES)!r}"
        )
    if len(set(required_connect)) != len(required_connect):
        blockers.append("required CONNECT database names must be unique")
    topology_sources = {
        source.source_key for source in sources if source.purpose == "topology"
    }
    if len(topology_sources) != 1:
        blockers.append(
            "matrix requires exactly one topology source, got "
            f"{sorted(topology_sources)!r}"
        )
    declared_relation_sources = {
        source.source_key
        for source in sources
        if source.purpose == "relation_inventory"
    }
    declared_service_sources = {
        source.source_key for source in sources if source.purpose == "service_ownership"
    }
    declared_principal_sources = {
        source.source_key
        for source in sources
        if source.purpose == "principal_inventory"
    }
    declared_policy_sources = {
        source.source_key for source in sources if source.purpose == "acl_policy"
    }
    declared_ownership_sources = declared_relation_sources.union(
        declared_service_sources
    )
    if not declared_ownership_sources:
        object_blockers.append(
            "matrix requires at least one typed ownership evidence source"
        )
    missing_relation_inputs = sorted(
        declared_relation_sources - relation_inventories.keys()
    )
    if missing_relation_inputs:
        object_blockers.append(
            "declared relation_inventory sources were not parsed: "
            f"{missing_relation_inputs!r}"
        )
    missing_service_inputs = sorted(declared_service_sources - service_manifests.keys())
    if missing_service_inputs:
        object_blockers.append(
            "declared service_ownership sources were not parsed: "
            f"{missing_service_inputs!r}"
        )
    if not declared_principal_sources:
        blockers.append(
            "matrix requires a typed principal_inventory source for every database"
        )
    missing_principal_inputs = sorted(
        declared_principal_sources - principal_inventories.keys()
    )
    if missing_principal_inputs:
        blockers.append(
            "declared principal_inventory sources were not parsed: "
            f"{missing_principal_inputs!r}"
        )
    if not declared_policy_sources:
        blockers.append(
            "matrix requires an independent typed acl_policy source for every database"
        )
    missing_policy_inputs = sorted(declared_policy_sources - acl_policies.keys())
    if missing_policy_inputs:
        blockers.append(
            f"declared acl_policy sources were not parsed: {missing_policy_inputs!r}"
        )
    excluded: list[str] = []
    objects: dict[
        tuple[str, str, EnumDatabaseGrantObjectType, str, str | None],
        ObjectEvidence,
    ] = {}
    for source_id, inventory in sorted(relation_inventories.items()):
        object_blockers.extend(_source_status_blockers(source_id, inventory))
        _append_inventory_objects(
            topology=topology,
            source_id=source_id,
            inventory=inventory,
            objects=objects,
            blockers=object_blockers,
            excluded=excluded,
        )
    for source_id, manifest in sorted(service_manifests.items()):
        object_blockers.extend(_service_status_blockers(source_id, manifest))
        _append_service_objects(
            topology=topology,
            source_id=source_id,
            manifest=manifest,
            objects=objects,
            blockers=object_blockers,
            excluded=excluded,
        )
    topology_privileges = _topology_privileges(topology)
    object_identities_by_topology_target: dict[
        tuple[str, str, EnumDatabaseGrantObjectType, str],
        list[tuple[str, str, EnumDatabaseGrantObjectType, str, str | None]],
    ] = defaultdict(list)
    for identity in objects:
        object_identities_by_topology_target[identity[:4]].append(identity)
    resolved_object_privileges: dict[
        tuple[
            str,
            str,
            EnumDatabaseGrantObjectType,
            str,
            str,
            str | None,
        ],
        set[EnumDatabasePrivilege],
    ] = defaultdict(set)
    explicit_object_grants = {
        key: privileges
        for key, privileges in topology_privileges.items()
        if key[2] in _OBJECT_TYPES and privileges
    }
    if objects and not explicit_object_grants:
        object_blockers.append(
            "topology declares zero explicit object grants for "
            f"{len(objects)} typed ownership objects"
        )
    for key, privileges in sorted(
        topology_privileges.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        principal, database_ref, object_type, schema_name, object_name = key
        if object_type not in _OBJECT_TYPES:
            continue
        topology_target = (
            database_ref,
            schema_name or "",
            object_type,
            object_name or "",
        )
        matching_identities = object_identities_by_topology_target.get(
            topology_target, []
        )
        if not matching_identities:
            object_blockers.append(
                f"topology grant {principal}:{object_type.value}:"
                f"{schema_name}.{object_name} targets no typed ownership object "
                f"({sorted(privilege.value for privilege in privileges)!r})"
            )
        elif len(matching_identities) > 1:
            signatures = sorted(
                identity[4] or "<missing>" for identity in matching_identities
            )
            object_blockers.append(
                f"topology grant {principal}:{object_type.value}:"
                f"{schema_name}.{object_name} is ambiguous across exact object "
                f"identities {signatures!r}; overload-specific targeting is required"
            )
        else:
            exact_identity = matching_identities[0]
            resolved_object_privileges[
                (
                    principal,
                    exact_identity[0],
                    exact_identity[2],
                    exact_identity[1],
                    exact_identity[3],
                    exact_identity[4],
                )
            ].update(privileges)

    for evidence in objects.values():
        obj = evidence.obj
        if (
            obj.object_type is EnumDatabaseGrantObjectType.FUNCTION
            and not obj.function_signature
        ):
            object_blockers.append(
                f"{obj.schema_ref}.{obj.object_ref}: ownership and ACL rendering "
                "require an explicit function_signature to avoid overload widening"
            )

    if not objects:
        object_blockers.append(
            "typed ownership evidence resolved to zero database objects"
        )
    object_databases = {identity[0] for identity in objects}
    for database_ref in sorted(set(topology.databases) - object_databases):
        object_blockers.append(
            f"{database_ref}: no typed ownership objects cover this topology database"
        )

    principal_inputs_by_database: dict[
        str,
        list[tuple[str, ModelApplicationDatabasePrincipalInventory]],
    ] = defaultdict(list)
    observed_owner_roles = tuple(
        sorted(
            {
                owner
                for inventory in principal_inventories.values()
                for owner in inventory.owner_refs
            }
        )
    )
    absent_owner_roles = tuple(
        sorted(
            {
                owner
                for inventory in principal_inventories.values()
                for owner in inventory.absent_owner_refs
            }
        )
    )
    observed_role_state_by_name: dict[
        str,
        ModelApplicationDatabaseObservedRoleState,
    ] = {}
    for source_id, principal_inventory in sorted(principal_inventories.items()):
        for observed_state in principal_inventory.observed_role_states:
            existing_state = observed_role_state_by_name.get(observed_state.role)
            if existing_state is not None and existing_state != observed_state:
                blockers.append(
                    f"{source_id}: observed role attributes for "
                    f"{observed_state.role!r} conflict across cluster inventories"
                )
            observed_role_state_by_name[observed_state.role] = observed_state
        if not _status_is_ready(principal_inventory.completion_status):
            blockers.append(
                f"{source_id}: completion_status="
                f"{principal_inventory.completion_status!r}"
            )
        if not _status_is_ready(principal_inventory.catalog_parity_status):
            blockers.append(
                f"{source_id}: catalog_parity_status="
                f"{principal_inventory.catalog_parity_status!r}: "
                f"{principal_inventory.reason}"
            )
        if (
            authorization_scope
            is EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT
        ):
            if principal_inventory.source_kind != (
                EnumApplicationDatabasePrincipalInventorySourceKind.AUTHORIZED_CATALOG
            ):
                blockers.append(
                    f"{source_id}: deployment authorization requires "
                    "source_kind='authorized_catalog'"
                )
            if (
                principal_inventory.activity_evidence is None
                or principal_inventory.catalog_query_sha256 is None
                or principal_inventory.catalog_result_sha256 is None
                or principal_inventory.catalog_query_source_key is None
                or principal_inventory.catalog_result_source_key is None
            ):
                blockers.append(
                    f"{source_id}: deployment authorization lacks durable full-day "
                    "activity and catalog query/result provenance"
                )
            else:
                catalog_result_key = principal_inventory.catalog_result_source_key
                activity_result_key = (
                    principal_inventory.activity_evidence.result_source_key
                )
                catalog_result = parsed_catalog_results.get(catalog_result_key)
                activity_result = parsed_activity_results.get(activity_result_key)
                if catalog_result is None:
                    blockers.append(
                        f"{source_id}: catalog result {catalog_result_key!r} was "
                        "not supplied as parsed typed content"
                    )
                if activity_result is None:
                    blockers.append(
                        f"{source_id}: activity result {activity_result_key!r} was "
                        "not supplied as parsed typed content"
                    )
                if catalog_result is not None and activity_result is not None:
                    semantic_violations = (
                        validate_application_database_principal_evidence(
                            principal_inventory,
                            catalog_result,
                            activity_result,
                        )
                    )
                    if semantic_violations:
                        blockers.extend(
                            f"{source_id}: typed evidence mismatch: {violation}"
                            for violation in semantic_violations
                        )
                    else:
                        verified_evidence.update(
                            {catalog_result_key, activity_result_key}
                        )
                evidence_bindings = (
                    (
                        principal_inventory.catalog_query_source_key,
                        principal_inventory.catalog_query_sha256,
                        "catalog_query_evidence",
                    ),
                    (
                        principal_inventory.catalog_result_source_key,
                        principal_inventory.catalog_result_sha256,
                        "catalog_result_evidence",
                    ),
                    (
                        principal_inventory.activity_evidence.query_source_key,
                        principal_inventory.activity_evidence.query_sha256,
                        "activity_query_evidence",
                    ),
                    (
                        principal_inventory.activity_evidence.result_source_key,
                        principal_inventory.activity_evidence.result_sha256,
                        "activity_result_evidence",
                    ),
                )
                for (
                    evidence_key,
                    evidence_digest,
                    evidence_purpose,
                ) in evidence_bindings:
                    evidence_source = source_by_key.get(evidence_key)
                    if evidence_source is None:
                        blockers.append(
                            f"{source_id}: evidence source {evidence_key!r} is absent "
                            "from the immutable source lock"
                        )
                    elif evidence_source.purpose != evidence_purpose:
                        blockers.append(
                            f"{source_id}: evidence source {evidence_key!r} has purpose "
                            f"{evidence_source.purpose!r}, expected {evidence_purpose!r}"
                        )
                    elif evidence_source.sha256 != evidence_digest:
                        blockers.append(
                            f"{source_id}: evidence digest {evidence_digest!r} does not "
                            f"match locked source {evidence_key!r} "
                            f"({evidence_source.sha256!r})"
                        )
                    elif (
                        evidence_purpose
                        in {
                            "catalog_result_evidence",
                            "activity_result_evidence",
                        }
                        and evidence_key not in verified_evidence
                    ):
                        blockers.append(
                            f"{source_id}: evidence source {evidence_key!r} was "
                            "hash-bound but its typed result content was not "
                            "semantically verified"
                        )
        elif principal_inventory.source_kind != (
            EnumApplicationDatabasePrincipalInventorySourceKind.SYNTHETIC_FIXTURE
        ):
            blockers.append(
                f"{source_id}: synthetic proof requires source_kind='synthetic_fixture'"
            )
        database = topology.databases.get(principal_inventory.database_ref)
        if (
            database is not None
            and database.physical_name != principal_inventory.physical_database
        ):
            blockers.append(
                f"{source_id}: physical database drift: topology="
                f"{database.physical_name!r} inventory="
                f"{principal_inventory.physical_database!r}"
            )
        principal_inputs_by_database[principal_inventory.database_ref].append(
            (source_id, principal_inventory)
        )

    policy_inputs_by_database: dict[
        str,
        list[tuple[str, ModelApplicationDatabaseAclPolicy]],
    ] = defaultdict(list)
    for source_id, policy in sorted(acl_policies.items()):
        if not _status_is_ready(policy.completion_status):
            blockers.append(
                f"{source_id}: completion_status={policy.completion_status!r}"
            )
        required_policy_kind = (
            EnumApplicationDatabaseAclPolicySourceKind.TOPOLOGY_CONTRACT
            if authorization_scope
            is EnumApplicationDatabaseAclAuthorizationScope.DEPLOYMENT
            else EnumApplicationDatabaseAclPolicySourceKind.SYNTHETIC_FIXTURE
        )
        if policy.source_kind is not required_policy_kind:
            blockers.append(
                f"{source_id}: {authorization_scope.value} authorization requires "
                f"ACL policy source_kind={required_policy_kind.value!r}"
            )
        database = topology.databases.get(policy.database_ref)
        if database is None:
            blockers.append(
                f"{source_id}: unknown database_ref {policy.database_ref!r}"
            )
            continue
        if database.physical_name != policy.physical_database:
            blockers.append(
                f"{source_id}: physical database drift: topology="
                f"{database.physical_name!r} policy={policy.physical_database!r}"
            )
        policy_inputs_by_database[policy.database_ref].append((source_id, policy))

    connection_policy_inputs: dict[
        str,
        list[tuple[str, ModelApplicationDatabaseConnectionPolicy]],
    ] = defaultdict(list)
    policy_governed_roles = {
        state.role
        for policy in acl_policies.values()
        for state in policy.governed_role_states
    }
    noncreatable_policy_roles = {
        state.role
        for policy in acl_policies.values()
        for state in policy.governed_role_states
        if not state.manage_attributes
    }
    for source_id, policy in sorted(acl_policies.items()):
        for connection_policy in policy.connection_policies:
            connection_policy_inputs[connection_policy.physical_database].append(
                (source_id, connection_policy)
            )

    allowed_connect_principals: dict[str, tuple[str, ...]] = {}
    observed_connect_principals: dict[str, tuple[str, ...]] = {}
    absent_connect_principals: dict[str, tuple[str, ...]] = {}
    observed_connect_database_owners: dict[str, str] = {}
    connection_database_refs: set[str] = set()
    for physical_database in required_connect:
        policies = connection_policy_inputs.get(physical_database, [])
        if len(policies) != 1:
            blockers.append(
                f"{physical_database}: requires exactly one CONNECT policy, got "
                f"{[source_id for source_id, _ in policies]!r}"
            )
            continue
        source_id, connection_policy = policies[0]
        database_ref = connection_policy.database_ref
        connection_database_refs.add(database_ref)
        inventory_inputs = principal_inputs_by_database.get(database_ref, [])
        if len(inventory_inputs) != 1:
            blockers.append(
                f"{physical_database}: requires exactly one principal inventory "
                f"for CONNECT policy {database_ref!r}, got "
                f"{[item[0] for item in inventory_inputs]!r}"
            )
            continue
        connection_inventory = inventory_inputs[0][1]
        if connection_inventory.physical_database != physical_database:
            blockers.append(
                f"{physical_database}: CONNECT inventory physical database is "
                f"{connection_inventory.physical_database!r}"
            )
            continue
        allowed = tuple(sorted(connection_policy.allowed_principals))
        connection_observed = tuple(sorted(connection_inventory.principal_refs))
        connection_absent = tuple(sorted(connection_inventory.absent_principal_refs))
        missing_allowed = sorted(
            set(allowed) - set(connection_observed) - set(connection_absent)
        )
        if missing_allowed:
            blockers.append(
                f"{physical_database}: CONNECT inventory lacks presence/absence "
                "evidence for allowed principals "
                f"{missing_allowed!r}"
            )
        unmanaged_absent = sorted(
            set(allowed)
            .intersection(connection_absent)
            .intersection(noncreatable_policy_roles)
        )
        if unmanaged_absent:
            blockers.append(
                f"{physical_database}: CONNECT-only policy cannot create absent "
                "external principals without governed role-state authorization: "
                f"{unmanaged_absent!r}"
            )
        parent_policy = acl_policies[source_id]
        classified_activity_principals = (
            set(allowed)
            .union(parent_policy.retained_administrative_principals)
            .union({parent_policy.migration_principal})
            .union({connection_inventory.database_owner_role})
            .union(connection_inventory.owner_refs)
        )
        unclassified_activity = sorted(
            set(connection_inventory.activity_principal_refs)
            - classified_activity_principals
        )
        if unclassified_activity:
            object_blockers.append(
                f"{physical_database}: full-day activity contains principals not "
                "classified as allowed, migration, administrative, or owner: "
                f"{unclassified_activity!r}"
            )
        if (
            connection_policy.observed_database_owner_role
            != connection_inventory.database_owner_role
        ):
            blockers.append(
                f"{physical_database}: CONNECT policy observed database owner "
                f"{connection_policy.observed_database_owner_role!r} disagrees "
                f"with catalog inventory {connection_inventory.database_owner_role!r}"
            )
        topology_database = topology.databases.get(database_ref)
        if topology_database is not None:
            topology_allowed = {
                principal
                for principal, principal_contract in topology_database.principals.items()
                if any(
                    grant.object_type is EnumDatabaseGrantObjectType.DATABASE
                    and EnumDatabasePrivilege.CONNECT in grant.privileges
                    for grant in principal_contract.grants
                )
            }
            if set(allowed) != topology_allowed:
                blockers.append(
                    f"{physical_database}: CONNECT policy disagrees with topology: "
                    f"missing={sorted(topology_allowed - set(allowed))!r} "
                    f"extra={sorted(set(allowed) - topology_allowed)!r}"
                )
        observed_connect_database_owners[physical_database] = (
            connection_policy.observed_database_owner_role
        )
        allowed_connect_principals[physical_database] = allowed
        observed_connect_principals[physical_database] = connection_observed
        absent_connect_principals[physical_database] = connection_absent
        _ = source_id

    unexpected_connection_policies = sorted(
        set(connection_policy_inputs) - set(required_connect)
    )
    if unexpected_connection_policies:
        blockers.append(
            "CONNECT policies include databases outside the required inventory: "
            f"{unexpected_connection_policies!r}"
        )
    consumed_inventory_refs = set(topology.databases).union(connection_database_refs)
    unexpected_inventory_refs = sorted(
        set(principal_inputs_by_database) - consumed_inventory_refs
    )
    if unexpected_inventory_refs:
        blockers.append(
            "principal inventories are not bound to topology or CONNECT policy: "
            f"{unexpected_inventory_refs!r}"
        )

    declared_principals: dict[str, tuple[str, ...]] = {}
    observed_principals: dict[str, tuple[str, ...]] = {}
    absent_principals: dict[str, tuple[str, ...]] = {}
    database_owners: dict[str, str] = {}
    observed_schema_owners: dict[str, dict[str, str]] = {}
    absent_schemas: dict[str, tuple[str, ...]] = {}
    observed_catalog_objects: list[ModelApplicationDatabaseCatalogObjectEvidence] = []
    principal_domains: dict[str, set[EnumDatabaseSchemaDomain]] = defaultdict(set)
    allowed_memberships: list[ModelApplicationDatabaseRoleMembership] = []
    governed_role_states: list[ModelApplicationDatabaseRoleState] = []
    retained_administrative_principals: set[str] = set()
    for database_ref, database in sorted(topology.databases.items()):
        declared = tuple(sorted(database.principals))
        declared_principals[database_ref] = declared

        inventory_inputs = principal_inputs_by_database.get(database_ref, [])
        if len(inventory_inputs) != 1:
            blockers.append(
                f"{database_ref}: requires exactly one principal inventory, got "
                f"{[source_id for source_id, _ in inventory_inputs]!r}"
            )
            observed: tuple[str, ...] = ()
            absent: tuple[str, ...] = ()
        else:
            observed = tuple(sorted(inventory_inputs[0][1].principal_refs))
            absent = tuple(sorted(inventory_inputs[0][1].absent_principal_refs))
            missing_declared = sorted(set(declared) - set(observed) - set(absent))
            if missing_declared:
                blockers.append(
                    f"{database_ref}: principal inventory lacks presence/absence "
                    "evidence for topology principals "
                    f"{missing_declared!r}"
                )
        observed_principals[database_ref] = observed
        absent_principals[database_ref] = absent
        if len(inventory_inputs) == 1:
            observed_schema_owner_values = dict(
                sorted(inventory_inputs[0][1].observed_schema_owners.items())
            )
            absent_schema_refs = tuple(
                sorted(inventory_inputs[0][1].absent_schema_refs)
            )
            missing_schema_evidence = sorted(
                set(database.schemas)
                - set(observed_schema_owner_values)
                - set(absent_schema_refs)
            )
            extra_schema_evidence = sorted(
                set(observed_schema_owner_values).union(absent_schema_refs)
                - set(database.schemas)
            )
            if missing_schema_evidence or extra_schema_evidence:
                blockers.append(
                    f"{database_ref}: target schema presence/absence evidence "
                    f"differs from topology: missing={missing_schema_evidence!r} "
                    f"extra={extra_schema_evidence!r}"
                )
            observed_catalog_objects.extend(inventory_inputs[0][1].observed_objects)
        else:
            observed_schema_owner_values = {}
            absent_schema_refs = ()
        observed_schema_owners[database_ref] = observed_schema_owner_values
        absent_schemas[database_ref] = absent_schema_refs

        policy_inputs = policy_inputs_by_database.get(database_ref, [])
        if len(policy_inputs) != 1:
            blockers.append(
                f"{database_ref}: requires exactly one independent ACL policy, got "
                f"{[source_id for source_id, _ in policy_inputs]!r}"
            )
            continue
        policy = policy_inputs[0][1]
        policy_principals = set(policy.principal_domains)
        declared_set = set(declared)
        if policy_principals != declared_set:
            blockers.append(
                f"{database_ref}: ACL policy principals differ from topology: "
                f"missing={sorted(declared_set - policy_principals)!r} "
                f"extra={sorted(policy_principals - declared_set)!r}"
            )
        expected_owners = set(database.owners)
        policy_owners = set(policy.migration_owner_roles)
        if policy_owners != expected_owners:
            blockers.append(
                f"{database_ref}: migration owner roles differ from topology: "
                f"missing={sorted(expected_owners - policy_owners)!r} "
                f"extra={sorted(policy_owners - expected_owners)!r}"
            )
        if policy.database_owner_role not in expected_owners:
            blockers.append(
                f"{database_ref}: database owner role "
                f"{policy.database_owner_role!r} is not a topology owner"
            )
        else:
            database_owners[database_ref] = policy.database_owner_role
        if len(inventory_inputs) == 1:
            principal_inventory = inventory_inputs[0][1]
            owner_evidence = set(principal_inventory.owner_refs).union(
                principal_inventory.absent_owner_refs
            )
            missing_owner_evidence = sorted(expected_owners - owner_evidence)
            if missing_owner_evidence:
                blockers.append(
                    f"{database_ref}: managed owner roles lack presence/absence "
                    f"evidence {missing_owner_evidence!r}"
                )
        if policy.migration_principal not in observed:
            blockers.append(
                f"{database_ref}: principal inventory omits migration principal "
                f"{policy.migration_principal!r}"
            )
        if policy.migration_principal in declared_set or (
            policy.migration_principal in expected_owners
        ):
            blockers.append(
                f"{database_ref}: migration principal must be distinct from "
                "workload and owner roles"
            )
        external_connect_roles = {
            principal
            for connection_policy in policy.connection_policies
            for principal in connection_policy.allowed_principals
        } - declared_set
        expected_governed_roles = (
            expected_owners.union(declared_set)
            .union({policy.migration_principal})
            .union(external_connect_roles)
        )
        actual_governed_roles = {state.role for state in policy.governed_role_states}
        if actual_governed_roles != expected_governed_roles:
            blockers.append(
                f"{database_ref}: governed role-state policy differs from topology "
                f"and migration roles: missing="
                f"{sorted(expected_governed_roles - actual_governed_roles)!r} "
                f"extra={sorted(actual_governed_roles - expected_governed_roles)!r}"
            )
        retained_admins = set(policy.retained_administrative_principals)
        retained_administrative_principals.update(retained_admins)
        if retained_admins & actual_governed_roles:
            blockers.append(
                f"{database_ref}: retained administrative principals overlap "
                "governed role-state policy"
            )
        missing_retained_admins = sorted(retained_admins - set(observed))
        if missing_retained_admins:
            blockers.append(
                f"{database_ref}: retained administrative principals are absent "
                f"from the observed census: {missing_retained_admins!r}"
            )
        for state in policy.governed_role_states:
            expected_kind = (
                "owner"
                if state.role in expected_owners
                else "workload"
                if state.role in declared_set
                else "migration"
                if state.role == policy.migration_principal
                else "external_connect"
            )
            expected_login = state.role in declared_set.union(external_connect_roles)
            expected_manage_attributes = state.role not in external_connect_roles
            if (
                state.role_kind != expected_kind
                or state.login is not expected_login
                or state.manage_attributes is not expected_manage_attributes
                or not state.manage_memberships
            ):
                blockers.append(
                    f"{database_ref}: governed role state for {state.role!r} must "
                    f"be kind={expected_kind!r} login={expected_login!r} "
                    f"manage_attributes={expected_manage_attributes!r} "
                    "manage_memberships=True"
                )
            if not state.manage_attributes:
                actual_role_state = observed_role_state_by_name.get(state.role)
                if actual_role_state is None:
                    blockers.append(
                        f"{database_ref}: non-mutating governed role {state.role!r} "
                        "lacks observed attribute evidence"
                    )
                elif any(
                    getattr(actual_role_state, field_name) != getattr(state, field_name)
                    for field_name in (
                        "login",
                        "superuser",
                        "bypass_rls",
                        "create_database",
                        "create_role",
                        "replication",
                        "inherit",
                    )
                ):
                    blockers.append(
                        f"{database_ref}: non-mutating governed role {state.role!r} "
                        "does not already match the safe desired attribute state"
                    )
        governed_role_states.extend(policy.governed_role_states)
        for owner in sorted(policy_owners & expected_owners):
            allowed_memberships.append(
                ModelApplicationDatabaseRoleMembership(
                    database_ref=database_ref,
                    role=owner,
                    member=policy.migration_principal,
                    admin_option=False,
                    inherit_option=False,
                    set_option=True,
                )
            )
        for principal_name, domains in policy.principal_domains.items():
            if principal_name in declared_set:
                principal_domains[principal_name].update(domains)

    rows: list[ModelApplicationDatabaseAclRow] = []
    for database_ref, database in sorted(topology.databases.items()):
        principal_universe = set(database.principals) | set(
            observed_principals[database_ref]
        )
        grantees = (PUBLIC_PRINCIPAL, *sorted(principal_universe))
        for principal in grantees:
            rows.append(
                ModelApplicationDatabaseAclRow(
                    principal=principal,
                    database_ref=database_ref,
                    physical_database=database.physical_name,
                    object_type=EnumDatabaseGrantObjectType.DATABASE,
                    privileges=tuple(
                        sorted(
                            topology_privileges.get(
                                (
                                    principal,
                                    database_ref,
                                    EnumDatabaseGrantObjectType.DATABASE,
                                    None,
                                    None,
                                ),
                                set(),
                            ),
                            key=lambda privilege: privilege.value,
                        )
                    ),
                )
            )
            for schema_name in sorted(database.schemas):
                rows.append(
                    ModelApplicationDatabaseAclRow(
                        principal=principal,
                        database_ref=database_ref,
                        physical_database=database.physical_name,
                        object_type=EnumDatabaseGrantObjectType.SCHEMA,
                        schema_ref=schema_name,
                        privileges=tuple(
                            sorted(
                                topology_privileges.get(
                                    (
                                        principal,
                                        database_ref,
                                        EnumDatabaseGrantObjectType.SCHEMA,
                                        schema_name,
                                        None,
                                    ),
                                    set(),
                                ),
                                key=lambda privilege: privilege.value,
                            )
                        ),
                    )
                )

    ordered_objects = tuple(
        sorted(
            (evidence.obj for evidence in objects.values()),
            key=lambda obj: (
                obj.database_ref,
                obj.schema_ref,
                obj.object_type.value,
                obj.object_ref,
                obj.function_signature or "",
            ),
        )
    )
    expected_catalog_object_ids: set[tuple[str, str, str, str]] = {
        (
            obj.catalog_kind,
            obj.schema_ref,
            obj.object_ref,
            obj.function_signature or "",
        )
        for obj in ordered_objects
    }
    observed_catalog_object_ids: set[tuple[str, str, str, str]] = {
        obj.identity for obj in observed_catalog_objects
    }
    if expected_catalog_object_ids != observed_catalog_object_ids:
        object_blockers.append(
            "live catalog object identities differ from the exact typed ownership "
            f"projection: missing="
            f"{sorted(expected_catalog_object_ids - observed_catalog_object_ids)!r} "
            f"extra={sorted(observed_catalog_object_ids - expected_catalog_object_ids)!r}"
        )
    for obj in ordered_objects:
        database = topology.databases[obj.database_ref]
        principal_universe = set(database.principals) | set(
            observed_principals[obj.database_ref]
        )
        for principal in (PUBLIC_PRINCIPAL, *sorted(principal_universe)):
            rows.append(
                ModelApplicationDatabaseAclRow(
                    principal=principal,
                    database_ref=obj.database_ref,
                    physical_database=obj.physical_database,
                    object_type=obj.object_type,
                    schema_ref=obj.schema_ref,
                    object_ref=obj.object_ref,
                    function_signature=obj.function_signature,
                    privileges=tuple(
                        sorted(
                            resolved_object_privileges.get(
                                (
                                    principal,
                                    obj.database_ref,
                                    obj.object_type,
                                    obj.schema_ref,
                                    obj.object_ref,
                                    obj.function_signature,
                                ),
                                set(),
                            ),
                            key=lambda privilege: privilege.value,
                        )
                    ),
                )
            )

    default_rows: list[ModelApplicationDatabaseDefaultAclRow] = []
    for database_ref, database in sorted(topology.databases.items()):
        principal_universe = set(database.principals) | set(
            observed_principals[database_ref]
        )
        grantees = (PUBLIC_PRINCIPAL, *sorted(principal_universe))
        for schema_name, schema in sorted(database.schemas.items()):
            for object_type in _OBJECT_TYPES:
                for grantee in grantees:
                    default_rows.append(
                        ModelApplicationDatabaseDefaultAclRow(
                            owner=schema.owner,
                            database_ref=database_ref,
                            physical_database=database.physical_name,
                            schema_ref=schema_name,
                            object_type=object_type,
                            grantee=grantee,
                        )
                    )

    scaffold_blockers = tuple(sorted(set(blockers)))
    unique_blockers = tuple(sorted({*blockers, *object_blockers}))
    matrix = ModelApplicationDatabaseAclMatrix(
        authorization_scope=authorization_scope,
        scaffold_status="BLOCKED" if scaffold_blockers else "READY",
        scaffold_blockers=scaffold_blockers,
        status="BLOCKED" if unique_blockers else "READY",
        sources=tuple(sorted(sources, key=lambda source: source.source_key)),
        verified_evidence_source_keys=tuple(sorted(verified_evidence)),
        declared_principals=declared_principals,
        observed_principals=observed_principals,
        absent_principals=absent_principals,
        observed_owner_roles=observed_owner_roles,
        absent_owner_roles=absent_owner_roles,
        observed_role_states=tuple(
            sorted(observed_role_state_by_name.values(), key=lambda state: state.role)
        ),
        governed_role_states=tuple(
            sorted(governed_role_states, key=lambda state: state.role)
        ),
        retained_administrative_principals=tuple(
            sorted(retained_administrative_principals)
        ),
        database_owners=database_owners,
        required_connect_databases=required_connect,
        observed_connect_database_owners=observed_connect_database_owners,
        allowed_connect_principals=allowed_connect_principals,
        observed_connect_principals=observed_connect_principals,
        absent_connect_principals=absent_connect_principals,
        schema_domains={
            database_ref: {
                schema_name: schema.domain
                for schema_name, schema in sorted(database.schemas.items())
            }
            for database_ref, database in sorted(topology.databases.items())
        },
        observed_schema_owners=observed_schema_owners,
        absent_schemas=absent_schemas,
        principal_domains={
            principal: tuple(sorted(domains, key=lambda domain: domain.value))
            for principal, domains in sorted(principal_domains.items())
        },
        allowed_memberships=tuple(
            sorted(
                allowed_memberships,
                key=lambda item: item.identity,
            )
        ),
        observed_objects=tuple(
            sorted(observed_catalog_objects, key=lambda obj: obj.identity)
        ),
        objects=ordered_objects,
        rows=tuple(
            sorted(
                rows,
                key=lambda row: tuple(str(item) for item in row.identity),
            )
        ),
        default_privileges=tuple(
            sorted(
                default_rows,
                key=lambda row: (
                    row.database_ref,
                    row.schema_ref,
                    row.owner,
                    row.object_type.value,
                    row.grantee,
                ),
            )
        ),
        blockers=unique_blockers,
        excluded_objects=tuple(sorted(set(excluded))),
    )
    shared_scaffold_violations = validate_application_database_acl_scaffold(
        matrix,
        require_safe_existing_roles=False,
    )
    scaffold_violations = validate_application_database_acl_scaffold(matrix)
    if scaffold_violations:
        rendered_scaffold_blockers = tuple(
            sorted(
                {
                    *matrix.scaffold_blockers,
                    *(
                        f"ACL scaffold policy violation: {violation}"
                        for violation in scaffold_violations
                    ),
                }
            )
        )
        matrix = matrix.model_copy(
            update={
                "scaffold_status": "BLOCKED",
                "scaffold_blockers": rendered_scaffold_blockers,
            }
        )
    if shared_scaffold_violations:
        matrix = matrix.model_copy(
            update={
                "status": "BLOCKED",
                "blockers": tuple(
                    sorted(
                        {
                            *matrix.blockers,
                            *(
                                f"ACL scaffold policy violation: {violation}"
                                for violation in shared_scaffold_violations
                            ),
                        }
                    )
                ),
            }
        )
    structural_violations = validate_application_database_acl_matrix(matrix)
    if structural_violations:
        matrix = matrix.model_copy(
            update={
                "status": "BLOCKED",
                "blockers": tuple(
                    sorted(
                        {
                            *matrix.blockers,
                            *(
                                f"ACL policy violation: {violation}"
                                for violation in structural_violations
                            ),
                        }
                    )
                ),
            }
        )
    return ModelApplicationDatabaseAclMatrix.model_validate(
        matrix.model_dump(mode="json")
    )


def validate_application_database_acl_matrix(
    matrix: ModelApplicationDatabaseAclMatrix,
) -> tuple[str, ...]:
    """Validate deny-by-default, ownership, completeness, and domain separation."""
    violations: list[str] = []
    declared_principals = {
        principal
        for principals in matrix.declared_principals.values()
        for principal in principals
    }
    matrix_principals = declared_principals.union(
        principal
        for principals in (
            *matrix.observed_principals.values(),
            *matrix.observed_connect_principals.values(),
            *matrix.absent_principals.values(),
            *matrix.absent_connect_principals.values(),
            *matrix.allowed_connect_principals.values(),
        )
        for principal in principals
    )
    for database_ref, owner in matrix.database_owners.items():
        if owner in matrix_principals:
            violations.append(
                f"runtime principal {owner!r} owns database {database_ref!r}"
            )
    for obj in matrix.objects:
        if not obj.target_materialized:
            violations.append(
                f"target object {obj.physical_database}.{obj.schema_ref}."
                f"{obj.object_ref} is not materialized at the rendered location"
            )
        if (
            obj.object_type is EnumDatabaseGrantObjectType.FUNCTION
            and obj.function_signature is None
        ):
            violations.append(
                f"routine {obj.schema_ref}.{obj.object_ref} lacks an exact signature"
            )
        if obj.owner in matrix_principals:
            violations.append(
                f"runtime principal {obj.owner!r} owns "
                f"{obj.schema_ref}.{obj.object_ref}"
            )
    for row in matrix.rows:
        if row.principal == PUBLIC_PRINCIPAL and row.privileges:
            violations.append(
                f"PUBLIC has {sorted(p.value for p in row.privileges)!r} on "
                f"{row.object_type.value}:{row.schema_ref}:{row.object_ref}"
            )

    domains_by_object = {obj.identity: obj.domain for obj in matrix.objects}
    for row in matrix.rows:
        if not row.privileges:
            continue
        if row.object_type is EnumDatabaseGrantObjectType.SCHEMA:
            domain = matrix.schema_domains.get(row.database_ref, {}).get(
                row.schema_ref or ""
            )
        elif row.object_type in _OBJECT_TYPES:
            object_identity = (
                row.database_ref,
                row.schema_ref or "",
                row.object_type,
                row.object_ref or "",
                row.function_signature,
            )
            domain = domains_by_object.get(object_identity)
        else:
            domain = None
        if domain is None:
            continue
        allowed = matrix.principal_domains.get(row.principal, ())
        if domain not in allowed:
            violations.append(
                f"principal {row.principal!r} has cross-domain privileges on "
                f"{row.schema_ref}.{row.object_ref} ({domain.value})"
            )
    prohibited = {
        EnumDatabasePrivilege.CREATE,
        EnumDatabasePrivilege.TEMPORARY,
        EnumDatabasePrivilege.TRIGGER,
        EnumDatabasePrivilege.REFERENCES,
        EnumDatabasePrivilege.TRUNCATE,
    }
    for acl_row in matrix.rows:
        if acl_row.principal in matrix_principals and prohibited & set(
            acl_row.privileges
        ):
            violations.append(
                f"runtime principal {acl_row.principal!r} has DDL privilege on "
                f"{acl_row.object_type.value}:{acl_row.schema_ref}:"
                f"{acl_row.object_ref}"
            )
    declared_by_database = {
        database_ref: set(principals)
        for database_ref, principals in matrix.declared_principals.items()
    }
    observed_by_database = {
        database_ref: set(principals)
        for database_ref, principals in matrix.observed_principals.items()
    }
    owner_roles = (
        set(matrix.database_owners.values())
        .union(obj.owner for obj in matrix.objects)
        .union(row.owner for row in matrix.default_privileges)
    )
    for membership in matrix.allowed_memberships:
        if membership.member in declared_by_database.get(
            membership.database_ref, set()
        ):
            violations.append(
                f"workload principal {membership.member!r} has permitted role "
                f"membership {membership.role!r}"
            )
        if membership.member not in observed_by_database.get(
            membership.database_ref, set()
        ):
            violations.append(
                f"membership principal {membership.member!r} is absent from the "
                "observed principal census"
            )
        if membership.role not in owner_roles:
            violations.append(
                f"permitted membership role {membership.role!r} is not an owner role"
            )
        if (
            membership.admin_option
            or membership.inherit_option
            or not membership.set_option
        ):
            violations.append(
                f"migration membership {membership.role!r} -> "
                f"{membership.member!r} must be SET-only"
            )
    for default_row in matrix.default_privileges:
        if default_row.grantee == PUBLIC_PRINCIPAL and default_row.privileges:
            violations.append(
                f"PUBLIC has future {default_row.object_type.value} privileges in "
                f"{default_row.schema_ref!r}"
            )
        if default_row.grantee in matrix_principals and default_row.privileges:
            violations.append(
                f"runtime principal {default_row.grantee!r} has broad future "
                f"{default_row.object_type.value} privileges in "
                f"{default_row.schema_ref!r}"
            )

    schema_usage = {
        (row.principal, row.database_ref, row.schema_ref)
        for row in matrix.rows
        if row.object_type is EnumDatabaseGrantObjectType.SCHEMA
        and EnumDatabasePrivilege.USAGE in row.privileges
    }
    for row in matrix.rows:
        if row.object_type not in _OBJECT_TYPES or not row.privileges:
            continue
        if (row.principal, row.database_ref, row.schema_ref) not in schema_usage:
            violations.append(
                f"principal {row.principal!r} has object privilege without schema "
                f"USAGE on {row.schema_ref!r}"
            )

    object_ids = {obj.identity for obj in matrix.objects}
    object_row_ids = {
        (
            row.database_ref,
            row.schema_ref,
            row.object_type,
            row.object_ref,
            row.function_signature,
        )
        for row in matrix.rows
        if row.object_type in _OBJECT_TYPES
    }
    if object_ids != object_row_ids:
        violations.append("object rows do not cover the exact ownership object set")
    expected_grantees_by_database = {
        database_ref: {
            row.principal
            for row in matrix.rows
            if row.database_ref == database_ref
            and row.object_type is EnumDatabaseGrantObjectType.DATABASE
        }
        for database_ref in {obj.database_ref for obj in matrix.objects}
    }
    for obj in matrix.objects:
        actual_grantees = {
            row.principal
            for row in matrix.rows
            if (
                row.database_ref,
                row.schema_ref,
                row.object_type,
                row.object_ref,
                row.function_signature,
            )
            == obj.identity
        }
        if actual_grantees != expected_grantees_by_database[obj.database_ref]:
            violations.append(
                f"{obj.schema_ref}.{obj.object_ref} lacks a complete principal row set"
            )
    return tuple(sorted(set(violations)))


def validate_application_database_acl_scaffold(
    matrix: ModelApplicationDatabaseAclMatrix,
    *,
    require_safe_existing_roles: bool = True,
) -> tuple[str, ...]:
    """Validate the additive P1 roles/schemas/CONNECT/default-ACL stage only."""
    violations: list[str] = []
    database_refs = set(matrix.declared_principals)
    if not database_refs:
        violations.append("scaffold declares zero topology databases")
    elif len(database_refs) != 1:
        violations.append("scaffold requires exactly one topology application database")
    if (
        set(matrix.observed_principals) != database_refs
        or set(matrix.absent_principals) != database_refs
    ):
        violations.append(
            "scaffold principal presence/absence database keys are incomplete"
        )
    if set(matrix.database_owners) != database_refs:
        violations.append("scaffold database owner map is incomplete")
    if set(matrix.schema_domains) != database_refs:
        violations.append("scaffold schema domain map is incomplete")
    if (
        set(matrix.observed_schema_owners) != database_refs
        or set(matrix.absent_schemas) != database_refs
    ):
        violations.append("scaffold schema presence/absence maps are incomplete")
    application_physical_databases = {
        row.physical_database for row in matrix.default_privileges
    }
    if len(application_physical_databases) != 1:
        violations.append(
            "scaffold requires exactly one application physical database target"
        )
    if any(
        _SQL_IDENTIFIER.fullmatch(database) is None
        for database in application_physical_databases
    ):
        violations.append("scaffold application database target is not canonical")
    row_physical_databases = {
        row.physical_database
        for row in matrix.rows
        if row.database_ref in database_refs
    }
    object_physical_databases = {
        obj.physical_database
        for obj in matrix.objects
        if obj.database_ref in database_refs
    }
    if row_physical_databases != application_physical_databases or (
        object_physical_databases
        and object_physical_databases != application_physical_databases
    ):
        violations.append(
            "scaffold application physical database target disagrees across rows, "
            "objects, and future-ACL cells"
        )
    if not application_physical_databases <= set(matrix.required_connect_databases):
        violations.append(
            "scaffold application physical database is absent from CONNECT scope"
        )

    required_connect = set(matrix.required_connect_databases)
    for name, values in (
        ("allowed CONNECT", matrix.allowed_connect_principals),
        ("observed CONNECT", matrix.observed_connect_principals),
        ("observed CONNECT owner", matrix.observed_connect_database_owners),
    ):
        if set(values) != required_connect:
            violations.append(f"{name} database keys do not cover required CONNECT set")

    declared = {
        principal
        for principals in matrix.declared_principals.values()
        for principal in principals
    }
    observed = {
        principal
        for principals in matrix.observed_principals.values()
        for principal in principals
    }
    connection_principals = {
        principal
        for principals in matrix.observed_connect_principals.values()
        for principal in principals
    }
    desired_connect = {
        principal
        for principals in matrix.allowed_connect_principals.values()
        for principal in principals
    }
    runtime_principals = declared.union(
        observed,
        connection_principals,
        desired_connect,
    )
    owner_roles = set(matrix.database_owners.values()) | {
        row.owner for row in matrix.default_privileges
    }
    if owner_roles & runtime_principals:
        violations.append("scaffold owner roles overlap runtime principal census")

    for database_ref in sorted(database_refs):
        expected_grantees = {
            PUBLIC_PRINCIPAL,
            *matrix.declared_principals[database_ref],
            *matrix.observed_principals[database_ref],
        }
        existing = set(matrix.observed_principals[database_ref])
        absent = set(matrix.absent_principals[database_ref])
        if existing & absent:
            violations.append(
                f"{database_ref}: principal presence/absence evidence overlaps"
            )
        if not set(matrix.declared_principals[database_ref]) <= existing.union(absent):
            violations.append(
                f"{database_ref}: scaffold lacks presence/absence evidence for "
                "declared principals"
            )
        schemas = set(matrix.schema_domains[database_ref])
        if not schemas:
            violations.append(f"{database_ref}: scaffold declares zero schemas")
        observed_schemas = set(matrix.observed_schema_owners.get(database_ref, {}))
        absent_schemas = set(matrix.absent_schemas.get(database_ref, ()))
        if observed_schemas & absent_schemas:
            violations.append(
                f"{database_ref}: schema presence/absence evidence overlaps"
            )
        if schemas != observed_schemas.union(absent_schemas):
            violations.append(
                f"{database_ref}: scaffold lacks exact target schema evidence"
            )
        database_rows = {
            row.principal: row
            for row in matrix.rows
            if row.database_ref == database_ref
            and row.object_type is EnumDatabaseGrantObjectType.DATABASE
        }
        if set(database_rows) != expected_grantees:
            violations.append(
                f"{database_ref}: database ACL cells do not cover exact principal census"
            )
        for schema_name in sorted(schemas):
            schema_rows = {
                row.principal: row
                for row in matrix.rows
                if row.database_ref == database_ref
                and row.object_type is EnumDatabaseGrantObjectType.SCHEMA
                and row.schema_ref == schema_name
            }
            if set(schema_rows) != expected_grantees:
                violations.append(
                    f"{database_ref}.{schema_name}: schema ACL cells do not cover "
                    "exact principal census"
                )
            owners = {
                row.owner
                for row in matrix.default_privileges
                if row.database_ref == database_ref and row.schema_ref == schema_name
            }
            if len(owners) != 1:
                violations.append(
                    f"{database_ref}.{schema_name}: requires exactly one schema owner"
                )
            expected_defaults = {
                (object_type, grantee)
                for object_type in _OBJECT_TYPES
                for grantee in expected_grantees
            }
            actual_defaults = {
                (row.object_type, row.grantee)
                for row in matrix.default_privileges
                if row.database_ref == database_ref and row.schema_ref == schema_name
            }
            if actual_defaults != expected_defaults:
                violations.append(
                    f"{database_ref}.{schema_name}: default ACL cells are incomplete"
                )

    for physical_database in sorted(required_connect):
        allowed = set(matrix.allowed_connect_principals.get(physical_database, ()))
        observed_connect = set(
            matrix.observed_connect_principals.get(physical_database, ())
        )
        absent_connect = set(
            matrix.absent_connect_principals.get(physical_database, ())
        )
        if not allowed:
            violations.append(f"{physical_database}: CONNECT allowlist is empty")
        if observed_connect & absent_connect:
            violations.append(
                f"{physical_database}: CONNECT presence/absence evidence overlaps"
            )
        if not allowed <= observed_connect.union(absent_connect):
            violations.append(
                f"{physical_database}: CONNECT allowlist lacks presence/absence evidence"
            )
    managed_owner_evidence = set(matrix.observed_owner_roles).union(
        matrix.absent_owner_roles
    )
    if not owner_roles <= managed_owner_evidence:
        violations.append("scaffold managed owners lack presence/absence evidence")
    governed_states = {state.role: state for state in matrix.governed_role_states}
    if not owner_roles <= {
        role for role, state in governed_states.items() if state.role_kind == "owner"
    }:
        violations.append("scaffold managed owners lack governed owner-role state")
    if not declared <= {
        role
        for role, state in governed_states.items()
        if state.role_kind == "workload" and state.login
    }:
        violations.append("scaffold workloads lack governed LOGIN role state")
    observed_states = {state.role: state for state in matrix.observed_role_states}
    globally_absent_roles = {
        principal
        for principals in (
            *matrix.absent_principals.values(),
            *matrix.absent_connect_principals.values(),
            matrix.absent_owner_roles,
        )
        for principal in principals
    }
    scaffold_safe_roles = set(globally_absent_roles)
    scaffold_safe_roles.update(
        role
        for role, governed_state in governed_states.items()
        if role in observed_states
        and _role_attribute_values(observed_states[role])
        == _role_attribute_values(governed_state)
    )
    intended_scaffold_roles = set(owner_roles)
    intended_scaffold_roles.update(
        principal
        for principals in matrix.allowed_connect_principals.values()
        for principal in principals
    )
    intended_scaffold_roles.update(
        row.principal
        for row in matrix.rows
        if row.principal != PUBLIC_PRINCIPAL
        and row.object_type
        in {
            EnumDatabaseGrantObjectType.DATABASE,
            EnumDatabaseGrantObjectType.SCHEMA,
        }
        and row.privileges
    )
    intended_scaffold_roles.update(
        principal
        for membership in matrix.allowed_memberships
        for principal in (membership.role, membership.member)
    )
    unsafe_scaffold_roles = sorted(intended_scaffold_roles - scaffold_safe_roles)
    if require_safe_existing_roles and unsafe_scaffold_roles:
        violations.append(
            "additive scaffold intended roles are not already safe or proven absent: "
            f"{unsafe_scaffold_roles!r}"
        )

    prohibited = {
        EnumDatabasePrivilege.CREATE,
        EnumDatabasePrivilege.TEMPORARY,
        EnumDatabasePrivilege.TRIGGER,
        EnumDatabasePrivilege.REFERENCES,
        EnumDatabasePrivilege.TRUNCATE,
    }
    for row in matrix.rows:
        if row.object_type not in {
            EnumDatabaseGrantObjectType.DATABASE,
            EnumDatabaseGrantObjectType.SCHEMA,
        }:
            continue
        if row.principal == PUBLIC_PRINCIPAL and row.privileges:
            violations.append("PUBLIC has privileges in the additive scaffold")
        if row.principal in runtime_principals and prohibited & set(row.privileges):
            violations.append(
                f"runtime principal {row.principal!r} has scaffold DDL privileges"
            )
        if row.object_type is EnumDatabaseGrantObjectType.SCHEMA and row.privileges:
            domain = matrix.schema_domains.get(row.database_ref, {}).get(
                row.schema_ref or ""
            )
            if domain not in matrix.principal_domains.get(row.principal, ()):
                violations.append(
                    f"principal {row.principal!r} has cross-domain scaffold access"
                )
    if any(row.privileges for row in matrix.default_privileges):
        violations.append("scaffold future-object defaults must be deny-by-default")

    for membership in matrix.allowed_memberships:
        if membership.role not in owner_roles:
            violations.append("scaffold membership targets a non-owner role")
        if membership.member in declared:
            violations.append("runtime workload has scaffold owner membership")
        if (
            membership.admin_option
            or membership.inherit_option
            or not membership.set_option
        ):
            violations.append("scaffold owner membership must be SET-only")
    return tuple(sorted(set(violations)))


def _quote_identifier(identifier: str) -> str:
    if _SQL_IDENTIFIER.fullmatch(identifier) is None:
        raise ValueError(f"Unsafe or non-canonical SQL identifier {identifier!r}")
    return f'"{identifier}"'


def _quote_sql_literal(value: str) -> str:
    """Quote one trusted typed value as a PostgreSQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def _managed_provenance_marker(
    matrix: ModelApplicationDatabaseAclMatrix,
    *,
    kind: str,
    name: str,
) -> str:
    """Return a deterministic source-lock marker for additive managed objects."""
    _ = matrix
    return f"omnibase_application_acl:v1:managed:{kind}:{name}"


def _sql_object_target(
    obj: ModelApplicationDatabaseAclObject,
) -> tuple[str, str, str]:
    """Return ownership keyword, ACL keyword, and exact SQL target."""
    qualified = (
        f"{_quote_identifier(obj.schema_ref)}.{_quote_identifier(obj.object_ref)}"
    )
    if obj.object_type is EnumDatabaseGrantObjectType.TABLE:
        owner_keyword = {
            "table": "TABLE",
            "view": "VIEW",
            "materialized_view": "MATERIALIZED VIEW",
        }[obj.catalog_kind]
        return owner_keyword, "TABLE", qualified
    if obj.object_type is EnumDatabaseGrantObjectType.SEQUENCE:
        return "SEQUENCE", "SEQUENCE", qualified
    if obj.object_type is EnumDatabaseGrantObjectType.TYPE:
        return "TYPE", "TYPE", qualified
    if obj.function_signature is None:
        raise ValueError(
            f"Function {obj.schema_ref}.{obj.object_ref} lacks function_signature"
        )
    keyword = "PROCEDURE" if obj.catalog_kind == "procedure" else "FUNCTION"
    return keyword, keyword, f"{qualified}{obj.function_signature}"


def render_application_database_acl_sql(
    matrix: ModelApplicationDatabaseAclMatrix,
    *,
    allow_synthetic_proof: bool = False,
    phase: EnumApplicationDatabaseAclRenderPhase = (
        EnumApplicationDatabaseAclRenderPhase.FULL
    ),
) -> str:
    """Render an atomic additive scaffold or the materialized-object full phase."""
    if phase is EnumApplicationDatabaseAclRenderPhase.FULL and matrix.status != "READY":
        raise ValueError(
            f"Cannot render blocked ACL matrix ({len(matrix.blockers)} blocker(s))"
        )
    if (
        phase is EnumApplicationDatabaseAclRenderPhase.SCAFFOLD
        and matrix.scaffold_status != "READY"
    ):
        raise ValueError(
            "Cannot render blocked ACL scaffold "
            f"({len(matrix.scaffold_blockers)} blocker(s))"
        )
    if (
        matrix.authorization_scope
        is EnumApplicationDatabaseAclAuthorizationScope.SYNTHETIC_PROOF
        and not allow_synthetic_proof
    ):
        raise ValueError(
            "Synthetic proof matrices require explicit allow_synthetic_proof=true"
        )
    scaffold_violations = validate_application_database_acl_scaffold(
        matrix,
        require_safe_existing_roles=(
            phase is EnumApplicationDatabaseAclRenderPhase.SCAFFOLD
        ),
    )
    violations = (
        tuple(
            sorted(
                set(scaffold_violations).union(
                    validate_application_database_acl_matrix(matrix)
                )
            )
        )
        if phase is EnumApplicationDatabaseAclRenderPhase.FULL
        else scaffold_violations
    )
    if violations:
        raise ValueError("Cannot render invalid ACL matrix: " + "; ".join(violations))

    declared_principals = sorted(
        {
            principal
            for principals in matrix.declared_principals.values()
            for principal in principals
        }
    )
    allowed_connect_principals = sorted(
        {
            principal
            for principals in matrix.allowed_connect_principals.values()
            for principal in principals
        }
    )
    governed_states = {state.role: state for state in matrix.governed_role_states}
    revocation_principals = sorted(
        set(declared_principals)
        .union(allowed_connect_principals)
        .union(
            principal
            for principals in (
                *matrix.observed_principals.values(),
                *matrix.observed_connect_principals.values(),
            )
            for principal in principals
        )
    )
    owners = sorted(
        set(matrix.database_owners.values())
        .union(obj.owner for obj in matrix.objects)
        .union(row.owner for row in matrix.default_privileges)
    )
    globally_observed_roles = sorted(
        {
            principal
            for principals in (
                *matrix.observed_principals.values(),
                *matrix.observed_connect_principals.values(),
                matrix.observed_owner_roles,
                tuple(matrix.observed_connect_database_owners.values()),
            )
            for principal in principals
        }
    )
    globally_absent_roles = {
        principal
        for principals in (
            *matrix.absent_principals.values(),
            *matrix.absent_connect_principals.values(),
            matrix.absent_owner_roles,
        )
        for principal in principals
    }
    application_physical_database = next(
        iter({row.physical_database for row in matrix.default_privileges})
    )
    application_database_ref = next(iter(matrix.declared_principals))
    application_desired_owner = matrix.database_owners[application_database_ref]
    schema_owners = {
        (row.database_ref, row.schema_ref): row.owner
        for row in matrix.default_privileges
    }
    observed_states = {state.role: state for state in matrix.observed_role_states}
    desired_object_owners: dict[tuple[str, str, str, str], str] = {
        (
            obj.catalog_kind,
            obj.schema_ref,
            obj.object_ref,
            obj.function_signature or "",
        ): obj.owner
        for obj in matrix.objects
    }

    def role_attribute_sql(role_name: str) -> tuple[str, ...]:
        state = governed_states[role_name]
        return (
            f"role.rolcanlogin IS DISTINCT FROM {'TRUE' if state.login else 'FALSE'}",
            "role.rolsuper IS DISTINCT FROM FALSE",
            "role.rolbypassrls IS DISTINCT FROM FALSE",
            "role.rolcreatedb IS DISTINCT FROM FALSE",
            "role.rolcreaterole IS DISTINCT FROM FALSE",
            "role.rolreplication IS DISTINCT FROM FALSE",
            "role.rolinherit IS DISTINCT FROM FALSE",
        )

    def role_attribute_clause(role_name: str) -> str:
        state = governed_states[role_name]
        return (
            f"{'LOGIN' if state.login else 'NOLOGIN'} NOSUPERUSER NOBYPASSRLS "
            "NOCREATEDB NOCREATEROLE NOREPLICATION NOINHERIT"
        )

    def role_state_match_clause(
        state: ProtocolApplicationDatabaseRoleAttributeState,
    ) -> str:
        attribute_columns = (
            ("login", "rolcanlogin"),
            ("superuser", "rolsuper"),
            ("bypass_rls", "rolbypassrls"),
            ("create_database", "rolcreatedb"),
            ("create_role", "rolcreaterole"),
            ("replication", "rolreplication"),
            ("inherit", "rolinherit"),
        )
        return (
            "("
            + " AND ".join(
                f"role.{column} IS {'TRUE' if getattr(state, field) else 'FALSE'}"
                for field, column in attribute_columns
            )
            + ")"
        )

    lines = [
        "-- Generated application-database ACL; do not hand edit.",
        f"-- Render phase: {phase.value}",
        "-- Source revisions:",
        *[
            f"--   {source.source_key}: {source.repository}@{source.revision} "
            f"{source.path} sha256:{source.sha256}"
            for source in matrix.sources
        ],
        "\\set ON_ERROR_STOP on",
        "BEGIN;",
        "DO $acl_database_guard$ BEGIN",
        f"  IF current_database() <> '{application_physical_database}' THEN",
        "    RAISE EXCEPTION 'application ACL connected to unexpected database %', "
        "current_database();",
        "  END IF;",
        "END $acl_database_guard$;",
        "LOCK TABLE pg_catalog.pg_authid IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_auth_members IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_database IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_namespace IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_class IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_attribute IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_proc IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_type IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_extension IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_depend IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_default_acl IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_description IN SHARE MODE;",
        "LOCK TABLE pg_catalog.pg_shdescription IN SHARE MODE;",
        "",
    ]
    lines.extend(
        [
            "DO $acl_evidence_guard$",
            "DECLARE actual_owner text;",
            "DECLARE actual_objects text[];",
            "BEGIN",
        ]
    )
    for role_name in globally_observed_roles:
        lines.extend(
            [
                "  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_authid role",
                f"                 WHERE role.rolname = '{role_name}') THEN",
                f"    RAISE EXCEPTION 'locked role census is stale: expected present role {role_name}';",
                "  END IF;",
            ]
        )
    for role_name, observed_state in sorted(observed_states.items()):
        accepted_state_clauses = [role_state_match_clause(observed_state)]
        governed_state = governed_states.get(role_name)
        if governed_state is not None and governed_state.manage_attributes:
            accepted_state_clauses.append(role_state_match_clause(governed_state))
        accepted_state_sql = " OR ".join(sorted(set(accepted_state_clauses)))
        lines.extend(
            [
                "  IF EXISTS (SELECT 1 FROM pg_catalog.pg_authid role",
                f"             WHERE role.rolname = '{role_name}'",
                f"               AND NOT ({accepted_state_sql})) THEN",
                f"    RAISE EXCEPTION 'locked role attribute census is stale for {role_name}';",
                "  END IF;",
            ]
        )
    for role_name, state in sorted(governed_states.items()):
        if state.manage_attributes:
            continue
        mismatch = " OR ".join(role_attribute_sql(role_name))
        lines.extend(
            [
                "  IF EXISTS (SELECT 1 FROM pg_catalog.pg_authid role",
                f"             WHERE role.rolname = '{role_name}'",
                f"               AND ({mismatch})) THEN",
                f"    RAISE EXCEPTION 'non-mutating governed role {role_name} has unsafe attributes';",
                "  END IF;",
            ]
        )
    allowed_membership_conditions = [
        f"(parent.rolname = '{membership.role}' AND member.rolname = '{membership.member}')"
        for membership in matrix.allowed_memberships
    ]
    allowed_membership_sql = (
        " AND NOT (" + " OR ".join(allowed_membership_conditions) + ")"
        if allowed_membership_conditions
        else ""
    )
    for role_name in sorted(globally_absent_roles.intersection(governed_states)):
        state = governed_states[role_name]
        if not state.manage_attributes:
            continue
        marker = _managed_provenance_marker(matrix, kind="role", name=role_name)
        mismatch = " OR ".join(
            (
                f"pg_catalog.shobj_description(role.oid, 'pg_authid') IS DISTINCT FROM '{marker}'",
                "role.rolpassword IS NOT NULL",
                *role_attribute_sql(role_name),
            )
        )
        lines.extend(
            [
                "  IF EXISTS (SELECT 1 FROM pg_catalog.pg_authid role",
                f"             WHERE role.rolname = '{role_name}'",
                f"               AND ({mismatch}))",
                "     OR EXISTS (",
                "       SELECT 1 FROM pg_catalog.pg_auth_members membership",
                "       JOIN pg_catalog.pg_authid parent ON parent.oid = membership.roleid",
                "       JOIN pg_catalog.pg_authid member ON member.oid = membership.member",
                f"       WHERE (parent.rolname = '{role_name}' OR member.rolname = '{role_name}')",
                f"       {allowed_membership_sql}",
                "     ) THEN",
                f"    RAISE EXCEPTION 'expected-absent role collision for {role_name}';",
                "  END IF;",
            ]
        )
    for physical_database, observed_owner in sorted(
        matrix.observed_connect_database_owners.items()
    ):
        accepted_owners = {observed_owner}
        if physical_database == application_physical_database:
            accepted_owners.add(application_desired_owner)
        owner_literals = ", ".join(f"'{owner}'" for owner in sorted(accepted_owners))
        lines.extend(
            [
                "  SELECT owner.rolname INTO actual_owner",
                "  FROM pg_catalog.pg_database database",
                "  JOIN pg_catalog.pg_authid owner ON owner.oid = database.datdba",
                f"  WHERE database.datname = '{physical_database}';",
                f"  IF actual_owner IS NULL OR actual_owner NOT IN ({owner_literals}) THEN",
                f"    RAISE EXCEPTION 'locked database owner census is stale for {physical_database}: %', actual_owner;",
                "  END IF;",
            ]
        )
    for (database_ref, schema_name), owner in sorted(schema_owners.items()):
        if schema_name is None:
            continue
        if schema_name in matrix.observed_schema_owners[database_ref]:
            observed_schema_owner = matrix.observed_schema_owners[database_ref][
                schema_name
            ]
            accepted_schema_owners = ", ".join(
                f"'{role}'" for role in sorted({observed_schema_owner, owner})
            )
            lines.extend(
                [
                    "  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace namespace",
                    f"                 WHERE namespace.nspname = '{schema_name}') THEN",
                    f"    RAISE EXCEPTION 'locked schema census is stale: expected present schema {schema_name}';",
                    "  END IF;",
                    "  IF EXISTS (SELECT 1 FROM pg_catalog.pg_namespace namespace",
                    "             JOIN pg_catalog.pg_authid schema_owner ON schema_owner.oid = namespace.nspowner",
                    f"             WHERE namespace.nspname = '{schema_name}'",
                    f"               AND schema_owner.rolname NOT IN ({accepted_schema_owners})) THEN",
                    f"    RAISE EXCEPTION 'locked schema owner census is stale for {schema_name}';",
                    "  END IF;",
                ]
            )
        else:
            marker = _managed_provenance_marker(
                matrix,
                kind="schema",
                name=schema_name,
            )
            lines.extend(
                [
                    "  IF EXISTS (",
                    "       SELECT 1 FROM pg_catalog.pg_namespace namespace",
                    "       JOIN pg_catalog.pg_authid owner ON owner.oid = namespace.nspowner",
                    f"       WHERE namespace.nspname = '{schema_name}'",
                    "         AND (",
                    f"           pg_catalog.obj_description(namespace.oid, 'pg_namespace') IS DISTINCT FROM '{marker}'",
                    f"           OR owner.rolname IS DISTINCT FROM '{owner}'",
                    "         )",
                    "     ) OR EXISTS (",
                    "       SELECT 1 FROM pg_catalog.pg_depend dependency",
                    "       JOIN pg_catalog.pg_namespace namespace",
                    "         ON namespace.oid = dependency.refobjid",
                    "       WHERE dependency.refclassid = 'pg_catalog.pg_namespace'::regclass",
                    f"         AND namespace.nspname = '{schema_name}'",
                    "     ) THEN",
                    f"    RAISE EXCEPTION 'expected-absent schema collision for {schema_name}';",
                    "  END IF;",
                ]
            )
    if phase is EnumApplicationDatabaseAclRenderPhase.FULL:
        managed_schema_literals = ", ".join(
            repr(schema_name)
            for schema_name in sorted(
                {
                    schema_name
                    for schemas in matrix.schema_domains.values()
                    for schema_name in schemas
                }
            )
        )
        expected_object_identities = tuple(
            sorted(
                "|".join(
                    (
                        obj.catalog_kind,
                        obj.schema_ref,
                        obj.object_ref,
                        obj.function_signature or "",
                    )
                )
                for obj in matrix.observed_objects
            )
        )
        expected_object_array = (
            "ARRAY["
            + ", ".join(
                _quote_sql_literal(identity) for identity in expected_object_identities
            )
            + "]::text[]"
            if expected_object_identities
            else "ARRAY[]::text[]"
        )
        lines.extend(
            [
                "  SELECT COALESCE(array_agg(catalog_identity), ARRAY[]::text[])",
                "  INTO actual_objects",
                "  FROM (",
                "    SELECT CASE relation.relkind",
                "             WHEN 'r' THEN 'table'",
                "             WHEN 'p' THEN 'table'",
                "             WHEN 'v' THEN 'view'",
                "             WHEN 'm' THEN 'materialized_view'",
                "             WHEN 'S' THEN 'sequence'",
                "           END || '|' || namespace.nspname || '|' ||",
                "           relation.relname || '|' AS catalog_identity",
                "    FROM pg_catalog.pg_class relation",
                "    JOIN pg_catalog.pg_namespace namespace ON namespace.oid = relation.relnamespace",
                f"    WHERE namespace.nspname IN ({managed_schema_literals})",
                "      AND relation.relkind IN ('r', 'p', 'v', 'm', 'S')",
                "    UNION ALL",
                "    SELECT CASE WHEN procedure.prokind = 'p'",
                "                THEN 'procedure' ELSE 'function' END || '|' ||",
                "           namespace.nspname || '|' || procedure.proname || '|' ||",
                "           '(' || pg_catalog.pg_get_function_identity_arguments(procedure.oid) || ')'",
                "    FROM pg_catalog.pg_proc procedure",
                "    JOIN pg_catalog.pg_namespace namespace ON namespace.oid = procedure.pronamespace",
                f"    WHERE namespace.nspname IN ({managed_schema_literals})",
                "      AND procedure.prokind IN ('f', 'p')",
                "    UNION ALL",
                "    SELECT 'type|' || namespace.nspname || '|' || type.typname || '|'",
                "    FROM pg_catalog.pg_type type",
                "    JOIN pg_catalog.pg_namespace namespace ON namespace.oid = type.typnamespace",
                "    LEFT JOIN pg_catalog.pg_class relation ON relation.oid = type.typrelid",
                f"    WHERE namespace.nspname IN ({managed_schema_literals})",
                "      AND type.typisdefined",
                "      AND ((type.typtype IN ('b', 'd', 'e', 'r', 'm') AND type.typelem = 0)",
                "           OR (type.typtype = 'c' AND relation.relkind = 'c'))",
                "  ) locked_object_census;",
                f"  IF cardinality(actual_objects) <> {len(expected_object_identities)}",
                f"     OR NOT (actual_objects @> {expected_object_array}",
                f"             AND actual_objects <@ {expected_object_array}) THEN",
                "    RAISE EXCEPTION 'locked object census is stale: %', actual_objects;",
                "  END IF;",
            ]
        )
        for observed_object in sorted(
            matrix.observed_objects,
            key=lambda obj: obj.identity,
        ):
            identity = observed_object.identity
            desired_owner = desired_object_owners[identity]
            accepted_object_owners = ", ".join(
                repr(owner) for owner in sorted({observed_object.owner, desired_owner})
            )
            if observed_object.catalog_kind in {
                "table",
                "view",
                "materialized_view",
                "sequence",
            }:
                relkind_condition = {
                    "table": "relation.relkind IN ('r', 'p')",
                    "view": "relation.relkind = 'v'",
                    "materialized_view": "relation.relkind = 'm'",
                    "sequence": "relation.relkind = 'S'",
                }[observed_object.catalog_kind]
                owner_query = [
                    "  SELECT owner.rolname INTO actual_owner",
                    "  FROM pg_catalog.pg_class relation",
                    "  JOIN pg_catalog.pg_namespace namespace ON namespace.oid = relation.relnamespace",
                    "  JOIN pg_catalog.pg_authid owner ON owner.oid = relation.relowner",
                    f"  WHERE namespace.nspname = '{observed_object.schema_ref}'",
                    f"    AND relation.relname = '{observed_object.object_ref}'",
                    f"    AND {relkind_condition};",
                ]
            elif observed_object.catalog_kind in {"function", "procedure"}:
                prokind = "p" if observed_object.catalog_kind == "procedure" else "f"
                owner_query = [
                    "  SELECT owner.rolname INTO actual_owner",
                    "  FROM pg_catalog.pg_proc procedure",
                    "  JOIN pg_catalog.pg_namespace namespace ON namespace.oid = procedure.pronamespace",
                    "  JOIN pg_catalog.pg_authid owner ON owner.oid = procedure.proowner",
                    f"  WHERE namespace.nspname = '{observed_object.schema_ref}'",
                    f"    AND procedure.proname = '{observed_object.object_ref}'",
                    f"    AND procedure.prokind = '{prokind}'",
                    "    AND '(' || pg_catalog.pg_get_function_identity_arguments(procedure.oid) || ')' = "
                    f"{_quote_sql_literal(observed_object.function_signature or '')};",
                ]
            else:
                owner_query = [
                    "  SELECT owner.rolname INTO actual_owner",
                    "  FROM pg_catalog.pg_type type",
                    "  JOIN pg_catalog.pg_namespace namespace ON namespace.oid = type.typnamespace",
                    "  JOIN pg_catalog.pg_authid owner ON owner.oid = type.typowner",
                    "  LEFT JOIN pg_catalog.pg_class relation ON relation.oid = type.typrelid",
                    f"  WHERE namespace.nspname = '{observed_object.schema_ref}'",
                    f"    AND type.typname = '{observed_object.object_ref}'",
                    "    AND type.typisdefined",
                    "    AND ((type.typtype IN ('b', 'd', 'e', 'r', 'm') AND type.typelem = 0)",
                    "         OR (type.typtype = 'c' AND relation.relkind = 'c'));",
                ]
            lines.extend(
                [
                    *owner_query,
                    f"  IF actual_owner IS NULL OR actual_owner NOT IN ({accepted_object_owners}) THEN",
                    "    RAISE EXCEPTION 'locked object owner census is stale for "
                    f"{observed_object.schema_ref}.{observed_object.object_ref}: %', actual_owner;",
                    "  END IF;",
                ]
            )
    lines.extend(["END", "$acl_evidence_guard$;", ""])

    for role_name, state in sorted(governed_states.items()):
        if not state.manage_attributes:
            continue
        quoted = _quote_identifier(role_name)
        attributes = role_attribute_clause(role_name)
        if role_name in globally_absent_roles:
            marker = _managed_provenance_marker(matrix, kind="role", name=role_name)
            lines.extend(
                [
                    "DO $acl_create_role$ BEGIN",
                    f"  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_authid WHERE rolname = '{role_name}') THEN",  # noqa: S608 -- strict identifier
                    f"    CREATE ROLE {quoted} {attributes};",
                    f"    COMMENT ON ROLE {quoted} IS '{marker}';",
                    "  END IF;",
                    "END $acl_create_role$;",
                ]
            )
        if phase is EnumApplicationDatabaseAclRenderPhase.FULL:
            lines.append(f"ALTER ROLE {quoted} {attributes};")
    membership_members = sorted(
        set(revocation_principals).union(owners)
        - set(matrix.retained_administrative_principals)
    )
    governed_parents = sorted(
        state.role for state in matrix.governed_role_states if state.manage_memberships
    )
    member_literals = ", ".join(repr(name) for name in membership_members)
    parent_literals = ", ".join(repr(name) for name in governed_parents)
    if phase is EnumApplicationDatabaseAclRenderPhase.FULL:
        lines.extend(
            [
                "DO $acl_membership$",
                "DECLARE membership_record record;",
                "BEGIN",
                "  FOR membership_record IN",
                "    SELECT parent.rolname AS parent_role,",
                "           member.rolname AS member_role,",
                "           grantor.rolname AS grantor_role",
                "    FROM pg_auth_members membership",
                "    JOIN pg_roles parent ON parent.oid = membership.roleid",
                "    JOIN pg_roles member ON member.oid = membership.member",
                "    JOIN pg_roles grantor ON grantor.oid = membership.grantor",
                f"    WHERE member.rolname IN ({member_literals})",
                f"       OR parent.rolname IN ({parent_literals})",
                "  LOOP",
                "    IF EXISTS (",
                "      SELECT 1 FROM pg_catalog.pg_auth_members current_membership",
                "      JOIN pg_catalog.pg_authid parent ON parent.oid = current_membership.roleid",
                "      JOIN pg_catalog.pg_authid member ON member.oid = current_membership.member",
                "      JOIN pg_catalog.pg_authid grantor ON grantor.oid = current_membership.grantor",
                "      WHERE parent.rolname = membership_record.parent_role",
                "        AND member.rolname = membership_record.member_role",
                "        AND grantor.rolname = membership_record.grantor_role",
                "    ) THEN",
                "      EXECUTE format('SET LOCAL ROLE %I', membership_record.grantor_role);",
                "      EXECUTE format(",
                "        'REVOKE %I FROM %I GRANTED BY %I CASCADE',",
                "        membership_record.parent_role,",
                "        membership_record.member_role,",
                "        membership_record.grantor_role",
                "      );",
                "      EXECUTE 'RESET ROLE';",
                "    END IF;",
                "  END LOOP;",
                "END",
                "$acl_membership$;",
            ]
        )
    for membership in matrix.allowed_memberships:
        lines.append(
            f"GRANT {_quote_identifier(membership.role)} TO "
            f"{_quote_identifier(membership.member)} WITH ADMIN "
            f"{'TRUE' if membership.admin_option else 'FALSE'}, INHERIT "
            f"{'TRUE' if membership.inherit_option else 'FALSE'}, SET "
            f"{'TRUE' if membership.set_option else 'FALSE'};"
        )

    database_literals = ", ".join(
        repr(database) for database in matrix.required_connect_databases
    )
    if phase is EnumApplicationDatabaseAclRenderPhase.FULL:
        lines.extend(
            [
                "DO $acl_database_grantors$",
                "DECLARE acl_record record;",
                "DECLARE grantee_sql text;",
                "BEGIN",
                "  FOR acl_record IN",
                "    SELECT database.datname AS database_name,",
                "           COALESCE(grantee.rolname, 'PUBLIC') AS grantee,",
                "           grantor.rolname AS grantor, acl.privilege_type",
                "    FROM pg_catalog.pg_database database",
                "    CROSS JOIN LATERAL pg_catalog.aclexplode(",
                "      COALESCE(database.datacl, pg_catalog.acldefault('d', database.datdba))",
                "    ) acl",
                "    LEFT JOIN pg_catalog.pg_authid grantee ON grantee.oid = acl.grantee",
                "    JOIN pg_catalog.pg_authid grantor ON grantor.oid = acl.grantor",
                f"    WHERE database.datname IN ({database_literals})",
                "      AND acl.grantee <> database.datdba",
                "  LOOP",
                "    grantee_sql := CASE WHEN acl_record.grantee = 'PUBLIC'",
                "                        THEN 'PUBLIC'",
                "                        ELSE format('%I', acl_record.grantee) END;",
                "    IF EXISTS (",
                "      SELECT 1 FROM pg_catalog.pg_database current_database_acl",
                "      CROSS JOIN LATERAL pg_catalog.aclexplode(",
                "        COALESCE(current_database_acl.datacl,",
                "                 pg_catalog.acldefault('d', current_database_acl.datdba))",
                "      ) current_acl",
                "      LEFT JOIN pg_catalog.pg_authid current_grantee ON current_grantee.oid = current_acl.grantee",
                "      JOIN pg_catalog.pg_authid current_grantor ON current_grantor.oid = current_acl.grantor",
                "      WHERE current_database_acl.datname = acl_record.database_name",
                "        AND COALESCE(current_grantee.rolname, 'PUBLIC') = acl_record.grantee",
                "        AND current_grantor.rolname = acl_record.grantor",
                "        AND current_acl.privilege_type = acl_record.privilege_type",
                "    ) THEN",
                "      EXECUTE format('SET LOCAL ROLE %I', acl_record.grantor);",
                "      EXECUTE format(",
                "        'REVOKE %s ON DATABASE %I FROM %s GRANTED BY %I CASCADE',",
                "        acl_record.privilege_type, acl_record.database_name,",
                "        grantee_sql, acl_record.grantor",
                "      );",
                "      EXECUTE 'RESET ROLE';",
                "    END IF;",
                "  END LOOP;",
                "END",
                "$acl_database_grantors$;",
            ]
        )

    database_rows = [
        row
        for row in matrix.rows
        if row.object_type is EnumDatabaseGrantObjectType.DATABASE
    ]
    for physical_database in matrix.required_connect_databases:
        connection_grantees = ", ".join(
            ["PUBLIC", *(_quote_identifier(name) for name in revocation_principals)]
        )
        quoted_database = _quote_identifier(physical_database)
        if phase is EnumApplicationDatabaseAclRenderPhase.FULL:
            lines.append(
                f"REVOKE ALL PRIVILEGES ON DATABASE {quoted_database} FROM "
                f"{connection_grantees} CASCADE;"
            )
        for principal in matrix.allowed_connect_principals[physical_database]:
            lines.append(
                f"GRANT CONNECT ON DATABASE {quoted_database} TO "
                f"{_quote_identifier(principal)};"
            )
    database_grantees = ", ".join(
        ["PUBLIC", *(_quote_identifier(name) for name in revocation_principals)]
    )
    for physical_database in (
        sorted({row.physical_database for row in database_rows})
        if phase is EnumApplicationDatabaseAclRenderPhase.FULL
        else ()
    ):
        quoted_database = _quote_identifier(physical_database)
        database_ref = next(
            row.database_ref
            for row in database_rows
            if row.physical_database == physical_database
        )
        lines.append(
            f"ALTER DATABASE {quoted_database} OWNER TO "
            f"{_quote_identifier(matrix.database_owners[database_ref])};"
        )
        lines.append(
            f"REVOKE ALL PRIVILEGES ON DATABASE {quoted_database} FROM "
            f"{database_grantees} CASCADE;"
        )
        for row in database_rows:
            if row.physical_database != physical_database or not row.privileges:
                continue
            privileges = ", ".join(p.value for p in row.privileges)
            lines.append(
                f"GRANT {privileges} ON DATABASE {quoted_database} TO "
                f"{_quote_identifier(row.principal)};"
            )

    schema_rows = [
        row
        for row in matrix.rows
        if row.object_type is EnumDatabaseGrantObjectType.SCHEMA
    ]
    schema_literals = ", ".join(
        repr(schema_name)
        for schema_name in sorted(
            {"public", *(schema_name for _, schema_name in schema_owners)}
        )
    )
    if phase is EnumApplicationDatabaseAclRenderPhase.FULL:
        lines.extend(
            [
                "DO $acl_schema_grantors$",
                "DECLARE acl_record record;",
                "DECLARE grantee_sql text;",
                "BEGIN",
                "  FOR acl_record IN",
                "    SELECT namespace.nspname AS schema_name,",
                "           COALESCE(grantee.rolname, 'PUBLIC') AS grantee,",
                "           grantor.rolname AS grantor, acl.privilege_type",
                "    FROM pg_catalog.pg_namespace namespace",
                "    CROSS JOIN LATERAL pg_catalog.aclexplode(",
                "      COALESCE(namespace.nspacl, pg_catalog.acldefault('n', namespace.nspowner))",
                "    ) acl",
                "    LEFT JOIN pg_catalog.pg_authid grantee ON grantee.oid = acl.grantee",
                "    JOIN pg_catalog.pg_authid grantor ON grantor.oid = acl.grantor",
                f"    WHERE namespace.nspname IN ({schema_literals})",
                "      AND acl.grantee <> namespace.nspowner",
                "  LOOP",
                "    grantee_sql := CASE WHEN acl_record.grantee = 'PUBLIC'",
                "                        THEN 'PUBLIC'",
                "                        ELSE format('%I', acl_record.grantee) END;",
                "    IF EXISTS (",
                "      SELECT 1 FROM pg_catalog.pg_namespace current_namespace",
                "      CROSS JOIN LATERAL pg_catalog.aclexplode(",
                "        COALESCE(current_namespace.nspacl,",
                "                 pg_catalog.acldefault('n', current_namespace.nspowner))",
                "      ) current_acl",
                "      LEFT JOIN pg_catalog.pg_authid current_grantee ON current_grantee.oid = current_acl.grantee",
                "      JOIN pg_catalog.pg_authid current_grantor ON current_grantor.oid = current_acl.grantor",
                "      WHERE current_namespace.nspname = acl_record.schema_name",
                "        AND COALESCE(current_grantee.rolname, 'PUBLIC') = acl_record.grantee",
                "        AND current_grantor.rolname = acl_record.grantor",
                "        AND current_acl.privilege_type = acl_record.privilege_type",
                "    ) THEN",
                "      EXECUTE format('SET LOCAL ROLE %I', acl_record.grantor);",
                "      EXECUTE format(",
                "        'REVOKE %s ON SCHEMA %I FROM %s GRANTED BY %I CASCADE',",
                "        acl_record.privilege_type, acl_record.schema_name,",
                "        grantee_sql, acl_record.grantor",
                "      );",
                "      EXECUTE 'RESET ROLE';",
                "    END IF;",
                "  END LOOP;",
                "END",
                "$acl_schema_grantors$;",
            ]
        )
    for (database_ref, schema_name), owner in sorted(schema_owners.items()):
        if schema_name is None:
            continue
        quoted_schema = _quote_identifier(schema_name)
        quoted_owner = _quote_identifier(owner)
        if schema_name in matrix.absent_schemas[database_ref]:
            marker = _managed_provenance_marker(
                matrix,
                kind="schema",
                name=schema_name,
            )
            lines.extend(
                [
                    "DO $acl_create_schema$ BEGIN",
                    "  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace",
                    f"                 WHERE nspname = '{schema_name}') THEN",
                    f"    CREATE SCHEMA {quoted_schema} AUTHORIZATION {quoted_owner};",
                    f"    COMMENT ON SCHEMA {quoted_schema} IS '{marker}';",
                    "  END IF;",
                    "END $acl_create_schema$;",
                ]
            )
        if phase is EnumApplicationDatabaseAclRenderPhase.FULL:
            lines.extend(
                [
                    f"ALTER SCHEMA {quoted_schema} OWNER TO {quoted_owner};",
                    f"REVOKE ALL PRIVILEGES ON SCHEMA {quoted_schema} FROM "
                    f"{database_grantees} CASCADE;",
                ]
            )
        for row in schema_rows:
            if (
                row.database_ref != database_ref
                or row.schema_ref != schema_name
                or not row.privileges
            ):
                continue
            privileges = ", ".join(p.value for p in row.privileges)
            lines.append(
                f"GRANT {privileges} ON SCHEMA {quoted_schema} TO "
                f"{_quote_identifier(row.principal)};"
            )
    if phase is EnumApplicationDatabaseAclRenderPhase.FULL:
        lines.append(
            f'REVOKE ALL PRIVILEGES ON SCHEMA "public" FROM {database_grantees} CASCADE;'
        )
    lines.append("")

    rows_by_object: dict[
        tuple[str, str, EnumDatabaseGrantObjectType, str, str | None],
        list[ModelApplicationDatabaseAclRow],
    ] = defaultdict(list)
    for row in matrix.rows:
        if row.object_type in _OBJECT_TYPES:
            rows_by_object[
                (
                    row.database_ref,
                    row.schema_ref or "",
                    row.object_type,
                    row.object_ref or "",
                    row.function_signature,
                )
            ].append(row)
    all_grantees = ", ".join(
        ["PUBLIC", *(_quote_identifier(name) for name in revocation_principals)]
    )
    ownership_order = {
        EnumDatabaseGrantObjectType.TABLE: 0,
        EnumDatabaseGrantObjectType.SEQUENCE: 1,
        EnumDatabaseGrantObjectType.TYPE: 2,
        EnumDatabaseGrantObjectType.FUNCTION: 3,
    }
    for obj in sorted(
        (matrix.objects if phase is EnumApplicationDatabaseAclRenderPhase.FULL else ()),
        key=lambda item: (
            ownership_order[item.object_type],
            item.database_ref,
            item.schema_ref,
            item.object_ref,
            item.function_signature or "",
        ),
    ):
        owner_keyword, acl_keyword, target = _sql_object_target(obj)
        dynamic_target = target.replace("%", "%%")
        dynamic_revoke_pattern = (
            f"REVOKE %s ON {acl_keyword} {dynamic_target} FROM %s GRANTED BY %I CASCADE"
        )
        if obj.object_type in {
            EnumDatabaseGrantObjectType.TABLE,
            EnumDatabaseGrantObjectType.SEQUENCE,
        }:
            default_acl_kind = (
                "s" if obj.object_type is EnumDatabaseGrantObjectType.SEQUENCE else "r"
            )
            relkind_condition = {
                "table": "object.relkind IN ('r', 'p')",
                "view": "object.relkind = 'v'",
                "materialized_view": "object.relkind = 'm'",
                "sequence": "object.relkind = 'S'",
            }[obj.catalog_kind]
            catalog_acl_lines = [
                "    FROM pg_catalog.pg_class object",
                "    JOIN pg_catalog.pg_namespace namespace ON namespace.oid = object.relnamespace",
                "    CROSS JOIN LATERAL pg_catalog.aclexplode(",
                f"      COALESCE(object.relacl, pg_catalog.acldefault('{default_acl_kind}', object.relowner))",
                "    ) acl",
                "    LEFT JOIN pg_catalog.pg_authid grantee ON grantee.oid = acl.grantee",
                "    JOIN pg_catalog.pg_authid grantor ON grantor.oid = acl.grantor",
                f"    WHERE namespace.nspname = '{obj.schema_ref}'",
                f"      AND object.relname = '{obj.object_ref}'",
                f"      AND {relkind_condition}",
                "      AND acl.grantee <> object.relowner",
            ]
        elif obj.object_type is EnumDatabaseGrantObjectType.FUNCTION:
            prokind = "p" if obj.catalog_kind == "procedure" else "f"
            catalog_acl_lines = [
                "    FROM pg_catalog.pg_proc object",
                "    JOIN pg_catalog.pg_namespace namespace ON namespace.oid = object.pronamespace",
                "    CROSS JOIN LATERAL pg_catalog.aclexplode(",
                "      COALESCE(object.proacl, pg_catalog.acldefault('f', object.proowner))",
                "    ) acl",
                "    LEFT JOIN pg_catalog.pg_authid grantee ON grantee.oid = acl.grantee",
                "    JOIN pg_catalog.pg_authid grantor ON grantor.oid = acl.grantor",
                f"    WHERE namespace.nspname = '{obj.schema_ref}'",
                f"      AND object.proname = '{obj.object_ref}'",
                f"      AND object.prokind = '{prokind}'",
                "      AND '(' || pg_catalog.pg_get_function_identity_arguments(object.oid) || ')' = "
                f"{_quote_sql_literal(obj.function_signature or '')}",
                "      AND acl.grantee <> object.proowner",
            ]
        else:
            catalog_acl_lines = [
                "    FROM pg_catalog.pg_type object",
                "    JOIN pg_catalog.pg_namespace namespace ON namespace.oid = object.typnamespace",
                "    CROSS JOIN LATERAL pg_catalog.aclexplode(",
                "      COALESCE(object.typacl, pg_catalog.acldefault('T', object.typowner))",
                "    ) acl",
                "    LEFT JOIN pg_catalog.pg_authid grantee ON grantee.oid = acl.grantee",
                "    JOIN pg_catalog.pg_authid grantor ON grantor.oid = acl.grantor",
                f"    WHERE namespace.nspname = '{obj.schema_ref}'",
                f"      AND object.typname = '{obj.object_ref}'",
                "      AND acl.grantee <> object.typowner",
            ]
        lines.extend(
            [
                "DO $acl_object_grantors$",
                "DECLARE acl_record record;",
                "DECLARE grantee_sql text;",
                "DECLARE temporary_schema_usage boolean;",
                "BEGIN",
                "  FOR acl_record IN",
                "    SELECT COALESCE(grantee.rolname, 'PUBLIC') AS grantee,",
                "           grantor.rolname AS grantor, acl.privilege_type",
                *catalog_acl_lines,
                "  LOOP",
                "    grantee_sql := CASE WHEN acl_record.grantee = 'PUBLIC'",
                "                        THEN 'PUBLIC'",
                "                        ELSE format('%I', acl_record.grantee) END;",
                "    IF EXISTS (",
                "      SELECT 1",
                *catalog_acl_lines,
                "        AND COALESCE(grantee.rolname, 'PUBLIC') = acl_record.grantee",
                "        AND grantor.rolname = acl_record.grantor",
                "        AND acl.privilege_type = acl_record.privilege_type",
                "    ) THEN",
                "      SELECT NOT pg_catalog.has_schema_privilege(",
                f"        acl_record.grantor, '{obj.schema_ref}', 'USAGE'",
                "      ) INTO temporary_schema_usage;",
                "      IF temporary_schema_usage THEN",
                "        EXECUTE format(",
                f"          'GRANT USAGE ON SCHEMA %I TO %I', '{obj.schema_ref}',",
                "          acl_record.grantor",
                "        );",
                "      END IF;",
                "      EXECUTE format('SET LOCAL ROLE %I', acl_record.grantor);",
                "      EXECUTE format(",
                f"        {_quote_sql_literal(dynamic_revoke_pattern)},",
                "        acl_record.privilege_type, grantee_sql, acl_record.grantor",
                "      );",
                "      EXECUTE 'RESET ROLE';",
                "      IF temporary_schema_usage THEN",
                "        EXECUTE format(",
                f"          'REVOKE USAGE ON SCHEMA %I FROM %I CASCADE', '{obj.schema_ref}',",
                "          acl_record.grantor",
                "        );",
                "      END IF;",
                "    END IF;",
                "  END LOOP;",
                "END",
                "$acl_object_grantors$;",
            ]
        )
        lines.append(
            f"ALTER {owner_keyword} {target} OWNER TO {_quote_identifier(obj.owner)};"
        )
        lines.append(
            f"REVOKE ALL PRIVILEGES ON {acl_keyword} {target} FROM "
            f"{all_grantees} CASCADE;"
        )
        if obj.object_type is EnumDatabaseGrantObjectType.TABLE:
            lines.extend(
                [
                    "DO $acl_column_grantors$",
                    "DECLARE acl_record record;",
                    "DECLARE grantee_sql text;",
                    "DECLARE temporary_schema_usage boolean;",
                    "BEGIN",
                    "  FOR acl_record IN",
                    "    SELECT attribute.attname AS column_name,",
                    "           COALESCE(grantee.rolname, 'PUBLIC') AS grantee,",
                    "           grantor.rolname AS grantor, acl.privilege_type",
                    "    FROM pg_catalog.pg_attribute attribute",
                    "    JOIN pg_catalog.pg_class relation ON relation.oid = attribute.attrelid",
                    "    JOIN pg_catalog.pg_namespace namespace ON namespace.oid = relation.relnamespace",
                    "    CROSS JOIN LATERAL pg_catalog.aclexplode(attribute.attacl) acl",
                    "    LEFT JOIN pg_catalog.pg_authid grantee ON grantee.oid = acl.grantee",
                    "    JOIN pg_catalog.pg_authid grantor ON grantor.oid = acl.grantor",
                    f"    WHERE namespace.nspname = '{obj.schema_ref}'",
                    f"      AND relation.relname = '{obj.object_ref}'",
                    "      AND attribute.attnum > 0 AND NOT attribute.attisdropped",
                    "      AND attribute.attacl IS NOT NULL",
                    "  LOOP",
                    "    grantee_sql := CASE WHEN acl_record.grantee = 'PUBLIC'",
                    "                        THEN 'PUBLIC'",
                    "                        ELSE format('%I', acl_record.grantee) END;",
                    "    IF EXISTS (",
                    "      SELECT 1 FROM pg_catalog.pg_attribute current_attribute",
                    "      JOIN pg_catalog.pg_class current_relation ON current_relation.oid = current_attribute.attrelid",
                    "      JOIN pg_catalog.pg_namespace current_namespace ON current_namespace.oid = current_relation.relnamespace",
                    "      CROSS JOIN LATERAL pg_catalog.aclexplode(current_attribute.attacl) current_acl",
                    "      LEFT JOIN pg_catalog.pg_authid current_grantee ON current_grantee.oid = current_acl.grantee",
                    "      JOIN pg_catalog.pg_authid current_grantor ON current_grantor.oid = current_acl.grantor",
                    f"      WHERE current_namespace.nspname = '{obj.schema_ref}'",
                    f"        AND current_relation.relname = '{obj.object_ref}'",
                    "        AND current_attribute.attname = acl_record.column_name",
                    "        AND COALESCE(current_grantee.rolname, 'PUBLIC') = acl_record.grantee",
                    "        AND current_grantor.rolname = acl_record.grantor",
                    "        AND current_acl.privilege_type = acl_record.privilege_type",
                    "    ) THEN",
                    "      SELECT NOT pg_catalog.has_schema_privilege(",
                    f"        acl_record.grantor, '{obj.schema_ref}', 'USAGE'",
                    "      ) INTO temporary_schema_usage;",
                    "      IF temporary_schema_usage THEN",
                    "        EXECUTE format(",
                    f"          'GRANT USAGE ON SCHEMA %I TO %I', '{obj.schema_ref}',",
                    "          acl_record.grantor",
                    "        );",
                    "      END IF;",
                    "      EXECUTE format('SET LOCAL ROLE %I', acl_record.grantor);",
                    "      EXECUTE format(",
                    "        'REVOKE %s (%I) ON TABLE %I.%I FROM %s GRANTED BY %I CASCADE',",
                    "        acl_record.privilege_type, acl_record.column_name,",
                    f"        '{obj.schema_ref}', '{obj.object_ref}', grantee_sql, acl_record.grantor",
                    "      );",
                    "      EXECUTE 'RESET ROLE';",
                    "      IF temporary_schema_usage THEN",
                    "        EXECUTE format(",
                    f"          'REVOKE USAGE ON SCHEMA %I FROM %I CASCADE', '{obj.schema_ref}',",
                    "          acl_record.grantor",
                    "        );",
                    "      END IF;",
                    "    END IF;",
                    "  END LOOP;",
                    "END",
                    "$acl_column_grantors$;",
                    "DO $acl_columns$",
                    "DECLARE column_name text;",
                    "BEGIN",
                    "  FOR column_name IN",
                    "    SELECT attribute.attname",
                    "    FROM pg_attribute attribute",
                    "    JOIN pg_class relation ON relation.oid = attribute.attrelid",
                    "    JOIN pg_namespace namespace ON namespace.oid = relation.relnamespace",
                    f"    WHERE namespace.nspname = '{obj.schema_ref}'",
                    f"      AND relation.relname = '{obj.object_ref}'",
                    "      AND attribute.attnum > 0 AND NOT attribute.attisdropped",
                    "  LOOP",
                    "    EXECUTE format(",
                    f"      'REVOKE ALL PRIVILEGES (%I) ON TABLE %I.%I FROM {all_grantees} CASCADE',",
                    f"      column_name, '{obj.schema_ref}', '{obj.object_ref}'",
                    "    );",
                    "  END LOOP;",
                    "END",
                    "$acl_columns$;",
                ]
            )
        for row in rows_by_object[obj.identity]:
            if not row.privileges:
                continue
            privileges = ", ".join(p.value for p in row.privileges)
            lines.append(
                f"GRANT {privileges} ON {acl_keyword} {target} TO "
                f"{_quote_identifier(row.principal)};"
            )

    default_keyword = {
        EnumDatabaseGrantObjectType.TABLE: "TABLES",
        EnumDatabaseGrantObjectType.SEQUENCE: "SEQUENCES",
        EnumDatabaseGrantObjectType.FUNCTION: "FUNCTIONS",
        EnumDatabaseGrantObjectType.TYPE: "TYPES",
    }
    default_owner_scope = {row.owner for row in matrix.default_privileges}
    if phase is EnumApplicationDatabaseAclRenderPhase.SCAFFOLD:
        # P1 may establish deny-by-default only for roles it creates from exact
        # absence evidence. Existing owners' legacy defaults remain untouched.
        default_owner_scope.intersection_update(globally_absent_roles)
    phase_default_rows = tuple(
        row for row in matrix.default_privileges if row.owner in default_owner_scope
    )
    global_default_identities = sorted(
        {(row.owner, row.object_type) for row in phase_default_rows},
        key=lambda item: (item[0], item[1].value),
    )
    default_owner_literals = ", ".join(
        _quote_sql_literal(owner) for owner in sorted(default_owner_scope)
    )
    managed_default_schema_literals = ", ".join(
        _quote_sql_literal(schema_name)
        for schema_name in sorted({row.schema_ref for row in phase_default_rows})
    )
    if phase_default_rows:
        lines.extend(
            [
                "DO $acl_default_grantor_guard$ BEGIN",
                "  IF EXISTS (",
                "    SELECT 1 FROM pg_catalog.pg_default_acl defaults",
                "    JOIN pg_catalog.pg_authid owner ON owner.oid = defaults.defaclrole",
                "    LEFT JOIN pg_catalog.pg_namespace namespace ON namespace.oid = defaults.defaclnamespace",
                "    CROSS JOIN LATERAL pg_catalog.aclexplode(defaults.defaclacl) acl",
                "    LEFT JOIN pg_catalog.pg_authid grantee ON grantee.oid = acl.grantee",
                "    JOIN pg_catalog.pg_authid grantor ON grantor.oid = acl.grantor",
                f"    WHERE owner.rolname IN ({default_owner_literals})",
                "      AND (defaults.defaclnamespace = 0",
                f"           OR namespace.nspname IN ({managed_default_schema_literals}))",
                "      AND grantor.oid <> defaults.defaclrole",
                "  ) THEN",
                "    RAISE EXCEPTION 'default ACL contains an unsupported alternate grantor';",
                "  END IF;",
                "END $acl_default_grantor_guard$;",
                "DO $acl_default_actual_rows$",
                "DECLARE acl_record record;",
                "DECLARE grantee_sql text;",
                "BEGIN",
                "  FOR acl_record IN",
                "    SELECT owner.rolname AS owner, namespace.nspname AS schema_name,",
                "           CASE defaults.defaclobjtype",
                "             WHEN 'r' THEN 'TABLES'",
                "             WHEN 'S' THEN 'SEQUENCES'",
                "             WHEN 'f' THEN 'FUNCTIONS'",
                "             WHEN 'T' THEN 'TYPES'",
                "           END AS object_keyword,",
                "           COALESCE(grantee.rolname, 'PUBLIC') AS grantee,",
                "           acl.privilege_type",
                "    FROM pg_catalog.pg_default_acl defaults",
                "    JOIN pg_catalog.pg_authid owner ON owner.oid = defaults.defaclrole",
                "    LEFT JOIN pg_catalog.pg_namespace namespace ON namespace.oid = defaults.defaclnamespace",
                "    CROSS JOIN LATERAL pg_catalog.aclexplode(defaults.defaclacl) acl",
                "    LEFT JOIN pg_catalog.pg_authid grantee ON grantee.oid = acl.grantee",
                f"    WHERE owner.rolname IN ({default_owner_literals})",
                "      AND (defaults.defaclnamespace = 0",
                f"           OR namespace.nspname IN ({managed_default_schema_literals}))",
                "      AND acl.grantee <> defaults.defaclrole",
                "  LOOP",
                "    grantee_sql := CASE WHEN acl_record.grantee = 'PUBLIC'",
                "                        THEN 'PUBLIC'",
                "                        ELSE format('%I', acl_record.grantee) END;",
                "    EXECUTE format('SET LOCAL ROLE %I', acl_record.owner);",
                "    IF acl_record.schema_name IS NULL THEN",
                "      EXECUTE format(",
                "        'ALTER DEFAULT PRIVILEGES FOR ROLE %I REVOKE %s ON %s FROM %s CASCADE',",
                "        acl_record.owner, acl_record.privilege_type,",
                "        acl_record.object_keyword, grantee_sql",
                "      );",
                "    ELSE",
                "      EXECUTE format(",
                "        'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA %I REVOKE %s ON %s FROM %s CASCADE',",
                "        acl_record.owner, acl_record.schema_name,",
                "        acl_record.privilege_type, acl_record.object_keyword, grantee_sql",
                "      );",
                "    END IF;",
                "    EXECUTE 'RESET ROLE';",
                "  END LOOP;",
                "END",
                "$acl_default_actual_rows$;",
            ]
        )
    default_revocation_principals = (PUBLIC_PRINCIPAL, *revocation_principals)
    for owner, object_type in global_default_identities:
        for principal in default_revocation_principals:
            grantee = (
                "PUBLIC"
                if principal == PUBLIC_PRINCIPAL
                else _quote_identifier(principal)
            )
            lines.append(
                "ALTER DEFAULT PRIVILEGES FOR ROLE "
                f"{_quote_identifier(owner)} REVOKE ALL PRIVILEGES ON "
                f"{default_keyword[object_type]} FROM {grantee} CASCADE;"
            )
    schema_default_identities = sorted(
        {(row.owner, row.schema_ref, row.object_type) for row in phase_default_rows},
        key=lambda item: (item[0], item[1], item[2].value),
    )
    for owner, schema_ref, object_type in schema_default_identities:
        for principal in default_revocation_principals:
            grantee = (
                "PUBLIC"
                if principal == PUBLIC_PRINCIPAL
                else _quote_identifier(principal)
            )
            lines.append(
                "ALTER DEFAULT PRIVILEGES FOR ROLE "
                f"{_quote_identifier(owner)} IN SCHEMA "
                f"{_quote_identifier(schema_ref)} REVOKE ALL PRIVILEGES ON "
                f"{default_keyword[object_type]} FROM {grantee} CASCADE;"
            )
    lines.append("COMMIT;")
    return "\n".join(lines) + "\n"


__all__ = [
    "PUBLIC_PRINCIPAL",
    "build_application_database_acl_matrix",
    "render_application_database_acl_sql",
    "validate_application_database_acl_scaffold",
    "validate_application_database_acl_matrix",
]
