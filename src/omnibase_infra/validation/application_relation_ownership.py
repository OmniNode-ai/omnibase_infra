# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed global application-relation ownership validation (OMN-15418).

Node contracts and service migration manifests reuse core's
``ModelDbTableDeclaration`` as their table-location source of truth. This module
projects those distributed declarations into one validation report and proves
that a supplied live inventory has exactly one writer/owner for every relation.
Readers remain explicit and never count as owners.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.models.contracts.subcontracts.model_db_table_declaration import (
    ModelDbTableDeclaration,
)
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.errors import ModelOnexError
from omnibase_infra.validation.enums.enum_application_database_object_kind import (
    EnumApplicationDatabaseObjectKind,
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
from omnibase_infra.validation.enums.enum_application_relation_violation import (
    EnumApplicationRelationViolation,
)
from omnibase_infra.validation.models.model_application_relation_declaration import (
    ModelApplicationRelationDeclaration,
)
from omnibase_infra.validation.models.model_application_relation_evidence_inventory import (
    ModelApplicationRelationEvidenceInventory,
)
from omnibase_infra.validation.models.model_application_relation_inventory import (
    ModelApplicationRelationInventory,
)
from omnibase_infra.validation.models.model_application_relation_ownership_report import (
    ModelApplicationRelationOwnershipReport,
)
from omnibase_infra.validation.models.model_application_relation_violation import (
    ModelApplicationRelationViolation,
)
from omnibase_infra.validation.models.model_live_application_relation import (
    ModelLiveApplicationRelation,
    RelationIdentity,
)
from omnibase_infra.validation.models.model_migration_ownership_manifest import (
    ModelMigrationOwnershipManifest,
)
from omnibase_infra.validation.models.model_node_ownership_document import (
    ModelNodeOwnershipDocument,
)


def _load_yaml_mapping(path: Path) -> Mapping[str, object]:
    # Why: PyYAML is a runtime dependency without complete inline typing.
    import yaml

    raw: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Expected YAML mapping in {path}, got {type(raw).__name__}")
    return raw


def load_application_relation_inventory(
    path: Path,
) -> ModelApplicationRelationInventory:
    """Load a minimal inventory or project the exact rich OMN-15423 evidence."""
    raw = _load_yaml_mapping(path)
    if "inventory_projection" not in raw:
        return ModelApplicationRelationInventory.model_validate(raw)

    evidence = ModelApplicationRelationEvidenceInventory.model_validate(raw)
    kind_map = {
        EnumApplicationInventoryObjectKind.TABLE: EnumApplicationRelationKind.TABLE,
        EnumApplicationInventoryObjectKind.VIEW: EnumApplicationRelationKind.VIEW,
        EnumApplicationInventoryObjectKind.MATERIALIZED_VIEW: (
            EnumApplicationRelationKind.MATERIALIZED_VIEW
        ),
        EnumApplicationInventoryObjectKind.FUNCTION: (
            EnumApplicationRelationKind.FUNCTION
        ),
    }
    live_relations = tuple(
        ModelLiveApplicationRelation(
            name=relation.name,
            database_ref=evidence.database_ref,
            schema=relation.target_schema,
            kind=kind_map[relation.kind],
            purpose=_inventory_relation_purpose(relation.name),
            domain=relation.domain,
        )
        for relation in evidence.relations
        if relation.kind in kind_map
    )
    excluded_database_objects = tuple(
        relation.name
        for relation in evidence.relations
        if relation.kind not in kind_map
    )
    return ModelApplicationRelationInventory(
        schema_version=evidence.schema_version,
        relations=live_relations,
        completion_status=evidence.completion_status,
        blocked_relations=evidence.blocked_relations,
        retained_live_census=evidence.retained_live_census,
        runtime_evidence={
            "full_day_datname_usename_activity": (
                evidence.runtime_evidence.full_day_datname_usename_activity
            ),
            "live_catalog_parity": evidence.runtime_evidence.live_catalog_parity,
        },
        source_relation_count=len(evidence.relations),
        excluded_database_objects=excluded_database_objects,
    )


def load_service_ownership_manifest(path: Path) -> ModelMigrationOwnershipManifest:
    """Load an OMN-15423 service manifest with strict typed table locations."""
    return ModelMigrationOwnershipManifest.model_validate(_load_yaml_mapping(path))


def _load_node_ownership_document(path: Path) -> ModelNodeOwnershipDocument:
    return ModelNodeOwnershipDocument.model_validate(_load_yaml_mapping(path))


def _table_purpose(table: ModelDbTableDeclaration) -> EnumApplicationRelationPurpose:
    normalized_role = table.role.strip().lower()
    if normalized_role == "migration_ledger" or normalized_role.endswith(
        "_migration_ledger"
    ):
        return EnumApplicationRelationPurpose.MIGRATION_LEDGER
    return EnumApplicationRelationPurpose.DATA


def _inventory_relation_purpose(name: str) -> EnumApplicationRelationPurpose:
    """Classify conventional migration-ledger relation names from live evidence."""
    normalized_name = name.strip().lower()
    if normalized_name in {"alembic_version", "migrations_log", "schema_migrations"}:
        return EnumApplicationRelationPurpose.MIGRATION_LEDGER
    if normalized_name.endswith("_schema_migrations"):
        return EnumApplicationRelationPurpose.MIGRATION_LEDGER
    return EnumApplicationRelationPurpose.DATA


def _lookup_violation_code(exc: ValueError) -> EnumApplicationRelationViolation:
    if str(exc).startswith("Unknown database_ref"):
        return EnumApplicationRelationViolation.UNKNOWN_DATABASE
    return EnumApplicationRelationViolation.UNKNOWN_SCHEMA


def _topology_domain(
    topology: ModelDeploymentTopology,
    table: ModelDbTableDeclaration,
    violations: list[ModelApplicationRelationViolation],
    source_path: Path,
) -> EnumDatabaseSchemaDomain | None:
    try:
        return topology.table_domain(table)
    except ValueError as exc:
        violations.append(
            ModelApplicationRelationViolation(
                code=_lookup_violation_code(exc),
                message=str(exc),
                relation_name=table.name,
                source_paths=(str(source_path),),
            )
        )
        return None


def _declaration_from_table(
    *,
    table: ModelDbTableDeclaration,
    authority: str,
    topology: ModelDeploymentTopology,
    source_path: Path,
    violations: list[ModelApplicationRelationViolation],
) -> ModelApplicationRelationDeclaration:
    is_reader = table.access == "read"
    return ModelApplicationRelationDeclaration(
        name=table.name,
        database_ref=table.database_ref,
        schema=table.schema,
        kind=EnumApplicationRelationKind.TABLE,
        purpose=_table_purpose(table),
        domain=_topology_domain(topology, table, violations, source_path),
        owner_declaration=None if is_reader else authority,
        readers=(authority,) if is_reader else (),
        access=table.access,
        role=table.role,
        source_path=str(source_path),
    )


def _status_is_blocking(status: str) -> bool:
    return status.strip().lower() not in {"complete", "pass", "passed", "verified"}


def _append_inventory_evidence_violations(
    inventory: ModelApplicationRelationInventory,
    violations: list[ModelApplicationRelationViolation],
) -> None:
    """Retain rich inventory blockers in the global fail-closed report."""
    if inventory.completion_status and _status_is_blocking(inventory.completion_status):
        violations.append(
            ModelApplicationRelationViolation(
                code=EnumApplicationRelationViolation.INCOMPLETE_CENSUS,
                message=(
                    "Application inventory completion_status is "
                    f"{inventory.completion_status!r}"
                ),
            )
        )

    for evidence_name, runtime_status in inventory.runtime_evidence.items():
        if _status_is_blocking(runtime_status.status):
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.INCOMPLETE_CENSUS,
                    message=(
                        f"Application inventory runtime evidence {evidence_name!r} "
                        f"is {runtime_status.status!r}: "
                        f"{runtime_status.reason or 'no reason supplied'}"
                    ),
                )
            )

    census = inventory.retained_live_census
    if census is not None and census.parity_status is not None:
        if _status_is_blocking(census.parity_status):
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.INCOMPLETE_CENSUS,
                    message=(
                        f"Application inventory retained census is "
                        f"{census.parity_status!r}: "
                        f"{census.reason or 'no reason supplied'}"
                    ),
                )
            )

    for blocked in inventory.blocked_relations:
        violations.append(
            ModelApplicationRelationViolation(
                code=EnumApplicationRelationViolation.BLOCKED_RELATION,
                message=f"Blocked {blocked.kind.value} {blocked.name!r}: {blocked.reason}",
                relation_name=blocked.name,
            )
        )


def _append_service_declarations(
    *,
    path: Path,
    manifest: ModelMigrationOwnershipManifest,
    topology: ModelDeploymentTopology,
    declarations: list[ModelApplicationRelationDeclaration],
    violations: list[ModelApplicationRelationViolation],
) -> None:
    authority = manifest.owner_declaration or f"service:{manifest.service}"
    if manifest.completion_status and _status_is_blocking(manifest.completion_status):
        violations.append(
            ModelApplicationRelationViolation(
                code=EnumApplicationRelationViolation.INCOMPLETE_CENSUS,
                message=(
                    f"Service manifest {manifest.service!r} completion_status is "
                    f"{manifest.completion_status!r}"
                ),
                source_paths=(str(path),),
            )
        )

    for evidence_name, runtime_status in manifest.runtime_evidence.items():
        if _status_is_blocking(runtime_status.status):
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.INCOMPLETE_CENSUS,
                    message=(
                        f"Service manifest {manifest.service!r} runtime evidence "
                        f"{evidence_name!r} is {runtime_status.status!r}: "
                        f"{runtime_status.reason or 'no reason supplied'}"
                    ),
                    source_paths=(str(path),),
                )
            )

    census = manifest.retained_live_census
    if census is not None and census.parity_status is not None:
        if _status_is_blocking(census.parity_status):
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.INCOMPLETE_CENSUS,
                    message=(
                        f"Service manifest {manifest.service!r} retained census is "
                        f"{census.parity_status!r}: {census.reason or 'no reason supplied'}"
                    ),
                    source_paths=(str(path),),
                )
            )

    for blocked in manifest.blocked_relations:
        violations.append(
            ModelApplicationRelationViolation(
                code=EnumApplicationRelationViolation.BLOCKED_RELATION,
                message=f"Blocked {blocked.kind.value} {blocked.name!r}: {blocked.reason}",
                relation_name=blocked.name,
                source_paths=(str(path),),
            )
        )

    tables_by_name: dict[str, list[ModelDbTableDeclaration]] = defaultdict(list)
    for table in manifest.db_io.db_tables:
        tables_by_name[table.name].append(table)
        if table.database_ref != manifest.target_database_ref:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.CONFLICTING_LOCATION,
                    message=(
                        f"Service manifest target_database_ref "
                        f"{manifest.target_database_ref!r} conflicts with table "
                        f"{table.name!r} database_ref {table.database_ref!r}"
                    ),
                    relation_name=table.name,
                    source_paths=(str(path),),
                )
            )
        declarations.append(
            _declaration_from_table(
                table=table,
                authority=authority,
                topology=topology,
                source_path=path,
                violations=violations,
            )
        )

    evidence_table_names: set[str] = set()
    for evidence in manifest.relation_evidence:
        if evidence.owner_declaration != authority:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.OWNER_MISMATCH,
                    message=(
                        f"Relation evidence owner {evidence.owner_declaration!r} does "
                        f"not match manifest owner {authority!r}"
                    ),
                    relation_name=evidence.name,
                    source_paths=(str(path),),
                )
            )

        if evidence.kind is EnumApplicationRelationKind.TABLE:
            evidence_table_names.add(evidence.name)
            candidates = tables_by_name.get(evidence.name, [])
            if len(candidates) != 1:
                violations.append(
                    ModelApplicationRelationViolation(
                        code=EnumApplicationRelationViolation.MISSING_TYPED_LOCATION,
                        message=(
                            f"Table evidence {evidence.name!r} resolves to "
                            f"{len(candidates)} typed db_io declarations"
                        ),
                        relation_name=evidence.name,
                        source_paths=(str(path),),
                    )
                )
                continue
            table = candidates[0]
            domain = _topology_domain(topology, table, violations, path)
            if domain is not None and domain is not evidence.domain:
                violations.append(
                    ModelApplicationRelationViolation(
                        code=EnumApplicationRelationViolation.DOMAIN_MISMATCH,
                        message=(
                            f"Table {table.name!r} evidence domain "
                            f"{evidence.domain.value} conflicts with topology domain "
                            f"{domain.value}"
                        ),
                        relation_name=table.name,
                        source_paths=(str(path),),
                    )
                )
            declarations.append(
                ModelApplicationRelationDeclaration(
                    name=table.name,
                    database_ref=table.database_ref,
                    schema=table.schema,
                    kind=EnumApplicationRelationKind.TABLE,
                    purpose=_table_purpose(table),
                    domain=domain,
                    owner_declaration=None,
                    readers=tuple(sorted(set(evidence.readers))),
                    access="read",
                    role=f"{table.role}_evidence_readers",
                    source_path=str(path),
                )
            )
            continue

        if evidence.database_ref is None or evidence.schema is None:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.MISSING_TYPED_LOCATION,
                    message=(
                        f"{evidence.kind.value} evidence {evidence.name!r} must "
                        "declare database_ref and schema"
                    ),
                    relation_name=evidence.name,
                    source_paths=(str(path),),
                )
            )
            continue
        try:
            domain = topology.schema_domain(evidence.database_ref, evidence.schema)
        except ValueError as exc:
            violations.append(
                ModelApplicationRelationViolation(
                    code=_lookup_violation_code(exc),
                    message=str(exc),
                    relation_name=evidence.name,
                    source_paths=(str(path),),
                )
            )
            domain = None
        if domain is not None and domain is not evidence.domain:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.DOMAIN_MISMATCH,
                    message=(
                        f"{evidence.kind.value} {evidence.name!r} evidence domain "
                        f"{evidence.domain.value} conflicts with topology domain "
                        f"{domain.value}"
                    ),
                    relation_name=evidence.name,
                    source_paths=(str(path),),
                )
            )
        declarations.append(
            ModelApplicationRelationDeclaration(
                name=evidence.name,
                database_ref=evidence.database_ref,
                schema=evidence.schema,
                kind=evidence.kind,
                purpose=EnumApplicationRelationPurpose.DATA,
                domain=domain,
                owner_declaration=authority,
                readers=tuple(sorted(set(evidence.readers))),
                access="read_write",
                role=evidence.kind.value,
                source_path=str(path),
            )
        )

    if census is not None and census.observed_base_tables is not None:
        named_tables = len(evidence_table_names)
        if census.observed_base_tables > named_tables:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.INCOMPLETE_CENSUS,
                    message=(
                        f"Service manifest {manifest.service!r} observed "
                        f"{census.observed_base_tables} base tables but names only "
                        f"{named_tables}; {census.observed_base_tables - named_tables} "
                        "live-only relations remain unresolved"
                    ),
                    source_paths=(str(path),),
                )
            )

    named_view_count = sum(
        evidence.kind
        in {
            EnumApplicationRelationKind.VIEW,
            EnumApplicationRelationKind.MATERIALIZED_VIEW,
        }
        for evidence in manifest.relation_evidence
    )
    if (
        census is not None
        and census.observed_views_and_materialized_views is not None
        and census.observed_views_and_materialized_views > named_view_count
    ):
        unresolved = census.observed_views_and_materialized_views - named_view_count
        violations.append(
            ModelApplicationRelationViolation(
                code=EnumApplicationRelationViolation.INCOMPLETE_CENSUS,
                message=(
                    f"Service manifest {manifest.service!r} observed "
                    f"{census.observed_views_and_materialized_views} views/materialized "
                    f"views but names only {named_view_count}; {unresolved} live-only "
                    "relations remain unresolved"
                ),
                source_paths=(str(path),),
            )
        )

    for database_object in manifest.database_objects:
        if database_object.kind is not EnumApplicationDatabaseObjectKind.FUNCTION:
            continue
        if database_object.owner_declaration != authority:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.OWNER_MISMATCH,
                    message=(
                        f"Function owner {database_object.owner_declaration!r} does "
                        f"not match manifest owner {authority!r}"
                    ),
                    relation_name=database_object.name,
                    source_paths=(str(path),),
                )
            )
        if database_object.database_ref is None or database_object.schema is None:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.MISSING_TYPED_LOCATION,
                    message=(
                        f"function {database_object.name!r} must declare database_ref "
                        "and schema"
                    ),
                    relation_name=database_object.name,
                    source_paths=(str(path),),
                )
            )
            continue
        try:
            domain = topology.schema_domain(
                database_object.database_ref, database_object.schema
            )
        except ValueError as exc:
            violations.append(
                ModelApplicationRelationViolation(
                    code=_lookup_violation_code(exc),
                    message=str(exc),
                    relation_name=database_object.name,
                    source_paths=(str(path),),
                )
            )
            domain = None
        if domain is not None and domain is not database_object.domain:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.DOMAIN_MISMATCH,
                    message=(
                        f"function {database_object.name!r} evidence domain "
                        f"{database_object.domain.value} conflicts with topology "
                        f"domain {domain.value}"
                    ),
                    relation_name=database_object.name,
                    source_paths=(str(path),),
                )
            )
        declarations.append(
            ModelApplicationRelationDeclaration(
                name=database_object.name,
                database_ref=database_object.database_ref,
                schema=database_object.schema,
                kind=EnumApplicationRelationKind.FUNCTION,
                purpose=EnumApplicationRelationPurpose.DATA,
                domain=domain,
                owner_declaration=authority,
                readers=tuple(sorted(set(database_object.readers))),
                access="read_write",
                role="function",
                source_path=str(path),
            )
        )


def _append_location_conflicts(
    declarations: Sequence[ModelApplicationRelationDeclaration],
    violations: list[ModelApplicationRelationViolation],
) -> None:
    locations_by_object: dict[
        tuple[str, EnumApplicationRelationKind], set[tuple[str, str]]
    ] = defaultdict(set)
    sources_by_object: dict[tuple[str, EnumApplicationRelationKind], set[str]] = (
        defaultdict(set)
    )
    for declaration in declarations:
        object_key = (declaration.name, declaration.kind)
        locations_by_object[object_key].add(
            (declaration.database_ref, declaration.schema)
        )
        sources_by_object[object_key].add(declaration.source_path)
    for (name, kind), locations in locations_by_object.items():
        if len(locations) > 1:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.CONFLICTING_LOCATION,
                    message=(
                        f"{kind.value} {name!r} has conflicting typed locations: "
                        f"{sorted(locations)!r}"
                    ),
                    relation_name=name,
                    source_paths=tuple(sorted(sources_by_object[(name, kind)])),
                )
            )


def validate_application_relation_ownership(
    *,
    topology: ModelDeploymentTopology,
    node_contract_paths: Sequence[Path],
    service_manifest_paths: Sequence[Path],
    inventory: ModelApplicationRelationInventory,
) -> ModelApplicationRelationOwnershipReport:
    """Project distributed declarations and validate exactly-one ownership."""
    declarations: list[ModelApplicationRelationDeclaration] = []
    violations: list[ModelApplicationRelationViolation] = []
    _append_inventory_evidence_violations(inventory, violations)

    for path in sorted(node_contract_paths):
        document = _load_node_ownership_document(path)
        authority = f"node:{document.name}"
        for table in document.db_io.db_tables:
            declarations.append(
                _declaration_from_table(
                    table=table,
                    authority=authority,
                    topology=topology,
                    source_path=path,
                    violations=violations,
                )
            )

    for path in sorted(service_manifest_paths):
        _append_service_declarations(
            path=path,
            manifest=load_service_ownership_manifest(path),
            topology=topology,
            declarations=declarations,
            violations=violations,
        )

    _append_location_conflicts(declarations, violations)

    declarations_by_identity: dict[
        RelationIdentity, list[ModelApplicationRelationDeclaration]
    ] = defaultdict(list)
    for declaration in declarations:
        declarations_by_identity[declaration.identity].append(declaration)

    duplicate_owner_identities: set[RelationIdentity] = set()
    for identity, candidates in declarations_by_identity.items():
        owners = [candidate for candidate in candidates if candidate.owner_declaration]
        if len(owners) <= 1:
            continue
        duplicate_owner_identities.add(identity)
        violations.append(
            ModelApplicationRelationViolation(
                code=EnumApplicationRelationViolation.DUPLICATE_OWNER,
                message=(
                    f"Declared relation {identity!r} has {len(owners)} owners: "
                    f"{[owner.owner_declaration for owner in owners]!r}"
                ),
                relation_name=identity[2],
                source_paths=tuple(owner.source_path for owner in owners),
            )
        )

    live_by_identity: dict[RelationIdentity, ModelLiveApplicationRelation] = {}
    for live_relation in inventory.relations:
        try:
            topology_domain = topology.schema_domain(
                live_relation.database_ref, live_relation.schema
            )
        except ValueError as exc:
            violations.append(
                ModelApplicationRelationViolation(
                    code=_lookup_violation_code(exc),
                    message=str(exc),
                    relation_name=live_relation.name,
                )
            )
        else:
            if (
                live_relation.domain is not None
                and live_relation.domain is not topology_domain
            ):
                violations.append(
                    ModelApplicationRelationViolation(
                        code=EnumApplicationRelationViolation.DOMAIN_MISMATCH,
                        message=(
                            f"Live {live_relation.kind.value} {live_relation.name!r} "
                            f"inventory domain {live_relation.domain.value} conflicts "
                            f"with topology domain {topology_domain.value}"
                        ),
                        relation_name=live_relation.name,
                    )
                )
        if live_relation.identity in live_by_identity:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.DUPLICATE_OWNER,
                    message=f"Live inventory repeats relation {live_relation.identity!r}",
                    relation_name=live_relation.name,
                )
            )
        live_by_identity[live_relation.identity] = live_relation

    for identity, live_relation in live_by_identity.items():
        candidates = declarations_by_identity.get(identity, [])
        owners = [candidate for candidate in candidates if candidate.owner_declaration]
        if not owners:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.MISSING_OWNER,
                    message=f"Live {live_relation.kind.value} {identity!r} has no owner",
                    relation_name=live_relation.name,
                )
            )
        elif len(owners) > 1 and identity not in duplicate_owner_identities:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.DUPLICATE_OWNER,
                    message=(
                        f"Live {live_relation.kind.value} {identity!r} has "
                        f"{len(owners)} owners: "
                        f"{[owner.owner_declaration for owner in owners]!r}"
                    ),
                    relation_name=live_relation.name,
                    source_paths=tuple(owner.source_path for owner in owners),
                )
            )
        for owner in owners:
            if owner.purpose is not live_relation.purpose:
                violations.append(
                    ModelApplicationRelationViolation(
                        code=EnumApplicationRelationViolation.PURPOSE_MISMATCH,
                        message=(
                            f"Relation {identity!r} inventory purpose "
                            f"{live_relation.purpose.value!r} conflicts with owner "
                            f"purpose {owner.purpose.value!r}"
                        ),
                        relation_name=live_relation.name,
                        source_paths=(owner.source_path,),
                    )
                )

    for identity, candidates in declarations_by_identity.items():
        if identity not in live_by_identity:
            violations.append(
                ModelApplicationRelationViolation(
                    code=EnumApplicationRelationViolation.UNKNOWN_RELATION,
                    message=f"Declared relation {identity!r} is absent from live inventory",
                    relation_name=identity[2],
                    source_paths=tuple(
                        sorted({candidate.source_path for candidate in candidates})
                    ),
                )
            )

    ordered_violations = tuple(
        sorted(
            violations,
            key=lambda violation: (
                violation.code.value,
                violation.relation_name or "",
                violation.message,
            ),
        )
    )
    return ModelApplicationRelationOwnershipReport(
        declarations=tuple(declarations),
        violations=ordered_violations,
    )


def assert_application_relation_ownership(
    report: ModelApplicationRelationOwnershipReport,
) -> None:
    """Raise one sanitized startup/CI error when global ownership is invalid."""
    if report.is_valid:
        return
    summaries = "; ".join(
        f"{violation.code.value}: {violation.message}"
        for violation in report.violations
    )
    raise ModelOnexError(
        f"Application relation ownership validation failed with "
        f"{len(report.violations)} violation(s): {summaries}"
    )


__all__ = [
    "EnumApplicationDatabaseObjectKind",
    "EnumApplicationRelationKind",
    "EnumApplicationRelationPurpose",
    "EnumApplicationRelationViolation",
    "ModelApplicationRelationInventory",
    "ModelApplicationRelationOwnershipReport",
    "ModelLiveApplicationRelation",
    "ModelMigrationOwnershipManifest",
    "assert_application_relation_ownership",
    "load_application_relation_inventory",
    "load_service_ownership_manifest",
    "validate_application_relation_ownership",
]
