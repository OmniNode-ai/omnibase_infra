# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Schemas an ownership manifest's ``database_objects`` entry is declared at.

An entry that names ``database_ref`` and ``schema`` is declared there. A sequence
listed as a bare name under its domain (``dependencies: [<table>]``, no location)
occupies the schemas of the tables it depends on: a serial sequence is created in
its table's schema, and where that table physically lives in ``public`` the bridge
in ``validator_physical_public_tables`` puts the sequence there too. Resolution
adds no name to that bridge. A bare sequence with no owning table declared in the
same manifest resolves nowhere, and a bare sequence whose domain differs from its
table's declared domain is refused.
"""

from __future__ import annotations

from pathlib import Path

from omnibase_infra.validation import validator_physical_public_tables as bridge
from omnibase_infra.validation.enums.enum_application_database_object_kind import (
    EnumApplicationDatabaseObjectKind,
)
from omnibase_infra.validation.models.model_database_object_evidence import (
    ModelDatabaseObjectEvidence,
)
from omnibase_infra.validation.models.model_migration_ownership_manifest import (
    ModelMigrationOwnershipManifest,
)

_ROUTINE_KINDS = frozenset(
    {
        EnumApplicationDatabaseObjectKind.FUNCTION,
        EnumApplicationDatabaseObjectKind.AGGREGATE,
        EnumApplicationDatabaseObjectKind.WINDOW_FUNCTION,
        EnumApplicationDatabaseObjectKind.PROCEDURE,
    }
)


def _dependencies(database_object: ModelDatabaseObjectEvidence) -> tuple[str, ...]:
    declared = (database_object.model_extra or {}).get("dependencies", ())
    if not isinstance(declared, list | tuple):
        return ()
    return tuple(name for name in declared if isinstance(name, str))


def _bare_sequence_schemas(
    database_object: ModelDatabaseObjectEvidence,
    manifest: ModelMigrationOwnershipManifest,
    path: Path,
) -> tuple[str, ...]:
    dependencies = _dependencies(database_object)
    schemas: set[str] = set()
    for table in manifest.db_io.db_tables:
        if table.name not in dependencies:
            continue
        schemas.add(table.schema)
        if table.name in bridge.PHYSICALLY_PUBLIC_TABLES.get(table.schema, ()):
            schemas.add("public")
    for relation in manifest.relation_evidence:
        if relation.name in dependencies and relation.domain != database_object.domain:
            raise ValueError(
                f"{path}: sequence {database_object.name!r} domain "
                f"{database_object.domain.value!r} differs from table "
                f"{relation.name!r} domain {relation.domain.value!r}"
            )
    return tuple(sorted(schemas))


def database_object_schemas(
    database_object: ModelDatabaseObjectEvidence,
    manifest: ModelMigrationOwnershipManifest,
    *,
    authority: str,
    path: Path,
) -> tuple[str, ...]:
    """Return the schemas a manifest object is declared at, empty when none."""
    if database_object.database_ref is not None and database_object.schema is not None:
        schemas: tuple[str, ...] = (database_object.schema,)
    elif database_object.kind is EnumApplicationDatabaseObjectKind.SEQUENCE:
        schemas = _bare_sequence_schemas(database_object, manifest, path)
    else:
        schemas = ()
    if not schemas:
        return ()
    if database_object.owner_declaration != authority:
        raise ValueError(
            f"{path}: catalog object {database_object.name!r} owner "
            f"{database_object.owner_declaration!r} conflicts with manifest "
            f"authority {authority!r}"
        )
    if (
        database_object.kind in _ROUTINE_KINDS
        and database_object.function_signature is None
    ):
        raise ValueError(
            f"{path}: routine object {database_object.name!r} requires an "
            "exact function_signature"
        )
    return schemas


__all__ = ["database_object_schemas"]
