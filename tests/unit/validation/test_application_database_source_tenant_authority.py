# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Transitive catalog authority extraction for source-tenant provenance."""

from __future__ import annotations

import pytest

from omnibase_infra.validation.application_database_source_tenant_authority import (
    ApplicationDatabaseAuthorityResolutionError,
    resolve_application_database_authority_columns,
)
from omnibase_infra.validation.models.model_application_database_routine_dependency_state import (
    ModelApplicationDatabaseRoutineDependencyState,
)

_TARGET_COLUMNS = ("state_id", "source_tenant_id", "payload")
_TARGET_COMPOSITE_TYPE_ID = 16_387


def _routine(
    object_id: int,
    name: str,
    source_body: str,
    *,
    language: str = "sql",
    argument_type_ids: tuple[int, ...] = (_TARGET_COMPOSITE_TYPE_ID,),
    referenced_routine_ids: tuple[int, ...] = (),
    referenced_target_columns: tuple[str, ...] = (),
    references_target_whole_row: bool = False,
    returns_trigger: bool = False,
    argument_names: tuple[str | None, ...] = (),
) -> ModelApplicationDatabaseRoutineDependencyState:
    return ModelApplicationDatabaseRoutineDependencyState(
        object_id=object_id,
        namespace="omninode_internal",
        name=name,
        language=language,
        source_body=source_body,
        argument_type_ids=argument_type_ids,
        argument_names=argument_names,
        returns_trigger=returns_trigger,
        referenced_routine_ids=referenced_routine_ids,
        referenced_target_columns=referenced_target_columns,
        references_target_whole_row=references_target_whole_row,
    )


def test_nested_whole_row_helper_resolves_source_tenant_transitively() -> None:
    inner = _routine(101, "source_tenant_key", "SELECT $1.source_tenant_id")
    wrapper = _routine(
        102,
        "nested_source_tenant_key",
        "SELECT omninode_internal.source_tenant_key($1)",
    )

    assert (
        resolve_application_database_authority_columns(
            target_columns=_TARGET_COLUMNS,
            target_composite_type_id=_TARGET_COMPOSITE_TYPE_ID,
            direct_referenced_columns=(),
            direct_whole_row_reference=False,
            root_routine_ids=(wrapper.object_id,),
            routines=(inner, wrapper),
            governed_schemas=("tenant", "omninode_internal", "platform_catalog"),
        )
        == _TARGET_COLUMNS
    )


def test_nested_helper_cycle_is_bounded_and_preserves_authority() -> None:
    first = _routine(
        201,
        "first_key",
        "SELECT omninode_internal.second_key($1)",
    )
    second = _routine(
        202,
        "second_key",
        "SELECT CASE WHEN $1.source_tenant_id IS NULL "
        "THEN omninode_internal.first_key($1) ELSE $1.source_tenant_id END",
    )

    assert (
        resolve_application_database_authority_columns(
            target_columns=_TARGET_COLUMNS,
            target_composite_type_id=_TARGET_COMPOSITE_TYPE_ID,
            direct_referenced_columns=(),
            direct_whole_row_reference=False,
            root_routine_ids=(first.object_id,),
            routines=(first, second),
            governed_schemas=("omninode_internal",),
        )
        == _TARGET_COLUMNS
    )


@pytest.mark.parametrize(
    ("root_routine_ids", "routines", "expected"),
    [
        ((999,), (), "catalog routine oid 999 is unresolved"),
        (
            (301,),
            (
                _routine(
                    301,
                    "dynamic_key",
                    "BEGIN EXECUTE query_text; RETURN NULL; END",
                    language="plpgsql",
                ),
            ),
            "dynamic SQL",
        ),
        (
            (302,),
            (
                _routine(
                    302,
                    "opaque_key",
                    "opaque_symbol",
                    language="c",
                ),
            ),
            "cannot inspect language",
        ),
        (
            (303,),
            (
                _routine(
                    303,
                    "missing_nested_key",
                    "SELECT omninode_internal.not_installed($1)",
                ),
            ),
            "qualified governed routine",
        ),
    ],
)
def test_unresolved_transitive_dependencies_fail_closed(
    root_routine_ids: tuple[int, ...],
    routines: tuple[ModelApplicationDatabaseRoutineDependencyState, ...],
    expected: str,
) -> None:
    with pytest.raises(ApplicationDatabaseAuthorityResolutionError, match=expected):
        resolve_application_database_authority_columns(
            target_columns=_TARGET_COLUMNS,
            target_composite_type_id=_TARGET_COMPOSITE_TYPE_ID,
            direct_referenced_columns=(),
            direct_whole_row_reference=False,
            root_routine_ids=root_routine_ids,
            routines=routines,
            governed_schemas=("omninode_internal",),
        )


def test_catalog_direct_columns_and_safe_helper_remain_exact() -> None:
    safe = _routine(401, "payload_key", "SELECT $1.payload")

    assert resolve_application_database_authority_columns(
        target_columns=_TARGET_COLUMNS,
        target_composite_type_id=_TARGET_COMPOSITE_TYPE_ID,
        direct_referenced_columns=("payload",),
        direct_whole_row_reference=False,
        root_routine_ids=(safe.object_id,),
        routines=(safe,),
        governed_schemas=("omninode_internal",),
    ) == ("payload",)


@pytest.mark.parametrize(
    "routine",
    [
        _routine(
            501,
            "opaque_row_hash",
            "SELECT pg_catalog.md5($1::text)",
        ),
        _routine(
            502,
            "opaque_trigger_hash",
            "BEGIN PERFORM pg_catalog.md5(NEW::text); RETURN NEW; END",
            argument_type_ids=(),
            returns_trigger=True,
        ),
    ],
)
def test_whole_row_consumption_fails_closed_to_every_target_column(
    routine: ModelApplicationDatabaseRoutineDependencyState,
) -> None:
    assert (
        resolve_application_database_authority_columns(
            target_columns=_TARGET_COLUMNS,
            target_composite_type_id=_TARGET_COMPOSITE_TYPE_ID,
            direct_referenced_columns=(),
            direct_whole_row_reference=False,
            root_routine_ids=(routine.object_id,),
            routines=(routine,),
            governed_schemas=("omninode_internal",),
        )
        == _TARGET_COLUMNS
    )


def test_catalog_whole_row_dependency_fails_closed_to_every_target_column() -> None:
    assert (
        resolve_application_database_authority_columns(
            target_columns=_TARGET_COLUMNS,
            target_composite_type_id=_TARGET_COMPOSITE_TYPE_ID,
            direct_referenced_columns=(),
            direct_whole_row_reference=True,
            root_routine_ids=(),
            routines=(),
            governed_schemas=("omninode_internal",),
        )
        == _TARGET_COLUMNS
    )


def test_named_composite_argument_consumption_fails_closed() -> None:
    routine = _routine(
        503,
        "named_row_hash",
        "SELECT pg_catalog.md5(row_value::text)",
        argument_names=("row_value",),
    )

    assert (
        resolve_application_database_authority_columns(
            target_columns=_TARGET_COLUMNS,
            target_composite_type_id=_TARGET_COMPOSITE_TYPE_ID,
            direct_referenced_columns=(),
            direct_whole_row_reference=False,
            root_routine_ids=(routine.object_id,),
            routines=(routine,),
            governed_schemas=("omninode_internal",),
        )
        == _TARGET_COLUMNS
    )


@pytest.mark.parametrize(
    "wrapper_body",
    [
        (
            "SELECT omninode_internal.source_tenant_key "
            "/* formatted call */ ($1.payload)"
        ),
        "SELECT source_tenant_key -- formatted call\n($1.payload)",
    ],
)
def test_comments_between_governed_routine_name_and_call_preserve_closure(
    wrapper_body: str,
) -> None:
    inner = _routine(
        504,
        "source_tenant_key",
        "SELECT source_tenant_id",
        argument_type_ids=(),
    )
    wrapper = _routine(505, "commented_wrapper", wrapper_body)

    assert resolve_application_database_authority_columns(
        target_columns=_TARGET_COLUMNS,
        target_composite_type_id=_TARGET_COMPOSITE_TYPE_ID,
        direct_referenced_columns=(),
        direct_whole_row_reference=False,
        root_routine_ids=(wrapper.object_id,),
        routines=(inner, wrapper),
        governed_schemas=("omninode_internal",),
    ) == ("source_tenant_id", "payload")
