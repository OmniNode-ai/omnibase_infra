# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Catalog-resolved authority closure for optional source-tenant provenance."""

from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Sequence

from omnibase_infra.validation.models.model_application_database_routine_dependency_state import (
    ModelApplicationDatabaseRoutineDependencyState,
)

_INSPECTABLE_ROUTINE_LANGUAGES = frozenset({"sql", "plpgsql"})
_DYNAMIC_SQL_PATTERN = re.compile(r"\bexecute\b", re.IGNORECASE)
_QUALIFIED_CALL_PATTERN = re.compile(
    r'(?<![a-zA-Z0-9_$])(?P<schema>"[^"]+"|[a-z_][a-z0-9_$]*)'
    r'\s*\.\s*(?P<name>"[^"]+"|[a-z_][a-z0-9_$]*)\s*\(',
    re.IGNORECASE,
)


def _is_identifier_character(character: str) -> bool:
    return (
        character == "_"
        or character == "$"
        or character.isalnum()
        or ord(character) >= 128
    )


def _has_escape_string_prefix(value: str, quote_index: int) -> bool:
    return (
        quote_index > 0
        and value[quote_index - 1].lower() == "e"
        and (quote_index == 1 or not _is_identifier_character(value[quote_index - 2]))
    )


def _dollar_quote_at(value: str, index: int) -> str | None:
    match = re.match(
        r"\$(?:(?:[a-z_]|[^\x00-\x7f])"
        r"(?:[a-z0-9_]|[^\x00-\x7f])*)?\$",
        value[index:],
        re.IGNORECASE,
    )
    return None if match is None else match.group(0)


def _without_sql_comments(value: str) -> str:
    """Replace PostgreSQL comments with whitespace without touching quoted text."""
    normalized = list(value)
    index = 0
    while index < len(value):
        if value.startswith("--", index):
            newline = value.find("\n", index + 2)
            stop = len(value) if newline < 0 else newline
            for offset in range(index, stop):
                normalized[offset] = " "
            index = stop
            continue
        if value.startswith("/*", index):
            depth = 1
            start = index
            index += 2
            while index < len(value) and depth:
                if value.startswith("/*", index):
                    depth += 1
                    index += 2
                elif value.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            for offset in range(start, index):
                if normalized[offset] not in {"\r", "\n"}:
                    normalized[offset] = " "
            continue
        character = value[index]
        if character in {"'", '"'}:
            quote = character
            escape_backslashes = character == "'" and _has_escape_string_prefix(
                value,
                index,
            )
            index += 1
            while index < len(value):
                if escape_backslashes and value[index] == "\\":
                    index += 2
                    continue
                if value[index] != quote:
                    index += 1
                    continue
                if index + 1 < len(value) and value[index + 1] == quote:
                    index += 2
                    continue
                index += 1
                break
            continue
        if character == "$":
            delimiter = _dollar_quote_at(value, index)
            if delimiter is not None:
                end = value.find(delimiter, index + len(delimiter))
                index = len(value) if end < 0 else end + len(delimiter)
                continue
        index += 1
    return "".join(normalized)


class ApplicationDatabaseAuthorityResolutionError(ValueError):
    """Raised when catalog authority cannot be resolved without guessing."""


def _normalize_identifier(value: str) -> str:
    normalized = value.strip()
    if normalized.startswith('"') and normalized.endswith('"'):
        normalized = normalized[1:-1].replace('""', '"')
    return normalized.lower()


def _identifier_is_referenced(text: str, identifier: str) -> bool:
    return bool(
        re.search(
            rf'(?<![a-zA-Z0-9_$])"?{re.escape(identifier)}"?'
            rf"(?![a-zA-Z0-9_$])",
            text,
            re.IGNORECASE,
        )
    )


def _unqualified_routine_is_called(text: str, name: str) -> bool:
    return bool(
        re.search(
            rf'(?<![a-zA-Z0-9_$."])"?{re.escape(name)}"?\s*\(',
            text,
            re.IGNORECASE,
        )
    )


def _composite_argument_is_consumed(source_body: str, position: int) -> bool:
    """Detect a composite argument used as a value rather than one named field."""
    return bool(
        re.search(
            rf"\${position}(?!\d)(?:\s*\.\s*\*|(?!\s*\.))",
            source_body,
            re.IGNORECASE,
        )
    )


def _named_composite_argument_is_consumed(
    source_body: str,
    argument_name: str,
) -> bool:
    """Detect a named composite argument consumed as a whole-row value."""
    escaped_quoted_name = re.escape(argument_name.replace('"', '""'))
    escaped_unquoted_name = re.escape(argument_name)
    return bool(
        re.search(
            rf'(?<![a-zA-Z0-9_$])(?:"{escaped_quoted_name}"|'
            rf"{escaped_unquoted_name})(?![a-zA-Z0-9_$])"
            rf"(?:\s*\.\s*\*|(?!\s*\.))",
            source_body,
            re.IGNORECASE,
        )
    )


def _trigger_row_is_consumed(source_body: str) -> bool:
    """Detect NEW/OLD whole-row use while excluding the structural RETURN row."""
    without_structural_return = re.sub(
        r"\breturn\s+(?:new|old)\b",
        "",
        source_body,
        flags=re.IGNORECASE,
    )
    return bool(
        re.search(
            r"\b(?:new|old)\b(?:\s*\.\s*\*|(?!\s*\.))",
            without_structural_return,
            re.IGNORECASE,
        )
    )


def resolve_application_database_authority_columns(
    *,
    target_columns: Sequence[str],
    target_composite_type_id: int,
    direct_referenced_columns: Sequence[str],
    direct_whole_row_reference: bool,
    root_routine_ids: Sequence[int],
    routines: Sequence[ModelApplicationDatabaseRoutineDependencyState],
    governed_schemas: Sequence[str],
) -> tuple[str, ...]:
    """Resolve columns used by an index, trigger, policy, or dependent view.

    PostgreSQL records the routine OID used by a parsed catalog expression, but
    traditional SQL and PL/pgSQL string bodies do not record nested calls in
    ``pg_depend``.  The resolver therefore starts from exact catalog OIDs, walks
    catalog-bound routine edges, and inspects only reachable routine definitions
    for target-column and governed nested-call references.  Unknown catalog OIDs,
    opaque row-taking routines, dynamic SQL, and missing governed callees fail
    closed.  The visited-OID set bounds recursive routine cycles.
    """
    ordered_target_columns = tuple(dict.fromkeys(target_columns))
    if len(ordered_target_columns) != len(tuple(target_columns)):
        raise ApplicationDatabaseAuthorityResolutionError(
            "target relation columns must be unique"
        )
    target_column_set = frozenset(ordered_target_columns)
    governed_schema_set = frozenset(
        _normalize_identifier(schema) for schema in governed_schemas
    )
    routines_by_id: dict[int, ModelApplicationDatabaseRoutineDependencyState] = {}
    governed_by_qualified_name: dict[
        tuple[str, str], list[ModelApplicationDatabaseRoutineDependencyState]
    ] = defaultdict(list)
    governed_by_name: dict[
        str, list[ModelApplicationDatabaseRoutineDependencyState]
    ] = defaultdict(list)
    for routine in routines:
        if routine.object_id in routines_by_id:
            raise ApplicationDatabaseAuthorityResolutionError(
                f"catalog routine oid {routine.object_id} is duplicated"
            )
        routines_by_id[routine.object_id] = routine
        if routine.namespace in governed_schema_set:
            governed_by_qualified_name[(routine.namespace, routine.name)].append(
                routine
            )
            governed_by_name[routine.name].append(routine)

    referenced_columns = (
        set(ordered_target_columns)
        if direct_whole_row_reference
        else set(direct_referenced_columns)
    )
    unknown_direct_columns = referenced_columns.difference(target_column_set)
    if unknown_direct_columns:
        raise ApplicationDatabaseAuthorityResolutionError(
            "catalog authority references unknown target columns: "
            f"{sorted(unknown_direct_columns)!r}"
        )

    pending: deque[int] = deque(dict.fromkeys(root_routine_ids))
    visited: set[int] = set()
    while pending:
        routine_id = pending.popleft()
        if routine_id in visited:
            continue
        resolved_routine = routines_by_id.get(routine_id)
        if resolved_routine is None:
            raise ApplicationDatabaseAuthorityResolutionError(
                f"catalog routine oid {routine_id} is unresolved"
            )
        visited.add(routine_id)

        unknown_routine_columns = set(
            resolved_routine.referenced_target_columns
        ).difference(target_column_set)
        if unknown_routine_columns:
            raise ApplicationDatabaseAuthorityResolutionError(
                f"catalog routine oid {routine_id} references unknown target "
                f"columns: {sorted(unknown_routine_columns)!r}"
            )
        referenced_columns.update(resolved_routine.referenced_target_columns)
        if resolved_routine.references_target_whole_row:
            referenced_columns.update(ordered_target_columns)
        pending.extend(resolved_routine.referenced_routine_ids)

        receives_target_row = (
            target_composite_type_id in resolved_routine.argument_type_ids
            or resolved_routine.returns_trigger
        )
        governed = resolved_routine.namespace in governed_schema_set
        if not governed and not receives_target_row:
            continue
        if resolved_routine.language not in _INSPECTABLE_ROUTINE_LANGUAGES:
            raise ApplicationDatabaseAuthorityResolutionError(
                f"catalog routine {resolved_routine.namespace}."
                f"{resolved_routine.name} cannot inspect language "
                f"{resolved_routine.language!r} safely"
            )
        if not resolved_routine.source_body or not resolved_routine.source_body.strip():
            raise ApplicationDatabaseAuthorityResolutionError(
                f"catalog routine {resolved_routine.namespace}."
                f"{resolved_routine.name} has no inspectable definition"
            )
        source_body = _without_sql_comments(resolved_routine.source_body)
        if _DYNAMIC_SQL_PATTERN.search(source_body):
            raise ApplicationDatabaseAuthorityResolutionError(
                f"catalog routine {resolved_routine.namespace}."
                f"{resolved_routine.name} uses dynamic SQL that cannot be "
                "resolved fail-closed"
            )
        composite_argument_positions = tuple(
            position
            for position, argument_type_id in enumerate(
                resolved_routine.argument_type_ids,
                start=1,
            )
            if argument_type_id == target_composite_type_id
        )
        argument_names = resolved_routine.argument_names
        if any(
            _composite_argument_is_consumed(source_body, position)
            or (
                bool(argument_names)
                and argument_names[position - 1] is not None
                and _named_composite_argument_is_consumed(
                    source_body,
                    argument_names[position - 1] or "",
                )
            )
            for position in composite_argument_positions
        ) or (
            resolved_routine.returns_trigger and _trigger_row_is_consumed(source_body)
        ):
            referenced_columns.update(ordered_target_columns)
        referenced_columns.update(
            column
            for column in ordered_target_columns
            if _identifier_is_referenced(source_body, column)
        )

        for match in _QUALIFIED_CALL_PATTERN.finditer(source_body):
            schema_name = _normalize_identifier(match.group("schema"))
            if schema_name not in governed_schema_set:
                continue
            routine_name = _normalize_identifier(match.group("name"))
            candidates = governed_by_qualified_name.get((schema_name, routine_name), ())
            if not candidates:
                raise ApplicationDatabaseAuthorityResolutionError(
                    "qualified governed routine "
                    f"{schema_name}.{routine_name} is unresolved"
                )
            pending.extend(candidate.object_id for candidate in candidates)

        for routine_name, candidates in governed_by_name.items():
            if _unqualified_routine_is_called(source_body, routine_name):
                pending.extend(candidate.object_id for candidate in candidates)

    return tuple(
        column for column in ordered_target_columns if column in referenced_columns
    )


__all__ = [
    "ApplicationDatabaseAuthorityResolutionError",
    "resolve_application_database_authority_columns",
]
