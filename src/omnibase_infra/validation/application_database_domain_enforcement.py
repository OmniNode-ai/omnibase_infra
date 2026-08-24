# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed application database domain enforcement (OMN-15361)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import sqlparse
from sqlparse.tokens import DML

from omnibase_core.enums.enum_database_schema_domain import EnumDatabaseSchemaDomain
from omnibase_core.models.core.model_deployment_topology import ModelDeploymentTopology
from omnibase_core.models.core.model_deployment_topology_database import (
    ModelDeploymentTopologyDatabase,
)
from omnibase_infra.topology.physical_schema_mapping import (
    INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
    TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359,
)
from omnibase_infra.validation.application_relation_ownership import (
    load_service_ownership_manifest,
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
from omnibase_infra.validation.models.model_application_database_catalog_identity import (
    ModelApplicationDatabaseCatalogIdentity,
)
from omnibase_infra.validation.models.model_application_database_function_state import (
    ModelApplicationDatabaseFunctionState,
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

CANONICAL_TENANT_PREDICATE = "tenant_id = current_setting('app.tenant_id', true)::uuid"

_EXPECTED_IDENTITY_ROOT_RELATIONS = {
    EnumApplicationDatabaseIdentityRoot.CANONICAL_TENANT: "tenants",
    EnumApplicationDatabaseIdentityRoot.PRE_TENANT_BOOTSTRAP: "bootstrap_tokens",
}
_EXPECTED_IDENTITY_ROOT_OPERATIONS = frozenset(
    EnumApplicationDatabaseIdentityRootOperation
)
_EXPECTED_CANONICAL_POLICY_ROLES = ("PUBLIC",)
_SQL_IDENTIFIER = (
    r'(?:"(?:[^"]|"")+"|'
    r"(?:[a-z_]|[^\x00-\x7f])(?:[a-z0-9_$]|[^\x00-\x7f])*)"
)
_OPTIONAL_ONLY_TARGET = r"(?:only\s+(?:\(\s*)?)?"
_SYSTEM_READ_SCHEMAS = frozenset({"information_schema", "pg_catalog"})
# OMN-16237: the static schema-qualification lint below must agree with the
# runtime grants system's physical_grant_schema_for_table() on which tables
# are logically tenant/omninode_internal domain but still physically created
# in `public` pending their OMN-15359 migration. This is the SAME allowlist
# (imported, never copied) -- a table only passes unqualified or explicitly
# public-qualified by being enumerated in one of these two frozensets; it is
# a narrow, enumerated pass-through, never a blanket public exemption.
# TRAP: use frozenset.union(), never the `|` operator, to combine these two
# sets. `validate_infra_union_usage`'s AST checker (omnibase_core
# UnionUsageChecker.visit_BinOp) flags ANY `ast.BinOp` with a `BitOr` operator
# as a type union candidate -- it cannot distinguish a runtime set-union
# expression from `X | Y` in an annotation. `A | B` here reads as a two-type
# non-Optional union and pushes the repo-wide INFRA_MAX_UNIONS threshold over
# by one, failing an unrelated gate for a reason that has nothing to do with
# type complexity. `.union()` is semantically identical and invisible to that
# checker.
_PHYSICALLY_PUBLIC_APPLICATION_TABLES: frozenset[str] = (
    TENANT_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359.union(
        INTERNAL_TABLES_PHYSICALLY_IN_PUBLIC_UNTIL_OMN15359
    )
)
_RELATION_OBJECT_KINDS = frozenset(
    {
        EnumApplicationInventoryObjectKind.TABLE,
        EnumApplicationInventoryObjectKind.FOREIGN_TABLE,
        EnumApplicationInventoryObjectKind.VIEW,
        EnumApplicationInventoryObjectKind.MATERIALIZED_VIEW,
    }
)
_ROUTINE_OBJECT_KINDS = frozenset(
    {
        EnumApplicationInventoryObjectKind.FUNCTION,
        EnumApplicationInventoryObjectKind.WINDOW_FUNCTION,
        EnumApplicationInventoryObjectKind.PROCEDURE,
        EnumApplicationInventoryObjectKind.AGGREGATE,
    }
)


@dataclass(frozen=True, slots=True)
class ApplicationDatabaseSqlTargetRequirement:
    """One kind- and overload-aware ownership requirement found in SQL."""

    schema: str
    name: str
    allowed_kinds: frozenset[EnumApplicationInventoryObjectKind]
    function_signature: str | None = None

    @property
    def location(self) -> tuple[str, str]:
        """Return the schema-qualified target location."""
        return (self.schema, self.name)


def application_database_function_definition_sha256(
    *,
    schema: str,
    name: str,
    signature: str,
    language: str,
    source_body: str,
    parsed_sql_body: str | None,
    security_definer: bool,
    leakproof: bool,
    volatility: str,
    parallel: str,
    config: Sequence[str],
    kind: str,
    strict: bool,
    returns_set: bool,
    result_type: str,
) -> str:
    """Fingerprint the complete security-relevant routine catalog definition."""
    payload = {
        "config": sorted(config),
        "kind": kind,
        "language": language,
        "leakproof": leakproof,
        "name": name,
        "parallel": parallel,
        "parsed_sql_body": parsed_sql_body,
        "result_type": result_type,
        "returns_set": returns_set,
        "schema": schema,
        "security_definer": security_definer,
        "signature": signature,
        "source_body": source_body,
        "strict": strict,
        "volatility": volatility,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _compile_relation_target(prefix: str) -> re.Pattern[str]:
    return re.compile(
        rf"{prefix}(?:(?P<schema>{_SQL_IDENTIFIER})\s*\.\s*)?"
        rf"(?P<name>{_SQL_IDENTIFIER})",
        re.IGNORECASE + re.DOTALL,
    )


# pattern, permits a CTE/table-function target, permits a system read schema
_RELATION_TARGETS: tuple[tuple[re.Pattern[str], bool, bool], ...] = (
    (
        _compile_relation_target(
            r"^\s*create\s+(?:(?:global|local)\s+)?"
            r"(?:(?:temporary|temp|unlogged)\s+)?(?:foreign\s+)?table\s+"
            r"(?:if\s+not\s+exists\s+)?"
        ),
        False,
        False,
    ),
    (
        _compile_relation_target(
            r"^\s*create\s+(?:or\s+replace\s+)?"
            r"(?:view|materialized\s+view|function|procedure|sequence|type|domain|aggregate)\s+"
            r"(?:if\s+not\s+exists\s+)?"
        ),
        False,
        False,
    ),
    (
        _compile_relation_target(
            r"^\s*(?:alter|drop)\s+"
            r"(?:foreign\s+table|table|view|materialized\s+view|function|"
            r"procedure|sequence|type|domain|aggregate)\s+"
            r"(?:if\s+exists\s+)?(?:only\s+)?"
        ),
        False,
        False,
    ),
    (
        _compile_relation_target(
            r"^\s*(?:insert\s+into|update|delete\s+from|merge\s+into|copy|"
            r"truncate(?:\s+table)?)\s+(?:only\s+)?"
        ),
        False,
        False,
    ),
    (_compile_relation_target(r"^\s*call\s+"), False, False),
    (
        _compile_relation_target(
            rf"^\s*create\s+(?:unique\s+)?index\s+(?:concurrently\s+)?"
            rf"(?:(?:if\s+not\s+exists\s+)?{_SQL_IDENTIFIER}"
            rf"(?:\s*\.\s*{_SQL_IDENTIFIER})?\s+)?on\s+"
            rf"{_OPTIONAL_ONLY_TARGET}"
        ),
        False,
        False,
    ),
    (
        _compile_relation_target(rf"^\s*table\s+{_OPTIONAL_ONLY_TARGET}"),
        False,
        False,
    ),
    (
        _compile_relation_target(
            r"^\s*comment\s+on\s+"
            r"(?:foreign\s+table|materialized\s+view|table|view|column|"
            r"function|procedure|aggregate|sequence|type|domain)\s+"
        ),
        False,
        False,
    ),
    (
        _compile_relation_target(r"\bexecute\s+(?:function|procedure)\s+"),
        False,
        False,
    ),
    (
        _compile_relation_target(
            r"^\s*(?:grant|revoke)\s+[\s\S]*?\s+on\s+"
            r"(?:(?:table|sequence|function|procedure|type|domain)\s+)?"
            r"(?!(?:database|schema)\b)"
        ),
        False,
        False,
    ),
    (_compile_relation_target(r"\breferences\s+"), False, False),
    (
        _compile_relation_target(
            rf"\b(?:from|join)\s+(?:lateral\s+)?{_OPTIONAL_ONLY_TARGET}"
        ),
        True,
        True,
    ),
    (
        _compile_relation_target(r"^\s*merge\b[\s\S]*?\busing\s+(?:only\s+)?"),
        True,
        True,
    ),
    (
        _compile_relation_target(r"^\s*delete\b[\s\S]*?\busing\s+(?:only\s+)?"),
        False,
        True,
    ),
    (
        _compile_relation_target(
            rf"^\s*(?:(?:create|alter)\s+policy\s+{_SQL_IDENTIFIER}|"
            rf"drop\s+policy\s+(?:if\s+exists\s+)?{_SQL_IDENTIFIER})\s+on\s+"
        ),
        False,
        False,
    ),
    (
        _compile_relation_target(
            rf"^\s*(?:(?:create\s+(?:or\s+replace\s+)?(?:constraint\s+)?|alter\s+)"
            rf"trigger\s+{_SQL_IDENTIFIER}[\s\S]*?\bon\s+|"
            rf"drop\s+trigger\s+(?:if\s+exists\s+)?{_SQL_IDENTIFIER}\s+on\s+)"
        ),
        False,
        False,
    ),
    (
        _compile_relation_target(
            r"^\s*refresh\s+materialized\s+view\s+(?:concurrently\s+)?"
        ),
        False,
        False,
    ),
    (
        _compile_relation_target(
            r"^\s*reindex\s+(?:\([^)]*\)\s*)?table\s+(?:concurrently\s+)?"
        ),
        False,
        False,
    ),
    (
        _compile_relation_target(r"^\s*cluster\s+(?:(?:\([^)]*\)|verbose)\s+)?"),
        False,
        False,
    ),
    (_compile_relation_target(r"\bpartition\s+of\s+"), False, False),
    (
        _compile_relation_target(r"\b(?:attach|detach)\s+partition\s+"),
        False,
        False,
    ),
    (_compile_relation_target(r"\binherits\s*\(\s*"), False, False),
    (_compile_relation_target(r"\binherit\s+"), False, False),
    (_compile_relation_target(r"\bas\s+table\s+"), False, False),
    (_compile_relation_target(r"\blike\s+"), False, False),
)
_LIST_TARGET = re.compile(
    rf"^\s*(?:lateral\s+)?{_OPTIONAL_ONLY_TARGET}"
    rf"(?:(?P<schema>{_SQL_IDENTIFIER})\s*\.\s*)?"
    rf"(?P<name>{_SQL_IDENTIFIER})",
    re.IGNORECASE,
)
_RELATION_LISTS: tuple[tuple[re.Pattern[str], bool, bool], ...] = (
    (
        re.compile(
            r"\bfrom\s+(?P<targets>[\s\S]*?)"
            r"(?=\bwhere\b|\bjoin\b|\bgroup\s+by\b|\border\s+by\b|"
            r"\blimit\b|\bunion\b|\breturning\b|\bcase\b|"
            r"\bfor\s+(?:update|share)\b|;|$)",
            re.IGNORECASE,
        ),
        True,
        True,
    ),
    (
        re.compile(
            r"^\s*(?:drop\s+(?:foreign\s+table|table|view|materialized\s+view|"
            r"function|procedure|aggregate|sequence|type)|"
            r"truncate(?:\s+table)?|lock(?:\s+table)?)\s+"
            r"(?:if\s+exists\s+)?(?P<targets>[\s\S]*?)"
            r"(?=\s+(?:cascade|restrict|nowait)\b|;|$)",
            re.IGNORECASE,
        ),
        False,
        False,
    ),
    (
        re.compile(
            r"^\s*(?:grant|revoke)\s+[\s\S]*?\s+on\s+"
            r"(?!(?:database|schema)\b)"
            r"(?:(?:table|sequence|function|procedure|type)\s+)?"
            r"(?P<targets>[\s\S]*?)\s+(?:to|from)\b",
            re.IGNORECASE,
        ),
        False,
        False,
    ),
    (
        re.compile(
            r"\binherits\s*\((?P<targets>[^)]*)\)",
            re.IGNORECASE,
        ),
        False,
        False,
    ),
    (
        re.compile(
            r"^\s*vacuum\s+(?:\([^)]*\)\s*)?"
            r"(?:(?:full|freeze|verbose|analyze)\s+)*"
            r"(?P<targets>[\s\S]*?)(?=;|$)",
            re.IGNORECASE,
        ),
        False,
        False,
    ),
    (
        re.compile(
            r"^\s*analyze\s+(?:\([^)]*\)\s*)?(?:verbose\s+)?"
            r"(?P<targets>[\s\S]*?)(?=;|$)",
            re.IGNORECASE,
        ),
        False,
        False,
    ),
)
_CREATED_APPLICATION_OBJECT = re.compile(
    rf"^\s*create\s+(?:or\s+replace\s+)?"
    rf"(?:(?:(?:global|local)\s+)?(?:temporary|temp|unlogged)\s+)?"
    rf"(?P<kind>foreign\s+table|materialized\s+view|table|view|function|"
    rf"procedure|sequence|type|domain|aggregate|extension)\s+"
    rf"(?:if\s+not\s+exists\s+)?"
    rf"(?:(?P<schema>{_SQL_IDENTIFIER})\s*\.\s*)?"
    rf"(?P<name>{_SQL_IDENTIFIER})",
    re.IGNORECASE,
)
_SELECT_INTO_TARGET = _compile_relation_target(
    r"^\s*(?:(?:temporary|temp|unlogged)\s+)?(?:table\s+)?"
)
_CREATED_KIND_MAP = {
    "foreign table": EnumApplicationInventoryObjectKind.FOREIGN_TABLE,
    "materialized view": EnumApplicationInventoryObjectKind.MATERIALIZED_VIEW,
    "table": EnumApplicationInventoryObjectKind.TABLE,
    "view": EnumApplicationInventoryObjectKind.VIEW,
    "function": EnumApplicationInventoryObjectKind.FUNCTION,
    "procedure": EnumApplicationInventoryObjectKind.PROCEDURE,
    "sequence": EnumApplicationInventoryObjectKind.SEQUENCE,
    "type": EnumApplicationInventoryObjectKind.TYPE,
    "domain": EnumApplicationInventoryObjectKind.TYPE,
    "aggregate": EnumApplicationInventoryObjectKind.AGGREGATE,
    "extension": EnumApplicationInventoryObjectKind.EXTENSION,
}
_NON_EXACT_MAINTENANCE_OPERATIONS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^\s*vacuum\b\s*(?:\([^)]*\)\s*)?"
        r"(?:(?:full|freeze|verbose|analyze)\b\s*)*;?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*analyze\b\s*(?:\([^)]*\)\s*)?(?:verbose\b\s*)?;?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*cluster\b\s*(?:(?:\([^)]*\)|verbose\b)\s*)?;?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*reindex\s+(?:\([^)]*\)\s*)?"
        r"(?:index|schema|database|system)\b[\s\S]*$",
        re.IGNORECASE,
    ),
)


def _unquote_identifier(token: str) -> str:
    token = token.strip()
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1].replace('""', '"')
    return token.lower()


def _main_dml_tail(statement: str) -> str:
    """Return the top-level DML tail so CTE-prefixed writes cannot evade anchors."""
    parsed = sqlparse.parse(statement)
    if not parsed:
        return statement
    parsed_statement = parsed[0]
    # sqlparse's bundled typing does not annotate Statement.get_type().
    statement_type = parsed_statement.get_type().upper()  # type: ignore[no-untyped-call]
    if statement_type not in {"INSERT", "UPDATE", "DELETE", "MERGE"}:
        return statement
    for index, token in enumerate(parsed_statement.tokens):
        if token.ttype is DML and token.normalized.upper() == statement_type:
            return "".join(str(item) for item in parsed_statement.tokens[index:])
    return statement


def _split_top_level_commas(value: str) -> tuple[str, ...]:
    """Split a SQL target list while preserving commas in calls and quoted text."""
    segments: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(value):
        character = value[index]
        if quote is not None:
            if character == quote:
                if index + 1 < len(value) and value[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if character in {"'", '"'}:
            quote = character
        elif character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            segments.append(value[start:index])
            start = index + 1
        index += 1
    segments.append(value[start:])
    return tuple(segment.strip() for segment in segments if segment.strip())


def _skip_whitespace(value: str, index: int) -> int:
    while index < len(value) and value[index].isspace():
        index += 1
    return index


def _balanced_parenthesized(
    value: str,
    start: int,
) -> tuple[str, int] | None:
    """Return one balanced SQL parenthesis body and the first following offset."""
    if start >= len(value) or value[start] != "(":
        return None
    depth = 0
    index = start
    while index < len(value):
        character = value[index]
        if character in {"'", '"'}:
            span_end = _quoted_sql_span_end(
                value,
                index,
                escape_backslashes=(
                    character == "'"
                    and (
                        _has_sql_prefix(value, index, "e")
                        or _has_sql_prefix(value, index, "u&")
                    )
                ),
            )
            if span_end is None:
                return None
            index = span_end
            continue
        if character == "$":
            dollar_quote = _dollar_quote_at(value, index)
            if dollar_quote is not None:
                end = value.find(dollar_quote, index + len(dollar_quote))
                if end < 0:
                    return None
                index = end + len(dollar_quote)
                continue
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return value[start + 1 : index], index + 1
        index += 1
    return None


def _identifier_at(value: str, index: int) -> tuple[str, int] | None:
    match = re.match(_SQL_IDENTIFIER, value[index:], re.IGNORECASE)
    if match is None:
        return None
    return match.group(0), index + match.end()


def _keyword_at(value: str, index: int, keyword: str) -> int | None:
    match = re.match(rf"{keyword}\b", value[index:], re.IGNORECASE)
    if match is None:
        return None
    return index + match.end()


def _leading_ctes(
    statement: str,
) -> tuple[bool, tuple[tuple[str, str], ...], str] | None:
    """Parse a leading WITH clause into ordered, independently scoped bodies."""
    index = _skip_whitespace(statement, 0)
    next_index = _keyword_at(statement, index, "with")
    if next_index is None:
        return None
    index = _skip_whitespace(statement, next_index)
    recursive_index = _keyword_at(statement, index, "recursive")
    recursive = recursive_index is not None
    if recursive_index is not None:
        index = _skip_whitespace(statement, recursive_index)

    definitions: list[tuple[str, str]] = []
    while True:
        identifier = _identifier_at(statement, index)
        if identifier is None:
            return None
        name_token, index = identifier
        index = _skip_whitespace(statement, index)
        if index < len(statement) and statement[index] == "(":
            columns = _balanced_parenthesized(statement, index)
            if columns is None:
                return None
            _, index = columns
            index = _skip_whitespace(statement, index)
        as_index = _keyword_at(statement, index, "as")
        if as_index is None:
            return None
        index = _skip_whitespace(statement, as_index)
        not_index = _keyword_at(statement, index, "not")
        if not_index is not None:
            index = _skip_whitespace(statement, not_index)
        materialized_index = _keyword_at(statement, index, "materialized")
        if materialized_index is not None:
            index = _skip_whitespace(statement, materialized_index)
        body = _balanced_parenthesized(statement, index)
        if body is None:
            return None
        body_sql, index = body
        definitions.append((_unquote_identifier(name_token), body_sql))
        index = _skip_whitespace(statement, index)
        if index >= len(statement) or statement[index] != ",":
            break
        index = _skip_whitespace(statement, index + 1)
    return recursive, tuple(definitions), statement[index:]


_VIEW_HEAD_MODIFIERS: tuple[str, ...] = (
    "or",
    "replace",
    "temp",
    "temporary",
    "unlogged",
    "recursive",
    "materialized",
)


def _view_query_offset(statement: str) -> int | None:
    """Return the offset of the query body in a ``CREATE ... VIEW ... AS`` statement.

    A view body may open with its own WITH clause, in which case the CTE names are
    only reachable past the ``AS``. ``_leading_ctes`` matches a WITH at offset zero,
    so without this the CTE names of a ``CREATE VIEW ... AS WITH ...`` statement are
    never collected and every later reference to one is misread as an unqualified
    application relation (OMN-15361).
    """
    index = _skip_whitespace(statement, 0)
    create_index = _keyword_at(statement, index, "create")
    if create_index is None:
        return None
    index = _skip_whitespace(statement, create_index)

    while True:
        for modifier in _VIEW_HEAD_MODIFIERS:
            candidate = _keyword_at(statement, index, modifier)
            if candidate is not None:
                index = _skip_whitespace(statement, candidate)
                break
        else:
            break

    view_index = _keyword_at(statement, index, "view")
    if view_index is None:
        return None
    index = _skip_whitespace(statement, view_index)

    for guard in ("if", "not", "exists"):
        candidate = _keyword_at(statement, index, guard)
        if candidate is not None:
            index = _skip_whitespace(statement, candidate)

    identifier = _identifier_at(statement, index)
    if identifier is None:
        return None
    _, index = identifier
    index = _skip_whitespace(statement, index)
    if index < len(statement) and statement[index] == ".":
        qualified = _identifier_at(statement, _skip_whitespace(statement, index + 1))
        if qualified is None:
            return None
        _, index = qualified
        index = _skip_whitespace(statement, index)

    if index < len(statement) and statement[index] == "(":
        columns = _balanced_parenthesized(statement, index)
        if columns is None:
            return None
        _, index = columns
        index = _skip_whitespace(statement, index)

    # ``WITH (security_invoker = true)`` is a view option list, not a CTE. It is only
    # consumed when a parenthesis actually follows, so a CTE WITH is never eaten here.
    options_index = _keyword_at(statement, index, "with")
    if options_index is not None:
        options_start = _skip_whitespace(statement, options_index)
        if options_start < len(statement) and statement[options_start] == "(":
            options = _balanced_parenthesized(statement, options_start)
            if options is None:
                return None
            _, index = options
            index = _skip_whitespace(statement, index)

    as_index = _keyword_at(statement, index, "as")
    if as_index is None:
        return None
    return _skip_whitespace(statement, as_index)


def _is_sql_identifier_character(character: str) -> bool:
    """Return whether a character can continue an unquoted PostgreSQL name."""
    return (
        character == "_"
        or character == "$"
        or character.isalnum()
        or ord(character) >= 128
    )


def _has_sql_prefix(value: str, quote_index: int, prefix: str) -> bool:
    start = quote_index - len(prefix)
    return (
        start >= 0
        and value[start:quote_index].lower() == prefix.lower()
        and (start == 0 or not _is_sql_identifier_character(value[start - 1]))
    )


def _dollar_quote_at(value: str, index: int) -> str | None:
    match = re.match(
        r"\$(?:(?:[a-z_]|[^\x00-\x7f])"
        r"(?:[a-z0-9_]|[^\x00-\x7f])*)?\$",
        value[index:],
        re.IGNORECASE,
    )
    return None if match is None else match.group(0)


def _quoted_sql_span_end(
    value: str,
    start: int,
    *,
    escape_backslashes: bool,
) -> int | None:
    quote = value[start]
    index = start + 1
    while index < len(value):
        character = value[index]
        if escape_backslashes and character == "\\":
            index += 2
            continue
        if character != quote:
            index += 1
            continue
        if index + 1 < len(value) and value[index + 1] == quote:
            index += 2
            continue
        return index + 1
    return None


def _postgresql_lexical_violations(value: str) -> tuple[str, ...]:
    """Reject lexical forms whose boundaries cannot be proven before sqlparse."""
    violations: list[str] = []
    index = 0
    while index < len(value):
        if value.startswith("--", index):
            newline = value.find("\n", index + 2)
            index = len(value) if newline < 0 else newline + 1
            continue
        if value.startswith("/*", index):
            depth = 1
            index += 2
            while index < len(value) and depth:
                if value.startswith("/*", index):
                    depth += 1
                    violations.append(
                        "nested block comments cannot be proven statically"
                    )
                    index += 2
                elif value.startswith("*/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            if depth:
                violations.append(
                    "unterminated block comment cannot be proven statically"
                )
            continue
        if (
            value[index : index + 2].lower() == "u&"
            and index + 2 < len(value)
            and value[index + 2] in {"'", '"'}
            and (index == 0 or not _is_sql_identifier_character(value[index - 1]))
        ):
            violations.append(
                "Unicode-escaped identifiers and literals cannot be proven statically"
            )
            index += 2
            continue
        character = value[index]
        if character in {"'", '"'}:
            span_end = _quoted_sql_span_end(
                value,
                index,
                escape_backslashes=(
                    character == "'"
                    and (
                        _has_sql_prefix(value, index, "e")
                        or _has_sql_prefix(value, index, "u&")
                    )
                ),
            )
            if span_end is None:
                violations.append(
                    "unterminated SQL literal or delimited identifier cannot be "
                    "proven statically"
                )
                break
            index = span_end
            continue
        if character == "$":
            delimiter = _dollar_quote_at(value, index)
            if delimiter is not None:
                end = value.find(delimiter, index + len(delimiter))
                if end < 0:
                    violations.append(
                        "unterminated dollar-quoted SQL body cannot be proven statically"
                    )
                    break
                index = end + len(delimiter)
                continue
        index += 1
    return tuple(dict.fromkeys(violations))


def _mask_sql_string_bodies(
    value: str,
    *,
    mask_quoted_identifiers: bool = False,
) -> str:
    """Blank quoted SQL bodies without changing offsets or target identifiers."""
    masked = list(value)
    index = 0
    while index < len(value):
        if value.startswith("--", index):
            newline = value.find("\n", index + 2)
            stop = len(value) if newline < 0 else newline
            for offset in range(index, stop):
                masked[offset] = " "
            index = stop
            continue
        if value.startswith("/*", index):
            end = value.find("*/", index + 2)
            stop = len(value) if end < 0 else end + 2
            for offset in range(index, stop):
                masked[offset] = " "
            index = stop
            continue
        character = value[index]
        if character == '"':
            span_end = _quoted_sql_span_end(
                value,
                index,
                escape_backslashes=False,
            )
            stop = len(value) if span_end is None else span_end
            if mask_quoted_identifiers:
                for offset in range(index, stop):
                    masked[offset] = " "
            index = stop
            continue
        if character == "'":
            span_end = _quoted_sql_span_end(
                value,
                index,
                escape_backslashes=(
                    _has_sql_prefix(value, index, "e")
                    or _has_sql_prefix(value, index, "u&")
                ),
            )
            stop = len(value) if span_end is None else span_end
            for offset in range(index, stop):
                masked[offset] = " "
            index = stop
            continue
        if character == "$":
            delimiter = _dollar_quote_at(value, index)
            if delimiter is not None:
                end = value.find(delimiter, index + len(delimiter))
                stop = len(value) if end < 0 else end + len(delimiter)
                for offset in range(index, stop):
                    masked[offset] = " "
                index = stop
                continue
        index += 1
    return "".join(masked)


def _contains_unquoted_keyword(value: str, keyword: str) -> bool:
    """Recognize a SQL keyword outside literals and delimited identifiers."""
    masked = _mask_sql_string_bodies(value, mask_quoted_identifiers=True)
    return re.search(rf"\b{keyword}\b", masked, re.IGNORECASE) is not None


def _without_leading_maintenance_options(value: str) -> str:
    """Blank one balanced VACUUM/ANALYZE/CLUSTER/REINDEX option group."""
    masked = _mask_sql_string_bodies(value, mask_quoted_identifiers=True)
    operation = re.match(
        r"^\s*(?:vacuum|analyze|cluster|reindex)\b",
        masked,
        re.IGNORECASE,
    )
    if operation is None:
        return value
    option_start = _skip_whitespace(value, operation.end())
    if option_start >= len(value) or value[option_start] != "(":
        return value
    options = _balanced_parenthesized(value, option_start)
    if options is None:
        return value
    _, option_end = options
    return (
        value[:option_start] + (" " * (option_end - option_start)) + value[option_end:]
    )


def _select_into_target_match(statement: str) -> re.Match[str] | None:
    """Find the top-level SELECT INTO target outside aliases and subqueries."""
    masked = _mask_sql_string_bodies(statement, mask_quoted_identifiers=True)
    if re.match(r"^\s*select\b", masked, re.IGNORECASE) is None:
        return None
    into_offsets = {
        match.start(): match.end()
        for match in re.finditer(r"\binto\b", masked, re.IGNORECASE)
    }
    depth = 0
    index = 0
    while index < len(masked):
        character = masked[index]
        if character == "(":
            depth += 1
        elif character == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and index in into_offsets:
            return _SELECT_INTO_TARGET.match(statement[into_offsets[index] :])
        index += 1
    return None


def _sql_literal_at(value: str, index: int) -> tuple[str, int] | None:
    """Decode one single- or dollar-quoted PostgreSQL literal at ``index``."""
    index = _skip_whitespace(value, index)
    if index >= len(value):
        return None
    if value[index] == "'":
        body: list[str] = []
        index += 1
        while index < len(value):
            character = value[index]
            if character != "'":
                body.append(character)
                index += 1
                continue
            if index + 1 < len(value) and value[index + 1] == "'":
                body.append("'")
                index += 2
                continue
            return "".join(body), index + 1
        return None
    if value[index] != "$":
        return None
    tag = re.match(r"\$(?:[a-z_][a-z0-9_]*)?\$", value[index:], re.IGNORECASE)
    if tag is None:
        return None
    delimiter = tag.group(0)
    body_start = index + len(delimiter)
    body_end = value.find(delimiter, body_start)
    if body_end < 0:
        return None
    return value[body_start:body_end], body_end + len(delimiter)


def _routine_language(statement: str) -> str | None:
    """Return a supported procedural language for one CREATE routine."""
    masked = _mask_sql_string_bodies(statement)
    if (
        re.match(
            r"^\s*create\s+(?:or\s+replace\s+)?(?:function|procedure)\b",
            masked,
            re.IGNORECASE,
        )
        is None
    ):
        return None
    language_match = re.search(
        rf"\blanguage\s+(?P<language>{_SQL_IDENTIFIER})",
        masked,
        re.IGNORECASE,
    )
    if language_match is None:
        return None
    language = _unquote_identifier(language_match.group("language"))
    return language if language in {"sql", "plpgsql"} else None


def _routine_sql_bodies(statement: str) -> tuple[tuple[str, str], ...]:
    """Extract statically inspectable SQL/PLpgSQL routine definition bodies."""
    language = _routine_language(statement)
    if language is None:
        return ()
    masked = _mask_sql_string_bodies(statement)
    language_match = re.search(
        rf"\blanguage\s+{_SQL_IDENTIFIER}",
        masked,
        re.IGNORECASE,
    )
    if language_match is None:  # pragma: no cover - guaranteed by helper
        return ()

    as_matches = tuple(re.finditer(r"\bas\b", masked, re.IGNORECASE))
    if as_matches:
        literal = _sql_literal_at(statement, as_matches[-1].end())
        if literal is not None:
            body, _ = literal
            return ((language, body),)

    if language == "sql":
        sql_body = re.search(
            r"\b(?:return\b|begin\s+atomic\b)",
            masked[language_match.end() :],
            re.IGNORECASE,
        )
        if sql_body is not None:
            start = language_match.end() + sql_body.start()
            return ((language, statement[start:]),)
    return ()


def _routine_body_is_uninspectable(statement: str) -> bool:
    """Fail closed when a supported routine body has no static representation."""
    return _routine_language(statement) is not None and not _routine_sql_bodies(
        statement
    )


def _explained_statement(value: str) -> str:
    """Return the statement wrapped by EXPLAIN, retaining original SQL offsets."""
    masked = _mask_sql_string_bodies(value)
    prefix = re.match(r"^\s*explain\b", masked, re.IGNORECASE)
    if prefix is None:
        return value
    wrapped = re.search(
        r"\b(?:select|insert|update|delete|merge|copy)\b",
        masked[prefix.end() :],
        re.IGNORECASE,
    )
    if wrapped is None:
        return value
    return value[prefix.end() + wrapped.start() :]


def _requires_dynamic_sql_rejection(statement: str) -> bool:
    """Reject procedural execution whose relation targets are runtime strings."""
    masked = _mask_sql_string_bodies(statement)
    if re.match(r"^\s*do\b", masked, re.IGNORECASE) is not None:
        return True
    return any(
        language == "plpgsql" and _contains_unquoted_keyword(body, "execute")
        for language, body in _routine_sql_bodies(statement)
    )


def _record_sql_target(
    *,
    schema_token: str | None,
    name_token: str,
    remaining: str,
    permits_ephemeral: bool,
    permits_system_read: bool,
    cte_names: set[str],
    application_schemas: set[str],
    violations: list[str],
    target_locations: list[tuple[str, str]],
) -> None:
    """Apply one topology-derived qualification verdict to a parsed target."""
    name = _unquote_identifier(name_token)
    if schema_token is None:
        if permits_ephemeral and (
            name in cte_names or remaining.lstrip().startswith("(")
        ):
            return
        if name in _PHYSICALLY_PUBLIC_APPLICATION_TABLES:
            return
        violations.append(
            f"application relation target {name!r} must be schema-qualified"
        )
        return
    schema = _unquote_identifier(schema_token)
    target = f"{schema}.{name}"
    if schema == "public":
        if name in _PHYSICALLY_PUBLIC_APPLICATION_TABLES:
            return
        violations.append(
            f"application relation target {target!r} is prohibited in public"
        )
    elif permits_system_read and schema in _SYSTEM_READ_SCHEMAS:
        return
    elif schema not in application_schemas:
        violations.append(
            f"application relation target {target!r} uses unknown topology schema"
        )
    else:
        target_locations.append((schema, name))


def _normalize_predicate(expression: str | None) -> str:
    if expression is None:
        return ""
    normalized = sqlparse.format(
        expression,
        keyword_case="lower",
        strip_comments=True,
        use_space_around_operators=True,
    ).lower()
    normalized = re.sub(
        r"'app\.tenant_id'\s*::\s*text",
        "'app.tenant_id'",
        normalized,
    )
    return re.sub(r"[\s()\"]+", "", normalized)


_NORMALIZED_TENANT_PREDICATE = _normalize_predicate(CANONICAL_TENANT_PREDICATE)


def _tenant_predicate(identity_column: str) -> str:
    return f"{identity_column} = current_setting('app.tenant_id', true)::uuid"


def _relation_label(state: ModelApplicationDatabaseRelationState) -> str:
    declaration = state.declaration
    return f"{declaration.schema}.{declaration.name}"


def _database_for(
    topology: ModelDeploymentTopology,
    database_ref: str,
    *,
    label: str,
    violations: list[str],
) -> ModelDeploymentTopologyDatabase | None:
    database = topology.databases.get(database_ref)
    if database is None:
        violations.append(
            f"{label}: database_ref {database_ref!r} is absent from typed topology"
        )
    return database


def _schema_owner_for(
    state: ModelApplicationDatabaseRelationState,
    topology: ModelDeploymentTopology,
    violations: list[str],
) -> str | None:
    declaration = state.declaration
    label = _relation_label(state)
    database = _database_for(
        topology,
        declaration.database_ref,
        label=label,
        violations=violations,
    )
    if database is None:
        return None
    schema = database.schemas.get(declaration.schema)
    if schema is None:
        violations.append(
            f"{label}: schema is absent from database_ref {declaration.database_ref!r} "
            "in typed topology"
        )
        return None
    if declaration.domain is None:
        violations.append(f"{label}: relation lacks one topology schema domain")
    elif schema.domain is not declaration.domain:
        violations.append(
            f"{label}: declared domain {declaration.domain.value!r} differs from "
            f"typed topology domain {schema.domain.value!r}"
        )
    return schema.owner


def _validate_tenant_isolation_evidence(
    evidence: ModelApplicationDatabaseTenantIsolationEvidence | None,
    *,
    label: str,
    surface: str,
    violations: list[str],
) -> None:
    if evidence is None:
        violations.append(f"{label}: {surface} requires behavioral evidence")
        return
    expected = evidence.expected_rows_by_tenant
    observed = evidence.observed_rows_by_tenant
    if set(expected) != set(observed):
        violations.append(
            f"{label}: {surface} tenant evidence does not cover the same tenants"
        )
    if expected != observed:
        violations.append(
            f"{label}: {surface} observed tenant results differ from expected results"
        )
    if any(count <= 0 for count in expected.values()):
        violations.append(
            f"{label}: {surface} evidence requires nonzero rows for every tenant"
        )
    if len(set(expected.values())) < 2:
        violations.append(
            f"{label}: {surface} evidence must discriminate tenant result sets"
        )
    if evidence.unset_context_rows != 0:
        violations.append(f"{label}: {surface} exposes rows with tenant context unset")
    if not evidence.malformed_context_denied:
        violations.append(f"{label}: {surface} does not deny malformed tenant context")


def _validate_function_state(
    state: ModelApplicationDatabaseRelationState,
    schema_owner: str | None,
    violations: list[str],
) -> None:
    label = _relation_label(state)
    function = state.function_state
    if function is None:
        violations.append(f"{label}: application function lacks catalog state")
        return
    if function.public_execute:
        violations.append(f"{label}: application function retains PUBLIC EXECUTE")
    if state.declaration.domain is EnumDatabaseSchemaDomain.TENANT:
        _validate_tenant_isolation_evidence(
            function.tenant_isolation_evidence,
            label=label,
            surface="tenant function",
            violations=violations,
        )
    if not function.security_definer:
        return
    if schema_owner is None or function.owner != schema_owner:
        violations.append(
            f"{label}: SECURITY DEFINER owner must equal typed topology schema owner "
            f"{schema_owner!r}; observed {function.owner!r}"
        )
    if function.audit_id is None:
        violations.append(f"{label}: SECURITY DEFINER requires an audit identifier")
    if function.definition_sha256 is None:
        violations.append(f"{label}: SECURITY DEFINER audit requires a definition hash")
    if function.audited_definition_sha256 is None:
        violations.append(
            f"{label}: SECURITY DEFINER requires an independently audited definition hash"
        )
    elif (
        function.definition_sha256 is not None
        and function.definition_sha256 != function.audited_definition_sha256
    ):
        violations.append(
            f"{label}: live SECURITY DEFINER definition does not match the audited "
            "definition hash"
        )
    if function.audited_definition_sha256 is not None and (
        function.audit_id is None
        or function.audited_definition_sha256 not in function.audit_id
    ):
        violations.append(
            f"{label}: SECURITY DEFINER audit identifier must bind its definition hash"
        )
    allowed_paths = {
        ("pg_catalog", "pg_temp"),
        ("pg_catalog", state.declaration.schema, "pg_temp"),
    }
    if function.search_path not in allowed_paths:
        violations.append(
            f"{label}: SECURITY DEFINER requires a fixed safe search_path of "
            "pg_catalog optionally followed by its topology schema, with pg_temp last"
        )


def _validate_tenant_table(
    state: ModelApplicationDatabaseRelationState,
    expected_runtime_principals: tuple[str, ...],
    violations: list[str],
) -> None:
    label = _relation_label(state)
    identity_column = state.tenant_identity_column
    if identity_column is None:
        violations.append(
            f"{label}: tenant table requires an explicit tenant_identity_column"
        )
        identity_column = "tenant_id"
    matching_columns = [
        column for column in state.columns if column.name == identity_column
    ]
    if len(matching_columns) != 1:
        violations.append(
            f"{label}: tenant identity column {identity_column!r} must exist exactly once"
        )
    else:
        column = matching_columns[0]
        if column.data_type.strip().lower() != "uuid":
            violations.append(f"{label}: tenant identity column must be UUID")
        if column.nullable:
            violations.append(f"{label}: tenant identity column must be NOT NULL")
        if column.default_expression is not None:
            violations.append(f"{label}: tenant identity column cannot have a default")

    root_contract = state.identity_root_contract
    if identity_column != "tenant_id" and root_contract is None:
        violations.append(
            f"{label}: non-canonical tenant identity requires an identity-root contract"
        )
    if root_contract is not None:
        expected_relation = _EXPECTED_IDENTITY_ROOT_RELATIONS[root_contract]
        if state.declaration.name != expected_relation:
            violations.append(
                f"{label}: identity-root contract {root_contract.value!r} is reserved "
                f"for relation {expected_relation!r}"
            )
        if state.primary_key_columns != (identity_column,):
            violations.append(
                f"{label}: identity-root column {identity_column!r} must be the exact "
                "primary key"
            )
        control = state.identity_root_control_state
        if control is None:
            violations.append(
                f"{label}: identity-root exception requires live control-operation "
                "evidence"
            )
        else:
            declared_root_operations = set(control.declared_operations)
            observed_root_operations = set(control.observed_operations)
            if declared_root_operations != _EXPECTED_IDENTITY_ROOT_OPERATIONS:
                violations.append(
                    f"{label}: identity-root control authority must declare the exact "
                    "tenant-creation and cross-tenant-enumeration operation set"
                )
            if observed_root_operations != declared_root_operations:
                violations.append(
                    f"{label}: identity-root observed control operations differ from "
                    "the declared operation set"
                )
            if len(control.behavioral_proof_ids) != len(control.observed_operations):
                violations.append(
                    f"{label}: identity-root control operations lack behavioral proof"
                )
            if control.role_can_login:
                violations.append(
                    f"{label}: identity-root control role must be non-runtime NOLOGIN"
                )
            if control.role_superuser:
                violations.append(
                    f"{label}: identity-root control role must not be superuser"
                )
            if not control.role_bypass_rls:
                violations.append(
                    f"{label}: FORCE RLS identity-root operations require an audited "
                    "BYPASSRLS control role, never a widened policy"
                )
            if control.runtime_membership_principals:
                violations.append(
                    f"{label}: runtime principals must have no direct or transitive "
                    "membership path to the identity-root control role; observed "
                    f"{control.runtime_membership_principals!r}"
                )
            if (
                control.runtime_set_role_denied_principals
                != expected_runtime_principals
            ):
                violations.append(
                    f"{label}: identity-root control authority requires SET ROLE "
                    "denial for the exact topology runtime-principal set "
                    f"{expected_runtime_principals!r}; observed "
                    f"{control.runtime_set_role_denied_principals!r}"
                )
    elif state.identity_root_control_state is not None:
        violations.append(
            f"{label}: identity-root control evidence requires an identity-root contract"
        )

    if not state.rls_enabled:
        violations.append(f"{label}: missing ENABLE ROW LEVEL SECURITY")
    if not state.rls_forced:
        violations.append(f"{label}: missing FORCE ROW LEVEL SECURITY")

    canonical_name = state.canonical_policy_name
    if canonical_name is None:
        violations.append(
            f"{label}: tenant table requires an explicit canonical_policy_name"
        )
        canonical_name = "tenant_isolation"
    canonical = [policy for policy in state.policies if policy.name == canonical_name]
    expected_predicate = (
        _NORMALIZED_TENANT_PREDICATE
        if identity_column == "tenant_id"
        else _normalize_predicate(_tenant_predicate(identity_column))
    )
    if len(canonical) != 1:
        violations.append(
            f"{label}: canonical policy {canonical_name!r} must exist exactly once"
        )
    else:
        policy = canonical[0]
        if not policy.permissive or policy.command != "ALL":
            violations.append(
                f"{label}: canonical policy must be the sole permissive ALL policy"
            )
        if policy.roles != _EXPECTED_CANONICAL_POLICY_ROLES:
            violations.append(
                f"{label}: canonical policy role scope must be exactly "
                f"{_EXPECTED_CANONICAL_POLICY_ROLES!r}; observed {policy.roles!r}"
            )
        if _normalize_predicate(policy.using_expression) != expected_predicate:
            violations.append(
                f"{label}: canonical policy USING predicate drift from declared "
                f"identity column {identity_column!r}"
            )
        if _normalize_predicate(policy.with_check_expression) != expected_predicate:
            violations.append(
                f"{label}: canonical policy WITH CHECK predicate drift from declared "
                f"identity column {identity_column!r}"
            )

    extras = [policy for policy in state.policies if policy.name != canonical_name]
    declared = set(state.declared_restrictive_policy_names)
    proofs = state.restrictive_policy_proofs
    for policy in extras:
        if policy.permissive:
            violations.append(
                f"{label}: extra permissive policy {policy.name!r} widens access"
            )
        if policy.name not in declared:
            violations.append(
                f"{label}: restrictive policy {policy.name!r} is not contract-declared"
            )
        if not proofs.get(policy.name):
            violations.append(
                f"{label}: restrictive policy {policy.name!r} lacks behavioral proof"
            )
    missing_policy_rows = declared.difference(policy.name for policy in extras)
    if missing_policy_rows:
        violations.append(
            f"{label}: declared restrictive policies are absent: "
            f"{sorted(missing_policy_rows)!r}"
        )
    undeclared_proofs = set(proofs).difference(declared)
    if undeclared_proofs:
        violations.append(
            f"{label}: behavioral proofs name undeclared restrictive policies: "
            f"{sorted(undeclared_proofs)!r}"
        )


def _validate_tenant_state(
    state: ModelApplicationDatabaseRelationState,
    schema_owner: str | None,
    expected_runtime_principals: tuple[str, ...],
    violations: list[str],
) -> None:
    kind = state.declaration.kind
    label = _relation_label(state)
    if kind is EnumApplicationRelationKind.TABLE:
        _validate_tenant_table(state, expected_runtime_principals, violations)
    elif kind is EnumApplicationRelationKind.VIEW:
        if state.security_invoker is not True:
            violations.append(f"{label}: tenant view must set security_invoker=true")
        _validate_tenant_isolation_evidence(
            state.view_tenant_isolation_evidence,
            label=label,
            surface="tenant view",
            violations=violations,
        )
    elif kind is EnumApplicationRelationKind.MATERIALIZED_VIEW:
        violations.append(
            f"{label}: tenant materialized view is prohibited; use a tenant table"
        )
    elif kind is EnumApplicationRelationKind.FOREIGN_TABLE:
        violations.append(
            f"{label}: tenant foreign table is prohibited; use a tenant table"
        )
    elif kind is EnumApplicationRelationKind.FUNCTION:
        _validate_function_state(state, schema_owner, violations)


def _validate_non_tenant_state(
    state: ModelApplicationDatabaseRelationState,
    schema_owner: str | None,
    violations: list[str],
) -> None:
    label = _relation_label(state)
    if any(column.name == "tenant_id" for column in state.columns):
        violations.append(
            f"{label}: internal/catalog relation cannot declare tenant_id"
        )
    if state.rls_enabled or state.rls_forced:
        violations.append(
            f"{label}: internal/catalog relation cannot enable tenant RLS"
        )
    if state.policies:
        violations.append(
            f"{label}: internal/catalog relation cannot declare tenant policy"
        )
    if state.tenant_identity_column is not None:
        violations.append(
            f"{label}: internal/catalog relation cannot declare tenant identity"
        )
    source_columns = [
        column for column in state.columns if column.name == "source_tenant_id"
    ]
    if source_columns:
        source = source_columns[0]
        if source.data_type.strip().lower() != "uuid":
            violations.append(f"{label}: source_tenant_id provenance must be UUID")
        if not source.nullable:
            violations.append(f"{label}: source_tenant_id provenance must be nullable")
        if source.default_expression is not None:
            violations.append(
                f"{label}: source_tenant_id provenance cannot have a default"
            )
        if state.source_tenant_provenance_contract != "non_authoritative_provenance":
            violations.append(
                f"{label}: source_tenant_id requires a non-authoritative provenance contract"
            )
        if "source_tenant_id" in state.primary_key_columns:
            violations.append(
                f"{label}: source_tenant_id provenance cannot be authoritative primary-key state"
            )

        generated_dependencies = {
            column.name: tuple(
                candidate.name
                for candidate in state.columns
                if column.generated_expression is not None
                and re.search(
                    rf"(?<![a-zA-Z0-9_]){re.escape(candidate.name)}"
                    r"(?![a-zA-Z0-9_])",
                    column.generated_expression,
                )
            )
            for column in state.columns
            if column.generated_expression is not None
        }

        def expand_generated_authority(columns: Sequence[str]) -> tuple[str, ...]:
            expanded: list[str] = []
            pending = list(columns)
            while pending:
                column_name = pending.pop(0)
                if column_name in expanded:
                    continue
                expanded.append(column_name)
                pending.extend(generated_dependencies.get(column_name, ()))
            return tuple(expanded)

        authoritative_uses: tuple[tuple[str, Sequence[str]], ...] = (
            (
                "uniqueness",
                expand_generated_authority(
                    tuple(
                        column
                        for column_set in state.unique_index_column_sets
                        for column in column_set
                    )
                ),
            ),
            (
                "foreign key",
                expand_generated_authority(
                    tuple(
                        column
                        for column_set in state.foreign_key_column_sets
                        for column in column_set
                    )
                ),
            ),
            ("partition", expand_generated_authority(state.partition_key_columns)),
            (
                "deduplication",
                expand_generated_authority(state.deduplication_key_columns),
            ),
            (
                "authorization",
                expand_generated_authority(state.authorization_dependency_columns),
            ),
            (
                "write eligibility",
                expand_generated_authority(state.write_eligibility_dependency_columns),
            ),
        )
        for authority_kind, columns in authoritative_uses:
            if "source_tenant_id" in columns:
                violations.append(
                    f"{label}: source_tenant_id provenance cannot drive "
                    f"{authority_kind}"
                )
    elif state.source_tenant_provenance_contract is not None:
        violations.append(
            f"{label}: source-tenant provenance contract requires source_tenant_id"
        )
    if state.declaration.kind is EnumApplicationRelationKind.FUNCTION:
        _validate_function_state(state, schema_owner, violations)


def validate_application_database_relation_states(
    states: Sequence[ModelApplicationDatabaseRelationState],
    topology: ModelDeploymentTopology,
) -> tuple[str, ...]:
    """Validate classification, topology placement, RLS, views, and functions."""
    violations: list[str] = []
    if not states:
        violations.append("application relation state set cannot be empty")
    identities = [state.declaration.identity for state in states]
    for identity, count in Counter(identities).items():
        if count != 1:
            violations.append(
                f"application relation {identity!r} has {count} catalog states; expected 1"
            )

    for state in states:
        declaration = state.declaration
        label = _relation_label(state)
        if declaration.owner_declaration is None:
            violations.append(f"{label}: relation lacks exactly one owner declaration")
        if declaration.schema == "public":
            violations.append(
                f"{label}: application relations are prohibited in public"
            )
        schema_owner = _schema_owner_for(state, topology, violations)
        domain = declaration.domain
        if domain is None:
            continue
        if domain is EnumDatabaseSchemaDomain.TENANT:
            database = topology.databases.get(declaration.database_ref)
            expected_runtime_principals = (
                ()
                if database is None
                else tuple(
                    sorted(binding.principal for binding in database.bindings.values())
                )
            )
            _validate_tenant_state(
                state,
                schema_owner,
                expected_runtime_principals,
                violations,
            )
        else:
            _validate_non_tenant_state(state, schema_owner, violations)
    return tuple(sorted(set(violations)))


def validate_application_database_catalog_census(
    states: Sequence[ModelApplicationDatabaseRelationState],
    observed: Sequence[ModelApplicationDatabaseCatalogIdentity],
    topology: ModelDeploymentTopology,
    *,
    authoritative_identities: Sequence[ModelApplicationDatabaseCatalogIdentity]
    | None = None,
) -> tuple[str, ...]:
    """Require an exact contract-to-catalog census, including absence of extras."""
    violations = list(validate_application_database_relation_states(states, topology))
    if not observed:
        violations.append("application catalog census cannot be empty")
    expected_identities = (
        [identity.identity for identity in authoritative_identities]
        if authoritative_identities is not None
        else [
            (
                state.declaration.schema,
                state.declaration.name,
                EnumApplicationInventoryObjectKind(state.declaration.kind.value),
                state.declaration.function_signature,
            )
            for state in states
        ]
    )
    if not expected_identities:
        violations.append("authoritative application catalog cannot be empty")
    observed_identities = [identity.identity for identity in observed]
    for identity, count in Counter(observed_identities).items():
        if count != 1:
            violations.append(
                f"application catalog identity {identity!r} appears {count} times"
            )
    missing = set(expected_identities).difference(observed_identities)
    extra = set(observed_identities).difference(expected_identities)

    def identity_sort_key(
        identity: tuple[
            str,
            str,
            EnumApplicationInventoryObjectKind,
            str | None,
        ],
    ) -> tuple[str, ...]:
        return tuple("" if item is None else str(item) for item in identity)

    if missing:
        violations.append(
            "application catalog census is missing declared objects: "
            f"{sorted(missing, key=identity_sort_key)!r}"
        )
    if extra:
        violations.append(
            "application catalog census contains undeclared objects: "
            f"{sorted(extra, key=identity_sort_key)!r}"
        )
    return tuple(sorted(set(violations)))


def application_database_created_catalog_identities(
    sql: str,
) -> tuple[ModelApplicationDatabaseCatalogIdentity, ...]:
    """Project application objects created by SQL into typed catalog identities."""
    if _postgresql_lexical_violations(sql):
        return ()
    identities: list[ModelApplicationDatabaseCatalogIdentity] = []
    cleaned = sqlparse.format(sql, strip_comments=True)
    for statement in sqlparse.split(cleaned):
        match = _CREATED_APPLICATION_OBJECT.match(statement)
        if match is None:
            select_statement = statement
            leading_ctes = _leading_ctes(statement)
            if leading_ctes is not None:
                _, _, select_statement = leading_ctes
            select_into = _select_into_target_match(select_statement)
            if select_into is None or select_into.group("schema") is None:
                continue
            identities.append(
                ModelApplicationDatabaseCatalogIdentity(
                    schema=_unquote_identifier(select_into.group("schema")),
                    name=_unquote_identifier(select_into.group("name")),
                    kind=EnumApplicationInventoryObjectKind.TABLE,
                )
            )
            continue
        kind_token = " ".join(match.group("kind").lower().split())
        schema_token = match.group("schema")
        if kind_token == "extension":
            schema_match = re.search(
                rf"\bschema\s+(?P<schema>{_SQL_IDENTIFIER})",
                statement[match.end() :],
                re.IGNORECASE,
            )
            if schema_match is None:
                continue
            schema_token = schema_match.group("schema")
        if schema_token is None:
            continue

        schema = _unquote_identifier(schema_token)
        name = _unquote_identifier(match.group("name"))
        kind = _CREATED_KIND_MAP[kind_token]
        function_signature: str | None = None
        remainder = statement[match.end() :]
        remainder_offset = _skip_whitespace(remainder, 0)
        if kind in {
            EnumApplicationInventoryObjectKind.FUNCTION,
            EnumApplicationInventoryObjectKind.PROCEDURE,
            EnumApplicationInventoryObjectKind.AGGREGATE,
        }:
            signature = _balanced_parenthesized(remainder, remainder_offset)
            if signature is not None:
                signature_body, _ = signature
                function_signature = f"({signature_body.strip()})"
            if (
                kind is EnumApplicationInventoryObjectKind.FUNCTION
                and _contains_unquoted_keyword(remainder, "window")
            ):
                kind = EnumApplicationInventoryObjectKind.WINDOW_FUNCTION
        elif kind is EnumApplicationInventoryObjectKind.TYPE:
            if remainder[remainder_offset:].startswith("("):
                kind = EnumApplicationInventoryObjectKind.BASE_TYPE
            elif re.match(
                r"as\s+range\b",
                remainder[remainder_offset:],
                re.IGNORECASE,
            ):
                kind = EnumApplicationInventoryObjectKind.RANGE_TYPE

        identities.append(
            ModelApplicationDatabaseCatalogIdentity(
                schema=schema,
                name=name,
                kind=kind,
                function_signature=function_signature,
            )
        )
        if kind is EnumApplicationInventoryObjectKind.RANGE_TYPE:
            multirange_match = re.search(
                rf"\bmultirange_type_name\s*=\s*"
                rf"(?:(?P<schema>{_SQL_IDENTIFIER})\s*\.\s*)?"
                rf"(?P<name>{_SQL_IDENTIFIER})",
                remainder,
                re.IGNORECASE,
            )
            if multirange_match is not None:
                multirange_schema = multirange_match.group("schema")
                identities.append(
                    ModelApplicationDatabaseCatalogIdentity(
                        schema=(
                            schema
                            if multirange_schema is None
                            else _unquote_identifier(multirange_schema)
                        ),
                        name=_unquote_identifier(multirange_match.group("name")),
                        kind=EnumApplicationInventoryObjectKind.MULTIRANGE_TYPE,
                    )
                )
    return tuple(identities)


def load_application_database_ownership_identities(
    manifest_paths: Sequence[Path],
) -> tuple[ModelApplicationDatabaseCatalogIdentity, ...]:
    """Load one exact owner-backed catalog identity from every manifest row."""
    declarations: dict[
        tuple[
            str,
            str,
            EnumApplicationInventoryObjectKind,
            str | None,
        ],
        list[tuple[str, str]],
    ] = {}

    def record(
        *,
        schema: str,
        name: str,
        kind: EnumApplicationInventoryObjectKind,
        function_signature: str | None,
        owner: str,
        source_path: Path,
    ) -> None:
        identity = (schema, name, kind, function_signature)
        declarations.setdefault(identity, []).append((owner, str(source_path)))

    routine_kinds = {
        EnumApplicationInventoryObjectKind.FUNCTION,
        EnumApplicationInventoryObjectKind.AGGREGATE,
        EnumApplicationInventoryObjectKind.WINDOW_FUNCTION,
        EnumApplicationInventoryObjectKind.PROCEDURE,
    }
    for path in sorted(manifest_paths):
        manifest = load_service_ownership_manifest(path)
        authority = manifest.owner_declaration or f"service:{manifest.service}"
        table_identities = {
            (table.database_ref, table.schema, table.name)
            for table in manifest.db_io.db_tables
        }
        for table in manifest.db_io.db_tables:
            record(
                schema=table.schema,
                name=table.name,
                kind=EnumApplicationInventoryObjectKind.TABLE,
                function_signature=None,
                owner=authority,
                source_path=path,
            )
        for relation in manifest.relation_evidence:
            if relation.database_ref is None or relation.schema is None:
                continue
            if relation.owner_declaration != authority:
                raise ValueError(
                    f"{path}: catalog relation {relation.name!r} owner "
                    f"{relation.owner_declaration!r} conflicts with manifest "
                    f"authority {authority!r}"
                )
            if relation.kind is EnumApplicationRelationKind.TABLE:
                typed_identity = (
                    relation.database_ref,
                    relation.schema,
                    relation.name,
                )
                if typed_identity not in table_identities:
                    raise ValueError(
                        f"{path}: table evidence {relation.name!r} lacks one typed "
                        "db_io ownership declaration"
                    )
                continue
            kind = EnumApplicationInventoryObjectKind(relation.kind.value)
            if kind in routine_kinds and relation.function_signature is None:
                raise ValueError(
                    f"{path}: routine relation {relation.name!r} requires an exact "
                    "function_signature"
                )
            record(
                schema=relation.schema,
                name=relation.name,
                kind=kind,
                function_signature=relation.function_signature,
                owner=relation.owner_declaration,
                source_path=path,
            )
        for database_object in manifest.database_objects:
            if database_object.database_ref is None or database_object.schema is None:
                continue
            if database_object.owner_declaration != authority:
                raise ValueError(
                    f"{path}: catalog object {database_object.name!r} owner "
                    f"{database_object.owner_declaration!r} conflicts with manifest "
                    f"authority {authority!r}"
                )
            kind = EnumApplicationInventoryObjectKind(database_object.kind.value)
            if kind in routine_kinds and database_object.function_signature is None:
                raise ValueError(
                    f"{path}: routine object {database_object.name!r} requires an "
                    "exact function_signature"
                )
            record(
                schema=database_object.schema,
                name=database_object.name,
                kind=kind,
                function_signature=database_object.function_signature,
                owner=database_object.owner_declaration,
                source_path=path,
            )

    duplicate_declarations = {
        identity: owners
        for identity, owners in declarations.items()
        if len(owners) != 1
    }
    if duplicate_declarations:
        raise ValueError(
            "application catalog identities require exactly one ownership "
            f"declaration: {duplicate_declarations!r}"
        )
    return tuple(
        ModelApplicationDatabaseCatalogIdentity(
            schema=schema,
            name=name,
            kind=kind,
            function_signature=function_signature,
        )
        for schema, name, kind, function_signature in sorted(
            declarations,
            key=lambda identity: tuple(
                "" if item is None else str(item) for item in identity
            ),
        )
    )


def _analyze_sql_fragment(
    fragment: str,
    *,
    visible_ctes: set[str],
    application_schemas: set[str],
    violations: list[str],
    target_locations: list[tuple[str, str]],
) -> None:
    """Analyze one SQL scope, recursively checking each data-modifying CTE body."""
    lexical_violations = _postgresql_lexical_violations(fragment)
    if lexical_violations:
        violations.extend(lexical_violations)
        return
    if _requires_dynamic_sql_rejection(fragment):
        violations.append(
            "procedural block contains dynamic SQL whose relation targets cannot "
            "be proven statically"
        )
        return
    if _routine_body_is_uninspectable(fragment):
        violations.append(
            "SQL or PL/pgSQL routine body cannot be proven statically; use a "
            "standard single-quoted, dollar-quoted, RETURN, or BEGIN ATOMIC body"
        )
        return

    for _language, body in _routine_sql_bodies(fragment):
        _analyze_sql_fragment(
            body,
            visible_ctes=set(visible_ctes),
            application_schemas=application_schemas,
            violations=violations,
            target_locations=target_locations,
        )

    masked_fragment = _mask_sql_string_bodies(fragment)
    target_fragment = _without_leading_maintenance_options(fragment)
    masked_target_fragment = _mask_sql_string_bodies(target_fragment)
    if any(
        pattern.fullmatch(masked_target_fragment)
        for pattern in _NON_EXACT_MAINTENANCE_OPERATIONS
    ):
        violations.append(
            "application maintenance operation must target an exact "
            "schema-qualified table target"
        )
    extension = re.match(
        rf"^\s*create\s+extension\s+(?:if\s+not\s+exists\s+)?"
        rf"(?P<name>{_SQL_IDENTIFIER})",
        masked_fragment,
        re.IGNORECASE,
    )
    if extension is not None:
        extension_schema = re.search(
            rf"\bwith\s+schema\s+(?P<schema>{_SQL_IDENTIFIER})",
            masked_fragment[extension.end() :],
            re.IGNORECASE,
        )
        if extension_schema is None:
            violations.append(
                "CREATE EXTENSION must be schema-qualified with an explicit topology "
                "WITH SCHEMA target"
            )
        else:
            schema = _unquote_identifier(extension_schema.group("schema"))
            if schema == "public":
                violations.append(
                    "application extension target schema 'public' is prohibited"
                )
            elif schema not in application_schemas:
                violations.append(
                    f"application extension target schema {schema!r} uses unknown "
                    "topology schema"
                )
    if (
        re.match(
            rf"^\s*create\s+type\s+"
            rf"(?:(?:{_SQL_IDENTIFIER})\s*\.\s*)?{_SQL_IDENTIFIER}\s+"
            r"as\s+range\b",
            masked_fragment,
            re.IGNORECASE,
        )
        is not None
        and re.search(r"\bmultirange_type_name\s*=", masked_fragment, re.IGNORECASE)
        is None
    ):
        violations.append(
            "CREATE TYPE AS RANGE requires an explicit MULTIRANGE_TYPE_NAME so "
            "the generated catalog identity is authoritative"
        )

    leading_ctes = _leading_ctes(fragment)
    if leading_ctes is not None:
        recursive, definitions, tail = leading_ctes
        definition_names = {name for name, _ in definitions}
        prior_names = set(visible_ctes)
        for name, body in definitions:
            body_names = (
                prior_names.union(definition_names) if recursive else set(prior_names)
            )
            _analyze_sql_fragment(
                body,
                visible_ctes=body_names,
                application_schemas=application_schemas,
                violations=violations,
                target_locations=target_locations,
            )
            prior_names.add(name)
        _analyze_sql_fragment(
            tail,
            visible_ctes=visible_ctes.union(definition_names),
            application_schemas=application_schemas,
            violations=violations,
            target_locations=target_locations,
        )
        return

    view_offset = _view_query_offset(fragment)
    if view_offset is not None:
        view_body_ctes = _leading_ctes(fragment[view_offset:])
        if view_body_ctes is not None:
            recursive, definitions, tail = view_body_ctes
            definition_names = {name for name, _ in definitions}
            prior_names = set(visible_ctes)
            for name, body in definitions:
                body_names = (
                    prior_names.union(definition_names)
                    if recursive
                    else set(prior_names)
                )
                _analyze_sql_fragment(
                    body,
                    visible_ctes=body_names,
                    application_schemas=application_schemas,
                    violations=violations,
                    target_locations=target_locations,
                )
                prior_names.add(name)
            # Re-join the view head to the post-WITH tail so the view's own name is
            # still validated as a real relation target. Only the CTE definitions are
            # elided here; they were each analyzed above under their own scope.
            _analyze_sql_fragment(
                f"{fragment[:view_offset]} {tail}",
                visible_ctes=visible_ctes.union(definition_names),
                application_schemas=application_schemas,
                violations=violations,
                target_locations=target_locations,
            )
            return

    explained = _explained_statement(target_fragment)
    main_dml_tail = _main_dml_tail(explained)
    variants: tuple[str, ...] = (target_fragment,)
    if main_dml_tail != target_fragment:
        variants = (target_fragment, main_dml_tail)
    is_privilege_statement = (
        re.match(
            r"^\s*(?:grant|revoke)\b",
            masked_target_fragment,
            re.IGNORECASE,
        )
        is not None
    )
    for variant in variants:
        masked_variant = _mask_sql_string_bodies(variant)
        for pattern, permits_ephemeral, permits_system_read in _RELATION_TARGETS:
            if permits_ephemeral and is_privilege_statement:
                # GRANT/REVOKE FROM names principals, not relation read targets.
                continue
            for match in pattern.finditer(masked_variant):
                _record_sql_target(
                    schema_token=match.group("schema"),
                    name_token=match.group("name"),
                    remaining=masked_variant[match.end() :],
                    permits_ephemeral=permits_ephemeral,
                    permits_system_read=permits_system_read,
                    cte_names=visible_ctes,
                    application_schemas=application_schemas,
                    violations=violations,
                    target_locations=target_locations,
                )
        select_into = _select_into_target_match(variant)
        if select_into is not None:
            _record_sql_target(
                schema_token=select_into.group("schema"),
                name_token=select_into.group("name"),
                remaining="",
                permits_ephemeral=False,
                permits_system_read=False,
                cte_names=visible_ctes,
                application_schemas=application_schemas,
                violations=violations,
                target_locations=target_locations,
            )

    for pattern, permits_ephemeral, permits_system_read in _RELATION_LISTS:
        if permits_ephemeral and is_privilege_statement:
            # REVOKE ... FROM names principals, not relation read targets.
            continue
        for list_match in pattern.finditer(masked_target_fragment):
            targets = _split_top_level_commas(list_match.group("targets"))
            for target in targets:
                target_match = _LIST_TARGET.match(target)
                if target_match is None:
                    continue
                _record_sql_target(
                    schema_token=target_match.group("schema"),
                    name_token=target_match.group("name"),
                    remaining=target[target_match.end() :],
                    permits_ephemeral=permits_ephemeral,
                    permits_system_read=permits_system_read,
                    cte_names=visible_ctes,
                    application_schemas=application_schemas,
                    violations=violations,
                    target_locations=target_locations,
                )


def _analyze_application_database_sql(
    sql: str,
    topology: ModelDeploymentTopology,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Return qualification violations and all qualified application targets."""
    violations: list[str] = []
    target_locations: list[tuple[str, str]] = []
    application_schemas = {
        schema_name
        for database in topology.databases.values()
        for schema_name in database.schemas
    }
    lexical_violations = _postgresql_lexical_violations(sql)
    if lexical_violations:
        return lexical_violations, ()
    cleaned = sqlparse.format(sql, strip_comments=True)
    for statement in sqlparse.split(cleaned):
        _analyze_sql_fragment(
            statement,
            visible_ctes=set(),
            application_schemas=application_schemas,
            violations=violations,
            target_locations=target_locations,
        )
    return tuple(sorted(set(violations))), tuple(sorted(set(target_locations)))


def lint_application_database_sql(
    sql: str,
    topology: ModelDeploymentTopology,
) -> tuple[str, ...]:
    """Reject public, unknown, or unqualified relation targets in changed SQL."""
    violations, _ = _analyze_application_database_sql(sql, topology)
    return violations


def application_database_sql_target_locations(
    sql: str,
    topology: ModelDeploymentTopology,
) -> tuple[tuple[str, str], ...]:
    """Return every qualified topology-application target referenced by SQL."""
    _, target_locations = _analyze_application_database_sql(sql, topology)
    return target_locations


_EXPLICIT_TARGET_KIND_MAP: dict[str, frozenset[EnumApplicationInventoryObjectKind]] = {
    "foreign table": frozenset({EnumApplicationInventoryObjectKind.FOREIGN_TABLE}),
    "table": frozenset({EnumApplicationInventoryObjectKind.TABLE}),
    "view": frozenset({EnumApplicationInventoryObjectKind.VIEW}),
    "materialized view": frozenset(
        {EnumApplicationInventoryObjectKind.MATERIALIZED_VIEW}
    ),
    "function": frozenset(
        {
            EnumApplicationInventoryObjectKind.FUNCTION,
            EnumApplicationInventoryObjectKind.WINDOW_FUNCTION,
        }
    ),
    "procedure": frozenset({EnumApplicationInventoryObjectKind.PROCEDURE}),
    "aggregate": frozenset({EnumApplicationInventoryObjectKind.AGGREGATE}),
    "sequence": frozenset({EnumApplicationInventoryObjectKind.SEQUENCE}),
    "type": frozenset(
        {
            EnumApplicationInventoryObjectKind.TYPE,
            EnumApplicationInventoryObjectKind.BASE_TYPE,
            EnumApplicationInventoryObjectKind.RANGE_TYPE,
            EnumApplicationInventoryObjectKind.MULTIRANGE_TYPE,
        }
    ),
    "domain": frozenset({EnumApplicationInventoryObjectKind.TYPE}),
    "column": _RELATION_OBJECT_KINDS,
}
_EXPLICIT_OBJECT_OPERATION = re.compile(
    r"^\s*(?P<operation>alter|drop)\s+"
    r"(?P<kind>foreign\s+table|materialized\s+view|table|view|function|"
    r"procedure|aggregate|sequence|type|domain)\s+"
    r"(?:if\s+exists\s+)?(?:only\s+)?(?P<targets>[\s\S]*)$",
    re.IGNORECASE,
)
_QUALIFIED_TARGET = re.compile(
    rf"^\s*(?P<schema>{_SQL_IDENTIFIER})\s*\.\s*"
    rf"(?P<name>{_SQL_IDENTIFIER})",
    re.IGNORECASE,
)
_COMMENT_OBJECT_OPERATION = re.compile(
    r"^\s*comment\s+on\s+"
    r"(?P<kind>foreign\s+table|materialized\s+view|table|view|column|function|"
    r"procedure|aggregate|sequence|type|domain)\s+"
    r"(?P<target>[\s\S]*?)\s+is\b",
    re.IGNORECASE,
)


def _target_requirement(
    target: str,
    *,
    allowed_kinds: frozenset[EnumApplicationInventoryObjectKind],
    routine_identity: bool,
    application_schemas: set[str],
) -> ApplicationDatabaseSqlTargetRequirement | None:
    """Parse one already qualification-checked target into exact authority shape."""
    match = _QUALIFIED_TARGET.match(target)
    if match is None:
        return None
    schema = _unquote_identifier(match.group("schema"))
    if schema not in application_schemas:
        return None
    name = _unquote_identifier(match.group("name"))
    signature: str | None = None
    if routine_identity:
        offset = _skip_whitespace(target, match.end())
        parenthesized = _balanced_parenthesized(target, offset)
        if parenthesized is not None:
            signature_body, _ = parenthesized
            signature = f"({signature_body.strip()})"
    return ApplicationDatabaseSqlTargetRequirement(
        schema=schema,
        name=name,
        allowed_kinds=allowed_kinds,
        function_signature=signature,
    )


def application_database_sql_target_requirements(
    sql: str,
    topology: ModelDeploymentTopology,
) -> tuple[ApplicationDatabaseSqlTargetRequirement, ...]:
    """Return kind- and overload-aware ownership requirements for SQL targets."""
    if _postgresql_lexical_violations(sql):
        return ()
    application_schemas = {
        schema_name
        for database in topology.databases.values()
        for schema_name in database.schemas
    }
    all_locations = set(application_database_sql_target_locations(sql, topology))
    created_locations = {
        (identity.schema, identity.name)
        for identity in application_database_created_catalog_identities(sql)
    }
    requirements: set[ApplicationDatabaseSqlTargetRequirement] = set()
    explicitly_typed_locations: set[tuple[str, str]] = set()
    cleaned = sqlparse.format(sql, strip_comments=True)
    for raw_statement in sqlparse.split(cleaned):
        statement = _mask_sql_string_bodies(raw_statement).rstrip("; ")
        explicit = _EXPLICIT_OBJECT_OPERATION.match(statement)
        if explicit is not None:
            kind_token = " ".join(explicit.group("kind").lower().split())
            allowed_kinds = _EXPLICIT_TARGET_KIND_MAP[kind_token]
            target_text = re.split(
                r"\s+(?:cascade|restrict)\b",
                explicit.group("targets"),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            targets = (
                _split_top_level_commas(target_text)
                if explicit.group("operation").lower() == "drop"
                else (target_text,)
            )
            routine_identity = bool(allowed_kinds.intersection(_ROUTINE_OBJECT_KINDS))
            for target in targets:
                requirement = _target_requirement(
                    target,
                    allowed_kinds=allowed_kinds,
                    routine_identity=routine_identity,
                    application_schemas=application_schemas,
                )
                if requirement is not None:
                    requirements.add(requirement)
                    explicitly_typed_locations.add(requirement.location)

        comment = _COMMENT_OBJECT_OPERATION.match(statement)
        if comment is not None:
            kind_token = " ".join(comment.group("kind").lower().split())
            allowed_kinds = _EXPLICIT_TARGET_KIND_MAP[kind_token]
            requirement = _target_requirement(
                comment.group("target"),
                allowed_kinds=allowed_kinds,
                routine_identity=bool(
                    allowed_kinds.intersection(_ROUTINE_OBJECT_KINDS)
                ),
                application_schemas=application_schemas,
            )
            if requirement is not None:
                requirements.add(requirement)
                explicitly_typed_locations.add(requirement.location)

        for call in re.finditer(
            rf"\bcall\s+(?P<target>{_SQL_IDENTIFIER}\s*\.\s*{_SQL_IDENTIFIER})",
            statement,
            re.IGNORECASE,
        ):
            requirement = _target_requirement(
                statement[call.start("target") :],
                allowed_kinds=frozenset({EnumApplicationInventoryObjectKind.PROCEDURE}),
                routine_identity=True,
                application_schemas=application_schemas,
            )
            if requirement is not None:
                requirements.add(requirement)
                explicitly_typed_locations.add(requirement.location)

        for execution in re.finditer(
            rf"\bexecute\s+(?P<kind>function|procedure)\s+"
            rf"(?P<target>{_SQL_IDENTIFIER}\s*\.\s*{_SQL_IDENTIFIER})",
            statement,
            re.IGNORECASE,
        ):
            execution_kind = execution.group("kind").lower()
            allowed_kinds = (
                frozenset(
                    {
                        EnumApplicationInventoryObjectKind.FUNCTION,
                        EnumApplicationInventoryObjectKind.WINDOW_FUNCTION,
                    }
                )
                if execution_kind == "function"
                else frozenset({EnumApplicationInventoryObjectKind.PROCEDURE})
            )
            requirement = _target_requirement(
                statement[execution.start("target") :],
                allowed_kinds=allowed_kinds,
                routine_identity=False,
                application_schemas=application_schemas,
            )
            if requirement is not None:
                # PostgreSQL trigger functions have catalog identity ``()`` even
                # when CREATE TRIGGER supplies TG_ARGV values.
                if execution_kind == "function":
                    requirement = ApplicationDatabaseSqlTargetRequirement(
                        schema=requirement.schema,
                        name=requirement.name,
                        allowed_kinds=requirement.allowed_kinds,
                        function_signature="()",
                    )
                requirements.add(requirement)
                explicitly_typed_locations.add(requirement.location)

        privilege = re.match(
            r"^\s*(?:grant|revoke)\s+[\s\S]*?\s+on\s+"
            r"(?:(?P<kind>foreign\s+table|materialized\s+view|table|view|function|"
            r"procedure|sequence|type|domain)\s+)?"
            r"(?P<targets>[\s\S]*?)\s+(?:to|from)\b",
            statement,
            re.IGNORECASE,
        )
        if privilege is not None:
            kind_token = privilege.group("kind")
            allowed_kinds = (
                _RELATION_OBJECT_KINDS
                if kind_token is None
                else _EXPLICIT_TARGET_KIND_MAP[" ".join(kind_token.lower().split())]
            )
            routine_identity = bool(allowed_kinds.intersection(_ROUTINE_OBJECT_KINDS))
            for target in _split_top_level_commas(privilege.group("targets")):
                requirement = _target_requirement(
                    target,
                    allowed_kinds=allowed_kinds,
                    routine_identity=routine_identity,
                    application_schemas=application_schemas,
                )
                if requirement is not None:
                    requirements.add(requirement)
                    explicitly_typed_locations.add(requirement.location)

    for schema, name in all_locations.difference(
        created_locations.union(explicitly_typed_locations)
    ):
        requirements.add(
            ApplicationDatabaseSqlTargetRequirement(
                schema=schema,
                name=name,
                allowed_kinds=_RELATION_OBJECT_KINDS,
            )
        )
    return tuple(
        sorted(
            requirements,
            key=lambda requirement: (
                requirement.schema,
                requirement.name,
                tuple(sorted(kind.value for kind in requirement.allowed_kinds)),
                requirement.function_signature or "",
            ),
        )
    )


def validate_application_database_pool_identities(
    identities: Sequence[ModelApplicationDatabasePoolIdentity],
    topology: ModelDeploymentTopology,
    *,
    database_ref: str = "application",
) -> tuple[str, ...]:
    """Prove one topology database with exact, distinct workload users."""
    violations: list[str] = []
    database = _database_for(
        topology,
        database_ref,
        label="application pool evidence",
        violations=violations,
    )
    if database is None:
        return tuple(sorted(set(violations)))
    expected_pool_users = {
        binding_name: binding.principal
        for binding_name, binding in database.bindings.items()
    }
    by_pool = {identity.pool: identity for identity in identities}
    if len(by_pool) != len(identities):
        violations.append("application pool evidence contains duplicate pool names")
    if set(by_pool) != set(expected_pool_users):
        violations.append(
            "application pool evidence must cover the exact typed topology binding set"
        )
    databases = {identity.current_database for identity in identities}
    if databases != {database.physical_name}:
        violations.append(
            "application pools must resolve to one physical database "
            f"{database.physical_name!r}; observed {sorted(databases)!r}"
        )
    for pool, expected_user in expected_pool_users.items():
        identity = by_pool.get(pool)
        if identity is not None and identity.current_user != expected_user:
            violations.append(
                f"pool {pool!r} expected current_user {expected_user!r}, got "
                f"{identity.current_user!r}"
            )
    users = [identity.current_user for identity in identities]
    if len(set(users)) != len(users):
        violations.append("application pool current_user values must be distinct")
    return tuple(sorted(set(violations)))


__all__ = [
    "ApplicationDatabaseSqlTargetRequirement",
    "CANONICAL_TENANT_PREDICATE",
    "application_database_created_catalog_identities",
    "application_database_function_definition_sha256",
    "application_database_sql_target_locations",
    "application_database_sql_target_requirements",
    "lint_application_database_sql",
    "load_application_database_ownership_identities",
    "validate_application_database_catalog_census",
    "validate_application_database_pool_identities",
    "validate_application_database_relation_states",
]
