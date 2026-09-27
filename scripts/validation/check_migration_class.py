#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Every forward migration declares its class; rule RB-2's barrier is computed here.

OMN-19344 (unified verification plan, row R7, rule RB). A PASS lab-pass receipt
names a composition that was good against the schema it ran on. Redeploying an
older composition over a schema that moved on since can fail, so rule RB makes a
rollback candidate eligible only when every migration applied after it is
``expand-only`` (additive and backward-compatible with the older code), or when
a forward-only migration's paired down-migration has itself been executed on the
lab and that execution is recorded against the migration id.

Until this change nothing declared a class in machine-readable form -- the words
appeared only in SQL comments, and inconsistently. This module owns:

* the declaration, ``config/migration_classes.yaml``: one entry per forward
  migration (the flat stream, every vendored node stream, and the intelligence
  stream), keyed by its path relative to ``docker/migrations``. Applied
  migration bytes are immutable (OMN-16705), so the class lives beside the SQL,
  never inside it.
* the DDL checker: a file declared ``expand-only`` must contain no destructive
  statement. The analyzer is deliberately conservative -- when in doubt it
  reports a finding, which forces the author to declare ``forward-only`` or
  ``contract``. A wrong ``forward-only`` costs an eligible rollback candidate;
  a wrong ``expand-only`` redeploys old code over a schema it cannot read. Only
  the first error is acceptable.
* the executed-down record, ``config/migration_down_executions.yaml``: a
  down-migration lifts RB-2's barrier only through a record of a PASS execution
  on a lab surface, bound to the migration id and to the sha256 of BOTH the
  forward and the down script, so editing either afterwards voids the record.
  A down-script merely existing on disk lifts nothing: none of the 65 files in
  ``docker/migrations/rollback/`` had ever been executed against a database by
  any workflow or test when this was written.

Classes:
  expand-only   additive only: new tables, nullable or defaulted columns, new
                indexes, new views, grants, comments, seed inserts. Never a
                rollback barrier.
  forward-only  anything else, or anything the analyzer cannot prove additive.
                A rollback barrier unless a recorded down execution lifts it.
  contract      the destructive half of an expand/contract pair (removes what
                current code no longer reads). A barrier exactly like
                forward-only; the name records intent for the reader.

An undeclared migration is refused by the checker and treated as forward-only
by :func:`rb2_barrier` (rule RB-2: "A migration with no declared class is
treated as forward-only").

Usage:
  check_migration_class.py                 # check the tree; exit 1 on violations
  check_migration_class.py --suggest       # print the analyzer's reading for
                                           # every UNDECLARED file (never writes)
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_MIGRATIONS_ROOT: Final[Path] = REPO_ROOT / "docker" / "migrations"
DEFAULT_MANIFEST: Final[Path] = REPO_ROOT / "config" / "migration_classes.yaml"
DEFAULT_EXECUTIONS: Final[Path] = (
    REPO_ROOT / "config" / "migration_down_executions.yaml"
)

EXPAND_ONLY: Final[str] = "expand-only"
FORWARD_ONLY: Final[str] = "forward-only"
CONTRACT: Final[str] = "contract"
CLASSES: Final[tuple[str, ...]] = (EXPAND_ONLY, FORWARD_ONLY, CONTRACT)

MANIFEST_SCHEMA_VERSION: Final[int] = 1

#: Surfaces a down-migration execution may be recorded from: the lab proof
#: surface classes of the pre-PR proof-surfaces runbook, plus the .201 compose
#: dev lane. A governed lane (stability-test, judge, a collaborator lane) or any
#: production surface is never a place to execute a down-migration, so a record
#: naming one is refused at load time.
_LAB_SURFACE: Final[re.Pattern[str]] = re.compile(
    r"^(mac-scratch|[0-9]{1,3}-scratchdb|dogfood-[0-9]{1,3}|compose-dev)"
    r"(:[A-Za-z0-9._-]+)?$"
)
_SHA256: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def enumerate_forward_migrations(root: Path) -> tuple[str, ...]:
    """Every forward migration the lane runners apply, keyed relative to ``root``.

    ``forward/*.sql`` (the flat stream), ``forward/nodes/<node>/*.sql`` (the
    vendored node streams, exactly the runner's ``-mindepth 2 -maxdepth 2``
    walk) and ``intelligence/*.sql``. ``forward/_ledger/`` holds the runner's
    own bootstrap, not a migration; ``rollback/`` holds down-scripts.
    """
    if not (root / "forward").is_dir():
        msg = f"forward-migration directory not found: {root / 'forward'}"
        raise ValueError(msg)
    keys: list[str] = [f"forward/{p.name}" for p in (root / "forward").glob("*.sql")]
    nodes = root / "forward" / "nodes"
    if nodes.is_dir():
        keys.extend(
            f"forward/nodes/{p.parent.name}/{p.name}" for p in nodes.glob("*/*.sql")
        )
    intelligence = root / "intelligence"
    if intelligence.is_dir():
        keys.extend(f"intelligence/{p.name}" for p in intelligence.glob("*.sql"))
    return tuple(sorted(keys))


# ---------------------------------------------------------------------------
# DDL analyzer
# ---------------------------------------------------------------------------


#: Emitted into the stripped text where the lexer could not find the end of a
#: literal or a dollar-quoted body. Everything after it is unreadable, so it is
#: a finding in its own right: an unbalanced quote must never hide a DROP.
_UNREADABLE: Final[str] = " __UNREADABLE__ "


def _strip(sql: str) -> str:
    """Comments removed, string literals emptied, dollar-quoted bodies isolated.

    A dollar-quoted body (a DO block, a function body) is code the migration
    runs or installs, so its CONTENT is analysed -- but lexed on its own, so an
    apostrophe inside ``$$the agent's id$$`` cannot open a literal that swallows
    the statements after the body. The delimiters become statement breaks. A
    single-quoted literal is data, so its content is dropped: ``COMMENT ON ...
    IS 'drop x'`` is not a DROP. ``E'...'`` literals honour backslash escapes.
    An unterminated literal or body emits :data:`_UNREADABLE`.
    """
    out: list[str] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if sql.startswith("--", i):
            j = sql.find("\n", i)
            i = n if j == -1 else j
            continue
        if sql.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if sql.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif sql.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            out.append(" ")
            continue
        if ch == "'":
            escape = (
                i > 0
                and sql[i - 1] in "Ee"
                and (i < 2 or not (sql[i - 2].isalnum() or sql[i - 2] == "_"))
            )
            i += 1
            closed = False
            while i < n:
                if (escape and sql[i] == "\\") or (
                    sql[i] == "'" and i + 1 < n and sql[i + 1] == "'"
                ):
                    i += 2
                elif sql[i] == "'":
                    i += 1
                    closed = True
                    break
                else:
                    i += 1
            out.append("''" if closed else _UNREADABLE)
            continue
        if ch == "$":
            m = re.match(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$", sql[i:])
            if m:
                tag = m.group(0)
                end = sql.find(tag, i + len(tag))
                if end == -1:
                    out.append(_UNREADABLE)
                    break
                out.append(" ; ")
                out.append(_strip(sql[i + len(tag) : end]))
                out.append(" ; ")
                i = end + len(tag)
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _segments(sql: str) -> list[str]:
    text = _strip(sql)
    segments = []
    for raw in text.split(";"):
        seg = re.sub(r"\s+", " ", raw).strip().upper()
        if seg:
            segments.append(seg)
    return segments


_IDENT: Final[str] = r'(?:"[^"]+"|[A-Z_][A-Z0-9_$]*)'
_QNAME: Final[str] = rf"{_IDENT}(?:\s*\.\s*{_IDENT})*"


def _norm_name(name: str) -> str:
    """``schema.relation``, lower-cased; an unqualified name is ``public.``."""
    parts = [p.strip().strip('"').lower() for p in name.split(".")]
    if len(parts) == 1:
        parts.insert(0, "public")
    return ".".join(parts[-2:])


def _created_tables(segments: Iterable[str]) -> set[str]:
    """Tables this file certainly creates: a plain ``CREATE TABLE`` only.

    ``CREATE TABLE IF NOT EXISTS`` is a no-op on a table that already exists, so
    a tightening after it may land on a table older code reads; it earns no
    exemption.
    """
    pattern = re.compile(
        rf"\bCREATE\s+(?:UNLOGGED\s+|TEMP(?:ORARY)?\s+)?TABLE\s+(?!IF\s+NOT\s+EXISTS\b)({_QNAME})"
    )
    created: set[str] = set()
    for seg in segments:
        for m in pattern.finditer(seg):
            created.add(_norm_name(m.group(1)))
    return created


def _split_top_level(text: str, sep: str = ",") -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    return [p for p in parts if p]


#: Anywhere in a segment. Each is destructive or opaque whatever it targets.
_DENY: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("unreadable text (unbalanced quote)", re.compile(r"__UNREADABLE__")),
    ("DROP", re.compile(r"\bDROP\b")),
    ("RENAME", re.compile(r"\bRENAME\b")),
    ("TRUNCATE", re.compile(r"\bTRUNCATE\b")),
    ("DELETE (data rewrite)", re.compile(r"\bDELETE\s+FROM\b")),
    (
        "UPDATE (data rewrite)",
        re.compile(
            rf"(?<!\bON )\bUPDATE\s+(?:ONLY\s+)?{_QNAME}(?:\s+(?:AS\s+)?{_IDENT})?\s+SET\b"
        ),
    ),
    (
        "ON CONFLICT DO UPDATE (data rewrite)",
        re.compile(r"\bON\s+CONFLICT\b.*\bDO\s+UPDATE\b"),
    ),
    ("REVOKE", re.compile(r"\bREVOKE\b")),
    (
        "EXECUTE (dynamic SQL the checker cannot read)",
        re.compile(r"(?<!\bGRANT )\bEXECUTE\b(?!\s+(?:FUNCTION|PROCEDURE|ON)\b)"),
    ),
    (
        "CREATE OR REPLACE (replaces an existing object)",
        re.compile(r"\bCREATE\s+OR\s+REPLACE\b"),
    ),
    ("CREATE RULE", re.compile(r"\bCREATE\s+RULE\b")),
    ("CALL (opaque procedure)", re.compile(r"\bCALL\s+[A-Z_\"]")),
    ("RESTRICTIVE policy", re.compile(r"\bAS\s+RESTRICTIVE\b")),
    ("SECURITY LABEL", re.compile(r"\bSECURITY\s+LABEL\b")),
)

#: A SELECT or PERFORM at the head of a statement (top level, or first in a
#: plpgsql block or branch). Any function it calls, other than the read-only
#: built-ins below, runs code the checker cannot read.
_CALLING_STATEMENT: Final[re.Pattern[str]] = re.compile(
    r"(?:^|\b(?:BEGIN|THEN|ELSE|LOOP|DO)\s+)(?:SELECT|PERFORM)\b(.*)$"
)
#: A name followed by ``(``, unless it follows ``AS`` (a column-alias list such
#: as ``unnest(...) AS k(key_position)`` is not a call).
_CALL_SITE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\bAS )\b([A-Z_][A-Z0-9_]*(?:\.[A-Z_][A-Z0-9_]*)?)\s*\("
)
_READ_ONLY_CALLS: Final[frozenset[str]] = frozenset(
    {
        # SQL syntax that looks like a call.
        "THEN",
        "ELSE",
        "WHEN",
        "END",
        "IN",
        "EXISTS",
        "ANY",
        "ALL",
        "VALUES",
        "CAST",
        "COALESCE",
        "NULLIF",
        "GREATEST",
        "LEAST",
        "CASE",
        "AND",
        "OR",
        "NOT",
        "FILTER",
        "OVER",
        "WHERE",
        "ON",
        "FROM",
        "SELECT",
        "ROW",
        "ARRAY",
        "DISTINCT",
        "USING",
        # Read-only built-ins.
        "COUNT",
        "MIN",
        "MAX",
        "SUM",
        "AVG",
        "BOOL_AND",
        "BOOL_OR",
        "ARRAY_AGG",
        "STRING_AGG",
        "NOW",
        "FORMAT",
        "TO_REGCLASS",
        "TO_REGTYPE",
        "TO_REGPROCEDURE",
        "TO_REGNAMESPACE",
        "TO_REGROLE",
        "CURRENT_SETTING",
        "SET_CONFIG",
        "LOWER",
        "UPPER",
        "LENGTH",
        "TRIM",
        "CONCAT",
        "REPLACE",
        "SUBSTRING",
        "QUOTE_IDENT",
        "QUOTE_LITERAL",
        "HAS_TABLE_PRIVILEGE",
        "HAS_SCHEMA_PRIVILEGE",
        "HAS_FUNCTION_PRIVILEGE",
        "PG_HAS_ROLE",
        "OBJ_DESCRIPTION",
        "COL_DESCRIPTION",
        "FORMAT_TYPE",
        "PG_GET_FUNCTION_IDENTITY_ARGUMENTS",
        "PG_GET_CONSTRAINTDEF",
        "PG_GET_INDEXDEF",
        "PG_GET_VIEWDEF",
        "PG_ADVISORY_XACT_LOCK",
        "PG_TRY_ADVISORY_XACT_LOCK",
        "GEN_RANDOM_UUID",
        "JSONB_BUILD_OBJECT",
        "JSONB_TYPEOF",
        "HAS_SEQUENCE_PRIVILEGE",
        "HAS_DATABASE_PRIVILEGE",
        "PG_GET_SERIAL_SEQUENCE",
        "PG_GET_USERBYID",
        "PG_GET_EXPR",
        "ACLEXPLODE",
        "UNNEST",
        "LEFT",
        "RIGHT",
        "CHAR_LENGTH",
        "CURRENT_DATABASE",
        "CURRENT_SCHEMA",
    }
)

_ALTER: Final[re.Pattern[str]] = re.compile(r"\bALTER\s+([A-Z]+(?:\s+[A-Z]+)?)\b")
_ALTER_TABLE: Final[re.Pattern[str]] = re.compile(
    rf"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?({_QNAME})\s*(.*)$"
)
_ADD_COLUMN: Final[re.Pattern[str]] = re.compile(
    r"^ADD\s+(?:COLUMN\s+)?(?:IF\s+NOT\s+EXISTS\s+)?(?!CONSTRAINT\b|PRIMARY\b|UNIQUE\b|CHECK\b|FOREIGN\b|EXCLUDE\b)"
)
_ON_TABLE_OBJECTS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "UNIQUE INDEX on an existing table",
        re.compile(rf"\bCREATE\s+UNIQUE\s+INDEX\b.*?\bON\s+(?:ONLY\s+)?({_QNAME})"),
    ),
    (
        "TRIGGER on an existing table",
        re.compile(rf"\bCREATE\s+(?:CONSTRAINT\s+)?TRIGGER\b.*?\bON\s+({_QNAME})"),
    ),
)


def _alter_table_findings(seg: str, created: set[str]) -> list[str]:
    m = _ALTER_TABLE.search(seg)
    if not m:
        return ["ALTER TABLE (unparseable)"]
    table, actions = _norm_name(m.group(1)), m.group(2)
    if table in created:
        # Created by a plain CREATE TABLE in this same file: no older
        # composition has read or written it.
        return []
    findings = []
    for action in _split_top_level(actions):
        if not _ADD_COLUMN.match(action):
            head = " ".join(action.split()[:3])
            if action.startswith("ADD"):
                findings.append(f"ALTER TABLE {table} {head} (ADD CONSTRAINT class)")
            elif "ROW LEVEL SECURITY" in action:
                findings.append(f"ALTER TABLE {table} {head} (ROW LEVEL SECURITY)")
            elif action.startswith("ALTER "):
                findings.append(f"ALTER TABLE {table} ALTER COLUMN ({head})")
            else:
                findings.append(f"ALTER TABLE {table} {head}")
            continue
        if re.search(r"\bNOT\s+NULL\b", action) and not re.search(
            r"\bDEFAULT\b", action
        ):
            findings.append(
                f"ALTER TABLE {table} ADD COLUMN ... NOT NULL without DEFAULT"
            )
    return findings


def _alter_findings(seg: str, created: set[str]) -> list[str]:
    findings: list[str] = []
    for m in _ALTER.finditer(seg):
        kind = m.group(1)
        rest = seg[m.start() :]
        if kind.startswith("TABLE"):
            findings.extend(_alter_table_findings(rest, created))
        elif kind.startswith("TYPE"):
            if not re.search(r"\bADD\s+VALUE\b", rest):
                findings.append("ALTER TYPE (other than ADD VALUE)")
        elif kind.startswith("DEFAULT PRIVILEGES"):
            continue  # a GRANT form is additive; a REVOKE form is caught by _DENY
        elif kind.startswith("COLUMN"):
            continue  # reached through its ALTER TABLE, already analysed
        else:
            findings.append(f"ALTER {kind.split()[0]}")
    return findings


def _call_findings(seg: str) -> list[str]:
    m = _CALLING_STATEMENT.search(seg)
    if not m:
        return []
    opaque = sorted(
        {
            name
            for name in _CALL_SITE.findall(m.group(1))
            if name.split(".")[-1] not in _READ_ONLY_CALLS
        }
    )
    return [
        f"function call {name}() (runs code the checker cannot read)" for name in opaque
    ]


def destructive_findings(sql: str) -> list[str]:
    """Every statement in ``sql`` that makes it NOT expand-only.

    Empty means every statement is one of the additive shapes rule RB-1(b)
    names. Conservative by construction: an unrecognised ALTER, dynamic SQL, a
    call into code it cannot read, an object replaced in place, or text it
    cannot lex is a finding.
    """
    segments = _segments(sql)
    created = _created_tables(segments)
    findings: list[str] = []
    for seg in segments:
        for label, pattern in _DENY:
            if pattern.search(seg):
                findings.append(label)
        findings.extend(_alter_findings(seg, created))
        findings.extend(_call_findings(seg))
        for label, pattern in _ON_TABLE_OBJECTS:
            for m in pattern.finditer(seg):
                if _norm_name(m.group(1)) not in created:
                    findings.append(f"{label} ({_norm_name(m.group(1))})")
    # Stable, de-duplicated, in first-seen order.
    return list(dict.fromkeys(findings))


def violations_for(key: str, text: str, declared: str | None) -> list[str]:
    """Why ``key`` (its bytes ``text``) may not carry the class ``declared``."""
    if declared is None:
        return [
            f"{key}: no declared class -- add it to config/migration_classes.yaml "
            f"as one of {', '.join(CLASSES)} (rule RB treats it as forward-only)"
        ]
    if declared not in CLASSES:
        return [
            f"{key}: {declared!r} is not a declared class (one of {', '.join(CLASSES)})"
        ]
    if declared != EXPAND_ONLY:
        return []
    return [
        f"{key}: declared expand-only but carries {finding}; declare it "
        "forward-only or contract, or make the migration additive"
        for finding in destructive_findings(text)
    ]


# ---------------------------------------------------------------------------
# Declarations and records
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        msg = f"declaration file not found: {path}"
        raise ValueError(msg)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"{path}: top level must be a mapping"
        raise ValueError(msg)
    if data.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        msg = f"{path}: schema_version must be {MANIFEST_SCHEMA_VERSION}"
        raise ValueError(msg)
    return data


def load_manifest(path: Path) -> dict[str, str]:
    """``{migration key: class}``. Raises on a malformed file, never returns {}."""
    data = _load_yaml(path)
    # safe_load keeps the LAST of two duplicated keys, so a second line could
    # silently override a class declaration; count the entry lines instead.
    entry_keys = re.findall(
        r"^[ ]{2}([^\s#][^:]*):[ ]", path.read_text(encoding="utf-8"), re.MULTILINE
    )
    duplicates = sorted(k for k, c in Counter(entry_keys).items() if c > 1)
    if duplicates:
        msg = f"{path}: duplicated migration key(s): {', '.join(duplicates)}"
        raise ValueError(msg)
    migrations = data.get("migrations")
    if not isinstance(migrations, dict) or not migrations:
        msg = f"{path}: 'migrations' must be a non-empty mapping"
        raise ValueError(msg)
    manifest: dict[str, str] = {}
    for key, value in migrations.items():
        if not isinstance(key, str) or not isinstance(value, str):
            msg = f"{path}: entry {key!r} must map a path string to a class string"
            raise ValueError(msg)
        manifest[key] = value
    return manifest


@dataclass(frozen=True)
class ModelDownExecution:
    """One recorded execution of a down-migration on a lab surface."""

    migration: str
    down_script: str
    forward_sha256: str
    down_sha256: str
    surface: str
    database: str
    executed_at: str
    outcome: str
    evidence: str

    def __post_init__(self) -> None:
        for name in (
            "migration",
            "down_script",
            "surface",
            "database",
            "executed_at",
            "outcome",
            "evidence",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                msg = f"down execution for {self.migration!r}: {name} is required"
                raise ValueError(msg)
        for name in ("forward_sha256", "down_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.match(value):
                msg = f"down execution for {self.migration!r}: {name} must be a sha256 hex digest"
                raise ValueError(msg)
        if not _LAB_SURFACE.match(self.surface):
            msg = (
                f"down execution for {self.migration!r}: surface {self.surface!r} is "
                "not a lab surface (mac-scratch, <host>-scratchdb, dogfood-<host> or "
                "compose-dev); a governed or production lane never records one"
            )
            raise ValueError(msg)
        if self.outcome not in ("PASS", "FAIL"):
            msg = f"down execution for {self.migration!r}: outcome must be PASS or FAIL"
            raise ValueError(msg)


def load_executions(path: Path) -> list[ModelDownExecution]:
    data = _load_yaml(path)
    rows = data.get("executions")
    if rows is None:
        rows = []
    if not isinstance(rows, list):
        msg = f"{path}: 'executions' must be a list"
        raise ValueError(msg)
    records = []
    for row in rows:
        if not isinstance(row, dict):
            msg = f"{path}: every execution must be a mapping"
            raise ValueError(msg)
        try:
            records.append(ModelDownExecution(**{str(k): v for k, v in row.items()}))
        except TypeError as exc:
            msg = f"{path}: malformed execution record {row.get('migration')!r}: {exc}"
            raise ValueError(msg) from exc
    return records


def _sha256_of(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record_problem(record: ModelDownExecution, root: Path) -> str | None:
    """Why ``record`` does not lift a barrier today, or None when it does."""
    if record.outcome != "PASS":
        return f"recorded outcome is {record.outcome}"
    fwd = _sha256_of(root / record.migration)
    if fwd is None:
        return f"forward script {record.migration} is absent"
    if fwd != record.forward_sha256:
        return f"forward_sha256 does not match the current bytes of {record.migration}"
    down = _sha256_of(root / record.down_script)
    if down is None:
        return f"down script {record.down_script} is absent"
    if down != record.down_sha256:
        return f"down_sha256 does not match the current bytes of {record.down_script}"
    return None


def rb2_barrier(
    key: str,
    declared: str | None,
    executions: Sequence[ModelDownExecution],
    root: Path,
) -> str | None:
    """Rule RB-2 for one applied migration: the barrier reason, or None.

    None means a composition older than ``key`` is NOT made ineligible by it:
    it is expand-only, or it is forward-only/contract with a PASS lab execution
    of its paired down-migration recorded against its id and current bytes.
    Anything else is a barrier, and the reason names the migration.
    """
    if declared == EXPAND_ONLY:
        return None
    if declared is None:
        return f"{key}: no declared class, so treated as forward-only with no recorded lab execution of a down-migration"
    if declared not in CLASSES:
        return f"{key}: undeclarable class {declared!r}, treated as forward-only"
    problems = []
    for record in executions:
        if record.migration != key:
            continue
        problem = _record_problem(record, root)
        if problem is None:
            return None
        problems.append(problem)
    if not problems:
        return f"{key}: {declared} with no recorded lab execution of a down-migration"
    return (
        f"{key}: {declared}; no recorded lab execution lifts it ({'; '.join(problems)})"
    )


def check_tree(
    root: Path,
    manifest: Mapping[str, str],
    executions: Sequence[ModelDownExecution],
) -> list[str]:
    """Every violation of the declaration and record contract, in a stable order."""
    violations: list[str] = []
    keys = enumerate_forward_migrations(root)
    for key in keys:
        text = (root / key).read_text(encoding="utf-8")
        violations.extend(violations_for(key, text, manifest.get(key)))
    present = set(keys)
    violations.extend(
        f"{key}: declared in config/migration_classes.yaml but no such forward migration exists"
        for key in sorted(set(manifest) - present)
    )
    for record in executions:
        if record.migration not in present:
            violations.append(
                f"{record.migration}: down execution recorded for a migration that does not exist"
            )
            continue
        problem = _record_problem(record, root)
        if problem is not None and record.outcome == "PASS":
            violations.append(
                f"{record.migration}: recorded down execution is stale ({problem}); "
                "re-execute and re-record it"
            )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_MIGRATIONS_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--executions", type=Path, default=DEFAULT_EXECUTIONS)
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="print the analyzer's reading for each undeclared migration; writes nothing",
    )
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        executions = load_executions(args.executions)
        if args.suggest:
            for key in enumerate_forward_migrations(args.root):
                if key in manifest:
                    continue
                findings = destructive_findings(
                    (args.root / key).read_text(encoding="utf-8")
                )
                reading = FORWARD_ONLY if findings else EXPAND_ONLY
                print(f"  {key}: {reading}  # {'; '.join(findings) or 'additive'}")
            return 0
        violations = check_tree(args.root, manifest, executions)
    except ValueError as exc:
        print(f"migration-class: ERROR {exc}", file=sys.stderr)
        return 2
    if violations:
        print(
            f"migration-class: {len(violations)} violation(s) (OMN-19344, rule RB):",
            file=sys.stderr,
        )
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        return 1
    barriers = sum(1 for v in manifest.values() if v != EXPAND_ONLY)
    print(
        f"migration-class: OK -- {len(manifest)} forward migrations declared "
        f"({barriers} rollback barriers, {len(executions)} recorded down executions)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
