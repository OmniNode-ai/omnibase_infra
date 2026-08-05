# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Shape helpers for the node-owned migration corpus (OMN-15376).

Parses ``docker/migrations/forward/nodes/<node>/*.sql`` well enough to answer
two questions that the shape-drift gate and its execution proof both need:

* which columns does a ``CREATE TABLE IF NOT EXISTS`` DECLARE, and
* which columns does the same file RECONCILE with a guarded
  ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS``.

The operator fence is READ FROM the single-sourced manifest
(``docker/migrations/forward/fenced-node-migrations.yaml``, OMN-15349) rather
than restated here: a second hand-maintained copy of that list is exactly the
cross-repo drift OMN-15336 was filed about. Before OMN-15349 this read from
the runner script's own literal copy; the runner no longer carries one (it
parses the same manifest at runtime), so this helper now points at the
manifest directly instead.

Ticket: OMN-15376
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_MIGRATIONS_DIR = REPO_ROOT / "docker" / "migrations" / "forward" / "nodes"
FORWARD_RUNNER = REPO_ROOT / "scripts" / "run-forward-migrations.sh"
FENCE_MANIFEST = (
    REPO_ROOT / "docker" / "migrations" / "forward" / "fenced-node-migrations.yaml"
)

_CREATE_TABLE_GUARDED = re.compile(
    r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+([A-Za-z0-9_.\"]+)\s*\(", re.I
)
_ADD_COLUMN_GUARDED = re.compile(
    r"ALTER\s+TABLE\s+([A-Za-z0-9_.\"]+)\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS\s+"
    r'("?[A-Za-z_][A-Za-z0-9_]*"?)',
    re.I,
)
_MANIFEST_ID_LINE = re.compile(r'^\s*-\s*id:\s*"([^"]*)"', re.MULTILINE)

# A leading keyword that marks a table-level constraint rather than a column.
_CONSTRAINT_HEADS = frozenset(
    {"CONSTRAINT", "PRIMARY", "UNIQUE", "CHECK", "FOREIGN", "EXCLUDE", "LIKE"}
)
_COLUMN_TERMINATORS = (
    "NOT NULL",
    "NULL",
    "PRIMARY KEY",
    "UNIQUE",
    "REFERENCES",
    "CHECK",
    "DEFAULT",
    "GENERATED",
    "COLLATE",
    "CONSTRAINT",
    "DEFERRABLE",
)


def mask_literals(sql: str) -> str:
    """Blank out comments, dollar-quoted bodies and string literals in place.

    Byte offsets are preserved so the mask can be used for paren/comma scanning
    while substrings are still taken from the ORIGINAL text.
    """
    out = list(sql)
    i, n = 0, len(sql)
    while i < n:
        if sql.startswith("--", i):
            end = sql.find("\n", i)
            end = n if end == -1 else end
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            end = n if end == -1 else end + 2
        elif sql.startswith("$$", i):
            end = sql.find("$$", i + 2)
            end = n if end == -1 else end + 2
        elif sql[i] == "'":
            end = _string_end(sql, i)
        else:
            i += 1
            continue
        for k in range(i, end):
            out[k] = " "
        i = end
    return "".join(out)


def _string_end(sql: str, start: int) -> int:
    j = start + 1
    n = len(sql)
    while j < n:
        if sql[j] == "'":
            if j + 1 < n and sql[j + 1] == "'":
                j += 2
                continue
            return j + 1
        j += 1
    return n


def _balanced(text: str, start: int) -> int:
    """Index just past the balanced paren group opening at ``text[start]``."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


@dataclass(frozen=True)
class DeclaredColumn:
    """One column of a ``CREATE TABLE``: identifier plus its type/default text."""

    name: str
    type_text: str
    default_text: str
    generated: bool

    def seed_ddl_fragment(self) -> str:
        """Column definition for a drift seed: type + DEFAULT, no constraints."""
        fragment = f"{self.name} {self.type_text}"
        if self.default_text and not self.generated:
            fragment += f" DEFAULT {self.default_text}"
        return fragment


@dataclass(frozen=True)
class GuardedTable:
    """A ``CREATE TABLE IF NOT EXISTS`` and the columns it declares."""

    qualified_name: str
    bare_name: str
    columns: tuple[DeclaredColumn, ...]


def _split_body_items(body: str) -> list[str]:
    masked = mask_literals(body)
    items, depth, start = [], 0, 0
    for i, ch in enumerate(masked):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            items.append(body[start:i])
            start = i + 1
    items.append(body[start:])
    cleaned = []
    for item in items:
        stripped = re.sub(r"--[^\n]*", "", item)
        stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.S).strip()
        if stripped:
            cleaned.append(" ".join(stripped.split()))
    return cleaned


def _parse_column(item: str) -> DeclaredColumn | None:
    match = re.match(r'("?[A-Za-z_][A-Za-z0-9_]*"?)\s+(.*)$', item, re.S)
    if match is None:
        return None
    rest = match.group(2)
    masked = mask_literals(rest)
    type_end = len(rest)
    default_text = ""
    generated = False
    i = 0
    while i < len(masked):
        if masked[i] == "(":
            i = _balanced(masked, i)
            continue
        upper = masked[i:].upper()
        hit = next(
            (
                kw
                for kw in _COLUMN_TERMINATORS
                if upper.startswith(kw)
                and (i == 0 or not (masked[i - 1].isalnum() or masked[i - 1] == "_"))
                and (
                    i + len(kw) >= len(masked)
                    or not (masked[i + len(kw)].isalnum() or masked[i + len(kw)] == "_")
                )
            ),
            None,
        )
        if hit is None:
            i += 1
            continue
        type_end = min(type_end, i)
        if hit == "DEFAULT":
            start = i + len(hit)
            # Whitespace is skipped against the ORIGINAL text: mask_literals()
            # blanks string bodies to spaces, so skipping on the mask would walk
            # straight past a literal default and yield "DEFAULT ::jsonb".
            while start < len(rest) and rest[start] == " ":
                start += 1
            end = _scan_default(masked, start)
            default_text = rest[start:end].strip()
            i = end
        elif hit == "GENERATED":
            generated = True
            i += len(hit)
        else:
            i += len(hit)
    return DeclaredColumn(
        name=match.group(1),
        type_text=rest[:type_end].strip(),
        default_text=default_text,
        generated=generated,
    )


def _scan_default(masked: str, start: int) -> int:
    i = start
    while i < len(masked):
        if masked[i] == "(":
            i = _balanced(masked, i)
            continue
        upper = masked[i:].upper()
        if any(
            upper.startswith(kw)
            and not (masked[i - 1].isalnum() or masked[i - 1] == "_")
            for kw in _COLUMN_TERMINATORS
        ):
            return i
        i += 1
    return len(masked)


def guarded_create_tables(sql: str) -> list[GuardedTable]:
    """Every ``CREATE TABLE IF NOT EXISTS`` in ``sql``, with declared columns."""
    masked = mask_literals(sql)
    tables: list[GuardedTable] = []
    for match in _CREATE_TABLE_GUARDED.finditer(masked):
        qualified = match.group(1)
        open_idx = match.end() - 1
        close_idx = _balanced(masked, open_idx) - 1
        body = sql[open_idx + 1 : close_idx]
        columns = []
        for item in _split_body_items(body):
            head = re.match(r'"?([A-Za-z_][A-Za-z0-9_]*)', item)
            if head is not None and head.group(1).upper() in _CONSTRAINT_HEADS:
                continue
            column = _parse_column(item)
            if column is not None:
                columns.append(column)
        tables.append(
            GuardedTable(
                qualified_name=qualified,
                bare_name=qualified.split(".")[-1].strip('"'),
                columns=tuple(columns),
            )
        )
    return tables


def reconciled_columns(sql: str, table: GuardedTable) -> set[str]:
    """Columns of ``table`` covered by a guarded ADD COLUMN in the same file."""
    masked = mask_literals(sql)
    covered: set[str] = set()
    for match in _ADD_COLUMN_GUARDED.finditer(masked):
        target = match.group(1).split(".")[-1].strip('"').lower()
        if target != table.bare_name.lower():
            continue
        covered.add(match.group(2).strip('"').lower())
    return covered


def fenced_migration_ids() -> frozenset[str]:
    """The operator fence baseline, read from the single-sourced manifest
    (OMN-15349) so this helper cannot drift from what the runners actually
    load. Note this is the BASELINE fence, not either runner's post-release
    effective fence — callers that need the effective (post-release) set for
    a specific lane must apply that lane's release policy themselves.
    """
    ids = tuple(_MANIFEST_ID_LINE.findall(FENCE_MANIFEST.read_text(encoding="utf-8")))
    if not ids:  # pragma: no cover - structural guard
        raise AssertionError(
            f"no fenced ids parsed from {FENCE_MANIFEST} — the manifest is "
            "missing, empty, or the fence seam moved; fix this reader, do not "
            "restate the list here."
        )
    return frozenset(ids)


def node_migration_files() -> list[tuple[str, Path]]:
    """Every vendored node migration as ``(namespaced_id, path)``, runner order."""
    files: list[tuple[str, Path]] = []
    for node_dir in sorted(p for p in NODE_MIGRATIONS_DIR.iterdir() if p.is_dir()):
        for sql_file in sorted(node_dir.glob("*.sql")):
            files.append((f"node:{node_dir.name}:{sql_file.name}", sql_file))
    return files
