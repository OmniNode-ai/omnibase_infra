#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Fail closed when a migration leaves a relation RLS-enabled without a policy.

OMN-17298. Row-level security that is switched on with no policy admitting
anyone is not a stricter boundary -- it is an outage. PostgreSQL's default for
a relation with ``ENABLE ROW LEVEL SECURITY`` and zero policies is to deny
every row to every non-owner, non-``BYPASSRLS`` principal, so the projection
writer stops writing and every read returns empty. The failure surfaces as
``InsufficientPrivilege`` (SQLSTATE 42501), which is indistinguishable at the
call site from a missing ``GRANT`` -- and that misreading is exactly what cost
the OMN-17298 investigation its first hour.

This has now happened twice on tenant-classified projection relations, which
is the threshold at which detection has to become a gate (Operating Rule 5):

  * migration 0032 (``node_projection_delegation``) dropped ``tenant_isolation``
    inside a ``DO $$ ... END$$`` block and recreated it in a standalone
    statement AFTER the block. The forward runner is
    ``psql -v ON_ERROR_STOP=1 -f <file>`` with no ``--single-transaction``, so
    ``END$$`` COMMITS: between that commit and the standalone ``CREATE POLICY``
    there is a real window in which the relation is FORCE-RLS with no policy at
    all. An interruption inside it (operator ^C, pod eviction, connection reset,
    OOM) leaves the relation permanently in that state. OMN-17288 superseded it
    with 0033, which moves the recreate inside the block.
  * the same shape was the first hypothesis for
    ``projection_delegation_inference_response_text`` under OMN-17298. It was
    NOT the cause there (a policy is present on the live dev lane -- see the
    ticket), but the hypothesis was only cheap to falsify because the shape is
    real and recurring.

TWO RULES, both static, both proven against real checked-in bytes by
``tests/ci/test_migration_rls_policy_atomicity.py``:

  RULE A -- POLICY PRESENCE. A file that turns RLS on for a relation
  (``ENABLE`` or ``FORCE ROW LEVEL SECURITY``) must also ``CREATE POLICY`` on
  that relation somewhere in the same file. Turning enforcement on and leaving
  the admitting rule to some other file is the defect class itself.

  RULE B -- POLICY ATOMICITY. If a file drops a policy on a relation from
  inside a dollar-quoted block, it must recreate a policy on that relation
  inside the SAME block. A drop that commits with ``END$$`` and a recreate that
  runs afterwards is the 0032 window above.

WHAT IS DELIBERATELY NOT CHECKED
  Whether the policy's predicate actually admits the writer at runtime. That is
  a live-database property (it depends on the ``app.tenant_id`` session GUC),
  and the authority for the predicate's SHAPE is already
  ``application_database_domain_enforcement.py``. This gate answers only the
  static question that one can answer statically: does an admitting rule exist
  at all, and does it exist without a committed window in which it does not.

EXEMPTION -- SUPERSEDED PREDECESSORS ONLY
  A migration recorded as superseded in
  ``docker/migrations/forward/_ledger/migration-supersessions.tsv`` is retired
  history. ``check_migration_append_only.py`` (OMN-16705) refuses to edit it,
  so a violation in such a file CANNOT be fixed in place -- the successor named
  in the ledger is the fix. This is not an allowlist: an entry only exists once
  a strictly-higher-ordinal successor has landed, so the exemption cannot be
  used to admit a new violation. 0032 is exempt on exactly these terms; 0033,
  its successor, passes both rules on its own bytes.

USAGE
  python3 scripts/validation/check_migration_rls_policy_atomicity.py
  python3 scripts/validation/check_migration_rls_policy_atomicity.py --root <dir>

EXIT CODES
  0 -- every scanned migration satisfies both rules
  1 -- at least one violation (printed, one per line, path-prefixed)
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_FORWARD_ROOT = Path("docker/migrations/forward")
_SUPERSESSIONS = _FORWARD_ROOT / "_ledger" / "migration-supersessions.tsv"

# An SQL identifier: bare, double-quoted, or schema-qualified with either half
# spelled in either style.
_IDENT = r'(?:"(?:[^"]|"")+"|[A-Za-z_][A-Za-z0-9_$]*)'
_QUALIFIED = rf"(?:{_IDENT}\s*\.\s*)?{_IDENT}"

# `NO FORCE` and `DISABLE` are excluded by the negative lookahead: they turn
# enforcement OFF or narrow it, and neither can strand a relation policy-less.
_RLS_ON = re.compile(
    rf"\bALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?(?:ONLY\s+)?({_QUALIFIED})\s+"
    r"(?!NO\s+FORCE)(?:ENABLE|FORCE)\s+ROW\s+LEVEL\s+SECURITY\b",
    re.IGNORECASE,
)
_CREATE_POLICY = re.compile(
    rf"\bCREATE\s+POLICY\s+{_IDENT}\s+ON\s+(?:ONLY\s+)?({_QUALIFIED})\b",
    re.IGNORECASE,
)
_DROP_POLICY = re.compile(
    rf"\bDROP\s+POLICY\s+(?:IF\s+EXISTS\s+)?{_IDENT}\s+ON\s+(?:ONLY\s+)?({_QUALIFIED})\b",
    re.IGNORECASE,
)
_DOLLAR_TAG = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$")


@dataclass(frozen=True, slots=True)
class _Scrubbed:
    """One migration's SQL with every non-code byte blanked to spaces.

    Offsets are preserved exactly, so a match position in ``text`` is a
    position in the original file and ``blocks`` can be compared against it
    directly. Blanking rather than deleting is what makes that true.
    """

    text: str
    blocks: tuple[tuple[int, int], ...]


def _relation(raw: str) -> str:
    """Normalize a possibly-qualified, possibly-quoted relation name.

    The schema half is dropped on purpose. These migrations address relations
    unqualified, qualified as ``public.``, and (post-cutover) under a domain
    schema, and all three spellings name the same physical table for the
    purposes of "was a policy created for the thing whose RLS you just turned
    on". Comparing the bare name is the conservative choice: it can only make
    the gate more permissive about spelling, never about the missing policy.
    """
    tail = raw.split(".")[-1].strip()
    if tail.startswith('"') and tail.endswith('"') and len(tail) >= 2:
        tail = tail[1:-1].replace('""', '"')
    return tail.lower()


def scrub(sql: str) -> _Scrubbed:
    """Blank comments and string literals; record dollar-quoted block spans.

    This is not optional cleverness. Migration 0033's own header comment quotes
    the three statements that made 0032 defective, verbatim:

        --   DROP POLICY IF EXISTS tenant_isolation ON delegation_events;
        --   CREATE POLICY tenant_isolation ON delegation_events ...;

    A regex run over raw bytes reads that prose as code and reports the fixed
    file as broken. Likewise ``RAISE EXCEPTION`` messages in these migrations
    discuss ROW LEVEL SECURITY in English. Both are blanked here.

    Dollar-quoted bodies are NOT blanked -- in a ``DO $$ ... $$`` block that
    body IS the code -- but their spans are recorded so Rule B can ask whether
    a drop and a create share one transaction.
    """
    out = list(sql)
    blocks: list[tuple[int, int]] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch == "-" and sql.startswith("--", i):
            end = sql.find("\n", i)
            end = n if end == -1 else end
            for j in range(i, end):
                out[j] = " "
            i = end
            continue
        if ch == "/" and sql.startswith("/*", i):
            # PostgreSQL block comments nest.
            depth = 1
            j = i + 2
            while j < n and depth:
                if sql.startswith("/*", j):
                    depth += 1
                    j += 2
                elif sql.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if ch == "$":
            opener = _DOLLAR_TAG.match(sql, i)
            if opener is not None:
                tag = opener.group(0)
                body_start = opener.end()
                close = sql.find(tag, body_start)
                if close == -1:
                    # Unterminated: treat the remainder as body rather than
                    # silently passing the file. Rule B then sees one block.
                    blocks.append((body_start, n))
                    i = n
                    continue
                blocks.append((body_start, close))
                # Blank only the delimiters; the body is real code.
                for k in range(i, body_start):
                    out[k] = " "
                for k in range(close, close + len(tag)):
                    out[k] = " "
                i = close + len(tag)
                continue
        i += 1
    return _Scrubbed(text="".join(out), blocks=tuple(blocks))


def _enclosing_block(
    pos: int, blocks: tuple[tuple[int, int], ...]
) -> tuple[int, int] | None:
    for span in blocks:
        if span[0] <= pos < span[1]:
            return span
    return None


def violations_for(path: Path, sql: str) -> list[str]:
    """Both rules, evaluated against one migration's scrubbed bytes."""
    scrubbed = scrub(sql)
    text = scrubbed.text
    found: list[str] = []

    created: dict[str, list[int]] = {}
    for match in _CREATE_POLICY.finditer(text):
        created.setdefault(_relation(match.group(1)), []).append(match.start())

    # RULE A -- policy presence.
    for match in _RLS_ON.finditer(text):
        relation = _relation(match.group(1))
        if relation not in created:
            found.append(
                f"{path}: RULE A -- turns ROW LEVEL SECURITY on for {relation!r} "
                "but never CREATEs a policy on it in the same file. A relation "
                "with RLS enabled and zero policies denies every row to every "
                "non-owner principal (42501 InsufficientPrivilege), which reads "
                "at the call site like a missing GRANT. Add the CREATE POLICY "
                "here, or do not enable RLS here."
            )

    # RULE B -- policy atomicity across the DO-block commit boundary.
    for match in _DROP_POLICY.finditer(text):
        block = _enclosing_block(match.start(), scrubbed.blocks)
        if block is None:
            continue
        relation = _relation(match.group(1))
        if not any(block[0] <= at < block[1] for at in created.get(relation, ())):
            found.append(
                f"{path}: RULE B -- DROPs a policy on {relation!r} inside a "
                "dollar-quoted block but does not CREATE one on it inside the "
                "same block. The forward runner uses `psql -v ON_ERROR_STOP=1 "
                "-f <file>` with no --single-transaction, so the block COMMITS "
                "at its terminator: a recreate placed after it leaves a real "
                "window in which the relation is enforcing RLS with no policy. "
                "Move the CREATE POLICY inside the block (see migration 0033)."
            )
    return found


def load_superseded(root: Path) -> frozenset[str]:
    """Artifact paths that a landed successor has already retired."""
    ledger = root / _SUPERSESSIONS
    if not ledger.is_file():
        return frozenset()
    superseded: set[str] = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        superseded.add(line.split("\t")[0].strip())
    return frozenset(superseded)


def check(root: Path) -> list[str]:
    forward = root / _FORWARD_ROOT
    if not forward.is_dir():
        raise SystemExit(f"forward-migration root not found: {forward}")
    superseded = load_superseded(root)
    found: list[str] = []
    for path in sorted(forward.rglob("*.sql")):
        artifact = path.relative_to(forward).as_posix()
        if artifact in superseded:
            continue
        found.extend(
            violations_for(
                path.relative_to(root),
                path.read_text(encoding="utf-8"),
            )
        )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (default: this script's repo).",
    )
    args = parser.parse_args(argv)
    found = check(args.root)
    for violation in found:
        print(violation, file=sys.stderr)
    if found:
        print(
            f"\n{len(found)} RLS-policy violation(s). See OMN-17298 and "
            "migration 0033 for the shape that passes.",
            file=sys.stderr,
        )
        return 1
    print("check_migration_rls_policy_atomicity: clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
