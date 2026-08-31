#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Ask a lane whether a migration is already applied -- from the table that gates.

OMN-17139.

The defect this closes
----------------------
Editing an already-applied migration in place bricks forward-migration on every
lane that applied the old bytes. ``check_migration_append_only.py`` refuses such
an edit unless the author declares a supersession, and the author is then
answering, by hand, one factual question: *has this migration already been
applied anywhere?*

On 2026-08-30 that question was answered with the wrong table. The probe run was

    SELECT to_regclass('public.onex_application_migration_manifest');

which returns NULL on every ``.201`` lane -- ``onex_application_migration_manifest``
is a TEMP table ``run-forward-migrations.sh`` creates from the checked-in TSV for
the duration of one bootstrap session, and it has never been a persistent
relation on any lane. A probe against it therefore reads "clean" unconditionally.
The supersession row committed on the strength of that answer
(``migration-supersessions.tsv``, OMN-16180) asserted the migration "has only
ever been applied by hand ... never through run-forward-migrations.sh". The dev
lane's canonical ledger said otherwise, in a row written by that very runner two
hours earlier:

    version    | node:node_projection_work_events:0001_create_work_events.sql
    checksum   | cba8013e...6664
    provenance | file:nodes/node_projection_work_events/0001_create_work_events.sql
    applied_at | 2026-08-30 04:59:57.430276+00

The gate reads ``platform_catalog.schema_migrations``. So does this probe. A
question answered against the wrong relation is not a weak answer, it is a
fabricated one -- and this tool exists so nobody has to hand-type the right query
under time pressure again.

What it reports, per database
-----------------------------
``APPLIED``    a row exists; its recorded checksum and the file's differ -> an
               in-place edit will brick this lane. Exit 1.
``CURRENT``    a row exists and its checksum equals the file on disk. Exit 1
               all the same when ``--artifact`` was given and the bytes are about
               to change: applied is applied.
``CLEAN``      no row for this version in this database.
``NO_LEDGER``  ``platform_catalog.schema_migrations`` does not exist here. Never
               reported as CLEAN: an absent ledger is an unanswered question, not
               a negative answer.

Exit status is 0 only when every database probed reports ``CLEAN``. ``NO_LEDGER``
exits 2 -- an indeterminate probe must not read as permission.

Usage::

    python scripts/migrations/check_migration_applied_on_lane.py \\
        --version node:node_projection_work_events:0001_create_work_events.sql \\
        --database omnidash_analytics \\
        --psql-exec '["ssh","jonah@192.168.86.201","docker","exec","-i",
                      "omnibase-infra-postgres","psql","-U","postgres"]'
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
APPLICATION_MANIFEST = FORWARD_DIR / "_ledger" / "application-migrations.tsv"

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VERSION_RE = re.compile(
    r"^node:[A-Za-z0-9_][A-Za-z0-9_.-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*[.]sql$"
)

# The table the forward-migration runner writes and gates on. Named once, here,
# so a future reader can see there is exactly one and it is not the manifest.
CANONICAL_LEDGER = "platform_catalog.schema_migrations"

LEDGER_PRESENT_SQL = f"SELECT to_regclass('{CANONICAL_LEDGER}');"

# The only interpolation is CANONICAL_LEDGER, a module constant naming the one
# table this tool is allowed to read; the version travels as a psql -v binding,
# never in the SQL text.
APPLIED_ROW_SQL = f"""
SELECT checksum, checksum_kind, provenance, applied_at
FROM {CANONICAL_LEDGER}
WHERE version = :'probe_version';
"""  # noqa: S608

STATUS_APPLIED = "APPLIED"
STATUS_CURRENT = "CURRENT"
STATUS_CLEAN = "CLEAN"
STATUS_NO_LEDGER = "NO_LEDGER"


class ProbeError(RuntimeError):
    """The probe could not be run at all."""


@dataclass(frozen=True)
class PsqlClient:
    """A psql invocation prefix. SQL travels on stdin, never in argv."""

    argv: tuple[str, ...]

    def rows(self, database: str, sql: str, *variables: str) -> list[list[str]]:
        if not IDENTIFIER_RE.match(database):
            raise ProbeError(f"unsafe database identifier: {database!r}")
        command = [
            *self.argv,
            "-X",
            "-q",
            "-v",
            "ON_ERROR_STOP=1",
            "-At",
            "-F",
            "\x1f",
            *[item for variable in variables for item in ("-v", variable)],
            "-d",
            database,
            "-f",
            "-",
        ]
        completed = subprocess.run(
            command, input=sql, capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            raise ProbeError(
                f"psql failed on {database} (exit {completed.returncode}): "
                f"{completed.stderr.strip()}"
            )
        return [
            line.split("\x1f") for line in completed.stdout.splitlines() if line != ""
        ]


@dataclass(frozen=True)
class ProbeResult:
    database: str
    status: str
    recorded_checksum: str = ""
    provenance: str = ""
    applied_at: str = ""


def declared_checksum(version: str) -> tuple[str, str]:
    """``(artifact_path, declared_checksum)`` for a manifest version."""
    if not APPLICATION_MANIFEST.is_file():
        raise ProbeError(f"{APPLICATION_MANIFEST} is missing")
    for line in APPLICATION_MANIFEST.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) >= 6 and fields[4] == version:
            return fields[0], fields[5]
    raise ProbeError(f"{version} is not declared in {APPLICATION_MANIFEST.name}")


def file_checksum(artifact_path: str) -> str:
    path = FORWARD_DIR / artifact_path
    if not path.is_file():
        raise ProbeError(f"{path} does not exist in the working tree")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(client: PsqlClient, database: str, version: str, on_disk: str) -> ProbeResult:
    present = client.rows(database, LEDGER_PRESENT_SQL)
    if not present or present[0][0] == "":
        return ProbeResult(database=database, status=STATUS_NO_LEDGER)

    rows = client.rows(database, APPLIED_ROW_SQL, f"probe_version={version}")
    if not rows:
        return ProbeResult(database=database, status=STATUS_CLEAN)

    checksum, _kind, provenance, applied_at = (rows[0] + ["", "", "", ""])[:4]
    return ProbeResult(
        database=database,
        status=STATUS_CURRENT if checksum == on_disk else STATUS_APPLIED,
        recorded_checksum=checksum,
        provenance=provenance,
        applied_at=applied_at,
    )


def _parse_psql_exec(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ("psql",)
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ProbeError("--psql-exec must be a JSON array of strings")
    if not parsed:
        raise ProbeError("--psql-exec must not be empty")
    return tuple(parsed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="node:<node>:<file>.sql")
    parser.add_argument(
        "--database",
        action="append",
        required=True,
        help="Database to probe (repeatable). Read-only.",
    )
    parser.add_argument("--psql-exec", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not VERSION_RE.match(args.version):
        print(
            f"[applied-probe] FATAL: --version must be node:<node>:<file>.sql, "
            f"got {args.version!r}",
            file=sys.stderr,
        )
        return 2

    try:
        artifact_path, declared = declared_checksum(args.version)
        on_disk = file_checksum(artifact_path)
        client = PsqlClient(argv=_parse_psql_exec(args.psql_exec))
        results = [probe(client, db, args.version, on_disk) for db in args.database]
    except ProbeError as exc:
        print(f"[applied-probe] FATAL: {exc}", file=sys.stderr)
        return 2

    print(f"[applied-probe] version   {args.version}")
    print(f"[applied-probe] artifact  {artifact_path}")
    print(f"[applied-probe] on disk   {on_disk}")
    print(f"[applied-probe] declared  {declared}")
    print(f"[applied-probe] ledger    {CANONICAL_LEDGER}")
    for result in results:
        line = f"[applied-probe]   {result.status:<10} {result.database}"
        if result.recorded_checksum:
            line += f" recorded={result.recorded_checksum}"
        if result.provenance:
            line += f" provenance={result.provenance}"
        if result.applied_at:
            line += f" applied_at={result.applied_at}"
        print(line)

    if any(r.status == STATUS_NO_LEDGER for r in results):
        print(
            f"[applied-probe] INDETERMINATE: {CANONICAL_LEDGER} is absent on at "
            "least one database probed. An absent ledger answers nothing.",
            file=sys.stderr,
        )
        return 2
    if any(r.status in (STATUS_APPLIED, STATUS_CURRENT) for r in results):
        print(
            "[applied-probe] APPLIED: this migration is recorded as applied. "
            "Editing its bytes in place will abort forward-migration on that "
            "lane. Add a new-ordinal successor, or -- when the rewrite is "
            "provably the same program -- earn a declaration with "
            "scripts/migrations/prove_migration_revision_equivalence.py "
            "(OMN-17139).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
