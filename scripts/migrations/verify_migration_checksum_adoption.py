# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Mechanically prove whether a non-canonical ledger checksum may be adopted.

OMN-15857.

Problem
-------
``docker/migrations/forward/_ledger/bootstrap.sql`` accepts exactly two
checksum spellings for a node row it adopts out of ``public.schema_migrations``:
a 64-hex ``content_sha256`` that equals the manifest checksum, or the literal
``applied-by-runner`` (which means "adopt the manifest checksum"). Anything else
falls through to ``RAISE EXCEPTION 'conflicting migration checksum for version %'``
and aborts the whole transaction -- so one hand-written sentinel blocks every
migration on the lane.

The ``.201`` stability-test lane carries seven such rows across two databases
(three ``hotfix-applied-by-codex``, four ``applied-manually-omn-11760``).

The obvious fix -- ``UPDATE schema_migrations SET checksum='applied-by-runner'``
-- is not a fix. It makes bootstrap.sql adopt the manifest checksum, which is a
silent assertion that the hand-applied SQL produced the same schema the
checked-in migration produces. That assertion is precisely what the sentinel is
flagging as *unproven*, so the one-line UPDATE destroys the only evidence that a
question exists.

What this tool does instead
---------------------------
For each non-canonical row it *proves or refutes* that assertion, mechanically:

1. Resolve the row's ``migration_id`` to a checked-in migration file.
2. Replay, into a throwaway scratch database on a scratch server, every
   migration that precedes the target in the same stream, then snapshot the
   scratch schema.
3. Apply the target migration, then snapshot again. The delta between the two
   snapshots is the target's **declared surface** -- exactly the objects this
   one file is responsible for, derived by execution rather than by parsing SQL.
4. Snapshot the same objects in the live database.
5. Compare declared surface against live, object by object: columns (type,
   nullability, default), constraints (definition), indexes (definition), view
   and matview definitions, enum labels.

Verdicts, one per row:

``equivalent``
    Every object in the declared surface exists live with a matching
    definition. The applied schema *is* what the checked-in migration produces,
    so adopting the manifest checksum states a fact rather than a hope.
``divergent``
    At least one object is missing or differs. A structural diff is recorded and
    the row is refused -- no adoption is emitted for it.
``unreachable``
    Equivalence could not be decided: no checked-in file for the version, the
    replay failed, or the database could not be read. Refused, same as
    divergent. Never silently treated as benign.
``legacy_attested``
    The version has no checked-in file but *is* declared in
    ``_ledger/legacy-node-migrations.tsv``. bootstrap.sql already accepts it via
    the ``legacy_attestation`` path, which deliberately proves a *source record*
    and not file bytes. Nothing to verify and nothing to adopt.

Only ``equivalent`` rows may be written into
``_ledger/verified-checksum-adoptions.tsv`` (``--emit-adoptions``). Every run
writes a receipt (``--receipt-out``) whose sha256 is recorded in the TSV, so an
adoption in the manifest is always traceable to the run that proved it.

This tool never writes to the audited database. The live connection is
read-only introspection; every mutation happens in the scratch server.

Usage
-----
Verify the .201 stability-test lane read-only (no local Postgres client needed
beyond a scratch server for the replay)::

    python scripts/migrations/verify_migration_checksum_adoption.py \\
        --database omnidash_analytics \\
        --psql-exec '["ssh","jonah@192.168.86.201","docker","exec","-i",
                      "omnibase-infra-stability-test-postgres","psql","-U","postgres"]' \\
        --receipt-out receipts/omn15857-stability-omnidash_analytics.json

Every psql invocation feeds SQL on **stdin** (``-f -``), never through ``-c``,
so wrapping the client in ``ssh``/``docker exec`` cannot mangle the statement:
argv carries only fixed flags and an identifier-validated database name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TOOL_VERSION = "1"
TICKET = "OMN-15857"

REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARD_DIR = REPO_ROOT / "docker" / "migrations" / "forward"
LEDGER_DIR = FORWARD_DIR / "_ledger"
APPLICATION_MANIFEST = LEDGER_DIR / "application-migrations.tsv"
LEGACY_DECLARATIONS = LEDGER_DIR / "legacy-node-migrations.tsv"
VERIFIED_ADOPTIONS = LEDGER_DIR / "verified-checksum-adoptions.tsv"
SKIP_MANIFEST = FORWARD_DIR.parent / "skip-manifest.yaml"

CANONICAL_CHECKSUM_RE = re.compile(r"^[0-9a-f]{64}$")
RUNNER_CHECKSUM = "applied-by-runner"
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NODE_VERSION_RE = re.compile(
    r"^node:(?P<node>[A-Za-z0-9_][A-Za-z0-9_.-]*):(?P<file>[A-Za-z0-9_][A-Za-z0-9_.-]*[.]sql)$"
)
DOCKER_VERSION_RE = re.compile(r"^docker/(?P<file>[A-Za-z0-9_][A-Za-z0-9_.-]*[.]sql)$")

VERDICT_EQUIVALENT = "equivalent"
VERDICT_DIVERGENT = "divergent"
VERDICT_UNREACHABLE = "unreachable"
VERDICT_LEGACY_ATTESTED = "legacy_attested"

ADOPTABLE_VERDICTS = frozenset({VERDICT_EQUIVALENT})


class VerificationError(RuntimeError):
    """A failure that must abort the run rather than be recorded as a verdict."""


# ---------------------------------------------------------------------------
# psql plumbing
# ---------------------------------------------------------------------------


def _validate_database_identifier(database: str) -> str:
    if not IDENTIFIER_RE.match(database):
        raise VerificationError(f"unsafe database identifier: {database!r}")
    return database


@dataclass(frozen=True)
class PsqlClient:
    """A psql invocation prefix. SQL always travels on stdin, never in argv."""

    argv: tuple[str, ...]
    label: str

    def run(
        self, database: str, sql: str, *, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        _validate_database_identifier(database)
        command = [
            *self.argv,
            "-X",
            "-q",
            "-v",
            "ON_ERROR_STOP=1",
            "-At",
            "-F",
            "\x1f",
            "-d",
            database,
            "-f",
            "-",
        ]
        completed = subprocess.run(
            command,
            input=sql,
            capture_output=True,
            text=True,
            check=False,
        )
        if check and completed.returncode != 0:
            raise VerificationError(
                f"psql failed on {self.label}/{database} "
                f"(exit {completed.returncode}): {completed.stderr.strip()}"
            )
        return completed

    def rows(self, database: str, sql: str) -> list[list[str]]:
        out = self.run(database, sql).stdout
        return [line.split("\x1f") for line in out.splitlines() if line != ""]

    def scalar(self, database: str, sql: str) -> str:
        rows = self.rows(database, sql)
        return rows[0][0] if rows else ""


# ---------------------------------------------------------------------------
# scratch server
# ---------------------------------------------------------------------------


def _postgres_bin_dir() -> Path | None:
    candidates: list[Path] = []
    env_dir = os.environ.get("ONEX_POSTGRES_BIN_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        sorted(Path("/opt/homebrew/opt").glob("postgresql@*/bin"), reverse=True)
    )
    candidates.extend(sorted(Path("/usr/lib/postgresql").glob("*/bin"), reverse=True))
    candidates.extend(
        sorted(Path("/usr/local/opt").glob("postgresql@*/bin"), reverse=True)
    )
    for candidate in candidates:
        if (candidate / "initdb").is_file() and (candidate / "pg_ctl").is_file():
            return candidate
    which = shutil.which("initdb")
    if which is not None:
        return Path(which).parent
    return None


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ScratchServer:
    """A disposable local Postgres used only to replay checked-in migrations."""

    def __init__(self, bin_dir: Path, base_dir: Path) -> None:
        self._bin_dir = bin_dir
        self._base = base_dir
        self._port = _free_port()
        self._data = base_dir / "data"
        self._socket_dir = base_dir / "sock"
        self.client = PsqlClient(
            argv=(
                str(bin_dir / "psql"),
                "-h",
                str(self._socket_dir),
                "-p",
                str(self._port),
                "-U",
                "postgres",
            ),
            label="scratch",
        )

    def start(self) -> None:
        self._socket_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(self._bin_dir / "initdb"),
                "-D",
                str(self._data),
                "-U",
                "postgres",
                "--auth=trust",
                "--encoding=UTF8",
                "--no-sync",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        start = subprocess.run(
            [
                str(self._bin_dir / "pg_ctl"),
                "-D",
                str(self._data),
                "-o",
                f"-p {self._port} -k {self._socket_dir} -c listen_addresses= -c fsync=off",
                "-w",
                "-t",
                "60",
                "-l",
                str(self._base / "postgres.log"),
                "start",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if start.returncode != 0:
            log = self._base / "postgres.log"
            text = log.read_text(errors="replace") if log.is_file() else ""
            raise VerificationError(
                f"scratch postgres did not start: {start.stderr!r}; log={text!r}"
            )

    def stop(self) -> None:
        subprocess.run(
            [
                str(self._bin_dir / "pg_ctl"),
                "-D",
                str(self._data),
                "-m",
                "immediate",
                "-w",
                "-t",
                "30",
                "stop",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def fresh_database(self) -> str:
        name = f"onex_verify_{uuid.uuid4().hex[:16]}"
        self.client.run("postgres", f'CREATE DATABASE "{name}"')
        return name

    def drop_database(self, name: str) -> None:
        _validate_database_identifier(name)
        self.client.run("postgres", f'DROP DATABASE IF EXISTS "{name}"', check=False)

    def server_version(self) -> str:
        return self.client.scalar("postgres", "SHOW server_version")


# ---------------------------------------------------------------------------
# schema snapshots
# ---------------------------------------------------------------------------

# One statement, one JSON value. View definitions and check-constraint expressions
# carry embedded newlines, so a row/column text transport would be ambiguous the
# moment it met a real view -- the snapshot travels as JSON for that reason.
_SNAPSHOT_SQL = """
SELECT json_build_object(
  'relations', COALESCE((
    SELECT json_agg(json_build_object(
      'name', c.relname,
      'relkind', c.relkind,
      'definition', CASE
        WHEN c.relkind IN ('v', 'm') THEN pg_get_viewdef(c.oid, true) ELSE NULL END))
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind IN ('r', 'p', 'v', 'm')
  ), '[]'::json),
  'columns', COALESCE((
    SELECT json_agg(json_build_object(
      'relation', c.relname,
      'name', a.attname,
      'type', format_type(a.atttypid, a.atttypmod),
      'nullability', CASE WHEN a.attnotnull THEN 'NOT NULL' ELSE 'NULL' END,
      'default', COALESCE(pg_get_expr(d.adbin, d.adrelid), '')))
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    JOIN pg_attribute a ON a.attrelid = c.oid
    LEFT JOIN pg_attrdef d ON d.adrelid = c.oid AND d.adnum = a.attnum
    WHERE n.nspname = 'public'
      AND c.relkind IN ('r', 'p', 'v', 'm')
      AND a.attnum > 0
      AND NOT a.attisdropped
  ), '[]'::json),
  'constraints', COALESCE((
    SELECT json_agg(json_build_object(
      'relation', c.relname,
      'name', con.conname,
      'definition', pg_get_constraintdef(con.oid)))
    FROM pg_constraint con
    JOIN pg_class c ON c.oid = con.conrelid
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public'
  ), '[]'::json),
  'indexes', COALESCE((
    SELECT json_agg(json_build_object(
      'relation', tablename, 'name', indexname, 'definition', indexdef))
    FROM pg_indexes WHERE schemaname = 'public'
  ), '[]'::json),
  'enums', COALESCE((
    SELECT json_agg(json_build_object('name', typname, 'labels', labels))
    FROM (
      SELECT t.typname,
             json_agg(e.enumlabel ORDER BY e.enumsortorder) AS labels
      FROM pg_type t
      JOIN pg_namespace n ON n.oid = t.typnamespace
      JOIN pg_enum e ON e.enumtypid = t.oid
      WHERE n.nspname = 'public'
      GROUP BY t.typname
    ) enum_labels
  ), '[]'::json)
)::text;
"""

_RELKIND_TO_KIND = {"r": "table", "p": "table", "v": "view", "m": "matview"}


def _normalize_sql_text(definition: str) -> str:
    """Collapse whitespace so pretty-printer differences are not divergence."""
    return re.sub(r"\s+", " ", definition).strip().rstrip(";")


def snapshot(client: PsqlClient, database: str) -> dict[str, dict[str, Any]]:
    """Structural snapshot of ``public`` keyed by ``<kind>:<name>``."""
    raw = json.loads(client.run(database, _SNAPSHOT_SQL).stdout.strip())

    objects: dict[str, dict[str, Any]] = {}
    by_relname: dict[str, dict[str, Any]] = {}

    for relation in raw["relations"]:
        kind = _RELKIND_TO_KIND[relation["relkind"]]
        entry: dict[str, Any] = {
            "kind": kind,
            "name": relation["name"],
            "columns": {},
            "constraints": {},
            "indexes": {},
        }
        if kind in {"view", "matview"}:
            entry["definition"] = _normalize_sql_text(relation["definition"] or "")
        objects[f"{kind}:{relation['name']}"] = entry
        by_relname[relation["name"]] = entry

    for column in raw["columns"]:
        column_entry = by_relname.get(column["relation"])
        if column_entry is not None:
            column_entry["columns"][column["name"]] = {
                "type": column["type"],
                "nullability": column["nullability"],
                "default": _normalize_sql_text(column["default"]),
            }

    for constraint in raw["constraints"]:
        constraint_entry = by_relname.get(constraint["relation"])
        if constraint_entry is not None:
            constraint_entry["constraints"][constraint["name"]] = _normalize_sql_text(
                constraint["definition"]
            )

    for index in raw["indexes"]:
        index_entry = by_relname.get(index["relation"])
        if index_entry is not None:
            index_entry["indexes"][index["name"]] = _normalize_sql_text(
                index["definition"]
            )

    for enum in raw["enums"]:
        objects[f"enum:{enum['name']}"] = {
            "kind": "enum",
            "name": enum["name"],
            "labels": list(enum["labels"]),
        }

    return objects


def declared_surface(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Objects the target migration created or altered, derived by execution."""
    surface: dict[str, dict[str, Any]] = {}
    for key, entry in after.items():
        if key not in before or before[key] != entry:
            surface[key] = entry
    return surface


def diff_object(declared: dict[str, Any], live: dict[str, Any] | None) -> list[str]:
    """Structural differences between a declared object and its live counterpart."""
    if live is None:
        return [f"absent: {declared['kind']} public.{declared['name']} does not exist"]

    findings: list[str] = []
    name = f"{declared['kind']} public.{declared['name']}"

    if declared["kind"] == "enum":
        if declared["labels"] != live.get("labels"):
            findings.append(
                f"{name}: enum labels declared={declared['labels']} live={live.get('labels')}"
            )
        return findings

    if "definition" in declared:
        if declared["definition"] != live.get("definition"):
            findings.append(
                f"{name}: definition differs\n  declared: {declared['definition']}"
                f"\n  live:     {live.get('definition')}"
            )

    for column, spec in declared["columns"].items():
        live_spec = live["columns"].get(column)
        if live_spec is None:
            findings.append(f"{name}: column {column!r} missing live")
        elif live_spec != spec:
            findings.append(
                f"{name}: column {column!r} declared={spec} live={live_spec}"
            )

    for constraint, definition in declared["constraints"].items():
        live_def = live["constraints"].get(constraint)
        if live_def is None:
            findings.append(f"{name}: constraint {constraint!r} missing live")
        elif live_def != definition:
            findings.append(
                f"{name}: constraint {constraint!r} declared={definition!r} live={live_def!r}"
            )

    for index, definition in declared["indexes"].items():
        live_def = live["indexes"].get(index)
        if live_def is None:
            findings.append(f"{name}: index {index!r} missing live")
        elif live_def != definition:
            findings.append(
                f"{name}: index {index!r} declared={definition!r} live={live_def!r}"
            )

    return findings


# ---------------------------------------------------------------------------
# manifests
# ---------------------------------------------------------------------------


def _read_tsv(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    return [
        line.split("\t")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() != ""
    ]


def load_manifest() -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for row in _read_tsv(APPLICATION_MANIFEST):
        artifact_path, stream, owner, domain, version, checksum = row
        manifest[version] = {
            "artifact_path": artifact_path,
            "migration_stream": stream,
            "owner": owner,
            "domain": domain,
            "checksum": checksum,
        }
    return manifest


def load_legacy_declarations() -> dict[str, dict[str, str]]:
    legacy: dict[str, dict[str, str]] = {}
    for row in _read_tsv(LEGACY_DECLARATIONS):
        stream, owner, domain, version, source_checksum, ticket = row
        legacy[version] = {
            "migration_stream": stream,
            "owner": owner,
            "domain": domain,
            "source_checksum": source_checksum,
            "ticket": ticket,
        }
    return legacy


def load_skipped_ids() -> set[str]:
    if not SKIP_MANIFEST.is_file():
        return set()
    return set(
        re.findall(
            r'^\s*-\s*id:\s*"([^"]*)"',
            SKIP_MANIFEST.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# replay sets
# ---------------------------------------------------------------------------


def _flat_migrations() -> list[Path]:
    return sorted(
        (p for p in FORWARD_DIR.glob("*.sql") if p.is_file()),
        key=lambda p: p.name,
    )


def resolve_replay_set(version: str) -> tuple[Path, list[Path]] | None:
    """Return ``(target_file, prefix_files)`` for a ledger version, or ``None``.

    The prefix is every checked-in migration in the same stream that precedes the
    target, so the target is applied onto the state it was authored against
    instead of onto an empty database.
    """
    node_match = NODE_VERSION_RE.match(version)
    if node_match is not None:
        node_dir = FORWARD_DIR / "nodes" / node_match.group("node")
        target = node_dir / node_match.group("file")
        if not target.is_file():
            return None
        siblings = sorted(
            (p for p in node_dir.glob("*.sql") if p.is_file()), key=lambda p: p.name
        )
        prefix = [p for p in siblings if p.name < target.name]
        return target, prefix

    docker_match = DOCKER_VERSION_RE.match(version)
    if docker_match is not None:
        target = FORWARD_DIR / docker_match.group("file")
        if not target.is_file():
            return None
        skipped = load_skipped_ids()
        prefix = [
            p
            for p in _flat_migrations()
            if p.name < target.name and f"docker/{p.name}" not in skipped
        ]
        return target, prefix

    return None


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


@dataclass
class RowVerdict:
    version: str
    database: str
    source_checksum: str
    source_set: str
    verdict: str
    reason: str = ""
    artifact_path: str = ""
    artifact_sha256: str = ""
    manifest_checksum: str = ""
    declared_objects: list[str] = field(default_factory=list)
    divergences: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "database": self.database,
            "source_checksum": self.source_checksum,
            "source_set": self.source_set,
            "verdict": self.verdict,
            "reason": self.reason,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "manifest_checksum": self.manifest_checksum,
            "declared_objects": sorted(self.declared_objects),
            "divergences": self.divergences,
        }


NON_CANONICAL_ROWS_SQL = """
SELECT migration_id, checksum, source_set
FROM public.schema_migrations
WHERE checksum !~ '^[0-9a-f]{64}$'
  AND checksum <> 'applied-by-runner'
ORDER BY migration_id;
"""


def discover_non_canonical_rows(
    client: PsqlClient, database: str
) -> list[tuple[str, str, str]]:
    present = client.scalar(database, "SELECT to_regclass('public.schema_migrations');")
    if present == "":
        raise VerificationError(
            f"{database}: public.schema_migrations is absent -- nothing to verify"
        )
    return [
        (row[0], row[1], row[2])
        for row in client.rows(database, NON_CANONICAL_ROWS_SQL)
    ]


def verify_row(
    *,
    version: str,
    source_checksum: str,
    source_set: str,
    database: str,
    live: PsqlClient,
    scratch: ScratchServer,
    manifest: dict[str, dict[str, str]],
    legacy: dict[str, dict[str, str]],
) -> RowVerdict:
    verdict = RowVerdict(
        version=version,
        database=database,
        source_checksum=source_checksum,
        source_set=source_set,
        verdict=VERDICT_UNREACHABLE,
    )

    replay = resolve_replay_set(version)
    if replay is None:
        if version in legacy:
            verdict.verdict = VERDICT_LEGACY_ATTESTED
            verdict.reason = (
                "no checked-in migration file; already declared in "
                f"_ledger/legacy-node-migrations.tsv under {legacy[version]['ticket']}, "
                "which bootstrap.sql imports as checksum_kind='legacy_attestation' "
                "(a source record, deliberately not a bytes claim)"
            )
            return verdict
        verdict.reason = (
            "no checked-in migration file resolves this version and it carries no "
            "legacy declaration -- equivalence cannot be decided"
        )
        return verdict

    target, prefix = replay
    verdict.artifact_path = str(target.relative_to(FORWARD_DIR))
    verdict.artifact_sha256 = file_sha256(target)
    verdict.manifest_checksum = manifest.get(version, {}).get("checksum", "")

    scratch_db = scratch.fresh_database()
    try:
        for migration in prefix:
            applied = scratch.client.run(
                scratch_db, migration.read_text(encoding="utf-8"), check=False
            )
            if applied.returncode != 0:
                verdict.reason = (
                    f"replay of prefix migration {migration.name} failed on the scratch "
                    f"server: {applied.stderr.strip()[:800]}"
                )
                return verdict

        before = snapshot(scratch.client, scratch_db)
        applied = scratch.client.run(
            scratch_db, target.read_text(encoding="utf-8"), check=False
        )
        if applied.returncode != 0:
            verdict.reason = (
                f"target migration {target.name} failed on the scratch server: "
                f"{applied.stderr.strip()[:800]}"
            )
            return verdict
        after = snapshot(scratch.client, scratch_db)
    finally:
        scratch.drop_database(scratch_db)

    surface = declared_surface(before, after)
    verdict.declared_objects = list(surface)
    if not surface:
        verdict.reason = (
            f"{target.name} declared no schema surface on replay -- there is nothing "
            "to compare, so equivalence cannot be asserted"
        )
        return verdict

    live_objects = snapshot(live, database)
    divergences: list[str] = []
    for key, declared in surface.items():
        divergences.extend(diff_object(declared, live_objects.get(key)))

    if divergences:
        verdict.verdict = VERDICT_DIVERGENT
        verdict.divergences = divergences
        verdict.reason = (
            f"{len(divergences)} structural difference(s) between the schema "
            f"{target.name} produces and the schema {database} actually carries"
        )
        return verdict

    verdict.verdict = VERDICT_EQUIVALENT
    verdict.reason = (
        f"all {len(surface)} object(s) declared by {target.name} exist in {database} "
        "with identical columns, constraints, indexes and definitions"
    )
    return verdict


# ---------------------------------------------------------------------------
# adoption emission
# ---------------------------------------------------------------------------

ADOPTION_COLUMNS = (
    "version",
    "source_checksum",
    "manifest_checksum",
    "ticket",
    "receipt_sha256",
    "verified_at",
)


def load_adoptions() -> dict[str, dict[str, str]]:
    adoptions: dict[str, dict[str, str]] = {}
    for row in _read_tsv(VERIFIED_ADOPTIONS):
        adoptions[row[0]] = dict(zip(ADOPTION_COLUMNS, row, strict=True))
    return adoptions


def write_adoptions(adoptions: dict[str, dict[str, str]]) -> None:
    lines = [
        "\t".join(adoptions[version][column] for column in ADOPTION_COLUMNS)
        for version in sorted(adoptions)
    ]
    VERIFIED_ADOPTIONS.write_text(
        "\n".join(lines) + "\n" if lines else "", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def _parse_psql_exec(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return ("psql",)
    parsed = json.loads(raw)
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise VerificationError("--psql-exec must be a JSON array of strings")
    if not parsed:
        raise VerificationError("--psql-exec must not be empty")
    return tuple(parsed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--database",
        action="append",
        required=True,
        help="Database to audit (repeatable). Read-only.",
    )
    parser.add_argument(
        "--psql-exec",
        default=None,
        help=(
            'JSON array for the psql invocation prefix, e.g. \'["ssh","host","docker",'
            '"exec","-i","container","psql","-U","postgres"]\'. Defaults to ["psql"].'
        ),
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        required=True,
        help="Path the JSON receipt is written to.",
    )
    parser.add_argument(
        "--emit-adoptions",
        action="store_true",
        help=(
            "Write proven-equivalent rows into "
            "_ledger/verified-checksum-adoptions.tsv. Divergent, unreachable and "
            "legacy-attested rows are never written."
        ),
    )
    parser.add_argument(
        "--lane",
        default=os.environ.get("ONEX_LANE", "unspecified"),
        help="Lane attribution recorded in the receipt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        psql_argv = _parse_psql_exec(args.psql_exec)
    except VerificationError as exc:
        print(f"[verify-adoption] FATAL: {exc}", file=sys.stderr)
        return 2

    live = PsqlClient(argv=psql_argv, label="live")
    manifest = load_manifest()
    legacy = load_legacy_declarations()

    bin_dir = _postgres_bin_dir()
    if bin_dir is None:
        print(
            "[verify-adoption] FATAL: no local initdb/pg_ctl found; a scratch "
            "Postgres is required to replay the checked-in migrations. Set "
            "ONEX_POSTGRES_BIN_DIR.",
            file=sys.stderr,
        )
        return 2

    verdicts: list[RowVerdict] = []
    with tempfile.TemporaryDirectory(prefix="onex-verify-adoption-") as tmp:
        scratch = ScratchServer(bin_dir, Path(tmp))
        scratch.start()
        try:
            scratch_version = scratch.server_version()
            for database in args.database:
                try:
                    rows = discover_non_canonical_rows(live, database)
                    live_version = live.scalar(database, "SHOW server_version")
                except VerificationError as exc:
                    print(f"[verify-adoption] FATAL: {exc}", file=sys.stderr)
                    return 2
                print(
                    f"[verify-adoption] {database}: {len(rows)} non-canonical row(s); "
                    f"live server_version={live_version} scratch={scratch_version}",
                    file=sys.stderr,
                )
                for version, checksum, source_set in rows:
                    verdict = verify_row(
                        version=version,
                        source_checksum=checksum,
                        source_set=source_set,
                        database=database,
                        live=live,
                        scratch=scratch,
                        manifest=manifest,
                        legacy=legacy,
                    )
                    verdicts.append(verdict)
                    print(
                        f"[verify-adoption]   {verdict.verdict.upper():<16} {version}",
                        file=sys.stderr,
                    )
        finally:
            scratch.stop()

    receipt = {
        "tool": "verify_migration_checksum_adoption.py",
        "tool_version": TOOL_VERSION,
        "ticket": TICKET,
        "lane": args.lane,
        "generated_at": datetime.now(UTC).isoformat(),
        "psql_exec": list(psql_argv),
        "databases": list(args.database),
        "scratch_server_version": scratch_version,
        "verdicts": [verdict.as_dict() for verdict in verdicts],
        "counts": {
            name: sum(1 for verdict in verdicts if verdict.verdict == name)
            for name in (
                VERDICT_EQUIVALENT,
                VERDICT_DIVERGENT,
                VERDICT_UNREACHABLE,
                VERDICT_LEGACY_ATTESTED,
            )
        },
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_out.write_text(payload, encoding="utf-8")
    receipt_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(
        f"[verify-adoption] receipt {args.receipt_out} sha256={receipt_sha}",
        file=sys.stderr,
    )

    if args.emit_adoptions:
        adoptions = load_adoptions()
        verified_at = datetime.now(UTC).strftime("%Y-%m-%d")
        emitted = 0
        for verdict in verdicts:
            if verdict.verdict not in ADOPTABLE_VERDICTS:
                continue
            if not CANONICAL_CHECKSUM_RE.match(verdict.manifest_checksum):
                # Service-owned 'docker/<file>.sql' rows are proven equivalent
                # above and recorded in the receipt, but they carry no row in
                # _ledger/application-migrations.tsv and bootstrap.sql never
                # compares their checksum (it imports source_set='node' rows
                # only).  There is no gate here to satisfy, so there is nothing
                # to declare -- the drift-detection gap those rows sit in is
                # OMN-15561, not this ticket.
                print(
                    f"[verify-adoption] {verdict.version}: verified but not "
                    "adoptable -- no application-manifest row, so bootstrap.sql "
                    "never gates on this checksum (see OMN-15561)",
                    file=sys.stderr,
                )
                continue
            adoptions[verdict.version] = {
                "version": verdict.version,
                "source_checksum": verdict.source_checksum,
                "manifest_checksum": verdict.manifest_checksum,
                "ticket": TICKET,
                "receipt_sha256": receipt_sha,
                "verified_at": verified_at,
            }
            emitted += 1
        write_adoptions(adoptions)
        print(
            f"[verify-adoption] wrote {emitted} adoption declaration(s) to "
            f"{VERIFIED_ADOPTIONS.relative_to(REPO_ROOT)}",
            file=sys.stderr,
        )

    refused = [
        verdict
        for verdict in verdicts
        if verdict.verdict in {VERDICT_DIVERGENT, VERDICT_UNREACHABLE}
    ]
    if refused:
        print(
            f"[verify-adoption] {len(refused)} row(s) refused adoption "
            "(divergent or unreachable) -- see receipt",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
