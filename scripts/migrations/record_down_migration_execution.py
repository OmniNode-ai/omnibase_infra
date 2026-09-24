#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Execute a down-migration on a lab scratch database and record the run.

OMN-19344 (plan row R7, rule RB-2). A forward-only migration is a rollback
barrier: no composition older than it is eligible, unless its paired
down-migration EXISTS and HAS BEEN EXECUTED on the lab, with the run recorded
against the migration id. This tool is the only writer of such a record.

What it does, all inside ONE fresh database it creates and drops itself
(``downproof_<stamp>``), on the postgres container you name:

  1. apply each ``--prerequisite`` forward migration, in the order given
  2. snapshot the catalog                                    -> S0
  3. apply the forward migration                             -> Sf
  4. apply the down-migration                                -> S1
  5. apply the forward migration again                       -> S2

PASS only when every psql exits 0 under ``ON_ERROR_STOP=1``, the down-migration
returned the catalog to exactly S0 (S1 == S0), and the forward re-applied to
exactly Sf (S2 == Sf). Anything else is FAIL, and a FAIL record never lifts a
barrier. It never touches an existing database, so it cannot damage a lane even
when pointed at one; the surface it records must still be a lab surface
(``check_migration_class.ModelDownExecution`` refuses any other).

The record binds the sha256 of both scripts, so editing either afterwards voids
it and ``check_migration_class.py`` fails CI until the run is repeated.

Usage::

    uv run python scripts/migrations/record_down_migration_execution.py \\
        --migration forward/031_create_llm_call_metrics_and_cost_aggregates.sql \\
        --down rollback/rollback_031_llm_call_metrics_and_cost_aggregates.sql \\
        --prerequisite forward/029_create_db_metadata.sql \\
        --container <scratch postgres container> \\
        --surface mac-scratch:<lane> --write
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
_CHECKER: Final[Path] = (
    REPO_ROOT / "scripts" / "validation" / "check_migration_class.py"
)

#: Every user-visible schema object whose presence a down-migration must undo.
SNAPSHOT_SQL: Final[str] = """
SELECT 'rel:' || n.nspname || '.' || c.relname || ':' || c.relkind::text
  FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
 WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
   AND n.nspname NOT LIKE 'pg_toast%'
UNION ALL
SELECT 'fn:' || n.nspname || '.' || p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')'
  FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
 WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
UNION ALL
SELECT 'type:' || n.nspname || '.' || t.typname
  FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace
 WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
   AND t.typtype IN ('e', 'd', 'c')
   AND NOT EXISTS (SELECT 1 FROM pg_class c WHERE c.reltype = t.oid)
UNION ALL
SELECT 'trg:' || tgrelid::regclass::text || '.' || tgname FROM pg_trigger WHERE NOT tgisinternal
UNION ALL
SELECT 'pol:' || schemaname || '.' || tablename || '.' || policyname FROM pg_policies
UNION ALL
SELECT 'col:' || table_schema || '.' || table_name || '.' || column_name || ':' || data_type
  FROM information_schema.columns
 WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
"""

Runner = Callable[[Sequence[str], str | None], subprocess.CompletedProcess[str]]


def _run(argv: Sequence[str], stdin: str | None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        input=stdin,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _load_checker() -> object:
    spec = importlib.util.spec_from_file_location("check_migration_class", _CHECKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class ModelRoundTrip:
    """What the four steps observed."""

    forward_exit: int
    down_exit: int
    reforward_exit: int
    before: frozenset[str]
    after_forward: frozenset[str]
    after_down: frozenset[str]
    after_reforward: frozenset[str]


def verdict(trip: ModelRoundTrip) -> tuple[str, str]:
    """``(PASS|FAIL, evidence)`` for one round trip. Pure, so it is tested."""
    problems = []
    for name in ("forward_exit", "down_exit", "reforward_exit"):
        if getattr(trip, name) != 0:
            problems.append(f"{name}={getattr(trip, name)}")
    if trip.after_forward == trip.before:
        problems.append("the forward migration changed no catalog object (vacuous)")
    left = sorted(trip.after_down - trip.before)
    missing = sorted(trip.before - trip.after_down)
    if left or missing:
        problems.append(
            f"down did not restore the pre-forward catalog: {len(left)} left behind "
            f"({', '.join(left[:3])}), {len(missing)} removed ({', '.join(missing[:3])})"
        )
    if trip.after_reforward != trip.after_forward:
        problems.append("forward re-applied after down did not reproduce its catalog")
    summary = (
        f"forward exit {trip.forward_exit}; down exit {trip.down_exit}; forward "
        f"re-apply exit {trip.reforward_exit}; catalog objects before={len(trip.before)} "
        f"after-forward={len(trip.after_forward)} after-down={len(trip.after_down)} "
        f"after-reforward={len(trip.after_reforward)}"
    )
    if problems:
        return "FAIL", f"{summary}; " + "; ".join(problems)
    return "PASS", f"{summary}; down restored the pre-forward catalog exactly"


def _psql(
    runner: Runner, container: str, database: str, sql: str, *, tuples: bool = False
) -> subprocess.CompletedProcess[str]:
    argv = ["docker", "exec", "-i", container, "psql", "-U", "postgres", "-d", database]
    argv += ["-v", "ON_ERROR_STOP=1", "-q"]
    if tuples:
        argv += ["-tA"]
    argv += ["-f", "-"]
    return runner(argv, sql)


def _snapshot(runner: Runner, container: str, database: str) -> frozenset[str]:
    result = _psql(runner, container, database, SNAPSHOT_SQL, tuples=True)
    if result.returncode != 0:
        msg = f"catalog snapshot failed: {result.stderr.strip()[:300]}"
        raise RuntimeError(msg)
    return frozenset(line for line in result.stdout.splitlines() if line.strip())


def round_trip(
    runner: Runner,
    container: str,
    database: str,
    root: Path,
    migration: str,
    down: str,
    prerequisites: Sequence[str],
) -> ModelRoundTrip:
    for key in prerequisites:
        pre = _psql(runner, container, database, (root / key).read_text("utf-8"))
        if pre.returncode != 0:
            msg = f"prerequisite {key} failed: {pre.stderr.strip()[:300]}"
            raise RuntimeError(msg)
    fwd_sql = (root / migration).read_text("utf-8")
    before = _snapshot(runner, container, database)
    fwd = _psql(runner, container, database, fwd_sql)
    after_forward = _snapshot(runner, container, database)
    dwn = _psql(runner, container, database, (root / down).read_text("utf-8"))
    after_down = _snapshot(runner, container, database)
    ref = _psql(runner, container, database, fwd_sql)
    after_reforward = _snapshot(runner, container, database)
    for label, result in (("forward", fwd), ("down", dwn), ("forward re-apply", ref)):
        if result.returncode != 0:
            print(f"{label} stderr: {result.stderr.strip()[:500]}", file=sys.stderr)
    return ModelRoundTrip(
        forward_exit=fwd.returncode,
        down_exit=dwn.returncode,
        reforward_exit=ref.returncode,
        before=before,
        after_forward=after_forward,
        after_down=after_down,
        after_reforward=after_reforward,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _IndentedDumper(yaml.SafeDumper):
    """Indent block sequences under their key, the layout yamlfmt enforces."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, False)


def _append_record(path: Path, record: dict[str, str]) -> None:
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    executions = list(data.get("executions") or [])
    executions.append(record)
    head = [line for line in text.splitlines() if line.startswith("#")]
    body = yaml.dump(
        {"schema_version": data["schema_version"], "executions": executions},
        Dumper=_IndentedDumper,
        sort_keys=False,
        width=100_000,
    )
    path.write_text("\n".join(head) + ("\n" if head else "") + body, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    checker = _load_checker()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--migration", required=True, help="forward key, e.g. forward/031_x.sql"
    )
    parser.add_argument(
        "--down", required=True, help="down key, e.g. rollback/rollback_031_x.sql"
    )
    parser.add_argument("--prerequisite", action="append", default=[])
    parser.add_argument("--container", required=True, help="scratch postgres container")
    parser.add_argument(
        "--surface", required=True, help="lab surface id, e.g. mac-scratch:<lane>"
    )
    parser.add_argument("--root", type=Path, default=checker.DEFAULT_MIGRATIONS_ROOT)  # type: ignore[attr-defined]
    parser.add_argument("--executions", type=Path, default=checker.DEFAULT_EXECUTIONS)  # type: ignore[attr-defined]
    parser.add_argument("--write", action="store_true", help="append the record")
    args = parser.parse_args(argv)

    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d%H%M%S")
    database = f"downproof_{stamp}"
    created = _run(
        [
            "docker",
            "exec",
            args.container,
            "psql",
            "-U",
            "postgres",
            "-d",
            "postgres",
            "-v",
            "ON_ERROR_STOP=1",
            "-c",
            f"CREATE DATABASE {database}",
        ],
        None,
    )
    if created.returncode != 0:
        print(f"could not create {database}: {created.stderr.strip()}", file=sys.stderr)
        return 2
    try:
        trip = round_trip(
            _run,
            args.container,
            database,
            args.root,
            args.migration,
            args.down,
            args.prerequisite,
        )
    finally:
        _run(
            [
                "docker",
                "exec",
                args.container,
                "psql",
                "-U",
                "postgres",
                "-d",
                "postgres",
                "-c",
                f"DROP DATABASE IF EXISTS {database}",
            ],
            None,
        )
    outcome, evidence = verdict(trip)
    executed_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "migration": args.migration,
        "down_script": args.down,
        "forward_sha256": _sha(args.root / args.migration),
        "down_sha256": _sha(args.root / args.down),
        "surface": args.surface,
        "database": f"{database} on container {args.container} (created and dropped by this run)",
        "executed_at": executed_at,
        "outcome": outcome,
        "evidence": evidence
        + (
            f"; prerequisites: {', '.join(args.prerequisite)}"
            if args.prerequisite
            else ""
        ),
    }
    checker.ModelDownExecution(**record)  # type: ignore[attr-defined]  # validates the surface
    print(yaml.safe_dump([record], sort_keys=False, width=100))
    if args.write:
        _append_record(args.executions, record)
        print(f"recorded in {args.executions}")
    return 0 if outcome == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
