#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17886: diff a live database's ``omninode_internal`` ACLs against the topology.

Reads relation ACLs, the schema ACL, default-privilege rules and relation owners
for one schema from a LIVE database, in a read-only session, and compares them in
both directions with what ``--profile``'s typed topology declares. See
``omnibase_infra.validation.omninode_internal_live_acl`` for what counts as a
finding.

Usage:
    python scripts/validation/check_omninode_internal_live_acl.py \\
        --profile onex-dev --dsn-env OMNINODE_INTERNAL_DB_URL

The DSN is read from the named environment variable and never printed.

Exit codes:
    0  no difference
    1  at least one finding
    2  usage error, unknown profile, or the database could not be read
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import psycopg2
import psycopg2.extras

from omnibase_infra.topology import SUPPORTED_TOPOLOGY_PROFILES, load_topology_profile
from omnibase_infra.validation.omninode_internal_live_acl import (
    DEFAULT_SCHEMA,
    LIVE_DEFAULT_ACL_QUERY,
    LIVE_RELATION_ACL_QUERY,
    LIVE_SCHEMA_ACL_QUERY,
    LiveAclReport,
    declared_acl,
    diff_live_acl,
)


def _read_live(dsn: str, schema: str) -> dict[str, list[dict[str, object]]]:
    """Run the three catalog reads in ONE read-only transaction."""
    connection = psycopg2.connect(dsn)
    try:
        connection.set_session(readonly=True, autocommit=False)
        with connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
            rows: dict[str, list[dict[str, object]]] = {}
            for key, statement in (
                ("relation", LIVE_RELATION_ACL_QUERY),
                ("schema", LIVE_SCHEMA_ACL_QUERY),
                ("default", LIVE_DEFAULT_ACL_QUERY),
            ):
                cursor.execute(statement, (schema,))
                rows[key] = [dict(row) for row in cursor.fetchall()]
        connection.rollback()
        return rows
    finally:
        connection.close()


def _report_json(report: LiveAclReport) -> str:
    return json.dumps(
        {
            "schema": report.schema,
            "clean": report.clean,
            "findings": list(report.findings),
            "observed_relation_count": report.observed_relation_count,
            "relation_owners": dict(report.relation_owners),
        },
        indent=2,
        sort_keys=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--profile",
        required=True,
        help="topology profile the database is expected to match",
    )
    parser.add_argument(
        "--dsn-env",
        required=True,
        help="name of the environment variable holding the DSN (never printed)",
    )
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    parser.add_argument("--database-ref", default="application")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)

    if args.profile not in SUPPORTED_TOPOLOGY_PROFILES:
        print(
            f"unknown topology profile {args.profile!r}; "
            f"expected one of {sorted(SUPPORTED_TOPOLOGY_PROFILES)}",
            file=sys.stderr,
        )
        return 2
    dsn = os.environ.get(args.dsn_env, "")
    if not dsn:
        print(f"environment variable {args.dsn_env} is unset or empty", file=sys.stderr)
        return 2

    declared = declared_acl(
        load_topology_profile(args.profile),
        database_ref=args.database_ref,
        schema=args.schema,
    )
    try:
        live = _read_live(dsn, args.schema)
    except psycopg2.Error as exc:
        print(
            f"could not read the live catalog: {type(exc).__name__}: "
            f"{str(exc).splitlines()[0] if str(exc) else ''}",
            file=sys.stderr,
        )
        return 2

    report = diff_live_acl(
        declared,
        relation_rows=live["relation"],
        schema_rows=live["schema"],
        default_rows=live["default"],
    )
    if args.json:
        print(_report_json(report))
    else:
        print(
            f"{args.schema}: {report.observed_relation_count} live relation(s), "
            f"{len(declared.table_grants)} declared table grant(s) for profile "
            f"{args.profile}"
        )
        for relation, owner in report.relation_owners.items():
            print(f"  owner {args.schema}.{relation} = {owner}")
        for line in report.findings:
            print(line)
        print("clean" if report.clean else f"{len(report.findings)} finding(s)")
    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
