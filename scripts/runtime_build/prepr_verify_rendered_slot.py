#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Verify a RENDERED pre-PR slot against the slot policy (OMN-18893).

Why a rendered-config gate exists at all
----------------------------------------
Every other isolation control in this design is a value written in a file:
a topic prefix, a database suffix, a consumer group, a port, a volume name.
Reviewing those files tells you what they say. It does not tell you what
compose will actually run, and the gap between the two is not hypothetical.

Three ways the gap opens, all of them silent:

1. **The ambient environment outranks a file COMPOSE READS.** Compose resolves
   an interpolation from the OS environment first and from ``--env-file`` only
   as a fallback. The lab host's interactive shell exports
   ``KAFKA_ENVIRONMENT`` set to ``local`` -- the dev lane's own consumer-group
   token -- so a slot brought up from a terminal inherits it and joins the dev
   lane's groups while every file on disk says it is isolated. Measured
   2026-09-21.

   The qualifier is load-bearing and is not pedantry. An env file the SHELL
   sources under ``set -a`` behaves the opposite way: it assigns, so it
   overwrites what the shell exported and the file wins. Both are precedence
   questions with the same surface shape and they resolve in opposite
   directions, and this author has already carried one conclusion to the
   other once, in a claim that reached a merged document before a reviewer
   refused to repeat the part they could not verify. Say which file.

2. **A compose merge appends where you expected it to replace.** ``ports`` and
   ``volumes`` are sequences, and an override file's sequence is APPENDED to
   the base's unless the entry carries ``!override``. A slot that published
   both ``8085`` and ``28085`` would collide with the dev lane on the first,
   and a slot mounting both the dev volume and its own at ``/app/logs`` would
   write into the dev lane's data.

3. **An edit to a base file lands outside this overlay.** A new unsuffixed DSN
   added to ``x-runtime-env`` is inherited by every slot service silently, and
   the slot then connects to a dev database as whichever principal that DSN
   names -- by default the superuser.

None of the three is visible in a review of the overlay, and all three are
visible in the rendered output. So the rendered output is what is checked, and
it is checked before anything starts rather than after something has gone
wrong.

The checks are DERIVED, never asserted
--------------------------------------
Every expectation below is computed from ``prepr_slot_policy.py`` for the slot
under test. Nothing here spells a port, a prefix or a database name, so a
change to the policy table cannot leave this gate passing against the old one.

The DSN check is the important one and it is deliberately a POSITIVE test on
the suffix rather than a negative test on a blocklist of dev names. A blocklist
answers "is this one of the databases I thought of"; the suffix rule answers
"is this the slot's own", which is the property that has to hold for a database
this code has never heard of.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepr_slot_policy import (
    EXIT_OK,
    EXIT_PROVENANCE_MISMATCH,
    EXIT_USAGE,
    SlotPolicy,
    resolve_slot,
)

#: The eleven services a slot always runs, plus the gateway it runs only under
#: ``--with-gateway``. Their compose keys, which are the dev lane's own service
#: names -- the slot renames the CONTAINERS, never the services, because the
#: services are inherited.
SLOT_SERVICES: tuple[str, ...] = (
    "omninode-runtime",
    "runtime-effects",
    "runtime-worker",
    "projection-api",
    "tenant-projection-writer",
    "projection-tenant-registry-writer",
    "projection-delegation-writer",
    "projection-registration-writer",
    "projection-savings-writer",
    "projection-tenant-credentials-writer",
    "projection-live-events-writer",
)
GATEWAY_SERVICE = "onex-api"

#: Every environment key whose value is a PostgreSQL DSN. Each one must name a
#: database carrying this slot's suffix and a principal carrying it too.
DSN_KEYS: tuple[str, ...] = (
    "OMNIBASE_INFRA_DB_URL",
    "OMNIINTELLIGENCE_DB_URL",
    "OMNIDASH_ANALYTICS_DB_URL",
    "OMNIMEMORY_DB_URL",
    "OMNINODE_INTERNAL_DB_URL",
    "ONEX_TENANT_DB_URL",
    "OMNINODE_CLOUD_DB_URL",
)

_DSN = re.compile(
    r"^postgres(?:ql)?://(?P<user>[^:/@]+):[^@]*@(?P<host>[^:/]+):\d+/(?P<db>[^?]+)"
)


def _container_prefixes(policy: SlotPolicy) -> str:
    return f"-prepr-{policy.slot}-"


def check(
    rendered: dict[str, Any], policy: SlotPolicy, expect_gateway: bool
) -> list[str]:
    """Return every mismatch. An empty list is the only passing result."""
    problems: list[str] = []
    services = rendered.get("services") or {}
    expected = set(SLOT_SERVICES) | ({GATEWAY_SERVICE} if expect_gateway else set())

    missing = sorted(expected - set(services))
    if missing:
        problems.append(
            f"the render is missing {len(missing)} slot service(s): {missing}. "
            f"A slot that starts a short set is a slot whose verdict covers less "
            f"than it claims."
        )
    unexpected = sorted(set(services) - expected)
    if unexpected:
        problems.append(
            f"the render carries {len(unexpected)} service(s) that are not the "
            f"slot's: {unexpected}. Each one is a shared dependency the fence "
            f"was supposed to exclude, and starting it would recreate a "
            f"dev-named container (OMN-13581)."
        )

    suffix = f"_{policy.db_slot}"
    for name in sorted(expected & set(services)):
        svc = services[name] or {}
        where = f"service '{name}'"

        container = svc.get("container_name") or ""
        if _container_prefixes(policy) not in container:
            problems.append(
                f"{where}: container_name '{container}' does not carry "
                f"'{_container_prefixes(policy)}'. An unslotted container name "
                f"collides with the dev lane's own container."
            )

        env = svc.get("environment") or {}

        got_ns = env.get("KAFKA_TOPIC_NAMESPACE")
        if got_ns != policy.topic_namespace:
            problems.append(
                f"{where}: KAFKA_TOPIC_NAMESPACE is {got_ns!r}, expected "
                f"{policy.topic_namespace!r}. Without it this service publishes "
                f"onto, and consumes from, the dev lane's own topics."
            )
        got_env = env.get("KAFKA_ENVIRONMENT")
        if got_env != policy.kafka_environment:
            problems.append(
                f"{where}: KAFKA_ENVIRONMENT is {got_env!r}, expected "
                f"{policy.kafka_environment!r}. This is the variable the "
                f"invoking shell silently overrides; a wrong value here means "
                f"this service joins the dev lane's consumer groups."
            )
        got_valkey = str(env.get("VALKEY_DB", ""))
        if got_valkey != str(policy.valkey_db_index):
            problems.append(
                f"{where}: VALKEY_DB is {got_valkey!r}, expected "
                f"{str(policy.valkey_db_index)!r}."
            )

        group = env.get("KAFKA_CONSUMER_GROUP")
        if group is not None and policy.db_slot not in group:
            problems.append(
                f"{where}: KAFKA_CONSUMER_GROUP {group!r} does not carry the "
                f"slot token {policy.db_slot!r}. These groups are spelled as "
                f"literals and a namespace does not move them, so an inherited "
                f"one SPLITS the dev lane's own partitions between its writer "
                f"and this one -- half the dev lane's rows stop landing, and "
                f"the symptom appears on the dev lane rather than here."
            )

        for key in DSN_KEYS:
            dsn = env.get(key)
            if not dsn:
                continue
            match = _DSN.match(dsn)
            if match is None:
                problems.append(f"{where}: {key} is not a parseable DSN.")
                continue
            db = match.group("db")
            user = match.group("user")
            if not db.endswith(suffix):
                problems.append(
                    f"{where}: {key} names database {db!r}, which does not end "
                    f"in {suffix!r}. This connects the slot to a DEV database."
                )
            if not user.endswith(suffix):
                problems.append(
                    f"{where}: {key} connects as {user!r}, which does not end in "
                    f"{suffix!r}. A shared principal reaches the dev databases, "
                    f"and the superuser is additionally exempt from the forced "
                    f"row-level security the tenant tables rely on."
                )

        published = {
            str(p.get("published"))
            for p in (svc.get("ports") or [])
            if isinstance(p, dict) and p.get("published") is not None
        }
        allowed = {
            str(policy.runtime_main_port),
            str(policy.runtime_effects_port),
            str(policy.gateway_port),
            str(policy.projection_api_port),
        }
        stray = sorted(published - allowed)
        if stray:
            problems.append(
                f"{where}: publishes {stray}, which is outside the slot's "
                f"reserved block {sorted(allowed)}. A compose merge APPENDS "
                f"sequences unless the override carries '!override', so an "
                f"inherited dev port surviving here is the expected shape of "
                f"this defect -- and it would bind the dev lane's port."
            )

        for vol in svc.get("volumes") or []:
            if not isinstance(vol, dict) or vol.get("type") != "volume":
                continue
            src = str(vol.get("source") or "")
            if "prepr" not in src:
                problems.append(
                    f"{where}: mounts named volume {src!r}, which is not "
                    f"slot-scoped. The inherited volume names are GLOBAL Docker "
                    f"names shared with the running dev lane, so this writes "
                    f"into the dev lane's data and a teardown with -v destroys it."
                )

    for key, spec in (rendered.get("volumes") or {}).items():
        name = str((spec or {}).get("name") or key)
        if f"prepr-{policy.slot}" not in name:
            problems.append(
                f"top-level volume '{key}' resolves to global name {name!r}, "
                f"which does not carry this slot. See the same failure above."
            )

    for key, spec in (rendered.get("networks") or {}).items():
        if not (spec or {}).get("external"):
            problems.append(
                f"network '{key}' is not external. A slot JOINS the dev lane's "
                f"network; creating it means a slot can own, and a slot's "
                f"teardown can remove, the network the dev lane runs on."
            )

    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prepr_verify_rendered_slot.py",
        description=(
            "Check a rendered compose configuration against the slot policy. "
            "Read-only: opens no connection and starts nothing."
        ),
    )
    parser.add_argument(
        "--rendered",
        required=True,
        help=(
            "the rendered compose configuration, as a path or '-' for stdin. "
            "The entrypoint always pipes it: a rendered configuration expands "
            "every interpolation, so on a real host it carries the broker, "
            "database and Keycloak credentials in clear, and it must not be "
            "written to disk."
        ),
    )
    parser.add_argument("--slot", required=True, type=int)
    parser.add_argument(
        "--expect-gateway",
        action="store_true",
        help="the render should also carry the slot's onex-api container.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        policy = resolve_slot(args.slot)
    except Exception as exc:  # noqa: BLE001 - the refusal carries its own text
        sys.stderr.write(f"[prepr-verify-rendered] {exc}\n")
        return int(EXIT_USAGE)

    source = "standard input" if args.rendered == "-" else args.rendered
    try:
        if args.rendered == "-":
            rendered = json.loads(sys.stdin.read())
        else:
            rendered = json.loads(Path(args.rendered).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        # Deliberately reports the SOURCE and the parse error, never the
        # content: an unparseable render is still a render, and echoing it
        # here would put every expanded credential in the log of a failing
        # run, which is exactly where people paste from.
        sys.stderr.write(
            f"[prepr-verify-rendered] cannot read the rendered configuration from "
            f"{source}: {type(exc).__name__}. Failing closed: a gate that cannot "
            f"read its own input has not passed, it has not run.\n"
        )
        return int(EXIT_PROVENANCE_MISMATCH)

    problems = check(rendered, policy, args.expect_gateway)
    if problems:
        sys.stderr.write(
            f"[prepr-verify-rendered] {len(problems)} mismatch(es) between the "
            f"rendered slot {policy.slot} and its policy:\n"
        )
        for problem in problems:
            sys.stderr.write(f"  - {problem}\n")
        return int(EXIT_PROVENANCE_MISMATCH)

    sys.stderr.write(
        f"[prepr-verify-rendered] slot {policy.slot} render matches policy on "
        f"every axis: container names, ports, topic namespace, consumer-group "
        f"environment, consumer groups, DSN database and principal suffixes, "
        f"named volumes and network externality.\n"
    )
    return int(EXIT_OK)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
