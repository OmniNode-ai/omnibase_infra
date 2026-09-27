#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Destroy ONE pre-PR verify slot and prove it is gone (OMN-18896, Task 7).

The teardown half of Task 7. The reaper and the heartbeat lease are not here.

Why this exists
---------------
``prepr_verify_lane.sh`` brings a slot up and leaves it running, because a
slot's job is to be exercised. Until this module existed every slot on the lab
host was then torn down BY HAND, five times on 2026-09-24, each lane working
out again what a slot leaves on the SHARED dev-lane servers: the compose
project's containers, volumes and images, the slot's prefixed topics and
consumer groups on the dev broker, its Valkey logical index, and its databases
and principals on the dev Postgres. A hand teardown that deletes by a loose
pattern can take the dev lane's own topics, groups or keys with it. So the
selection lives here, as functions a test runs against listings that carry the
dev lane's names beside the slot's.

What it touches, and the fence on each
--------------------------------------
* containers, volumes, images: selected by the compose-project LABEL of the
  slot's project, which ``prepr_slot_policy.assert_target_is_a_pool_slot``
  confirms is a pool slot and not a declared lane. The dependency network is
  external and shared, and is never touched.
* topics and consumer groups: names beginning ``<slot token>.``, where the
  token must match ``^prepr[1-9][0-9]*$``. An empty or dev-shaped token is
  refused before any listing is read, because an empty prefix selects
  everything.
* Valkey: ``FLUSHDB`` on the slot's own index only; index 0 is the dev lane's
  and is refused.
* Postgres: ``provision_db_slot.sh --drop``, which enumerates from the catalog
  and fences every row by name AND by slot-group membership.
* the slot's staging root, only under ``/var/tmp/onex-prepr``.

Then every axis is read back. A zero counts only beside a positive control
that reads non-zero on the same server (rule 16): a broker listing that
returned nothing because the credential was wrong reads exactly like a clean
teardown otherwise.

Credentials travel by environment into ``docker exec -e NAME``, never on a
command line, and nothing here prints one.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PROVISION_DB = REPO_ROOT / "scripts" / "provision_db_slot.sh"


def _load_policy() -> Any:
    name = "prepr_slot_policy"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


policy = _load_policy()

#: The dev lane's dependency servers, which a slot borrows. Read and written
#: here only through the slot's own selectors.
BROKER_CONTAINER = "omnibase-infra-redpanda"
VALKEY_CONTAINER = "omnibase-infra-valkey"
POSTGRES_CONTAINER = "omnibase-infra-postgres"
#: The dev lane's compose project, read as the containers positive control.
DEV_PROJECT = "omnibase-infra"

STAGING_PARENT = Path("/var/tmp/onex-prepr")  # noqa: S108 - the pool's declared staging parent, shared with prepr_verify_lane.sh
SLOT_TOKEN_RE = re.compile(r"^prepr[1-9][0-9]*$")
#: rpk accepts many names per delete; chunked so one argv stays well inside
#: ARG_MAX on a slot that provisioned ~1700 topics.
DELETE_CHUNK = 200

RPK_ENV = ("RPK_USER", "RPK_PASS", "RPK_SASL_MECHANISM")
VALKEY_ENV = ("REDISCLI_AUTH",)

Runner = Callable[
    [list[str], "dict[str, str] | None"], "subprocess.CompletedProcess[str]"
]


class Selectors(NamedTuple):
    compose_project: str
    db_slot: str
    #: Every prefix a slot-owned topic or consumer group begins with.
    name_prefixes: tuple[str, ...]
    valkey_db_index: int


def selectors_for(slot_policy: Any) -> Selectors:
    """Derive the teardown selectors, refusing any that could reach the dev lane."""
    policy.assert_target_is_a_pool_slot(slot_policy.compose_project)
    tokens = {
        "db_slot": slot_policy.db_slot,
        "topic_namespace": slot_policy.topic_namespace,
        "kafka_environment": slot_policy.kafka_environment,
    }
    for field, token in tokens.items():
        if not SLOT_TOKEN_RE.fullmatch(token or ""):
            raise policy.RefusalError(
                policy.EXIT_USAGE,
                f"{field} '{token}' is not a slot token (^prepr[1-9][0-9]*$). "
                f"A teardown selects by this prefix on the dev lane's shared "
                f"servers, and an empty or dev-shaped prefix would select the "
                f"dev lane's own names.",
            )
    if int(slot_policy.valkey_db_index) <= 0:
        raise policy.RefusalError(
            policy.EXIT_USAGE,
            f"Valkey index {slot_policy.valkey_db_index} belongs to the dev "
            f"lane; a slot teardown flushes only its own non-zero index.",
        )
    prefixes = tuple(
        sorted({f"{slot_policy.topic_namespace}.", f"{slot_policy.kafka_environment}."})
    )
    return Selectors(
        compose_project=slot_policy.compose_project,
        db_slot=slot_policy.db_slot,
        name_prefixes=prefixes,
        valkey_db_index=int(slot_policy.valkey_db_index),
    )


def resolve_staging_root(slot_policy: Any, override: str | None) -> Path:
    """The slot's staging root, only ever strictly below the pool parent."""
    default = STAGING_PARENT / f"slot-{slot_policy.slot}"
    if override is None:
        return default
    candidate = Path(os.path.normpath(override))
    if not candidate.is_absolute() or STAGING_PARENT not in candidate.parents:
        raise policy.RefusalError(
            policy.EXIT_USAGE,
            f"staging root '{override}' is not strictly below {STAGING_PARENT}; "
            f"a teardown deletes this directory recursively.",
        )
    return candidate


def parse_rpk_column(listing: str, column: str) -> list[str]:
    """Read one named column of an rpk table.

    Refuses output with no header row, because an rpk error printed in place
    of a table must not parse as "the slot owns nothing".
    """
    lines = [line for line in listing.splitlines() if line.strip()]
    if not lines or column not in lines[0].split():
        raise ValueError(f"rpk output has no '{column}' header: {listing[:200]!r}")
    index = lines[0].split().index(column)
    names: list[str] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) > index:
            names.append(fields[index])
    return names


def select_slot_names(names: Iterable[str], prefixes: Iterable[str]) -> list[str]:
    wanted = tuple(prefixes)
    if not wanted or any(not p or not p.endswith(".") for p in wanted):
        raise ValueError(f"refusing unanchored prefixes {wanted!r}")
    return sorted({n for n in names if n.startswith(wanted)})


def _chunks(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def default_runner(
    argv: list[str], env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    # stdin is /dev/null: provision_db_slot.sh's container psql path runs
    # `docker run -i`, which otherwise reads a caller's stdin (FRICTION row
    # lane=pg-conn-slots, 2026-09-24).
    return subprocess.run(
        argv,
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )


class _Teardown:
    def __init__(
        self, sel: Selectors, runner: Runner, base_env: Mapping[str, str]
    ) -> None:
        self.sel = sel
        self.run = runner
        self.env = dict(base_env)
        self.errors: list[str] = []

    # -- helpers ----------------------------------------------------------
    def _exec(self, container: str, names: tuple[str, ...], *cmd: str) -> list[str]:
        argv = ["docker", "exec"]
        for name in names:
            argv += ["-e", name]
        return argv + [container, *cmd]

    def _checked(
        self, argv: list[str], what: str, env: dict[str, str] | None = None
    ) -> str:
        result = self.run(argv, env if env is not None else self.env)
        if result.returncode != 0:
            self.errors.append(
                f"{what}: exit {result.returncode}: {(result.stderr or '').strip()[:300]}"
            )
            return ""
        return result.stdout or ""

    def _lines(self, out: str) -> list[str]:
        return sorted({line.strip() for line in out.splitlines() if line.strip()})

    def _scalar(self, out: str) -> int:
        try:
            return int(out.strip().splitlines()[-1])
        except (IndexError, ValueError):
            return -1

    # -- listings ---------------------------------------------------------
    def containers(self, project: str) -> list[str]:
        return self._lines(
            self._checked(
                [
                    "docker",
                    "ps",
                    "-a",
                    "-q",
                    "--filter",
                    f"label=com.docker.compose.project={project}",
                ],
                f"list containers of {project}",
            )
        )

    def volumes(self, project: str) -> list[str]:
        return self._lines(
            self._checked(
                [
                    "docker",
                    "volume",
                    "ls",
                    "-q",
                    "--filter",
                    f"label=com.docker.compose.project={project}",
                ],
                f"list volumes of {project}",
            )
        )

    def images(self, project: str) -> list[str]:
        return self._lines(
            self._checked(
                [
                    "docker",
                    "images",
                    "-q",
                    "--filter",
                    f"label=com.docker.compose.project={project}",
                ],
                f"list images of {project}",
            )
        )

    def _rpk_names(self, kind: str, column: str) -> list[str] | None:
        out = self._checked(
            self._exec(BROKER_CONTAINER, RPK_ENV, "rpk", kind, "list"),
            f"rpk {kind} list",
        )
        try:
            return parse_rpk_column(out, column)
        except ValueError as exc:
            self.errors.append(f"rpk {kind} list unreadable: {exc}")
            return None

    def topics(self) -> list[str] | None:
        return self._rpk_names("topic", "NAME")

    def groups(self) -> list[str] | None:
        return self._rpk_names("group", "GROUP")

    def valkey_size(self, index: int) -> int:
        return self._scalar(
            self._checked(
                self._exec(
                    VALKEY_CONTAINER,
                    VALKEY_ENV,
                    "valkey-cli",
                    "-n",
                    str(index),
                    "DBSIZE",
                ),
                f"valkey DBSIZE {index}",
            )
        )

    def _psql(self, sql: str) -> int:
        user = self.env.get("POSTGRES_USER", "")
        if not user:
            self.errors.append(
                "POSTGRES_USER is not set; the Postgres readback cannot run"
            )
            return -1
        return self._scalar(
            self._checked(
                [
                    "docker",
                    "exec",
                    POSTGRES_CONTAINER,
                    "psql",
                    "-U",
                    user,
                    "-d",
                    "postgres",
                    "-tAc",
                    sql,
                ],
                "postgres readback",
            )
        )

    def slot_databases(self) -> int:
        return self._psql(
            f"SELECT count(*) FROM pg_database WHERE datname LIKE '%\\_{self.sel.db_slot}'"  # noqa: S608 - db_slot passed SLOT_TOKEN_RE
        )

    def slot_roles(self) -> int:
        return self._psql(
            f"SELECT count(*) FROM pg_roles WHERE rolname LIKE '%\\_{self.sel.db_slot}'"  # noqa: S608 - db_slot passed SLOT_TOKEN_RE
        )

    def all_databases(self) -> int:
        return self._psql("SELECT count(*) FROM pg_database")

    # -- the run ----------------------------------------------------------
    def plan(self, staging_root: Path) -> dict[str, Any]:
        topics = self.topics() or []
        groups = self.groups() or []
        return {
            "containers": self.containers(self.sel.compose_project),
            "volumes": self.volumes(self.sel.compose_project),
            "images": self.images(self.sel.compose_project),
            "topics": select_slot_names(topics, self.sel.name_prefixes),
            "groups": select_slot_names(groups, self.sel.name_prefixes),
            "valkey_index": self.sel.valkey_db_index,
            "valkey_keys": self.valkey_size(self.sel.valkey_db_index),
            "databases": self.slot_databases(),
            "roles": self.slot_roles(),
            "staging_root": str(staging_root) if staging_root.exists() else None,
        }

    def destroy(self, planned: dict[str, Any], staging_root: Path) -> None:
        # Containers first: a live consumer keeps its group from being deleted
        # and its pool keeps sessions open on the slot's databases.
        if planned["containers"]:
            self._checked(
                ["docker", "rm", "-f", *planned["containers"]], "remove containers"
            )
        for chunk in _chunks(planned["groups"], DELETE_CHUNK):
            self._checked(
                self._exec(BROKER_CONTAINER, RPK_ENV, "rpk", "group", "delete", *chunk),
                "delete groups",
            )
        for chunk in _chunks(planned["topics"], DELETE_CHUNK):
            self._checked(
                self._exec(BROKER_CONTAINER, RPK_ENV, "rpk", "topic", "delete", *chunk),
                "delete topics",
            )
        self._checked(
            self._exec(
                VALKEY_CONTAINER,
                VALKEY_ENV,
                "valkey-cli",
                "-n",
                str(self.sel.valkey_db_index),
                "FLUSHDB",
            ),
            "flush the slot's Valkey index",
        )
        drop_env = dict(self.env)
        drop_env["ONEX_DB_SLOT"] = self.sel.db_slot
        self._checked(
            ["bash", str(PROVISION_DB), "--drop"],
            "provision_db_slot.sh --drop",
            env=drop_env,
        )
        if planned["volumes"]:
            self._checked(
                ["docker", "volume", "rm", *planned["volumes"]], "remove volumes"
            )
        if planned["images"]:
            self._checked(["docker", "rmi", "-f", *planned["images"]], "remove images")
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=False)

    def readback(self, staging_root: Path) -> tuple[dict[str, int], dict[str, int]]:
        topics = self.topics()
        groups = self.groups()
        residue = {
            "containers": len(self.containers(self.sel.compose_project)),
            "volumes": len(self.volumes(self.sel.compose_project)),
            "images": len(self.images(self.sel.compose_project)),
            "topics": -1
            if topics is None
            else len(select_slot_names(topics, self.sel.name_prefixes)),
            "groups": -1
            if groups is None
            else len(select_slot_names(groups, self.sel.name_prefixes)),
            "valkey_keys": self.valkey_size(self.sel.valkey_db_index),
            "databases": self.slot_databases(),
            "roles": self.slot_roles(),
            "staging_root": int(staging_root.exists()),
        }
        controls = {
            "dev_containers": len(self.containers(DEV_PROJECT)),
            "other_topics": -1 if topics is None else len(topics) - residue["topics"],
            "other_groups": -1 if groups is None else len(groups) - residue["groups"],
            "valkey_index0_keys": self.valkey_size(0),
            "databases": self.all_databases(),
        }
        return residue, controls


def run_teardown(
    slot_policy: Any,
    *,
    runner: Runner = default_runner,
    staging_root: Path,
    base_env: Mapping[str, str],
    plan_only: bool = False,
    report_path: Path | None = None,
) -> int:
    sel = selectors_for(slot_policy)
    td = _Teardown(sel, runner, base_env)
    planned = td.plan(staging_root)
    report: dict[str, Any] = {
        "schema": "onex.prepr.slot-teardown.v1",
        "slot": slot_policy.slot,
        "compose_project": sel.compose_project,
        "planned": planned,
    }
    if plan_only:
        report["verdict"] = "PLAN"
        code = policy.EXIT_OK if not td.errors else policy.EXIT_TEARDOWN_INCOMPLETE
    else:
        if not td.errors:
            td.destroy(planned, staging_root)
        residue, controls = td.readback(staging_root)
        report["residue"] = residue
        report["controls"] = controls
        if td.errors or any(v != 0 for v in residue.values()):
            report["verdict"] = "INCOMPLETE"
        elif any(v <= 0 for v in controls.values()):
            report["verdict"] = "UNPROVEN"
        else:
            report["verdict"] = "CLEAN"
        code = (
            policy.EXIT_OK
            if report["verdict"] == "CLEAN"
            else policy.EXIT_TEARDOWN_INCOMPLETE
        )
    report["errors"] = td.errors
    text = json.dumps(report, indent=2, sort_keys=True)
    if report_path is not None:
        report_path.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    return int(code)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepr_teardown_slot.py", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--slot", type=int, required=True)
    parser.add_argument("--staging-root", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--report-out", default=None)
    args = parser.parse_args(argv)
    try:
        slot_policy = policy.resolve_slot(args.slot)
        staging_root = resolve_staging_root(slot_policy, args.staging_root)
        return run_teardown(
            slot_policy,
            staging_root=staging_root,
            base_env=os.environ,
            plan_only=args.plan_only,
            report_path=Path(args.report_out) if args.report_out else None,
        )
    except policy.RefusalError as exc:
        sys.stderr.write(f"[prepr-teardown] REFUSED: {exc}\n")
        return int(exc.code)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
