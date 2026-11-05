#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Destroy ONE pre-PR verify slot and prove it is gone (OMN-18896, Task 7).

Shared-service cleanup, resource leases and abandoned-slot reaping.

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
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
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
#: here only through the slot's own selectors. Resolve by compose label
#: (project + service), not by container name: a timed-out compose recreate
#: leaves a hash-prefixed container name. Same pattern as compose_container()
#: in scripts/smoke/smoke_delegation.sh.
BROKER_SERVICE = "redpanda"
VALKEY_SERVICE = "valkey"
POSTGRES_SERVICE = "postgres"
#: The dev lane's compose project, read as the containers positive control.
DEV_PROJECT = "omnibase-infra"

STAGING_PARENT = Path("/var/tmp/onex-prepr")  # noqa: S108 - the pool's declared staging parent, shared with prepr_verify_lane.sh
SLOT_TOKEN_RE = re.compile(r"^prepr[1-9][0-9]*$")
#: rpk accepts many names per delete; chunked so one argv stays well inside
#: ARG_MAX on a slot that provisioned ~1700 topics.
DELETE_CHUNK = 200
GROUP_DRAIN_TIMEOUT_S = 60.0
GROUP_DRAIN_POLL_S = 2.0
#: OMN-17427: topic deletes run against the SHARED dev broker, so they go in
#: small chunks and each chunk waits for the broker to report healthy before
#: the next. On 2026-10-20 ~1845 prepr2 topics deleted in back-to-back chunks
#: of 200 stalled partition removal and crashed the dev broker (Redpanda
#: v24.2.7, smp 1), which then asserted on every restart and took the dev bus
#: down. A broker that does not come back healthy stops the deletes: the
#: remaining topics are residue the readback reports, never a reason to push on.
TOPIC_DELETE_CHUNK = 25
BROKER_HEALTH_ATTEMPTS = 12
BROKER_HEALTH_POLL_SECONDS = 5.0

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
        self._servers: dict[str, str | None] = {}
        self.tenant_authentication: dict[str, Any] | None = None

    # -- helpers ----------------------------------------------------------
    def server(self, service: str) -> str | None:
        if service not in self._servers:
            names = self._lines(
                self._checked(
                    [
                        "docker",
                        "ps",
                        "--filter",
                        f"label=com.docker.compose.project={DEV_PROJECT}",
                        "--filter",
                        f"label=com.docker.compose.service={service}",
                        "--filter",
                        "status=running",
                        "--format",
                        "{{.Names}}",
                    ],
                    f"resolve {service} in compose project {DEV_PROJECT}",
                )
            )
            self._servers[service] = None
            if not names:
                self.errors.append(
                    f"no running {service} container in compose project {DEV_PROJECT}"
                )
            elif len(names) > 1:
                self.errors.append(
                    f"ambiguous running {service} containers in compose project "
                    f"{DEV_PROJECT}: {', '.join(names)}; refusing to guess"
                )
            else:
                self._servers[service] = names[0]
        return self._servers[service]

    def _exec(
        self, service: str, names: tuple[str, ...], *cmd: str
    ) -> list[str] | None:
        container = self.server(service)
        if container is None:
            return None
        argv = ["docker", "exec"]
        for name in names:
            argv += ["-e", name]
        return argv + [container, *cmd]

    def _checked(
        self, argv: list[str] | None, what: str, env: dict[str, str] | None = None
    ) -> str:
        if argv is None:
            return ""
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
            self._exec(BROKER_SERVICE, RPK_ENV, "rpk", kind, "list"),
            f"rpk {kind} list",
        )
        try:
            return parse_rpk_column(out, column)
        except ValueError as exc:
            self.errors.append(f"rpk {kind} list unreadable: {exc}")
            return None

    def broker_healthy(self) -> bool:
        """The shared broker reports ``Healthy: true``, polled a bounded number of times."""
        argv = self._exec(BROKER_SERVICE, RPK_ENV, "rpk", "cluster", "health")
        if argv is None:
            return False
        for attempt in range(BROKER_HEALTH_ATTEMPTS):
            result = self.run(argv, self.env)
            if result.returncode == 0 and re.search(
                r"^Healthy:\s+true\s*$", result.stdout or "", re.MULTILINE
            ):
                return True
            if attempt + 1 < BROKER_HEALTH_ATTEMPTS:
                time.sleep(BROKER_HEALTH_POLL_SECONDS)
        return False

    def topics(self) -> list[str] | None:
        return self._rpk_names("topic", "NAME")

    def groups(self) -> list[str] | None:
        return self._rpk_names("group", "GROUP")

    def valkey_size(self, index: int) -> int:
        return self._scalar(
            self._checked(
                self._exec(
                    VALKEY_SERVICE,
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
        container = self.server(POSTGRES_SERVICE)
        if container is None:
            return -1
        return self._scalar(
            self._checked(
                [
                    "docker",
                    "exec",
                    container,
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
        # Offboard while the gateway and database can still demonstrate a
        # working credential followed by an authentication refusal.
        if not self.revoke_tenant(staging_root):
            return
        # Containers first: a live consumer keeps its group from being deleted
        # and its pool keeps sessions open on the slot's databases.
        if planned["containers"]:
            before = len(self.errors)
            self._checked(
                ["docker", "rm", "-f", *planned["containers"]], "remove containers"
            )
            if len(self.errors) != before:
                return
        self.drain_groups(planned["groups"])
        for chunk in _chunks(planned["topics"], TOPIC_DELETE_CHUNK):
            self._checked(
                self._exec(BROKER_SERVICE, RPK_ENV, "rpk", "topic", "delete", *chunk),
                "delete topics",
            )
            if not self.broker_healthy():
                self.errors.append(
                    "delete topics: stopped, the shared broker did not report "
                    f"healthy within {BROKER_HEALTH_ATTEMPTS} reads after a chunk; "
                    "the remaining slot topics are left for a later teardown"
                )
                break
        self._checked(
            self._exec(
                VALKEY_SERVICE,
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
        # Keep the compose files and tenant credential for a rerun if any
        # earlier step failed.
        if self.errors:
            return
        if staging_root.exists():
            try:
                shutil.rmtree(staging_root, ignore_errors=False)
            except OSError as exc:
                self.errors.append(f"remove staging root: {str(exc)[:300]}")

    def drain_groups(self, pending: list[str]) -> None:
        """Stopped consumers remain active until their broker session expires.

        A transient delete failure is not terminal if a later authoritative
        listing proves the slot groups gone. Every retry derives fresh exact
        slot-prefixed names; an unreadable listing stays a failure.
        """
        if not pending:
            return
        deadline = time.monotonic() + GROUP_DRAIN_TIMEOUT_S
        while True:
            for chunk in _chunks(pending, DELETE_CHUNK):
                command = self._exec(
                    BROKER_SERVICE, RPK_ENV, "rpk", "group", "delete", *chunk
                )
                if command is None:
                    return
                self.run(command, self.env)
            groups = self.groups()
            if groups is None:
                return
            pending = select_slot_names(groups, self.sel.name_prefixes)
            if not pending:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.errors.append(
                    "slot consumer groups remain after the bounded session drain"
                )
                return
            time.sleep(min(GROUP_DRAIN_POLL_S, remaining))

    def revoke_tenant(self, staging_root: Path) -> bool:
        credential = staging_root / "tenant-state" / "credential.env"
        if not credential.exists():
            # A slot without a gateway, or a bring-up that failed before mint.
            return True
        env = dict(self.env)
        env["COMPOSE_PROJECT"] = self.sel.compose_project
        env["LAB_TENANT_SLUG"] = f"onex-{self.sel.db_slot}"
        before = len(self.errors)
        output = self._checked(
            [
                "bash",
                str(REPO_ROOT / "scripts" / "smoke" / "smoke_delegation.sh"),
                "--target",
                "compose",
                "--tenant-lifecycle",
                "revoke",
            ],
            "revoke slot tenant and prove authentication refusal",
            env=env,
        )
        if len(self.errors) != before:
            return False
        try:
            proofs = [
                json.loads(line) for line in output.splitlines() if line.startswith("{")
            ]
        except json.JSONDecodeError:
            self.errors.append(
                "slot tenant offboard returned malformed authentication proof"
            )
            return False
        if not proofs or not (
            proofs[-1].get("revoked") is True
            and proofs[-1].get("auth_before") == 200
            and proofs[-1].get("auth_after") in (401, 403)
        ):
            self.errors.append("slot tenant offboard returned no authentication proof")
            return False
        self.tenant_authentication = {
            key: proofs[-1][key]
            for key in ("tenant_id", "revoked", "auth_before", "auth_after")
        }
        return True

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
    if td.tenant_authentication is not None:
        report["tenant_authentication"] = td.tenant_authentication
    text = json.dumps(report, indent=2, sort_keys=True)
    if report_path is not None:
        report_path.write_text(text + "\n", encoding="utf-8")
    sys.stdout.write(text + "\n")
    return int(code)


# Resource lease and reaper (OMN-18896)

LEASE_DESCRIPTION = """Heartbeat and reap the two pre-PR slots (OMN-18896).

The host-local resource lease outlives a killed pool driver. It is separate
from the pool's admission lease: acquiring an expired admission lease cannot
erase evidence of resources still awaiting cleanup. All resource-lease writes
and the entire reap share a kernel lock. Reaping also takes the bring-up's
compose-project lock and delegates every deletion to prepr_teardown_slot.sh.
Missing or unreadable evidence is never permission to delete resources.

The default heartbeat interval is 30 seconds and expiry is five minutes after
the last heartbeat. Keep a complete manual proof (bring-up, exercises and
teardown) inside one `run --slot N --holder LANE -- bash proof.sh` command.
The numbered pool does this per phase using one token for the whole run.
Standalone bring-up alone leaves a five-minute exercise window after exit.
Only canonical /var/tmp/onex-prepr/slot-N staging roots are admitted.

`reap --plan-only` reports eligibility without deleting resources. A normal
tick persists ~/.omnibase/state/prepr-slots/last-reap.json; UNKNOWN and
INCOMPLETE exit nonzero and retain the lease for inspection or retry. ABSENT
means no resource lease was found, not that legacy resources were inspected.
ONEX_PREPR_LEASE_DIR and ONEX_LANE_LOCK_DIR isolate filesystem-only tests;
the installer uses the same default state and locks as the slot entrypoints.
"""

RUNTIME = "scripts/runtime_build/"
FILES = tuple(
    RUNTIME + name
    for name in (
        "prepr_slot_policy.py",
        "prepr_teardown_slot.py",
        "prepr_teardown_slot.sh",
        "lane_lock.py",
        "lane_lock.sh",
    )
) + ("scripts/provision_db_slot.sh", "scripts/smoke/smoke_delegation.sh")
UNIT = "onex-prepr-slot-reaper"


def _unit_quote(value: str) -> str:
    if any(c in value for c in "\n\r\0"):
        raise ValueError("unit path contains a control character")
    return (
        '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'
    )


def install_reaper(prefix: Path, unit_dir: Path, *, enable: bool) -> dict[str, object]:
    """Install an immutable pre-PR slot reaper and its systemd user timer.

    Run on a Linux host offering the numbered shared-service slots. No sudo and
    no credentials are installed. --no-enable prepares the same artifacts without
    contacting systemd, for review and hosted-runner smoke tests.

    On the slot host, with the operator's existing ~/.omnibase/.env and Docker:
        python3 scripts/runtime_build/prepr_teardown_slot.py install-reaper
        systemctl --user start onex-prepr-slot-reaper.service
        systemctl --user list-timers onex-prepr-slot-reaper.timer
        journalctl --user -u onex-prepr-slot-reaper.service -n 30
    Read ~/.omnibase/state/prepr-slots/last-reap.json for the first tick's verdict.
    The user manager must already be configured to run while logged out; this
    installer does not change host login settings. Remove the schedule with
    `systemctl --user disable --now onex-prepr-slot-reaper.timer` before removing
    its units or installed snapshots. Legacy resources without a lease require
    manual inspection and the existing prepr_teardown_slot.sh entrypoint.
    """
    digest = hashlib.sha256()
    for name in FILES:
        digest.update(name.encode() + b"\0" + (REPO_ROOT / name).read_bytes())
    release = prefix / "releases" / digest.hexdigest()
    if not release.exists():
        staging = prefix / "releases" / f".install-{uuid.uuid4().hex}"
        try:
            for name in FILES:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(REPO_ROOT / name, target)
            staging.rename(release)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    helper = release / RUNTIME / "prepr_teardown_slot.py"
    service = f"""[Unit]
Description=Reap abandoned numbered pre-PR slots (OMN-18896)

[Service]
Type=oneshot
ExecStart={_unit_quote("/usr/bin/python3")} {_unit_quote(str(helper))} reap
TimeoutStartSec=1300
KillMode=control-group
UMask=0077
NoNewPrivileges=true
Environment=PYTHONDONTWRITEBYTECODE=1
"""
    timer = f"""[Unit]
Description=Inspect numbered pre-PR slot leases every minute

[Timer]
OnBootSec=60s
OnUnitActiveSec=60s
AccuracySec=5s
Unit={UNIT}.service

[Install]
WantedBy=timers.target
"""
    unit_dir.mkdir(parents=True, exist_ok=True)
    for suffix, contents in (("service", service), ("timer", timer)):
        path = unit_dir / f"{UNIT}.{suffix}"
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(contents)
        temporary.replace(path)
    if enable:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True, timeout=30)
        subprocess.run(
            ["systemctl", "--user", "enable", "--now", f"{UNIT}.timer"],
            check=True,
            timeout=30,
        )
    return {"helper": str(helper), "unit_dir": str(unit_dir), "enabled": enable}


LEASE_SCHEMA = "onex.prepr.slot-lease.v1"
TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
LEASE_DEFAULT_TTL = 300.0
LEASE_DEFAULT_INTERVAL = 30.0


class LeaseRefusedError(RuntimeError):
    """The slot's ownership cannot be established safely."""


def lease_slot_policy(slot: int) -> Any:
    if slot not in policy.SLOTS:
        raise LeaseRefusedError(f"slot {slot!r} is outside the closed pool policy")
    return policy.assert_target_is_a_pool_slot(policy.SLOTS[slot].compose_project)


def state_root() -> Path:
    return Path(
        os.environ.get(
            "ONEX_PREPR_LEASE_DIR", str(Path.home() / ".omnibase/state/prepr-slots")
        )
    )


def record_path(root: Path, slot: int) -> Path:
    lease_slot_policy(slot)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root / f"slot-{slot}.json"


def lane_lock_path(slot: int) -> Path:
    directory = Path(
        os.environ.get(
            "ONEX_LANE_LOCK_DIR", str(Path.home() / ".omnibase/state/lane-locks")
        )
    )
    return directory / f"{lease_slot_policy(slot).compose_project}.lock"


@contextlib.contextmanager
def locked(path: Path, *, wait: bool = False) -> Iterator[int]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise LeaseRefusedError("slot lock is held by another process") from exc
        # Close our descriptor on exit rather than explicitly unlocking its
        # open-file description. A surviving teardown child inherits this fd
        # and must retain the lane lock after a timeout or killed reaper.
        yield stream.fileno()


def process_start(pid: int) -> str:
    """Linux boot + process start ticks distinguish PID reuse and host reboot."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return f"{boot}:{stat[19]}"
    except (OSError, IndexError):
        return "unknown"


def process_alive(pid: int, recorded_start: str) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        if Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    start = process_start(pid)
    return start == "unknown" or recorded_start == "unknown" or start == recorded_start


def owner_alive(record: dict[str, Any]) -> bool:
    return process_alive(record["owner_pid"], record["owner_start"]) or (
        "child_pid" in record
        and process_alive(record["child_pid"], record["child_start"])
    )


def validate_staging(slot: int, staging: str) -> None:
    # Unattended deletion is narrower than the manual teardown's override.
    # Custom staging roots require manual teardown; a record cannot nominate
    # some other slot's tree (or a symlink to it) for a periodic reaper.
    expected = STAGING_PARENT / f"slot-{slot}"
    if Path(staging) != expected or expected.resolve() != expected:
        raise LeaseRefusedError(
            "unattended cleanup requires the slot's canonical staging root"
        )


def read_record(root: Path, slot: int) -> dict[str, Any]:
    record = json.loads(record_path(root, slot).read_text())
    if (
        not isinstance(record, dict)
        or record.get("schema") != LEASE_SCHEMA
        or record.get("slot") != slot
    ):
        raise LeaseRefusedError("unrecognized slot lease schema or slot identity")
    for key in ("token", "holder"):
        if not isinstance(record.get(key), str) or not TOKEN_RE.fullmatch(record[key]):
            raise LeaseRefusedError(f"invalid lease {key}")
    for key in ("created", "heartbeat", "expires"):
        if type(record.get(key)) not in (float, int) or not math.isfinite(record[key]):
            raise LeaseRefusedError(f"invalid lease {key}")
    if (
        type(record.get("owner_pid")) is not int
        or not 1 < record["owner_pid"] <= 2**31 - 1
        or not isinstance(record.get("owner_start"), str)
        or not record["owner_start"]
    ):
        raise LeaseRefusedError("invalid lease owner")
    if "child_pid" in record and (
        type(record["child_pid"]) is not int
        or not 1 < record["child_pid"] <= 2**31 - 1
        or not isinstance(record.get("child_start"), str)
        or not record["child_start"]
    ):
        raise LeaseRefusedError("invalid lease child")
    validate_staging(slot, record.get("staging_root", ""))
    return record


def write_record(root: Path, slot: int, record: dict[str, Any]) -> None:
    path = record_path(root, slot)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        json.dump(record, stream, sort_keys=True)
        stream.write("\n")
    tmp.replace(path)


def claim(
    root: Path,
    slot: int,
    token: str,
    holder: str,
    staging: str,
    *,
    ttl: float = LEASE_DEFAULT_TTL,
) -> dict[str, Any]:
    validate_staging(slot, staging)
    if not TOKEN_RE.fullmatch(token) or not TOKEN_RE.fullmatch(holder):
        raise LeaseRefusedError("invalid holder or run token")
    with locked(record_path(root, slot).with_suffix(".guard")):
        now = time.time()
        if record_path(root, slot).exists():
            old = read_record(root, slot)
            if old["token"] != token or owner_alive(old):
                raise LeaseRefusedError(
                    f"slot {slot} belongs to {old['holder']}; wait for CLEAN teardown or reaper"
                )
        record = {
            "schema": LEASE_SCHEMA,
            "slot": slot,
            "token": token,
            "holder": holder,
            "owner_pid": os.getpid(),
            "owner_start": process_start(os.getpid()),
            "created": now,
            "heartbeat": now,
            "expires": now + ttl,
            "staging_root": staging,
        }
        write_record(root, slot, record)
        return record


def heartbeat(
    root: Path, slot: int, token: str, *, ttl: float = LEASE_DEFAULT_TTL
) -> None:
    with locked(record_path(root, slot).with_suffix(".guard"), wait=True):
        record = read_record(root, slot)
        if record["token"] != token or record["owner_pid"] != os.getpid():
            raise LeaseRefusedError("heartbeat is not from the recorded owner")
        record.update(heartbeat=time.time(), expires=time.time() + ttl)
        write_record(root, slot, record)


def check_owner(root: Path, slot: int, token: str) -> None:
    with locked(record_path(root, slot).with_suffix(".guard"), wait=True):
        record = read_record(root, slot)
        if (
            record["token"] != token
            or not owner_alive(record)
            or record["expires"] <= time.time()
        ):
            raise LeaseRefusedError("the inherited resource lease has no live owner")


def clear(root: Path, slot: int, token: str) -> None:
    with locked(record_path(root, slot).with_suffix(".guard"), wait=True):
        path = record_path(root, slot)
        if not path.exists():
            return
        record = read_record(root, slot)
        if token and record["token"] != token:
            raise LeaseRefusedError("CLEAN teardown belongs to another run")
        if not token and owner_alive(record):
            raise LeaseRefusedError("manual teardown cannot release a live run's lease")
        path.unlink()


def authorize_teardown(root: Path, slot: int, token: str) -> None:
    with locked(record_path(root, slot).with_suffix(".guard"), wait=True):
        if not record_path(root, slot).exists():
            return
        record = read_record(root, slot)
        if (token and record["token"] != token) or (not token and owner_alive(record)):
            raise LeaseRefusedError(
                "teardown would interrupt another run's resource lease"
            )


def teardown_slot(
    slot: int, staging: str, root: Path, *, lock_fd: int | None = None
) -> bool:
    report = root / f"slot-{slot}.teardown.json"
    report.unlink(missing_ok=True)
    env = dict(os.environ)
    env["ONEX_PREPR_REAPER_HELD"] = str(slot)
    env["ONEX_LANE_LOCK_HELD"] = lease_slot_policy(slot).compose_project
    done = subprocess.run(
        [
            "bash",
            str(SCRIPT_DIR / "prepr_teardown_slot.sh"),
            "--slot",
            str(slot),
            "--reason",
            "OMN-18896 expired heartbeat and dead owner",
            "--staging-root",
            staging,
            "--lock-timeout",
            "0",
            "--report-out",
            str(report),
        ],
        env=env,
        check=False,
        timeout=600,
        pass_fds=() if lock_fd is None else (lock_fd,),
    )
    try:
        return (
            done.returncode == 0
            and json.loads(report.read_text()).get("verdict") == "CLEAN"
        )
    except (OSError, ValueError):
        return False


def reap_slot(
    root: Path,
    slot: int,
    *,
    teardown: Callable[[int, str], bool] | None = None,
    plan_only: bool = False,
    now: float | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"slot": slot, "verdict": "UNKNOWN"}
    lease_slot_policy(slot)
    try:
        with locked(record_path(root, slot).with_suffix(".guard")):
            if not record_path(root, slot).exists():
                return {**result, "verdict": "ABSENT"}
            record = read_record(root, slot)
            result["holder"] = record["holder"]
            if record["expires"] > (time.time() if now is None else now) or owner_alive(
                record
            ):
                return {**result, "verdict": "HELD"}
            if plan_only:
                return {**result, "verdict": "WOULD_REAP"}
            with locked(lane_lock_path(slot)) as lane_fd:
                clean = (
                    teardown
                    or (lambda s, p: teardown_slot(s, p, root, lock_fd=lane_fd))
                )(slot, record["staging_root"])
                if clean:
                    record_path(root, slot).unlink()
                return {**result, "verdict": "REAPED" if clean else "INCOMPLETE"}
    except LeaseRefusedError as exc:
        return {
            **result,
            "verdict": "BUSY" if "lock is held" in str(exc) else "UNKNOWN",
            "reason": str(exc),
        }
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        # Report only the exception type; subprocess output may carry credentials.
        return {**result, "reason": type(exc).__name__}


RUN_GATE = """import os, sys
fd = int(sys.argv[1])
if os.read(fd, 1) != b"1":
    sys.exit(125)
os.close(fd)
try:
    os.execvp(sys.argv[2], sys.argv[2:])
except OSError as exc:
    print(f"prepr lease run: cannot exec {sys.argv[2]}: {exc}", file=sys.stderr)
    sys.exit(127)
"""


def run_command(args: argparse.Namespace) -> int:
    root, token = state_root(), args.token or uuid.uuid4().hex
    staging = args.staging_root or str(STAGING_PARENT / f"slot-{args.slot}")
    claim(root, args.slot, token, args.holder, staging, ttl=args.ttl)
    env = {**os.environ, "ONEX_PREPR_LEASE_TOKEN": token}
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise LeaseRefusedError("run requires a command after --")
    # The child waits on a pipe until its pid is in the lease, then execs the
    # command (same pid and start time). An owner killed before recording it
    # closes the pipe, and the command never runs untracked by the reaper.
    gate_read, gate_write = os.pipe()
    try:
        child = subprocess.Popen(
            [sys.executable, "-c", RUN_GATE, str(gate_read), *command],
            env=env,
            start_new_session=True,
            pass_fds=(gate_read,),
        )
        os.close(gate_read)
        with locked(record_path(root, args.slot).with_suffix(".guard"), wait=True):
            record = read_record(root, args.slot)
            record.update(child_pid=child.pid, child_start=process_start(child.pid))
            write_record(root, args.slot, record)
        os.write(gate_write, b"1")
    finally:
        os.close(gate_write)

    def forward(signum: int, _frame: Any) -> None:
        if child.poll() is None:
            os.killpg(child.pid, signum)

    previous = {
        sig: signal.signal(sig, forward)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    try:
        while True:
            try:
                code = child.wait(timeout=args.interval)
                return code if code >= 0 else 128 - code
            except subprocess.TimeoutExpired:
                # CLEAN teardown can clear the record just before its shell
                # exits. Leave it absent; never recreate a completed lease.
                if record_path(root, args.slot).exists():
                    heartbeat(root, args.slot, token, ttl=args.ttl)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()


def lease_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=LEASE_DESCRIPTION)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run", help="keep a resource lease while a slot command runs")
    run.add_argument("--slot", type=int, choices=sorted(policy.SLOTS), required=True)
    run.add_argument("--holder", required=True)
    run.add_argument("--token", default="")
    run.add_argument("--staging-root")
    run.add_argument("--ttl", type=float, default=LEASE_DEFAULT_TTL)
    run.add_argument("--interval", type=float, default=LEASE_DEFAULT_INTERVAL)
    run.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("check", "clear", "authorize-teardown"):
        cmd = sub.add_parser(name)
        cmd.add_argument(
            "--slot", type=int, choices=sorted(policy.SLOTS), required=True
        )
    reap = sub.add_parser(
        "reap", help="inspect every slot; delete only expired, dead-owner slots"
    )
    reap.add_argument("--plan-only", action="store_true")
    reap.add_argument("--report-out", type=Path)
    install = sub.add_parser(
        "install-reaper", help="snapshot the reaper and install its systemd user timer"
    )
    install.add_argument(
        "--prefix", type=Path, default=Path.home() / ".local/share/onex-prepr-reaper"
    )
    install.add_argument(
        "--unit-dir", type=Path, default=Path.home() / ".config/systemd/user"
    )
    install.add_argument("--no-enable", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "install-reaper":
        sys.stdout.write(
            json.dumps(
                install_reaper(
                    args.prefix.resolve(),
                    args.unit_dir.resolve(),
                    enable=not args.no_enable,
                ),
                sort_keys=True,
            )
            + "\n"
        )
        return 0
    try:
        if args.action == "run":
            if not 0 < args.interval < args.ttl or not math.isfinite(args.ttl):
                parser.error("require 0 < interval < finite ttl")
            return run_command(args)
        if args.action in ("check", "clear", "authorize-teardown"):
            token = os.environ.get("ONEX_PREPR_LEASE_TOKEN", "")
            {
                "check": check_owner,
                "clear": clear,
                "authorize-teardown": authorize_teardown,
            }[args.action](state_root(), args.slot, token)
            return 0
        results = [
            reap_slot(state_root(), slot, plan_only=args.plan_only)
            for slot in sorted(policy.SLOTS)
        ]
        report = (
            json.dumps(
                {
                    "schema": "onex.prepr.slot-reap.v1",
                    "observed_at": time.time(),
                    "slots": results,
                },
                sort_keys=True,
            )
            + "\n"
        )
        report_path = args.report_out or state_root() / "last-reap.json"
        temporary = report_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(report)
        temporary.replace(report_path)
        sys.stdout.write(report)
        return int(any(r["verdict"] in ("UNKNOWN", "INCOMPLETE") for r in results))
    except (LeaseRefusedError, OSError, ValueError) as exc:
        print(
            f"prepr resource lease refused: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 8


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] in (
        "run",
        "check",
        "clear",
        "authorize-teardown",
        "reap",
        "install-reaper",
    ):
        return lease_main(argv)
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
