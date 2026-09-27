# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Route each dev rebuild command to exactly one deploy-agent instance (OMN-19506).

WHY THIS EXISTS
---------------
Task B3 of the second-deploy-slot plan (epic OMN-19500) runs a second
deploy-agent instance for the dev lane, on the .202 host (lane ``dev-202``),
beside the one on .201. Both read the one control topic. The wire field
``runtime_lane`` stays ``dev``: a new value would touch five repositories and two
releases. Instead a declared table, ``config/deploy_lane_routing.yaml``, maps a
dev command's requesting repository to one instance and names a default.

WHAT THE MODEL FORCED (OMN-19506 AC4, TLC logs on the ticket)
-------------------------------------------------------------
``DeployRouting.tla`` checks "no command accepted by two instances, none by
zero" and three mutants each break it:

* **One consumer group for both instances** delivers each record to ONE of
  them, so a command routed to the other is skipped by the only reader it
  reaches: zero acceptors. So each instance declares its own group.
* **Routing with the table the instance's own code loaded** breaks the moment
  the two hosts self-update at different times: one reads the new table, the
  other the old, and both accept. So the route is read from the table AT THE
  COMMAND'S ``git_ref``, which makes it a pure function of the command. Every CI
  command carries an omnibase_infra commit sha there, omnimarket-requested ones
  included.
* **Publishing a rejection for a command routed elsewhere** races the other
  instance's acceptance at the redeploy effect. So the skip publishes nothing.

A ref that is not a sha (an operator's ``origin/dev``), or a sha that predates
the table, takes the DEFAULT instance. The default is pinned by a test to
``dev-201``, so every version of the table agrees on it, which keeps that case a
pure function of the command too.

RESIDUAL, STATED
----------------
A sha the clone cannot resolve even after a fetch takes the default, with a
warning line. Refusing would stall the one .201 agent on every such command
today; the cost, once a second instance is live, is that an instance that can
read the table and one that cannot may disagree. Both fetch the same origin, so
this needs a fetch failure on one host at the moment of the command.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

from deploy_agent.events import EnumRuntimeLane, ModelRebuildRequested

logger = logging.getLogger(__name__)

#: Repository-relative path of the table, read at a command's ref.
ROUTING_TABLE_RELPATH: Final = "config/deploy_lane_routing.yaml"

#: Names the instance this process is. Optional: without it the instance is
#: resolved from the host name through the table's ``hostnames``.
ENV_INSTANCE: Final = "DEPLOY_AGENT_INSTANCE"

#: The lanes the table routes. Only the dev lane has two instances; a command
#: for any other lane is decided by the lane fence alone, as before.
ROUTED_LANES: Final = frozenset({EnumRuntimeLane.DEV})

#: The default every version of the table must name (pinned by a test).
PINNED_DEFAULT_INSTANCE: Final = "dev-201"

#: ``requested_by`` of a command the CI rebuild trigger published.
_CI_REQUESTER_RE: Final = re.compile(r"^gha/([a-z][a-z0-9_]*)/")
_SHA_RE: Final = re.compile(r"^[0-9a-f]{40}$")

GIT_TIMEOUT_SECONDS: Final = 30


class RoutingTableError(RuntimeError):
    """The routing table is missing, unreadable or inconsistent."""


class ModelAgentInstance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    hostnames: tuple[str, ...]
    consumer_group: str


class ModelRoute(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime_lane: EnumRuntimeLane
    requester_repository: str
    instance: str


class ModelRoutingTable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default_instance: str
    instances: dict[str, ModelAgentInstance]
    routes: tuple[ModelRoute, ...]
    #: OMN-19507. Hosts no instance may name and no agent may run as, each with
    #: the reason it is out of the deploy pool.
    excluded_hosts: dict[str, str] = Field(default_factory=dict)

    def route(self, runtime_lane: EnumRuntimeLane, requested_by: str) -> str:
        """The instance that runs a command, from this table alone."""
        repository = requester_repository(requested_by)
        for route in self.routes:
            if (
                route.runtime_lane is runtime_lane
                and route.requester_repository == repository
            ):
                return route.instance
        return self.default_instance


def requester_repository(requested_by: str) -> str | None:
    """``gha/<repo>/...`` -> ``<repo>``; any other requester -> ``None``."""
    match = _CI_REQUESTER_RE.match(requested_by)
    return match.group(1) if match else None


def parse_routing_table(text: str) -> ModelRoutingTable:
    """Parse and validate the table. Every inconsistency refuses, none defaults."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise RoutingTableError(f"routing table is not YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise RoutingTableError("routing table must be a mapping")

    raw_instances = raw.get("instances")
    if not isinstance(raw_instances, dict) or not raw_instances:
        raise RoutingTableError("routing table declares no instances")
    excluded = _parse_excluded_hosts(raw.get("excluded_hosts"))
    instances: dict[str, ModelAgentInstance] = {}
    groups: set[str] = set()
    hostnames: set[str] = set()
    for name, spec in raw_instances.items():
        if not isinstance(spec, dict):
            raise RoutingTableError(f"instance {name!r} must be a mapping")
        group = str(spec.get("consumer_group") or "").strip()
        if not group:
            raise RoutingTableError(f"instance {name!r} declares no consumer_group")
        if group in groups:
            raise RoutingTableError(
                f"consumer group {group!r} is declared by two instances; each "
                "instance must read every record, so each needs its own group"
            )
        groups.add(group)
        names = tuple(str(h).strip().lower() for h in spec.get("hostnames") or ())
        for host in names:
            if host in excluded:
                raise RoutingTableError(
                    f"instance {name!r} names host {host!r}, which is excluded from "
                    f"the deploy pool: {excluded[host]}"
                )
            if host in hostnames:
                raise RoutingTableError(f"hostname {host!r} names two instances")
            hostnames.add(host)
        instances[str(name)] = ModelAgentInstance(
            name=str(name), hostnames=names, consumer_group=group
        )

    default = str(raw.get("default_instance") or "").strip()
    if not default:
        raise RoutingTableError(
            "routing table declares no default_instance; a command whose "
            "requester matches no route would have no instance, so the agent "
            "refuses to start"
        )
    if default not in instances:
        raise RoutingTableError(f"default_instance {default!r} is not an instance")

    routes: list[ModelRoute] = []
    seen: set[tuple[EnumRuntimeLane, str]] = set()
    for entry in raw.get("routes") or ():
        if not isinstance(entry, dict):
            raise RoutingTableError(f"route {entry!r} must be a mapping")
        try:
            lane = EnumRuntimeLane(str(entry.get("runtime_lane")))
        except ValueError as exc:
            raise RoutingTableError(f"route {entry!r} names an unknown lane") from exc
        repository = str(entry.get("requester_repository") or "").strip()
        instance = str(entry.get("instance") or "").strip()
        if not repository:
            raise RoutingTableError(f"route {entry!r} names no requester_repository")
        if instance not in instances:
            raise RoutingTableError(
                f"route {entry!r} names instance {instance!r}, which is not declared"
            )
        key = (lane, repository)
        if key in seen:
            raise RoutingTableError(f"two routes for {lane.value}/{repository}")
        seen.add(key)
        routes.append(
            ModelRoute(
                runtime_lane=lane, requester_repository=repository, instance=instance
            )
        )

    return ModelRoutingTable(
        default_instance=default,
        instances=instances,
        routes=tuple(routes),
        excluded_hosts=excluded,
    )


def _parse_excluded_hosts(raw: object) -> dict[str, str]:
    """``excluded_hosts:`` -- lowercased first hostname label -> reason.

    OMN-19507. A host is out of the deploy pool by a ruling, and the reason is
    required so the table says which one. Absent means none.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RoutingTableError("excluded_hosts must map a hostname to its reason")
    excluded: dict[str, str] = {}
    for host, reason in raw.items():
        name = str(host).strip().lower()
        why = str(reason or "").strip()
        if not name or not why:
            raise RoutingTableError(
                f"excluded host {host!r} needs a hostname and a reason"
            )
        excluded[name] = why
    return excluded


#: The agent's own clone: this file is ``<repo>/scripts/deploy-agent/deploy_agent/``.
#: Self-update keeps it at the tracking ref together with the code, which the
#: deploy-source clone (``REPO_DIR``) is not: that one sits at whatever ref the
#: last job built, which can predate this table.
AGENT_CLONE_ROOT: Final = Path(__file__).resolve().parents[3]


def load_routing_table(repo_root: str | Path | None = None) -> ModelRoutingTable:
    """The table shipped with this process's code. Used for identity, the
    consumer group and the default, never for a command's route."""
    path = Path(repo_root if repo_root is not None else AGENT_CLONE_ROOT) / (
        ROUTING_TABLE_RELPATH
    )
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RoutingTableError(f"routing table unreadable at {path}: {exc}") from exc
    return parse_routing_table(text)


def resolve_instance(
    table: ModelRoutingTable,
    *,
    env: dict[str, str] | None = None,
    hostname: str | None = None,
) -> ModelAgentInstance:
    """Which instance this process is: ``DEPLOY_AGENT_INSTANCE``, else the host.

    Refuses rather than guessing. An agent that does not know which instance it
    is cannot tell its own commands from the other's.
    """
    source = os.environ if env is None else env
    host = (hostname or socket.gethostname()).strip().lower().split(".")[0]
    # OMN-19507: checked before the instance name, so an env file copied onto an
    # excluded host cannot name its way into the pool.
    if host in table.excluded_hosts:
        raise RoutingTableError(
            f"host {host!r} is excluded from the deploy pool: "
            f"{table.excluded_hosts[host]}"
        )
    named = source.get(ENV_INSTANCE, "").strip()
    if named:
        if named not in table.instances:
            raise RoutingTableError(
                f"{ENV_INSTANCE}={named!r} names no instance in the routing table "
                f"(declared: {', '.join(sorted(table.instances))})"
            )
        return table.instances[named]
    for instance in table.instances.values():
        if host in instance.hostnames:
            return instance
    raise RoutingTableError(
        f"host {host!r} matches no instance's hostnames in the routing table and "
        f"{ENV_INSTANCE} is unset; set it on the unit"
    )


TableAtRef = Callable[[str], "ModelRoutingTable | None"]


class GitTableAtRef:
    """Reads the table as committed at a sha in the deploy clone.

    Returns ``None`` when the sha predates the table (the file does not exist
    there). Raises ``RoutingTableError`` when the sha cannot be read even after
    one fetch, which the caller turns into the default with a warning.
    """

    def __init__(
        self,
        repo_dir: str,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.repo_dir = repo_dir
        self._run = run or self._default_run

    @staticmethod
    def _default_run(argv: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            argv, capture_output=True, text=True, check=False, timeout=timeout
        )

    def _has_commit(self, sha: str) -> bool:
        probe = self._run(
            ["git", "-C", self.repo_dir, "cat-file", "-e", f"{sha}^{{commit}}"],
            timeout=GIT_TIMEOUT_SECONDS,
        )
        return probe.returncode == 0

    def __call__(self, sha: str) -> ModelRoutingTable | None:
        if not self._has_commit(sha):
            self._run(
                ["git", "-C", self.repo_dir, "fetch", "--quiet", "--no-tags", "origin"],
                timeout=GIT_TIMEOUT_SECONDS,
            )
            if not self._has_commit(sha):
                raise RoutingTableError(f"commit {sha} is not in {self.repo_dir}")
        shown = self._run(
            ["git", "-C", self.repo_dir, "show", f"{sha}:{ROUTING_TABLE_RELPATH}"],
            timeout=GIT_TIMEOUT_SECONDS,
        )
        if shown.returncode != 0:
            # The commit exists, so a failed show means the file is absent there.
            return None
        return parse_routing_table(shown.stdout)


class ModelRoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instance: str
    #: Where the route came from, for the journal line.
    basis: str


class DeployRouter:
    """This process's routing: who it is, which group it reads with, and whether
    a command is its own."""

    def __init__(
        self,
        table: ModelRoutingTable,
        instance: ModelAgentInstance,
        table_at_ref: TableAtRef,
    ) -> None:
        if table.default_instance != PINNED_DEFAULT_INSTANCE:
            raise RoutingTableError(
                f"default_instance must be {PINNED_DEFAULT_INSTANCE!r}: a ref that "
                "names no table routes to the default, and only a default every "
                "table version agrees on keeps that a function of the command"
            )
        self.table = table
        self.instance = instance
        self._table_at_ref = table_at_ref

    @property
    def consumer_group(self) -> str:
        return self.instance.consumer_group

    def decide(self, cmd: ModelRebuildRequested) -> ModelRoutingDecision:
        default = self.table.default_instance
        ref = cmd.git_ref or ""
        if not _SHA_RE.match(ref):
            return ModelRoutingDecision(
                instance=default, basis=f"default (ref {ref!r} is not a sha)"
            )
        try:
            at_ref = self._table_at_ref(ref)
        except RoutingTableError as exc:
            logger.warning(
                "routing: table at %s unreadable (%s); %s takes the default %s "
                "friction_type=routing_table_unreadable_at_ref",
                ref[:12],
                exc,
                cmd.correlation_id,
                default,
            )
            return ModelRoutingDecision(
                instance=default, basis="default (table unreadable at ref)"
            )
        if at_ref is None:
            return ModelRoutingDecision(
                instance=default, basis=f"default (no table at {ref[:12]})"
            )
        return ModelRoutingDecision(
            instance=at_ref.route(cmd.runtime_lane, cmd.requested_by),
            basis=f"table at {ref[:12]}",
        )

    def is_mine(self, cmd: ModelRebuildRequested) -> tuple[bool, ModelRoutingDecision]:
        decision = self.decide(cmd)
        return decision.instance == self.instance.name, decision


def build_router_from_env(deploy_source_dir: str) -> DeployRouter:
    """The agent's router, or ``RoutingTableError`` (the agent refuses to start).

    Identity and group come from the table shipped with this code. A command's
    route is read at its own ref in the deploy-source clone, where every
    omnibase_infra commit a CI command names is (or is one fetch away).
    """
    table = load_routing_table()
    instance = resolve_instance(table)
    return DeployRouter(table, instance, GitTableAtRef(deploy_source_dir))
