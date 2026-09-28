#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Run a runtime pull request on any free lab host before it merges (OMN-18893).

Why this exists
---------------
Operator ruling 2026-09-27: runtime PRs deploy to any free runtime lane on any
lab machine, are tested, and merge when everything clears. Until this file every
pre-merge runtime proof either queued on the one ``.201`` dev lane under a
ledger lease or was run by hand by a ``prove-<host>`` lane that already knew
which host was free. Nothing read the whole pool and picked.

The pool has two kinds of member. An ``isolated`` member is a lab host that
offers ONE isolated slot (the laptop-bundle compose project
``omnibase-infra-local``, with its own Postgres, Redpanda and Valkey) on the
ports the declared ``.201`` lanes already hold. A ``prepr-slot`` member is one
of the two numbered pre-PR slots on ``.201`` (``omnibase-infra-prepr-1``/``-2``),
brought up through ``prepr_verify_lane.sh`` (the one entrypoint rule 24(e)
sanctions, which refuses every declared lane by name) and destroyed through
``prepr_teardown_slot.sh``. Its project and ports are read from
``prepr_slot_policy.py``, never from the pool config.

What one ``run`` does
---------------------
pick a free host -> take the host lease -> snapshot -> clone (a local test
merge of the PR head into dev, never pushed) -> build and boot -> probe (image
identity, migration gate, main and effects health, wiring failures, the PR's
own live readbacks) -> the PR's focused tests -> teardown -> zero-residue
readback against the snapshot -> release the lease -> one PASS/FAIL readback a
PR body can cite.

Teardown and release run whatever happened before them. A slot is never a
declared lane, so there is no lane image to put back: restoration IS the
teardown, and ``restored=yes`` is printed only when every residue count reads
zero beside a positive control that reads non-zero on the same host.

The lease
---------
The unit of leasing is the host, because the slot's project name and ports are
fixed. The lease is a directory on the host (``mkdir`` is atomic on macOS and
Linux) holding ``lease.json`` with the holder and an absolute expiry. An
expired lease is broken; a live one held by someone else is refused. The ledger
is read, never written: a live ``HOLD`` naming the host's surface, placed by
another lane, makes the host busy. The HOLD and RELEASE rows themselves are
the calling lane's to write through ``/omni:ledger-msg``; this file prints the
text for them.

Exit status: 0 PASS, 1 FAIL, 2 INCONCLUSIVE (the stack never became
provable for a reason outside the PR), 3 no free host, 4 lease refused,
5 usage or configuration error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Protocol

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
POOL_CONFIG = REPO_ROOT / "config" / "prepr_runtime_pool.yaml"
LANE_MANIFEST = REPO_ROOT / "deploy" / "lane-census" / "lane-manifest.yaml"
PROVE_SH = SCRIPT_DIR / "prepr_pool_prove.sh"
SLOT_POLICY_PY = SCRIPT_DIR / "prepr_slot_policy.py"

KINDS = ("isolated", "prepr-slot")
DEFAULT_POSITIVE_CONTROL = "omnibase-infra-dogfood"

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INCONCLUSIVE = 2
EXIT_NO_FREE_HOST = 3
EXIT_LEASE_REFUSED = 4
EXIT_USAGE = 5

PHASES = ("snap-pre", "clone", "build", "probe", "tests", "teardown", "snap-post")

# Prepended to every remote command: a non-login ssh shell on a Docker Desktop
# Mac has no docker on PATH, and a bare `docker ps` there reads zero containers
# and is false (STATUS ledger:4019).
REMOTE_PATH = (
    "export PATH=/Applications/Docker.app/Contents/Resources/bin:/usr/local/bin:"
    "$HOME/.local/bin:/opt/homebrew/bin:$PATH; "
)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class PoolHost:
    name: str
    ssh_target: str
    surface: str
    os: str
    status: str
    reason: str = ""
    # "isolated": build with a private DOCKER_CONFIG that has no credential
    # store, for a host whose keychain cannot unlock over ssh
    docker_config: str = "host"
    # "isolated" (the laptop-bundle slot) or "prepr-slot" (a numbered .201 slot)
    kind: str = "isolated"
    slot: int = 0
    # the lane-manifest host this member runs on
    machine: str = ""
    # the compose project the teardown counts as its non-zero control, read only
    positive_control: str = DEFAULT_POSITIVE_CONTROL
    # derived at load: the project, ports and lease this member owns
    compose_project: str = ""
    ports: tuple[int, ...] = ()
    main_port: int = 8085
    effects_port: int = 8086
    lease_dir: str = ""


@dataclass(frozen=True)
class PoolConfig:
    compose_project: str
    ports: tuple[int, ...]
    lease_dir: str
    work_root_prefix: str
    model_endpoint: str
    max_load_ratio: float
    hosts: tuple[PoolHost, ...]

    def host(self, name: str) -> PoolHost:
        for h in self.hosts:
            if name in (h.name, h.ssh_target):
                return h
        raise KeyError(f"host {name!r} is not in the pool config")


def _slot_policy() -> ModuleType:
    """prepr_slot_policy.py, the one table a .201 slot number becomes ports in."""
    spec = importlib.util.spec_from_file_location("prepr_slot_policy", SLOT_POLICY_PY)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {SLOT_POLICY_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pool_config(path: Path = POOL_CONFIG) -> PoolConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version")
    slot = raw["slot"]
    hosts = []
    policy: ModuleType | None = None
    for h in raw["hosts"]:
        status = h["status"]
        if status not in ("pool", "excluded"):
            raise ValueError(f"{path}: host {h['name']} has status {status!r}")
        if str(h.get("docker_config", "host")) not in ("host", "isolated"):
            raise ValueError(
                f"{path}: host {h['name']} docker_config must be host or isolated"
            )
        if status == "excluded" and not h.get("reason"):
            raise ValueError(f"{path}: excluded host {h['name']} states no reason")
        kind = str(h.get("kind", "isolated"))
        if kind not in KINDS:
            raise ValueError(f"{path}: host {h['name']} has kind {kind!r}")
        slot_no = int(h.get("slot", 0))
        if kind == "prepr-slot":
            # The project and ports come from the slot policy, which refuses a
            # slot number outside the pool; nothing in the pool file can name them.
            policy = policy or _slot_policy()
            try:
                sp = policy.resolve_slot(slot_no)
            except policy.RefusalError as exc:
                raise ValueError(
                    f"{path}: host {h['name']} names slot {slot_no}, not a pre-PR slot"
                ) from exc
            project = str(sp.compose_project)
            ports: tuple[int, ...] = (
                int(sp.runtime_main_port),
                int(sp.runtime_effects_port),
                int(sp.gateway_port),
                int(sp.projection_api_port),
            )
            main_port, effects_port = ports[0], ports[1]
            lease_dir = f"{slot['lease_dir']}-prepr-{slot_no}"
        else:
            if slot_no:
                raise ValueError(
                    f"{path}: host {h['name']} names a slot but is not a prepr-slot"
                )
            project = str(slot["compose_project"])
            ports = tuple(int(p) for p in slot["ports"])
            main_port, effects_port = 8085, 8086
            lease_dir = str(slot["lease_dir"])
        hosts.append(
            PoolHost(
                name=h["name"],
                ssh_target=str(h["ssh_target"]),
                surface=h["surface"],
                os=h["os"],
                status=status,
                reason=str(h.get("reason", "")).strip(),
                docker_config=str(h.get("docker_config", "host")),
                kind=kind,
                slot=slot_no,
                machine=str(h.get("machine", h["name"])),
                positive_control=str(
                    h.get("positive_control", DEFAULT_POSITIVE_CONTROL)
                ),
                compose_project=project,
                ports=ports,
                main_port=main_port,
                effects_port=effects_port,
                lease_dir=lease_dir,
            )
        )
    surfaces = [h.surface for h in hosts]
    if len(set(surfaces)) != len(surfaces):
        raise ValueError(f"{path}: two pool members share a ledger surface")
    return PoolConfig(
        compose_project=slot["compose_project"],
        ports=tuple(int(p) for p in slot["ports"]),
        lease_dir=slot["lease_dir"],
        work_root_prefix=slot["work_root_prefix"],
        model_endpoint=slot["model_endpoint"],
        max_load_ratio=float(slot["max_load_ratio"]),
        hosts=tuple(hosts),
    )


def declared_manifest_hosts(path: Path = LANE_MANIFEST) -> set[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return set(raw.get("hosts", {}))


def declared_lane_projects(path: Path = LANE_MANIFEST) -> set[str]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {
        str(lane["compose_project"])
        for lane in raw.get("lanes", {}).values()
        if lane.get("compose_project")
    }


# --------------------------------------------------------------------------- transport


class Transport(Protocol):
    """How the driver reaches a host. The real one is ssh; tests pass a fake."""

    def run(self, host: PoolHost, command: str, timeout: float) -> tuple[int, str]: ...

    def put(self, host: PoolHost, local: Path, remote: str) -> int: ...


class SshTransport:
    def __init__(self, connect_timeout: int = 10) -> None:
        self._opts = [
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={connect_timeout}",
        ]

    def run(self, host: PoolHost, command: str, timeout: float) -> tuple[int, str]:
        try:
            proc = subprocess.run(
                ["ssh", *self._opts, host.ssh_target, REMOTE_PATH + command],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            out = exc.stdout if isinstance(exc.stdout, str) else ""
            return 124, out + f"\n[timeout after {timeout:.0f}s]"
        return proc.returncode, proc.stdout + proc.stderr

    def put(self, host: PoolHost, local: Path, remote: str) -> int:
        return subprocess.run(
            ["scp", "-q", *self._opts, str(local), f"{host.ssh_target}:{remote}"],
            check=False,
        ).returncode


# --------------------------------------------------------------------------- ledger


_ROW = re.compile(r"^(\S+Z) \| (HOLD|RELEASE) \|")


def _field(row: str, key: str) -> str | None:
    m = re.search(rf"(?:^|\| |\s){key}=([^\s|]+)", row)
    return m.group(1) if m else None


def _parse_ts(value: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def live_surface_holds(
    ledger_lines: Sequence[str], now: dt.datetime
) -> dict[str, tuple[str, str, str]]:
    """Map surface -> (hold id, lane, until) for every HOLD still in force.

    A HOLD is in force until its ``until=`` passes or a RELEASE cites its id
    (``re=<id>``). Rows without ``surface=`` are not host leases and are ignored.
    """
    holds: dict[str, tuple[str, str, str, dt.datetime]] = {}
    released: set[str] = set()
    for line in ledger_lines:
        m = _ROW.match(line)
        if not m:
            continue
        if m.group(2) == "RELEASE":
            ref = _field(line, "re")
            if ref:
                released.add(ref)
            continue
        surface = _field(line, "surface")
        until = _field(line, "until")
        if not surface or not until:
            continue
        until_ts = _parse_ts(until)
        if until_ts is None:
            continue
        hold_id = _field(line, "id") or f"{m.group(1)}-{_field(line, 'lane')}"
        holds[hold_id] = (surface, _field(line, "lane") or "?", until, until_ts)
    live: dict[str, tuple[str, str, str]] = {}
    for hold_id, (surface, lane, until, until_ts) in holds.items():
        if hold_id in released or until_ts <= now:
            continue
        live[surface] = (hold_id, lane, until)
    return live


# --------------------------------------------------------------------------- probe


@dataclass
class HostState:
    host: PoolHost
    verdict: str  # FREE BUSY OFFLINE OVERLOADED EXCLUDED
    detail: str = ""
    cores: int = 0
    load1: float = 0.0
    slot_containers: int = 0
    slot_listeners: int = 0
    other_projects: dict[str, int] = field(default_factory=dict)
    lease: dict[str, str] | None = None

    @property
    def load_ratio(self) -> float:
        return self.load1 / self.cores if self.cores else float("inf")


def _probe_command(cfg: PoolConfig, host: PoolHost) -> str:
    ports = "|".join(str(p) for p in host.ports)
    return (
        "echo CORES=$(getconf _NPROCESSORS_ONLN); "
        "echo LOAD=$( (cut -d' ' -f1 /proc/loadavg 2>/dev/null) || "
        "(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1}') ); "
        f"echo SLOT=$(docker ps -a --filter label=com.docker.compose.project={host.compose_project} -q | wc -l); "
        "echo LISTEN=$( (ss -ltn 2>/dev/null || lsof -nP -iTCP -sTCP:LISTEN) "
        f"| grep -cE ':({ports})\\b'); "
        "echo PROJECTS=$(docker ps --format '{{.Label \"com.docker.compose.project\"}}' "
        "| sort | uniq -c | awk '{printf \"%s:%s,\", $2, $1}'); "
        f"echo LEASE=$(cat $HOME/{host.lease_dir}/lease.json 2>/dev/null | tr -d '\\n')"
    )


def parse_probe(host: PoolHost, rc: int, out: str) -> HostState:
    if rc != 0 or "CORES=" not in out:
        return HostState(
            host,
            "OFFLINE",
            detail=out.strip().splitlines()[-1:][0] if out.strip() else f"rc={rc}",
        )
    kv = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            kv[k.strip()] = v.strip()
    state = HostState(host, "FREE")
    try:
        state.cores = int(kv.get("CORES", "0"))
        state.load1 = float(kv.get("LOAD", "inf") or "inf")
        state.slot_containers = int(kv.get("SLOT", "0").split()[0] or 0)
        state.slot_listeners = int(kv.get("LISTEN", "0").split()[0] or 0)
    except ValueError:
        return HostState(host, "OFFLINE", detail=f"unparseable probe: {out[:200]!r}")
    for item in kv.get("PROJECTS", "").split(","):
        if ":" in item:
            name, _, count = item.partition(":")
            state.other_projects[name or "(none)"] = int(count)
    if kv.get("LEASE"):
        try:
            state.lease = json.loads(kv["LEASE"])
        except json.JSONDecodeError:
            state.lease = {"holder": "?", "until": "unparseable"}
    return state


def assess(
    states: Sequence[HostState],
    cfg: PoolConfig,
    holds: Mapping[str, tuple[str, str, str]],
    now: dt.datetime,
    me: str | None = None,
) -> list[HostState]:
    """Turn raw probes into FREE / BUSY / OVERLOADED / OFFLINE / EXCLUDED."""
    for s in states:
        if s.host.status == "excluded":
            s.verdict, s.detail = "EXCLUDED", s.host.reason
            continue
        if s.verdict == "OFFLINE":
            continue
        reasons = []
        if s.lease:
            until = _parse_ts(s.lease.get("until", ""))
            if until is not None and until > now and s.lease.get("holder") != me:
                reasons.append(
                    f"lease held by {s.lease.get('holder')} until {s.lease.get('until')}"
                )
        hold = holds.get(s.host.surface)
        if hold and hold[1] != me:
            reasons.append(f"ledger HOLD {hold[0]} by {hold[1]} until {hold[2]}")
        if s.slot_containers:
            reasons.append(
                f"{s.slot_containers} {s.host.compose_project} containers present"
            )
        if s.slot_listeners:
            reasons.append(f"{s.slot_listeners} listeners on the slot ports")
        if reasons:
            s.verdict, s.detail = "BUSY", "; ".join(reasons)
        elif s.load_ratio > cfg.max_load_ratio:
            s.verdict = "OVERLOADED"
            s.detail = f"load {s.load1:.1f} on {s.cores} cores (ratio {s.load_ratio:.2f} > {cfg.max_load_ratio})"
        else:
            s.verdict, s.detail = "FREE", f"load {s.load1:.1f} on {s.cores} cores"
    return list(states)


def survey(
    cfg: PoolConfig,
    transport: Transport,
    holds: Mapping[str, tuple[str, str, str]],
    now: dt.datetime,
    me: str | None = None,
) -> list[HostState]:
    states = []
    for host in cfg.hosts:
        if host.status == "excluded":
            states.append(HostState(host, "EXCLUDED"))
            continue
        rc, out = transport.run(host, _probe_command(cfg, host), timeout=30)
        states.append(parse_probe(host, rc, out))
    return assess(states, cfg, holds, now, me)


def pick(states: Sequence[HostState]) -> HostState | None:
    """The least-loaded free isolated member, else the least-loaded free slot.

    An isolated member owns its whole stack; a .201 pre-PR slot shares the dev
    lane's servers and the busiest lab host, so it is taken only when every
    isolated member is busy, offline or overloaded.
    """
    free = [s for s in states if s.verdict == "FREE"]
    return (
        min(free, key=lambda s: (s.host.kind == "prepr-slot", s.load_ratio))
        if free
        else None
    )


# --------------------------------------------------------------------------- lease


def acquire_lease(
    cfg: PoolConfig,
    transport: Transport,
    host: PoolHost,
    holder: str,
    until: dt.datetime,
    now: dt.datetime,
) -> tuple[bool, str]:
    """Take the host lease atomically; break it only when it has expired."""
    lease = f"$HOME/{host.lease_dir}"
    body = json.dumps(
        {
            "holder": holder,
            "until": until.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "taken": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    cmd = (
        f"mkdir -p $(dirname {lease}) && if mkdir {lease} 2>/dev/null; then "
        f"printf '%s' {shlex.quote(body)} > {lease}/lease.json && echo ACQUIRED; "
        f"else echo HELD; cat {lease}/lease.json 2>/dev/null; fi"
    )
    rc, out = transport.run(host, cmd, timeout=30)
    if rc == 0 and "ACQUIRED" in out:
        return True, "acquired"
    if "HELD" not in out:
        return False, f"lease command failed rc={rc}: {out.strip()[:200]}"
    held_raw = out.split("HELD", 1)[1].strip()
    try:
        held = json.loads(held_raw) if held_raw else {}
    except json.JSONDecodeError:
        held = {}
    held_until = _parse_ts(str(held.get("until", "")))
    if held.get("holder") == holder:
        return True, "already held by this holder"
    if held_until is not None and held_until <= now:
        # Expired: break it and try once more, still atomically.
        transport.run(host, f"rm -rf {lease}", timeout=30)
        rc, out = transport.run(host, cmd, timeout=30)
        if rc == 0 and "ACQUIRED" in out:
            return (
                True,
                f"acquired after breaking the lease {held.get('holder')} let expire at {held.get('until')}",
            )
    return False, f"held by {held.get('holder', '?')} until {held.get('until', '?')}"


def release_lease(
    cfg: PoolConfig, transport: Transport, host: PoolHost, holder: str
) -> tuple[bool, str]:
    lease = f"$HOME/{host.lease_dir}"
    cmd = (
        f"H=$(cat {lease}/lease.json 2>/dev/null); "
        f'case "$H" in *\'"holder": "{holder}"\'*) rm -rf {lease} && echo RELEASED;; '
        "'') echo ABSENT;; *) echo FOREIGN;; esac"
    )
    rc, out = transport.run(host, cmd, timeout=30)
    if "RELEASED" in out:
        return True, "released"
    if "ABSENT" in out:
        return True, "no lease was present"
    return False, f"lease is not this holder's (rc={rc}): {out.strip()[:200]}"


# --------------------------------------------------------------------------- verdict


@dataclass
class Readback:
    checks: dict[str, bool]
    notes: list[str]
    restored: bool
    residue: str

    @property
    def outcome(self) -> str:
        if not self.checks.get("stack_built", False):
            return "INCONCLUSIVE"
        return "PASS" if all(self.checks.values()) else "FAIL"


def failed_contracts(probe: str) -> dict[str, set[str]]:
    """Per container, the contracts the non-strict wiring pass gave up on."""
    found: dict[str, set[str]] = {}
    for container, names in re.findall(r"^failed-contracts (\S+): ?(.*)$", probe, re.M):
        found[container] = set(names.split())
    return found


def failure_signatures(probe: str) -> dict[tuple[str, str], str]:
    """(container, contract) -> the error class and message, handler name removed.

    Two contracts that fail for the same reason (say, a projection handler with
    no configured DSN on the laptop bundle) carry the same signature.
    """
    sigs: dict[tuple[str, str], str] = {}
    for container, name, reason in re.findall(
        r"^\s*failed-contract-reason (\S+) (\S+): (.*)$", probe, re.M
    ):
        body = re.sub(r"^handler=\S+:\s*", "", reason)
        sigs[(container, name)] = ": ".join(body.split(": ")[:2])
    return sigs


def judge(
    outputs: Mapping[str, str],
    base_probe: str | None = None,
    ports: tuple[int, int] = (8085, 8086),
) -> Readback:
    """Decide the run from the phase outputs of prepr_pool_prove.sh.

    Pure: every PASS/FAIL line a PR body cites is derived here from text, so a
    test can hand it the text a broken stack prints and see the FAIL.

    ``base_probe`` is the probe output of the same bundle built at dev on the
    same host (a base control). With it, a contract that fails to wire at dev
    too is dev-inherited and not held against the PR; only a contract that
    fails at the head and wires at the base is. Without it, any failed contract
    is the PR's (interim recipes, common frame 8).

    ``ports`` are the member's runtime main and effects ports: 8085/8086 on an
    isolated slot, the slot policy's ports on a .201 pre-PR slot.
    """
    build = outputs.get("build", "")
    probe = outputs.get("probe", "")
    tests = outputs.get("tests", "")
    teardown = outputs.get("teardown", "")
    clone = outputs.get("clone", "")
    checks: dict[str, bool] = {}
    notes: list[str] = []

    checks["head_matches"] = "match=NO" not in clone and "match=yes" in clone
    # A pre-PR slot migrates fresh databases before it starts anything. A
    # migration that fails there built fine and is the PR's finding (FAIL), not
    # an unprovable stack (prepr_verify_lane.sh step 10).
    migration_failed = bool(re.search(r"^slot-migration FAILED", build, re.M))
    checks["stack_built"] = bool(re.search(r"build rc=0\b", build)) and (
        bool(re.search(r"up rc=0\b", build)) or migration_failed
    )
    ident = re.findall(
        r"^file (\S+) image=(\S+) build-tree=(\S+) match=(\w+)", probe, re.M
    )
    checks["image_identity"] = bool(ident) and all(m[3] == "yes" for m in ident)
    for port in (str(ports[0]), str(ports[1])):
        m = re.search(
            rf"^port {port} HTTP (\d+) status (\S+) healthy (\S+)", probe, re.M
        )
        # A runtime that serves 200 with healthy True is up and consuming; a
        # "degraded" status names projections that persist nothing, which the
        # wiring check below attributes to the PR or to dev.
        ok = m is not None and m.group(1) == "200" and m.group(3) == "True"
        checks[f"health_{port}"] = ok
        if not ok:
            notes.append(f"port {port}: {m.group(0) if m else 'no health line'}")
        elif m is not None and m.group(2) != "healthy":
            notes.append(f"port {port} reports status {m.group(2)}")
    gate = re.search(r"^/?\S*migration-gate health=(\w+)", probe, re.M)
    checks["migration_gate_healthy"] = (
        gate is not None and gate.group(1) == "healthy" and not migration_failed
    )
    if migration_failed:
        notes.append(
            "the slot's forward or intelligence migration failed on fresh databases"
        )
    dups = re.findall(r"dup-dispatcher=(\d+)", probe)
    head_failed = failed_contracts(probe)
    base_failed = failed_contracts(base_probe) if base_probe is not None else {}
    sigs = failure_signatures(probe)
    new_failures: list[str] = []
    for container, names in head_failed.items():
        base_names = base_failed.get(container, set())
        extra = sorted(names - base_names)
        inherited = len(names) - len(extra)
        if inherited:
            notes.append(
                f"{container}: {inherited} contract(s) fail to wire at dev too (base control)"
            )
        # A contract the PR adds that fails for exactly the reason a contract dev
        # already has fails for is the same environment gap, not the PR's defect.
        inherited_sigs = {
            sigs[(container, n)] for n in names & base_names if (container, n) in sigs
        }
        for n in extra:
            sig = sigs.get((container, n))
            if base_probe is not None and sig and sig in inherited_sigs:
                notes.append(
                    f"{container}:{n} fails to wire for the reason dev's own failures share ({sig})"
                )
            else:
                new_failures.append(f"{container}:{n}")
    # both lines must be present: an absent failed-contracts line is an unread
    # log, not a clean one
    # a container that logged a wiring failure but names no failed contract is a
    # log this parser could not read, never a clean one
    unread = [
        c
        for c, fails, _ in re.findall(
            r"^(\S+) lines=\d+ autowire-fail=(\d+) dup-dispatcher=(\d+)", probe, re.M
        )
        if fails != "0" and not head_failed.get(c)
    ]
    if unread:
        notes.append("wiring failures logged but not named: " + ", ".join(unread))
    checks["no_wiring_failures"] = (
        bool(dups)
        and len(head_failed) == len(dups)
        and all(d == "0" for d in dups)
        and not new_failures
        and not unread
    )
    if new_failures:
        notes.append(
            "wiring failures the base does not have: " + ", ".join(new_failures[:8])
        )
    rcs = re.findall(r"^(\S+) focused rc=(\d+)", tests, re.M)
    failed_ids = re.findall(r"^FAILED (\S+)", tests, re.M)
    # pytest rc 1 = tests failed; a dev control at rc 1 means the test fails at
    # dev too. rc 4/5 (no such test at dev) means the PR introduced it.
    dev_rc = {
        i: rc
        for _, i, rc in re.findall(r"^dev-control (\S+) (\S+) rc=(\d+)", tests, re.M)
    }
    # the phase prints only the tail of pytest's output, so the dev-control
    # lines (one per failed test) are the complete list
    all_failed = sorted(set(failed_ids) | set(dev_rc))
    inherited_tests = [i for i in all_failed if dev_rc.get(i) == "1"]
    pr_tests = [i for i in all_failed if dev_rc.get(i) != "1"]
    focused_ok = bool(rcs)
    for repo, rc in rcs:
        if rc == "0":
            continue
        if rc == "1" and all_failed and not pr_tests:
            continue
        focused_ok = False
        notes.append(f"{repo} focused tests rc={rc}")
    if inherited_tests:
        notes.append(
            "fails at dev too (dev control): " + ", ".join(inherited_tests[:5])
        )
    if pr_tests:
        notes.append("fails at the head only: " + ", ".join(pr_tests[:5]))
    checks["focused_tests"] = focused_ok

    residue_m = re.search(
        r"containers=(\d+) volumes=(\d+) networks=(\d+) images=(\d+) listeners=(\d+) workdir=(\w+)",
        teardown,
    )
    control = re.search(r"positive control \S+ containers (\d+)", teardown)
    restored = bool(
        residue_m
        and all(v == "0" for v in residue_m.groups()[:5])
        and residue_m.group(6) == "gone"
        and control
        and int(control.group(1)) > 0
    )
    residue = residue_m.group(0) if residue_m else "no zero-residue readback"
    if residue_m and not control:
        notes.append("zero residue read without a non-zero positive control")
    diff_m = re.search(r"snapshot-diff=(\d+)", outputs.get("snap-post", ""))
    if diff_m is None:
        restored = False
        notes.append("no pre/post snapshot diff")
    elif diff_m.group(1) != "0":
        restored = False
        notes.append(
            f"host differs from its pre-run snapshot in {diff_m.group(1)} lines"
        )
    # a .201 pre-PR slot is destroyed by prepr_teardown_slot.sh, whose own
    # readback covers the dev lane's shared servers (databases, roles, topics,
    # groups, Valkey keys); anything but CLEAN is residue
    slot_td = re.search(r"slot-teardown verdict=(\w+)", teardown)
    if slot_td is not None and slot_td.group(1) != "CLEAN":
        restored = False
        notes.append(f"slot teardown verdict {slot_td.group(1)}")
    residue += f", snapshot-diff={diff_m.group(1) if diff_m else 'missing'}"
    return Readback(checks=checks, notes=notes, restored=restored, residue=residue)


def render_readback(
    rb: Readback,
    host: PoolHost,
    params: Mapping[str, str],
    started: str,
    finished: str,
) -> str:
    subject = []
    if params.get("INFRA_PR"):
        subject.append(
            f"omnibase_infra#{params['INFRA_PR']} head {params.get('INFRA_HEAD', '?')[:10]}"
        )
    if params.get("MARKET_PR"):
        subject.append(
            f"omnimarket#{params['MARKET_PR']} head {params.get('MARKET_HEAD', '?')[:10]}"
        )
    lines = [
        f"LAB PROOF {rb.outcome}: {' + '.join(subject) or 'dev'} on {host.name} ({host.ssh_target}), "
        f"{'pre-PR slot' if host.kind == 'prepr-slot' else 'isolated'} project {host.compose_project}, "
        f"{started} to {finished}",
    ]
    for name, ok in rb.checks.items():
        lines.append(f"  {'ok  ' if ok else 'FAIL'} {name}")
    for note in rb.notes:
        lines.append(f"  note: {note}")
    lines.append(f"  restored={'yes' if rb.restored else 'no'} ({rb.residue})")
    lines.append("  method: scripts/runtime_build/prepr_runtime_pool.py (OMN-18893)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- run


def read_params(path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        # remove ONE pair of surrounding quotes; a value like SQL may itself end
        # in a quote character that belongs to it
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        params[key.strip()] = value
    return params


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def run_proof(
    cfg: PoolConfig,
    transport: Transport,
    params_path: Path,
    holder: str,
    ttl_minutes: int,
    host_name: str | None,
    ledger_lines: Sequence[str],
    now_fn: Callable[[], dt.datetime] = _utcnow,
    log: Callable[[str], None] = lambda s: print(s, file=sys.stderr),
    base_probe: str | None = None,
    with_base_control: bool = False,
) -> tuple[int, str]:
    params = read_params(params_path)
    if not params.get("INFRA_PR") and not params.get("MARKET_PR"):
        return EXIT_USAGE, "params name neither INFRA_PR nor MARKET_PR"
    now = now_fn()
    holds = live_surface_holds(ledger_lines, now)
    states = survey(cfg, transport, holds, now, me=holder)
    if host_name:
        chosen = next(
            (s for s in states if host_name in (s.host.name, s.host.ssh_target)), None
        )
        if chosen is None or chosen.verdict != "FREE":
            detail = (
                f"{chosen.verdict}: {chosen.detail}" if chosen else "not in the pool"
            )
            return EXIT_NO_FREE_HOST, f"host {host_name} is not free ({detail})"
    else:
        chosen = pick(states)
        if chosen is None:
            return EXIT_NO_FREE_HOST, "no free host: " + "; ".join(
                f"{s.host.name} {s.verdict} {s.detail}" for s in states
            )
    host = chosen.host
    until = now + dt.timedelta(minutes=ttl_minutes)
    ok, why = acquire_lease(cfg, transport, host, holder, until, now)
    if not ok:
        return EXIT_LEASE_REFUSED, f"lease on {host.name} refused: {why}"
    log(f"lease {host.name}: {why}; until {until:%Y-%m-%dT%H:%M:%SZ}")
    log(
        "ledger: write HOLD surface="
        f"{host.surface} until={until:%Y-%m-%dT%H:%M:%SZ} through /omni:ledger-msg now"
    )

    started = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    ports = (host.main_port, host.effects_port)
    released, rwhy = False, "not attempted"
    base_text = base_probe
    try:
        outputs = _run_stack(
            cfg,
            transport,
            host,
            params,
            params_path,
            f"pool-{holder}-{now:%H%M%S}",
            holder,
            with_tests=True,
            log=log,
        )
        rb = judge(outputs, base_text, ports)
        wiring_or_health = not (
            rb.checks.get("no_wiring_failures", False)
            and rb.checks.get("health_8085", False)
            and rb.checks.get("health_8086", False)
        )
        if (
            base_text is None
            and with_base_control
            and rb.checks.get("stack_built")
            and wiring_or_health
        ):
            # A base control at dev on the same host, so a failure dev already
            # has is reported as dev-inherited instead of blamed on the PR.
            base_params = {
                k: v
                for k, v in params.items()
                if k
                not in ("INFRA_PR", "INFRA_HEAD", "MARKET_PR", "MARKET_HEAD", "TESTS")
            }
            base_out = _run_stack(
                cfg,
                transport,
                host,
                base_params,
                params_path.with_suffix(".base"),
                f"pool-{holder}-base-{now:%H%M%S}",
                holder,
                with_tests=False,
                log=log,
            )
            base_text = base_out.get("probe", "")
            rb = judge(outputs, base_text, ports)
            rb.notes.append("base control run at dev on the same host")
            base_rb = judge(base_out, None, ports)
            if not base_rb.restored:
                rb.restored = False
                rb.notes.append(f"base control left residue: {base_rb.residue}")
    finally:
        released, rwhy = release_lease(cfg, transport, host, holder)
        log(f"lease {host.name}: {rwhy}")
    if not released:
        rb.notes.append(f"lease not released: {rwhy}")
    text = render_readback(
        rb, host, params, started, now_fn().strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    # the ledger grammar takes PASS, FAIL or ABORTED on a surface RELEASE; an
    # INCONCLUSIVE run is released as ABORTED with the verdict in its text
    release_result = rb.outcome if rb.outcome in ("PASS", "FAIL") else "ABORTED"
    text += (
        f"\n  ledger: RELEASE re=<your HOLD id> surface={host.surface} "
        f"result={release_result} restored={'yes' if rb.restored else 'no'}"
    )
    code = {"PASS": EXIT_PASS, "FAIL": EXIT_FAIL}.get(rb.outcome, EXIT_INCONCLUSIVE)
    return code, text


# Set by the driver from the pool config; a params file cannot override them.
RESERVED_PARAMS = (
    "TAG",
    "W",
    "MODEL_ENDPOINT",
    "DOCKER_CONFIG_MODE",
    "SLOT_KIND",
    "PREPR_SLOT",
    "PROJECT",
    "MAIN_PORT",
    "EFFECTS_PORT",
    "SLOT_PORTS",
    "POSITIVE_CONTROL",
    "REASON",
)


def _run_stack(
    cfg: PoolConfig,
    transport: Transport,
    host: PoolHost,
    params: Mapping[str, str],
    params_path: Path,
    tag: str,
    holder: str,
    with_tests: bool,
    log: Callable[[str], None],
) -> dict[str, str]:
    """One stack on the leased host: every phase, then ALWAYS teardown."""
    # A path on the REMOTE host, private to this run and removed at the end.
    remote_dir = f"/tmp/{tag}"  # noqa: S108
    # keyed by member as well as holder: one lane may hold both .201 slots at
    # once, and a shared work directory would let one run's teardown remove the
    # other's tree
    work = f"$HOME/{cfg.work_root_prefix}{holder}-{host.name}"
    env_lines = [
        f"{k}={shlex.quote(v)}" for k, v in params.items() if k not in RESERVED_PARAMS
    ]
    env_lines += [
        f"TAG={tag}",
        f"W={work}",
        f"MODEL_ENDPOINT={shlex.quote(cfg.model_endpoint)}",
        f"DOCKER_CONFIG_MODE={shlex.quote(host.docker_config)}",
        f"SLOT_KIND={host.kind}",
        f"PREPR_SLOT={host.slot}",
        f"PROJECT={host.compose_project}",
        f"MAIN_PORT={host.main_port}",
        f"EFFECTS_PORT={host.effects_port}",
        f"SLOT_PORTS={shlex.quote('|'.join(str(p) for p in host.ports))}",
        f"POSITIVE_CONTROL={shlex.quote(host.positive_control)}",
        f"REASON={shlex.quote(f'lab pool run {tag} by {holder} (OMN-18893)')}",
    ]
    local_env = params_path.with_suffix(".resolved.env")
    local_env.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    phases = ["snap-pre", "clone", "build", "probe"] + (["tests"] if with_tests else [])
    # probe waits up to PROBE_WAIT_S (default 1500 s) for the runtimes to settle
    # a .201 slot bring-up waits for the pool-wide build lock, builds, migrates
    # and waits for readiness inside the one build phase
    build_budget = 7200 if host.kind == "prepr-slot" else 3600
    budget = {"clone": 900, "build": build_budget, "probe": 1800, "tests": 2400}
    outputs: dict[str, str] = {}

    def phase_run(phase: str, timeout: float) -> int:
        rc, out = transport.run(
            host,
            f"bash {remote_dir}/prove.sh {remote_dir}/params.env {phase}",
            timeout=timeout,
        )
        outputs[phase] = out
        log(f"[{host.name} {tag} {phase} rc={rc}]\n{out.rstrip()}")
        return rc

    try:
        transport.run(host, f"mkdir -p {remote_dir}", timeout=30)
        if transport.put(host, PROVE_SH, f"{remote_dir}/prove.sh") or transport.put(
            host, local_env, f"{remote_dir}/params.env"
        ):
            raise RuntimeError("copy to host failed")
        for phase in phases:
            if phase_run(phase, budget.get(phase, 120)) != 0 and phase in (
                "clone",
                "build",
            ):
                break
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        # teardown below must run whatever failed above
        log(f"run aborted before teardown: {exc}")
    finally:
        for phase in ("teardown", "snap-post"):
            phase_run(phase, 600)
        transport.run(host, f"rm -rf {remote_dir} /tmp/{tag}-*", timeout=30)
    return outputs


# --------------------------------------------------------------------------- cli


def _ledger_lines(path: str | None) -> list[str]:
    if not path:
        return []
    return Path(path).read_text(encoding="utf-8").splitlines()


def _status_table(states: Sequence[HostState]) -> str:
    rows = ["host             target           surface          verdict     detail"]
    for s in states:
        projects = ",".join(f"{k}:{v}" for k, v in sorted(s.other_projects.items()))
        detail = s.detail + (f" [running: {projects}]" if projects else "")
        rows.append(
            f"{s.host.name:<16} {s.host.ssh_target:<16} {s.host.surface:<16} {s.verdict:<11} {detail}"
        )
    return "\n".join(rows)


def main(argv: Sequence[str] | None = None, transport: Transport | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0] if __doc__ else ""
    )
    parser.add_argument("--config", type=Path, default=POOL_CONFIG)
    parser.add_argument(
        "--ledger",
        default=os.environ.get("ONEX_LEDGER_PATH"),
        help="rolling work ledger, read only, for live surface HOLDs (default $ONEX_LEDGER_PATH)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("status", help="probe every pool host, read only")
    st.add_argument("--json", action="store_true")
    st.add_argument("--holder")
    pk = sub.add_parser("pick", help="print the free host a run would take")
    pk.add_argument("--holder")
    rn = sub.add_parser("run", help="prove one PR on a free host")
    rn.add_argument(
        "--params",
        type=Path,
        required=True,
        help="KEY=VALUE file, see prepr_pool_prove.sh",
    )
    rn.add_argument("--holder", required=True, help="your ledger lane name")
    rn.add_argument("--ttl-minutes", type=int, default=75)
    rn.add_argument("--host", help="insist on this pool host")
    rn.add_argument(
        "--base-control",
        action="store_true",
        help="when health or wiring fails, build dev on the same host and hold only new failures against the PR",
    )
    rn.add_argument(
        "--base-probe",
        type=Path,
        help="probe output of an earlier base control at the same dev, instead of building one",
    )
    rl = sub.add_parser("release", help="release a lease this holder left behind")
    rl.add_argument("--host", required=True)
    rl.add_argument("--holder", required=True)
    args = parser.parse_args(argv)

    try:
        cfg = load_pool_config(args.config)
    except (OSError, ValueError, KeyError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    tp = transport or SshTransport()
    now = _utcnow()
    if args.cmd in ("status", "pick"):
        holds = live_surface_holds(_ledger_lines(args.ledger), now)
        states = survey(cfg, tp, holds, now, me=args.holder)
        if args.cmd == "pick":
            chosen = pick(states)
            print(chosen.host.name if chosen else "none")
            return EXIT_PASS if chosen else EXIT_NO_FREE_HOST
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "host": s.host.name,
                            "ssh_target": s.host.ssh_target,
                            "surface": s.host.surface,
                            "kind": s.host.kind,
                            "compose_project": s.host.compose_project,
                            "verdict": s.verdict,
                            "detail": s.detail,
                            "cores": s.cores,
                            "load1": s.load1,
                            "running_projects": s.other_projects,
                        }
                        for s in states
                    ],
                    indent=2,
                )
            )
        else:
            print(_status_table(states))
        return EXIT_PASS
    if args.cmd == "release":
        ok, why = release_lease(cfg, tp, cfg.host(args.host), args.holder)
        print(why)
        return EXIT_PASS if ok else EXIT_LEASE_REFUSED
    code, text = run_proof(
        cfg,
        tp,
        args.params,
        args.holder,
        args.ttl_minutes,
        args.host,
        _ledger_lines(args.ledger),
        base_probe=args.base_probe.read_text(encoding="utf-8")
        if args.base_probe
        else None,
        with_base_control=args.base_control,
    )
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
