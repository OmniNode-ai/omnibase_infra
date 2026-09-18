#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-run runner routing: the I/O boundary around the routing COMPUTE node.

WHAT THIS IS, AND WHAT IT IS NOT. A hosted ``route`` job runs this once per
workflow run. It PROBES -- org runner capacity, this repository's visibility,
the lab-load record -- assembles a typed request, and asks
``node_ci_runner_route_compute`` where the run's jobs should execute. The
DECISION is not here: it is the node's handler, declared by its contract
(OMN-18412). There is exactly one implementation of it, and this file cannot
drift from it because it does not contain one.

WHERE EACH INPUT COMES FROM, all of them declared rather than literal:

  thresholds, labels, refusal policy  the node's contract.yaml `config:` block
  fleet inventory (expected count)    config/runner_fleet.yaml
  hosted workflow list                config/runner_routing_policy.yaml
  the seam ceiling                    the caller's runner variable, READ ONLY
  capacity / visibility / lab load    probed here, at this boundary

THE SEAM IS READ, NEVER WRITTEN. CLAUDE.md rule 14 makes the routing variables
single-owner: two lanes read-modify-writing them produce a last-writer-wins
clobber with no error. Routing treats the seam as an upper bound and has no
code path that PATCHes any variable.

CONSEQUENCE, STATED UP FRONT: while a repository's seam reads
``["ubuntu-latest"]`` the node returns it verbatim on every run and placement is
byte-identically what it is today. Do not read a green route job as evidence
that jobs moved to the lab -- read the decision record's ``reason``.

FAIL-CLOSED. Every probe failure is modelled as a named error class rather than
an absent reading, and the node resolves each to GitHub-hosted. This file
catches everything at the boundary so even a crash hands the calling run a
usable ``runs-on`` -- with one deliberate exception: a REFUSAL (a private
repository whose only allowed placement is hosted) exits non-zero, because
there is no placement to hand back.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

# THE NODE IS IMPORTED LAZILY, INSIDE THE FUNCTIONS THAT DECIDE, and that is a
# constraint rather than a style choice. `scripts/ci/probe_lab_load.py` imports
# this module for its PROBES and runs on the fleet image's system python3,
# which has neither PyYAML nor pydantic; ten of twelve scheduled probe runs
# once failed on exactly such a module-scope import. Everything above the I/O
# boundary therefore stays standard-library only, and the decision -- which
# runs on a hosted runner with the package installed -- imports what it needs
# where it needs it.
if TYPE_CHECKING:  # pragma: no cover - typing only
    from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_policy import (
        ModelCIRunnerRoutePolicy,
    )
    from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_request import (
        ModelCIRunnerRouteRequest,
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

ORG = "OmniNode-ai"
NODE_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "src/omnibase_infra/nodes/node_ci_runner_route_compute/contract.yaml"
)
DEFAULT_POLICY = Path("config/runner_routing_policy.yaml")
FLEET_INVENTORY = Path("config/runner_fleet.yaml")

# The exit status a refusal uses. Distinct from 1 so the calling workflow can
# tell "the router refused this placement" from "the module itself broke", and
# so the workflow's fail-closed floor -- which exists to hand a crashed router's
# run a usable runs-on -- cannot mistake a deliberate refusal for a crash and
# paper over it with hosted labels.
REFUSED_EXIT_CODE = 3


def load_contract_policy(path: Path = NODE_CONTRACT) -> ModelCIRunnerRoutePolicy:
    """Parse the node contract's ``config:`` block into the typed policy.

    Every field is required by the model, so a contract missing one fails here,
    naming it, rather than at the first run that needed it (CLAUDE.md rule 8).
    PyYAML is imported locally: the self-hosted lab-load probe imports this
    module for its probe functions and runs on a bare interpreter that does not
    have it.
    """
    import yaml

    from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_policy import (
        ModelCIRunnerRoutePolicy,
    )

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    config = loaded.get("config")
    if not isinstance(config, dict):
        raise KeyError(f"{path} is missing the required 'config:' block")
    return ModelCIRunnerRoutePolicy.model_validate(config)


def load_contract_runner_group(path: Path = NODE_CONTRACT) -> str:
    """The runner group capacity is observed from, read WITHOUT pydantic.

    Same contract file and same field as :func:`load_contract_policy`; this
    reader exists because of who calls it, not because there is a second
    policy. ``load_contract_policy`` imports the typed model, which pulls in
    ``omnibase_infra`` and therefore ``omnibase_core`` and pydantic. The
    ``saturation-record`` job in ``dev-lane-liveness.yml`` runs on a bare
    hosted image's ``python3`` with no ``uv sync`` behind it -- deliberately,
    because that monitor has to survive the saturation it reports on -- and
    has exactly one question for the contract: which runner group to probe.
    Asking it through the typed loader means asking it to install the
    repository first.

    This is the same stdlib-plus-PyYAML shape as
    :func:`load_fleet_expected_count` and :func:`load_hosted_workflows`, for
    the same reason those have it, and PyYAML is what the caller already
    relies on -- ``runner_saturation_record.py`` imports it at module scope
    two steps later in the same job.

    ONE VALUE, TWO READERS, PINNED. ``tests/ci/test_runner_route_decision.py``
    asserts this returns exactly ``load_contract_policy().runner_group``, so
    the two cannot drift into two policies that agree until someone edits one.
    """
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    config = loaded.get("config")
    if not isinstance(config, dict):
        raise KeyError(f"{path} is missing the required 'config:' block")
    runner_group = config.get("runner_group")
    if not isinstance(runner_group, str) or not runner_group:
        raise KeyError(f"{path} config block is missing a usable 'runner_group'")
    return runner_group


# The runner class this router places work on. The router decides between the
# shared action fleet and GitHub-hosted runners; it never routes deploy- or
# verify-class jobs, which are pinned by label at the workflow.
ROUTED_RUNNER_CLASS = "action"


def load_fleet_expected_count(path: Path = FLEET_INVENTORY) -> int:
    """Declared capacity of the routed fleet, summed across every host.

    Read here rather than restated in the contract so the routing floor tracks
    a fleet resize instead of going stale against it -- the failure that made
    the previous literal floor equal to the whole fleet when it was capped.

    OMN-17477 made this a SUM over the declared host inventory rather than one
    host's scalar, because the fleet stopped being one machine. Two properties
    of that sum matter and neither is incidental:

    1. It is summed PER CLASS. A verify-class runner on a second host cannot
       pick up an action-class job, so counting it here would raise the
       degraded floor by capacity that can never satisfy the jobs the floor
       guards -- the router would read a short fleet as healthy.
    2. It agrees with what the fleet probe COUNTS. ``probe_fleet`` filters the
       org runner registry by the ``omnibase-ci`` label, so the denominator
       here must be the runners carrying that label and no others. A numerator
       and a denominator counting different sets is a floor that means nothing.

    A config with no inventory is a single-host fleet and falls back to the
    scalar, which is the value this function has always returned.
    """
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")

    hosts = loaded.get("hosts")
    if isinstance(hosts, list) and hosts:
        total = 0
        for row in hosts:
            if not isinstance(row, dict):
                raise ValueError(f"{path} 'hosts' entries must be mappings")
            classes = row.get("classes") or []
            if ROUTED_RUNNER_CLASS not in classes:
                continue
            count = row.get("expected_count")
            if not isinstance(count, int) or count < 0:
                raise KeyError(
                    f"{path} host {row.get('host')!r} is missing a usable "
                    "'expected_count'"
                )
            total += count
        if total <= 0:
            raise KeyError(
                f"{path} declares no host carrying the {ROUTED_RUNNER_CLASS!r} "
                "class; the routed fleet's declared capacity would be zero"
            )
        return total

    expected = loaded.get("expected_count")
    if not isinstance(expected, int) or expected <= 0:
        raise KeyError(f"{path} is missing a usable 'expected_count'")
    return expected


def load_hosted_workflows(path: Path = DEFAULT_POLICY) -> tuple[str, ...]:
    """Workflows whose routed jobs stay hosted regardless of capacity.

    DELIBERATELY NOT ``hosted_runner_allowlist``. That list marks files
    containing an intentionally bare hosted job, which is why ``ci.yml`` is on
    it -- for its one lightweight summary aggregator. An early build read that
    list here and a dry run against live data pinned all 54 of ci.yml's jobs
    hosted forever, making the mechanism a silent no-op on its largest consumer
    while every unit test still passed. The two lists answer different
    questions.

    This stays in the routing policy file rather than the contract because it
    is per-repository data, not a threshold: the contract declares the rule,
    the policy file names this repository's exceptions to it.
    """
    import yaml

    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    route = loaded.get("route")
    if not isinstance(route, dict):
        raise KeyError(f"{path} is missing the required 'route:' section")
    declared = route.get("hosted_workflows")
    if not isinstance(declared, list):
        raise KeyError(f"{path} route section is missing 'hosted_workflows'")
    return tuple(str(item) for item in declared)


# --------------------------------------------------------------------------
# I/O boundary. Probes live here so ``decide`` stays pure and testable.
# --------------------------------------------------------------------------


def probe_fleet(token: str | None, runner_group: str, api_url: str) -> dict[str, Any]:
    """Read org self-hosted runner busy/idle counts. Never raises."""
    if not token:
        return {"ok": False, "error": "missing_token"}
    # `api_url` originates in the GITHUB_API_URL environment variable. Pin the
    # scheme rather than suppressing the audit: a file:// or custom-scheme base
    # would make the probe read something other than the API, and the probe's
    # answer decides where jobs execute.
    if not api_url.startswith("https://"):
        return {"ok": False, "error": "bad_api_scheme"}
    runners: list[dict[str, Any]] = []
    try:
        for page in range(1, 11):
            url = f"{api_url}/orgs/{ORG}/actions/runners?per_page=100&page={page}"
            request = urllib.request.Request(  # noqa: S310 -- scheme pinned to https above
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/vnd.github+json",
                },
            )
            with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310 -- scheme pinned to https above
                payload = json.loads(response.read().decode("utf-8"))
            page_runners = payload.get("runners")
            if not isinstance(page_runners, list):
                return {"ok": False, "error": "malformed_json"}
            runners.extend(item for item in page_runners if isinstance(item, dict))
            if len(page_runners) < 100:
                break
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"http_{exc.code}"}
    except (urllib.error.URLError, TimeoutError):
        return {"ok": False, "error": "timeout"}
    except (ValueError, TypeError):
        return {"ok": False, "error": "malformed_json"}

    grouped = [
        item
        for item in runners
        if any(
            isinstance(label, dict) and label.get("name") == runner_group
            for label in (item.get("labels") or [])
        )
    ]
    if not grouped:
        # An empty result is not evidence of an idle fleet; it is evidence the
        # probe found nothing (CLAUDE.md rule 16).
        return {"ok": False, "error": "empty_fleet"}
    online = sum(1 for item in grouped if item.get("status") == "online")
    # A runner reporting offline while busy is demonstrably executing a job --
    # the registry read is stale, the listener is not dead (OMN-16030).
    busy = sum(1 for item in grouped if item.get("busy") is True)
    return {"ok": True, "online": online, "busy": busy, "total": len(grouped)}


def probe_repo_visibility(token: str | None, repository: str, api_url: str) -> str:
    """Read THIS repository's visibility from the API. Never raises.

    READ AT DECISION TIME, not declared in the policy file. A declared list of
    private repositories is a second copy of a fact GitHub already owns, and it
    goes stale silently the first time a repository changes visibility -- the
    same class of defect as the runner-routing table that rule 14 tells you not
    to enumerate from. The sibling `private-repo-runner-placement` gate reads
    live visibility for the same reason.

    ``unknown`` on every fault, and ``unknown`` is NOT a refusal -- see
    ``apply_visibility_rule`` for why that asymmetry is deliberate.
    """
    if not token:
        return "unknown"
    # Scheme-pinned for the same reason probe_fleet pins it: the answer decides
    # where jobs execute, so it may only ever come from the real API.
    if not api_url.startswith("https://"):
        return "unknown"
    if not repository or repository.count("/") != 1:
        return "unknown"
    try:
        request = urllib.request.Request(  # noqa: S310 -- scheme pinned to https above
            f"{api_url}/repos/{repository}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310 -- scheme pinned to https above
            payload = json.loads(response.read().decode("utf-8"))
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        ValueError,
        TypeError,
    ):
        return "unknown"
    if not isinstance(payload, dict):
        return "unknown"
    if "visibility" in payload:
        return normalise_visibility(payload.get("visibility"))
    # `private` is the older boolean form of the same fact; read it only when
    # `visibility` is absent, never as a second opinion that could disagree.
    private = payload.get("private")
    if private is True:
        return "private"
    if private is False:
        return "public"
    return "unknown"


def probe_lab_saturation_from_fleet(
    token: str | None, runner_group: str, api_url: str
) -> dict[str, Any]:
    """The lab-load record the self-hosted ``lab-load-probe`` job publishes,
    sourced from the org runner registry's busy/idle counts rather than this
    container's own ``/proc/loadavg``.

    WHY NOT ``probe_local_lab_load``'s ratio, from inside this job. The fleet
    runners are Docker containers on the ``.201`` lab host, and a container's
    own ``load1 / os.cpu_count()`` reading does not describe the host it runs
    on: measured live in the same window, the SAME machine read 1.524x
    (load1 48.76 / 32 cores) from an on-host dry run and 0.2232x from inside
    one of its own runner containers. ``max_lab_load_ratio`` is calibrated
    against the host-scoped number, so feeding it the container-scoped one is
    not "slightly off" -- it is a different, uncalibrated quantity that
    happens to share a threshold.

    No host-published load metric exists to read instead (checked: no
    node-exporter/cAdvisor surface and no deploy-agent host-metrics artifact
    anywhere in this repo), so this reads the one live number the probe CAN
    measure honestly from inside a container with nothing but outbound HTTPS:
    the org runner registry's busy/idle counts for ``runner_group`` --
    ``probe_fleet``'s own signal, the same one S4 already trusts for fleet
    saturation.

    Free memory stays LOCALLY sourced (``_free_mem_mib``): OMN-17392's OOM
    finding was about the runner container's own memory pressure, which a
    container-scoped reading answers correctly -- unlike load, memory here is
    the resource actually being contended for host-adjacent test runs.
    """
    fleet = probe_fleet(token, runner_group, api_url)
    if not fleet.get("ok"):
        return {"ok": False, "error": fleet.get("error", "probe_failed")}
    online = fleet.get("online")
    busy = fleet.get("busy")
    if not isinstance(online, int) or not isinstance(busy, int):
        return {"ok": False, "error": "unexpected_shape"}
    free_mem_mib = _free_mem_mib()
    if free_mem_mib is None:
        return {"ok": False, "error": "mem_unreadable"}
    ratio = (busy / online) if online else 1.0
    return {
        "ok": True,
        "age_seconds": 0,
        "hosts": [
            {
                "label": f"org:{runner_group}",
                "ratio": round(ratio, 4),
                "free_mem_mib": free_mem_mib,
            }
        ],
    }


def read_lab_record(path: Path | None, now: datetime | None = None) -> dict[str, Any]:
    """Read the freshness-bounded lab-load record. Never raises."""
    if path is None or not path.exists():
        return {"ok": False, "error": "no_record"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ok": False, "error": "unreadable_record"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "unreadable_record"}
    sampled_at = payload.get("sampled_at")
    hosts = payload.get("hosts")
    if not isinstance(sampled_at, str) or not isinstance(hosts, list):
        return {"ok": False, "error": "unreadable_record"}
    try:
        sampled = datetime.fromisoformat(sampled_at.replace("Z", "+00:00"))
    except ValueError:
        return {"ok": False, "error": "unreadable_record"}
    reference = now or datetime.now(UTC)
    age = int((reference - sampled).total_seconds())
    return {"ok": True, "age_seconds": max(age, 0), "hosts": hosts}


def probe_local_lab_load() -> dict[str, Any]:
    """Read THIS host's own load1/cores and free memory, the way the pre-push
    picker does. Used by ``--lab-probe-local`` for the lab-host dry run, and by
    the self-hosted lab-load probe job in the saturation monitor.
    """
    try:
        load1 = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        free_mem_mib = _free_mem_mib()
    except OSError:
        return {"ok": False, "error": "probe_failed"}
    if free_mem_mib is None:
        return {"ok": False, "error": "mem_unreadable"}
    return {
        "ok": True,
        "age_seconds": 0,
        "hosts": [
            {
                "label": os.uname().nodename,
                "ratio": round(load1 / cores, 4),
                "free_mem_mib": free_mem_mib,
            }
        ],
    }


def _free_mem_mib() -> int | None:
    """Available memory in MiB on Linux and macOS. ``None`` when unreadable --
    never a guessed value, because assumed headroom is the failure class.
    """
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
        except (OSError, ValueError, IndexError):
            return None
        return None
    try:
        import subprocess

        pagesize = int(
            subprocess.run(
                ["/usr/sbin/sysctl", "-n", "hw.pagesize"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            ).stdout.strip()
        )
        vm_stat = subprocess.run(
            ["/usr/bin/vm_stat"], capture_output=True, text=True, check=True, timeout=5
        ).stdout
        free_pages = 0
        for line in vm_stat.splitlines():
            if line.startswith(
                ("Pages free:", "Pages inactive:", "Pages speculative:")
            ):
                free_pages += int(line.split(":")[1].strip().rstrip("."))
        return (free_pages * pagesize) // (1024 * 1024)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------
# Request assembly + CLI.
# --------------------------------------------------------------------------


def build_request(
    *,
    event_name: str,
    head_repo: str,
    repository: str,
    workflow_path: str,
    seam_json: str,
    public_json: str,
    visibility: str,
    fleet: dict[str, Any],
    lab: dict[str, Any],
    hosted_workflows: tuple[str, ...],
    force: str,
    policy: ModelCIRunnerRoutePolicy,
    fleet_expected_count: int,
) -> ModelCIRunnerRouteRequest:
    """Turn raw probe results into the node's typed request.

    A probe result that cannot be understood becomes a NAMED failure class, not
    a silently absent field: "the fleet is idle" and "nobody asked the fleet"
    must not collapse into the same reading (CLAUDE.md rule 16).
    """
    from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_force import (
        EnumCIRunnerRouteForce,
    )
    from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_visibility import (
        EnumCIRunnerRouteVisibility,
    )
    from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_fleet_observation import (
        ModelCIRunnerFleetObservation,
    )
    from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_lab_host import (
        ModelCIRunnerLabHost,
    )
    from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_lab_observation import (
        ModelCIRunnerLabObservation,
    )
    from omnibase_infra.nodes.node_ci_runner_route_compute.models.model_ci_runner_route_request import (
        ModelCIRunnerRouteRequest,
    )

    fleet_ok = bool(fleet.get("ok"))
    online = fleet.get("online")
    busy = fleet.get("busy")
    if fleet_ok and not (isinstance(online, int) and isinstance(busy, int)):
        fleet_ok, online, busy = False, None, None
        fleet = {"error": "unexpected_shape"}
    observation = ModelCIRunnerFleetObservation(
        ok=fleet_ok,
        error=str(fleet.get("error") or ""),
        online=online if fleet_ok else None,
        busy=busy if fleet_ok else None,
    )

    lab_hosts: list[ModelCIRunnerLabHost] = []
    lab_ok = bool(lab.get("ok"))
    lab_error = str(lab.get("error") or "")
    if lab_ok:
        for host in lab.get("hosts") or []:
            if not isinstance(host, dict):
                lab_ok, lab_error = False, "malformed_host"
                break
            ratio, free_mem = host.get("ratio"), host.get("free_mem_mib")
            if not isinstance(ratio, (int, float)) or not isinstance(free_mem, int):
                lab_ok, lab_error = False, "malformed_host"
                break
            lab_hosts.append(
                ModelCIRunnerLabHost(
                    label=str(host.get("label") or ""),
                    ratio=float(ratio),
                    free_mem_mib=free_mem,
                )
            )
    age = lab.get("age_seconds")
    lab_observation = ModelCIRunnerLabObservation(
        ok=lab_ok,
        error=lab_error,
        age_seconds=age if lab_ok and isinstance(age, int) else None,
        hosts=tuple(lab_hosts) if lab_ok else (),
    )

    return ModelCIRunnerRouteRequest(
        github_event=event_name,
        head_repo=head_repo,
        repository=repository,
        workflow_path=workflow_path,
        seam_json=seam_json,
        public_json=public_json,
        visibility=EnumCIRunnerRouteVisibility(normalise_visibility(visibility)),
        fleet=observation,
        fleet_expected_count=fleet_expected_count,
        lab=lab_observation,
        hosted_workflows=hosted_workflows,
        force=EnumCIRunnerRouteForce(force),
        policy=policy,
    )


def normalise_visibility(value: Any) -> str:
    """Map a raw visibility reading onto the three the node knows.

    ``internal`` is PRIVATE: it is not publicly readable and its hosted minutes
    are billed exactly like a private repository's, which is what the ruling is
    about. Treating it as public because the API spells it differently would be
    a silent exemption.
    """
    if value in {"public", "private", "unknown"}:
        return str(value)
    if value == "internal":
        return "private"
    return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--head-repo", default="")
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow-path", required=True)
    parser.add_argument("--seam-json", default="")
    parser.add_argument("--public-json", default="")
    parser.add_argument("--contract", type=Path, default=NODE_CONTRACT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--fleet-inventory", type=Path, default=FLEET_INVENTORY)
    parser.add_argument("--lab-record", type=Path, default=None)
    parser.add_argument(
        "--lab-probe-local",
        action="store_true",
        help="probe THIS host's own load instead of reading a record (lab-host dry run)",
    )
    parser.add_argument(
        "--fleet-json", type=Path, default=None, help="pre-probed fleet payload"
    )
    parser.add_argument(
        "--repo-visibility",
        choices=["public", "private", "unknown"],
        default=None,
        help="this repository's visibility; probed from the API when omitted. "
        "A private repository is never placed on a hosted runner (OMN-18412).",
    )
    parser.add_argument(
        "--force",
        choices=["auto", "fleet", "hosted"],
        default="auto",
        help="operator override. Subject to every safety rule rather than a "
        "bypass of them: fleet cannot widen past the seam or take a fork, and "
        "hosted is still refused for a private repository.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the decision, write nothing to GITHUB_OUTPUT, always exit 0",
    )
    # GITHUB_API_URL is injected by the Actions runner and names the GitHub REST
    # API this run executes against. It is not a first-party service endpoint and
    # has no routing-authority/integration-catalog entry to resolve from -- the
    # same contract the existing fleet canary already operates under.
    # probe_fleet() scheme-pins the value to https before issuing any request.
    # fmt: off
    default_api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")  # url-authority-ok: Actions-injected GitHub REST base, not an ONEX-routed service endpoint; probe_fleet() scheme-pins it to https before any request
    # fmt: on
    parser.add_argument("--api-url", default=default_api_url)
    args = parser.parse_args(argv)

    from omnibase_infra.nodes.node_ci_runner_route_compute.handlers.handler_ci_runner_route import (
        HandlerCIRunnerRoute,
    )
    from omnibase_infra.nodes.node_ci_runner_route_compute.models.enum_ci_runner_route_decision import (
        EnumCIRunnerRouteDecision,
    )

    policy = load_contract_policy(args.contract)
    hosted_workflows = load_hosted_workflows(args.policy)
    fleet_expected_count = load_fleet_expected_count(args.fleet_inventory)

    if args.fleet_json is not None:
        try:
            fleet = json.loads(args.fleet_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            fleet = {"ok": False, "error": "unreadable_record"}
    else:
        token = os.environ.get("RUNNER_FLEET_STATUS_TOKEN") or os.environ.get(
            "CROSS_REPO_PAT"
        )
        fleet = probe_fleet(token, policy.runner_group, args.api_url)

    lab = (
        probe_local_lab_load()
        if args.lab_probe_local
        else read_lab_record(args.lab_record)
    )

    # The job token is enough: repository metadata is readable with the default
    # `metadata: read` every Actions token carries, so this needs no new
    # credential and no new scope. The fleet-status tokens are accepted as a
    # fallback only so a dry run outside Actions can still resolve the fact.
    visibility = args.repo_visibility or probe_repo_visibility(
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("RUNNER_FLEET_STATUS_TOKEN")
        or os.environ.get("CROSS_REPO_PAT"),
        args.repository,
        args.api_url,
    )

    request = build_request(
        event_name=args.event_name,
        head_repo=args.head_repo or "",
        repository=args.repository,
        workflow_path=args.workflow_path,
        seam_json=args.seam_json or "",
        public_json=args.public_json or "",
        visibility=visibility,
        fleet=fleet
        if isinstance(fleet, dict)
        else {"ok": False, "error": "unexpected_shape"},
        lab=lab
        if isinstance(lab, dict)
        else {"ok": False, "error": "unexpected_shape"},
        hosted_workflows=hosted_workflows,
        force=args.force,
        policy=policy,
        fleet_expected_count=fleet_expected_count,
    )

    decision = HandlerCIRunnerRoute().handle(request)
    record = decision.model_dump(mode="json")
    print(json.dumps(record, indent=2, sort_keys=True))

    labels = list(decision.runs_on)
    refused = decision.decision is EnumCIRunnerRouteDecision.BLOCKED
    if refused:
        print(
            f"::error title=Runner route refused::{decision.reason_wire} -- this "
            "repository is private and the only placement its policy allows is "
            "a GitHub-hosted runner, which the 2026-09-14 operator ruling "
            "forbids. Fix the cause named in the reason; do not place the job "
            "hosted."
        )
    else:
        print(
            f"::notice title=Runner route::{decision.decision.value} "
            f"({decision.reason_wire}) -> {json.dumps(labels)}"
        )

    if args.dry_run:
        # A dry run computes and reports; it gates nothing and cannot fail a run.
        return 0

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write(f"labels={json.dumps(labels)}\n")
            handle.write(f"runs_on={json.dumps(labels)}\n")
            handle.write(f"decision={decision.decision.value}\n")
            handle.write(f"reason={decision.reason_wire}\n")
    artifact = Path(
        os.environ.get("RUNNER_ROUTE_ARTIFACT", "runner-route-decision.json")
    )
    artifact.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    # The outputs and the record are written FIRST and the refusal is signalled
    # by the exit status, in that order: a refused run must still leave an
    # auditable decision record, and the consuming workflow must still see a
    # `labels=` line so its fail-closed floor does not overwrite the refusal
    # with a hosted placement.
    if refused:
        return REFUSED_EXIT_CODE
    return 0


if __name__ == "__main__":
    sys.exit(main())
