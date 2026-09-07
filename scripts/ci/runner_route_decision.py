#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Per-run runner routing decision (OMN-18031).

WHAT THIS IS. A hosted ``route`` job runs this once per workflow run and emits a
``runs-on`` label set that the heavy jobs consume via
``needs.route.outputs.labels``. The choice is made from live signals -- org
runner busy/idle counts and lab host load -- with the trusted seam variable
(``OMNI_TRUSTED_CI_RUNS_ON_JSON``) read as a CEILING.

THE SEAM IS READ, NEVER WRITTEN. CLAUDE.md rule 14 makes the routing variables
single-owner: two lanes read-modify-writing them produce a last-writer-wins
clobber with no error. This module therefore treats the seam as an upper bound
on what routing may choose and has no code path that PATCHes any variable.

CONSEQUENCE, STATED UP FRONT: with the seam at its current live value of
``["ubuntu-latest"]`` (the deliberate OMN-16682 re-flip), step S2 below returns
the seam verbatim on EVERY run and this mechanism is byte-identically inert.
That is intended. This ticket lands a mechanism proven safe and inert; placement
changes only after a separate single-owner claim flips the seam, which is
OMN-16682 and not this ticket. Do not read a green route job as evidence that
jobs moved to the lab -- read the decision artifact's ``reason``.

THE DECISION IS AN ORDERED ELIMINATION, AND EVERY STEP CAN ONLY MOVE THE ANSWER
TOWARD HOSTED:

  S0  workflow path is in the policy's hosted allowlist  -> hosted
  S1  fork / untrusted PR event                          -> hosted (public var)
  S2  seam names no self-hosted label                    -> the seam, VERBATIM
  S3  any probe error, of any class                      -> hosted
  S4  fleet saturated (idle floor or busy fraction)      -> hosted
  S5  fleet degraded (too few online at all)             -> hosted
  S6  lab record missing / stale / loaded / mem-starved  -> hosted
  S7  otherwise                                          -> the seam labels

FAIL-CLOSED IS THE WHOLE SAFETY STORY. Nothing raises to the caller: the
boundary catches every exception and degrades to hosted, so even a crashing
route job hands the run a usable ``runs-on`` rather than an empty one. An
unreadable probe is never "assume ample" -- that assumption is the failure class
the pre-push picker's own fail-closed rules already exist to prevent.

NEVER-WIDEN IS MECHANICAL, NOT A CONVENTION. ``decide`` re-checks the label set
``_decide_unchecked`` actually returned against that run's ceiling as the last
act before emitting, and a violation degrades to hosted rather than raising, and
``tests/ci/test_runner_route_decision.py::test_never_widens_beyond_ceiling``
sweeps the cross-product of every input dimension asserting a self-hosted label
can appear only when the seam already contained one.

FORK ISOLATION LIVES HERE (S1), not at the 46 selector call sites in ci.yml,
precisely so it cannot be edited wrong in one of them (OMN-16683/16684).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ORG = "OmniNode-ai"
DEFAULT_POLICY = Path("config/runner_routing_policy.yaml")

# Every threshold the decision reads. A missing key raises at load rather than
# silently defaulting (CLAUDE.md rule 8): a defaulted threshold is how a routing
# gate quietly stops gating.
REQUIRED_POLICY_KEYS: tuple[str, ...] = (
    "policy_version",
    "runner_group",
    "lab_labels",
    "hosted_labels",
    "min_idle_runners",
    "max_busy_fraction",
    "min_online_runners",
    "lab_record_max_age_seconds",
    "max_lab_load_ratio",
    "min_lab_free_mem_mib",
    "hosted_workflows",
)

REQUIRED_ALERT_KEYS: tuple[str, ...] = (
    "busy_fraction_threshold",
    "lab_load_ratio_threshold",
    "sustained_samples",
)


@dataclass(frozen=True)
class RouteDecision:
    """The emitted decision, complete enough to audit without the run log."""

    labels: list[str]
    decision: str
    reason: str
    policy_version: int
    inputs: dict[str, Any] = field(default_factory=dict)
    decided_at: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": "runner_route_decision/v1",
            "labels": self.labels,
            "decision": self.decision,
            "reason": self.reason,
            "policy_version": self.policy_version,
            "decided_at": self.decided_at or datetime.now(UTC).isoformat(),
            "inputs": self.inputs,
        }


def load_route_policy(path: Path) -> dict[str, Any]:
    """Load the ``route:`` section, failing on any missing threshold."""
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if "route" not in loaded or not isinstance(loaded["route"], dict):
        raise KeyError(f"{path} is missing the required 'route:' section")
    section = loaded["route"]
    for key in REQUIRED_POLICY_KEYS:
        if key not in section:
            raise KeyError(f"{path} route section is missing required key {key!r}")
    return section


def hosted_workflows(policy: dict[str, Any]) -> list[str]:
    """Workflows whose routed jobs stay hosted regardless of capacity.

    DELIBERATELY NOT ``hosted_runner_allowlist``. That list marks files
    containing an intentionally bare ``runs-on: ubuntu-latest`` job, which is
    why ``ci.yml`` is on it -- for its one lightweight CI Summary aggregator.
    An early build of this module read that list here, and a dry run against
    live data returned ``policy_allowlist`` for ``ci.yml``: all 54 of its jobs
    would have been pinned hosted forever, making the mechanism a silent no-op
    on its single largest consumer while every unit test still passed. The two
    lists answer different questions.
    """
    return [str(item) for item in policy.get("hosted_workflows", [])]


def _parse_labels(raw: str | None) -> list[str] | None:
    """Parse a runner-variable JSON array. ``None`` on anything unusable."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    if not all(isinstance(item, str) for item in parsed):
        return None
    return [str(item) for item in parsed]


def _is_self_hosted(labels: list[str]) -> bool:
    return "self-hosted" in labels


def assert_never_widens(
    returned: list[str], ceiling: list[str], hosted: list[str]
) -> None:
    """Last act before emitting: the answer is hosted, or within the ceiling.

    This is a runtime assertion and not only a test because the failure it
    guards -- untrusted or unbudgeted code reaching self-hosted compute -- is
    the one outcome this design may never produce. A raise here is caught by
    the boundary and degrades to hosted.
    """
    if set(returned) == set(hosted):
        return
    if not set(returned) <= set(ceiling):
        raise AssertionError(
            f"routing widened beyond the seam ceiling: returned={returned} ceiling={ceiling}"
        )
    if "self-hosted" in returned and "self-hosted" not in ceiling:
        raise AssertionError(f"routing invented a self-hosted label: ceiling={ceiling}")


def _decide_unchecked(
    *,
    event_name: str,
    head_repo: str | None,
    repository: str,
    workflow_path: str,
    seam_json: str | None,
    public_json: str | None,
    fleet: Any,
    lab: Any,
    policy: dict[str, Any],
    allowlist: list[str] | None = None,
) -> RouteDecision:
    """The ordered elimination itself. Callers use ``decide``, which re-checks
    this function's ANSWER against the ceiling before anyone can act on it.
    """
    hosted = [str(item) for item in policy.get("hosted_labels", ["ubuntu-latest"])]
    version = int(policy.get("policy_version", 0))
    decided_at = datetime.now(UTC).isoformat()

    def hosted_result(
        reason: str,
        *,
        labels: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> RouteDecision:
        # `extra` is an explicit mapping rather than **kwargs on purpose: with
        # **kwargs, mypy cannot rule out a caller's dict supplying `labels`
        # itself, which would let a probe payload silently override the chosen
        # runner labels. Making it a separate parameter removes that path.
        chosen = labels if labels else hosted
        return RouteDecision(
            labels=chosen,
            decision="hosted",
            reason=reason,
            policy_version=version,
            decided_at=decided_at,
            inputs={
                "event_name": event_name,
                "head_repo": head_repo,
                "repository": repository,
                "workflow_path": workflow_path,
                "seam_json": seam_json,
                **(extra or {}),
            },
        )

    try:
        ceiling = _parse_labels(seam_json)
        if ceiling is None:
            # An unparseable or unset seam is a config fault, not a licence to
            # pick. Hosted, and say so.
            return hosted_result("probe_error:seam_unparseable")

        # --- S0: the policy pins this workflow hosted, for reasons that
        # outrank capacity entirely (fleet canaries must not share fate with
        # the fleet; ECR pushes need clean egress; fork-only verification).
        if workflow_path in set(allowlist or []):
            return hosted_result("policy_allowlist")

        # --- S1: fork isolation. INVIOLABLE, and deliberately ahead of every
        # capacity signal: untrusted code never reaches self-hosted compute at
        # any idle level (OMN-16683/16684).
        is_fork_pr = event_name == "pull_request" and head_repo != repository
        if is_fork_pr or event_name == "pull_request_target":
            public = _parse_labels(public_json) or hosted
            if _is_self_hosted(public):
                # A misconfigured public variable can never widen a fork onto
                # the fleet; the built-in hosted set wins.
                public = hosted
            return hosted_result("fork_isolation", labels=public)

        # --- S2: the seam is the ceiling. If it names no self-hosted label
        # there is nothing to decide -- return it VERBATIM (not normalised to
        # ubuntu-latest, so a seam naming another hosted image is honoured).
        if not _is_self_hosted(ceiling):
            return RouteDecision(
                labels=ceiling,
                decision="hosted",
                reason="seam_ceiling_hosted",
                policy_version=version,
                decided_at=decided_at,
                inputs={
                    "event_name": event_name,
                    "head_repo": head_repo,
                    "repository": repository,
                    "workflow_path": workflow_path,
                    "seam_json": seam_json,
                },
            )

        # --- S3: probe integrity. Anything that cannot PROVE capacity is
        # hosted, including a shape we did not expect.
        if not isinstance(fleet, dict):
            return hosted_result("probe_error:unexpected_shape")
        if not fleet.get("ok"):
            error_class = str(fleet.get("error") or "unknown")
            return hosted_result(f"probe_error:{error_class}")
        online = fleet.get("online")
        busy = fleet.get("busy")
        if (
            not isinstance(online, int)
            or not isinstance(busy, int)
            or online < 0
            or busy < 0
        ):
            return hosted_result("probe_error:unexpected_shape")

        fleet_inputs = {"fleet": {"online": online, "busy": busy}}

        # --- S4: saturation, on either independent threshold. The idle floor
        # is HEADROOM (a measured CI fan-out is ~47 jobs), not a capacity
        # match; the busy fraction sits well below the observed peak because
        # busy was measured swinging 65 -> 10 of 88 inside two minutes, and a
        # threshold near the peak would flap.
        idle = online - busy
        busy_fraction = (busy / online) if online else 1.0
        if idle < int(policy["min_idle_runners"]) or busy_fraction >= float(
            policy["max_busy_fraction"]
        ):
            return hosted_result(
                "fleet_saturated",
                extra={
                    **fleet_inputs,
                    "idle": idle,
                    "busy_fraction": round(busy_fraction, 4),
                },
            )

        # --- S5: a fleet too small to trust at all, a lower and separate
        # floor from the canary's own offline thresholds.
        if online < int(policy["min_online_runners"]):
            return hosted_result("fleet_degraded", extra=fleet_inputs)

        # --- S6: the lab half. A hosted route job cannot reach the lab
        # directly (the pre-push probe reaches it over ssh to tailnet/RFC1918
        # addresses), so this record arrives asynchronously from the
        # saturation monitor and is FRESHNESS-BOUNDED. Stale is unknown, and
        # unknown is hosted -- never "assume ample".
        if not isinstance(lab, dict) or not lab.get("ok"):
            return hosted_result("lab_unknown", extra=fleet_inputs)
        age = lab.get("age_seconds")
        if not isinstance(age, int) or age > int(policy["lab_record_max_age_seconds"]):
            return hosted_result("lab_unknown", extra=fleet_inputs)
        hosts = lab.get("hosts")
        if not isinstance(hosts, list) or not hosts:
            return hosted_result("lab_unknown", extra=fleet_inputs)
        for host in hosts:
            if not isinstance(host, dict):
                return hosted_result("lab_unknown", extra=fleet_inputs)
            ratio = host.get("ratio")
            free_mem = host.get("free_mem_mib")
            if not isinstance(ratio, (int, float)) or not isinstance(free_mem, int):
                return hosted_result("lab_unknown", extra=fleet_inputs)
            # Load ranks, memory ADMITS (OMN-17392): a host at 0.10x load with
            # 2.5 GiB free is the target that cost an OMN-17316 landing hours
            # of OOM kills.
            if float(ratio) > float(policy["max_lab_load_ratio"]):
                return hosted_result(
                    "lab_saturated",
                    extra={**fleet_inputs, "lab_host": host.get("label")},
                )
            if free_mem < int(policy["min_lab_free_mem_mib"]):
                return hosted_result(
                    "lab_saturated",
                    extra={**fleet_inputs, "lab_host": host.get("label")},
                )

        # --- S7: capacity is available and the seam permits it. The
        # never-widen check is NOT made here -- checking `ceiling` against
        # itself is a tautology that would pass a widened answer. It is made in
        # `decide` below, against the labels this function actually returned.
        return RouteDecision(
            labels=ceiling,
            decision="self_hosted",
            reason="capacity_available",
            policy_version=version,
            decided_at=decided_at,
            inputs={
                "event_name": event_name,
                "head_repo": head_repo,
                "repository": repository,
                "workflow_path": workflow_path,
                "seam_json": seam_json,
                "fleet": {"online": online, "busy": busy},
                "idle": idle,
                "busy_fraction": round(busy_fraction, 4),
                "lab": {"age_seconds": age, "hosts": hosts},
            },
        )
    except Exception as exc:  # noqa: BLE001 -- the boundary is the point
        # A crashing route job must still hand the run a usable runs-on.
        return RouteDecision(
            labels=hosted,
            decision="hosted",
            reason=f"probe_error:internal:{type(exc).__name__}",
            policy_version=version,
            decided_at=decided_at,
            inputs={"event_name": event_name, "workflow_path": workflow_path},
        )


def decide(**kwargs: Any) -> RouteDecision:
    """Choose a runs-on label set. Never raises; every fault degrades to hosted.

    THE NEVER-WIDEN CHECK LIVES HERE, ON THE WAY OUT, and it is the only place
    it can be honest. An earlier build called ``assert_never_widens(ceiling,
    ceiling, hosted)`` from inside the S7 branch and the module docstring
    claimed that as the mechanical guard. It is a TAUTOLOGY -- it compares the
    ceiling with itself and never looks at what was returned. Proven by
    injecting a widening bug into S7: the runtime check passed and the widened
    label set was emitted; only the cross-product sweep in
    ``tests/ci/test_runner_route_decision.py`` caught it. A guard that cannot
    fail is documentation, not enforcement (CLAUDE.md rule 5).

    Checking the ANSWER, on every path, is what makes it enforcement: a future
    edit to any of the eight return points is re-validated here against the
    seam that run actually carried, and a violation degrades to hosted rather
    than raising -- an unroutable run is worse than a hosted one.
    """
    decision = _decide_unchecked(**kwargs)
    policy = kwargs["policy"]
    hosted = [str(item) for item in policy.get("hosted_labels", ["ubuntu-latest"])]
    ceiling = _parse_labels(kwargs.get("seam_json")) or []
    try:
        assert_never_widens(decision.labels, ceiling, hosted)
    except AssertionError as exc:
        return RouteDecision(
            labels=hosted,
            decision="hosted",
            reason="never_widen_violation",
            policy_version=int(policy.get("policy_version", 0)),
            decided_at=decision.decided_at,
            inputs={**decision.inputs, "violation": str(exc)},
        )
    return decision


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--head-repo", default=None)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow-path", required=True)
    parser.add_argument("--seam-json", default=None)
    parser.add_argument("--public-json", default=None)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
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
        "--dry-run",
        action="store_true",
        help="print the decision, write nothing to GITHUB_OUTPUT, always exit 0",
    )
    # GITHUB_API_URL is injected by the Actions runner and names the GitHub REST
    # API this run executes against. It is not a first-party service endpoint and
    # has no routing-authority/integration-catalog entry to resolve from -- the
    # same contract the existing fleet canary already operates under
    # (scripts/ci/runner_fleet_canary.sh: GITHUB_API_URL:-https://api.github.com).
    # probe_fleet() scheme-pins the value to https before issuing any request.
    # fmt: off
    default_api_url = os.environ.get("GITHUB_API_URL", "https://api.github.com")  # url-authority-ok: Actions-injected GitHub REST base, not an ONEX-routed service endpoint; probe_fleet() scheme-pins it to https before any request
    # fmt: on
    parser.add_argument("--api-url", default=default_api_url)
    args = parser.parse_args(argv)

    policy = load_route_policy(args.policy)
    allowlist = hosted_workflows(policy)

    if args.fleet_json is not None:
        try:
            fleet = json.loads(args.fleet_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            fleet = {"ok": False, "error": "unreadable_record"}
    else:
        token = os.environ.get("RUNNER_FLEET_STATUS_TOKEN") or os.environ.get(
            "CROSS_REPO_PAT"
        )
        fleet = probe_fleet(token, str(policy["runner_group"]), args.api_url)

    lab = (
        probe_local_lab_load()
        if args.lab_probe_local
        else read_lab_record(args.lab_record)
    )

    decision = decide(
        event_name=args.event_name,
        head_repo=args.head_repo,
        repository=args.repository,
        workflow_path=args.workflow_path,
        seam_json=args.seam_json,
        public_json=args.public_json,
        fleet=fleet,
        lab=lab,
        policy=policy,
        allowlist=allowlist,
    )

    record = decision.to_record()
    print(json.dumps(record, indent=2, sort_keys=True))
    print(
        f"::notice title=Runner route::{decision.decision} "
        f"({decision.reason}) -> {json.dumps(decision.labels)}"
    )

    if args.dry_run:
        # A dry run computes and reports; it gates nothing and cannot fail a run.
        return 0

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with Path(output_path).open("a", encoding="utf-8") as handle:
            handle.write(f"labels={json.dumps(decision.labels)}\n")
            handle.write(f"decision={decision.decision}\n")
            handle.write(f"reason={decision.reason}\n")
    artifact = Path(
        os.environ.get("RUNNER_ROUTE_ARTIFACT", "runner-route-decision.json")
    )
    artifact.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
