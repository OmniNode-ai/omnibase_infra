#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Stability-lane refresh health-gate + receipt emission [OMN-14263 / OMN-14873].

Standalone health-gate for ``refresh_stability_lane.sh``. Every check is
required (fail-closed AND-of-all-checks, not best-effort):

  1. **Digest changed** -- each of the 4 core services' running container
     image ID differs from the pre-refresh snapshot (a no-op refresh that
     leaves the same image running would otherwise silently "pass").
  2. **Manifest count** -- ``/v1/introspection/manifest`` contract count is
     >= ``--min-contracts`` (a floor, never silently lowered).
  3. **Health endpoint** [strictness fixed in OMN-17563] -- ``/health``
     returns HTTP 200 with a JSON body whose top-level ``status`` is exactly
     ``healthy``. ``degraded``/``unhealthy`` FAIL regardless of any nested
     ``details.healthy`` flag, and a body with no readable ``status`` fails
     closed. Until OMN-17563 this check ORed ``details.healthy`` in, so it
     signed off a lane the lane's own ``--degraded-policy fail`` container
     probe called unhealthy ten minutes later -- see ``health_payload.py``.
  4. **Cluster health** -- ``rpk cluster health`` inside the broker container
     reports healthy.
  5. **Consumer groups Stable** -- every group declared in
     ``consumer_groups_stability.yaml`` reports ``state == "Stable"`` via
     ``rpk group describe``.
  6. **Revision readback** -- the ``org.opencontainers.image.revision`` label
     on each core container equals the intended new ref (or a prefix thereof,
     tolerating short/full SHA differences).
  7. **Partition headroom** [OMN-14013] -- live ``topic_partitions_per_shard``
     cluster config vs. the live total partition count across all topics.
     ``rpk cluster health`` has NO concept of partition-allocation headroom --
     it reported ``Healthy: true`` right up to (and past) the lane's
     allocation ceiling, silently blocking every new topic create. At/over the
     cap is a genuine FAIL (a new topic create WILL fail); crossing the warn
     threshold (default 80%) is surfaced in the report but does not flip
     ``overall`` -- it is an early-warning signal, not by itself a broken
     lane.

``refresh_stability_lane.sh`` calls this script both for the post-refresh gate
and, on a FAIL, again against the rolled-back state to confirm rollback
actually restored health.

Exit codes:
    0 - PASS (all requested checks succeeded)
    1 - FAIL (a genuine check failure -- digest not changed, manifest floor
        not met, unhealthy, cluster unhealthy, a group not Stable, a revision
        mismatch, or partition usage at/over the cap)
    2 - INFRA_ERROR (could not run a check at all -- docker/curl/rpk
        unavailable, container missing, etc.) -- distinguished from a
        genuine FAIL so the caller does not conflate "couldn't check" with
        "checked and it's broken"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

# Sibling module in this same directory. These verifiers are executed as
# scripts (``python scripts/runtime_build/verify_*.py`` or ``uv run python
# <path>``), so ``sys.path[0]`` is this directory and the plain import
# resolves. Sharing the verdict rather than re-copying it is the point:
# OMN-17563 was one defect that had already been duplicated into both files.
from health_payload import (
    HEALTH_POLICY_STATUS_ONLY_STRICT,
    HealthVerdict,
    evaluate_health_body,
    unreachable_verdict,
)

# OCI label stamped from VCS_REF/GIT_SHA at build time (Dockerfile.runtime).
_REVISION_LABEL = "org.opencontainers.image.revision"

DEFAULT_MIN_CONTRACTS = 288
# OMN-14013: fraction of topic_partitions_per_shard in use that triggers a
# visible (non-blocking) WARN. `rpk cluster health` never surfaces this on its
# own -- see the module docstring's check #7.
DEFAULT_PARTITION_WARN_THRESHOLD = 0.8


# ─── Data structures ────────────────────────────────────────────────────────


@dataclass
class ServiceDigestCheck:
    """Digest-changed + revision-readback result for one core service."""

    service: str
    container: str
    pre_image_id: str | None
    post_image_id: str | None
    digest_changed: bool
    revision_label: str | None
    expected_revision: str
    revision_match: bool
    error: str | None = None


@dataclass
class ConsumerGroupCheck:
    """Stable-state result for one declared consumer group."""

    group: str
    state: str | None
    stable: bool
    error: str | None = None


@dataclass
class PartitionHeadroomCheck:
    """Partition-allocation headroom -- invisible to `rpk cluster health` (OMN-14013).

    ``rpk cluster health`` reports on broker/replica state only; it has no
    concept of "no partition-allocation headroom left" and stays green right
    up to (and past) a single-shard broker's ``topic_partitions_per_shard``
    ceiling (OMN-14013: the stability-test lane was observed at 6994/7000,
    and later 7046/7000 -- OVER cap -- with `rpk cluster health` reporting
    `Healthy: true` throughout both times). This check queries the live cap
    and the live total partition count directly so that condition is visible
    BEFORE the next topic-create silently fails.

    ``at_or_over_cap`` participates in the overall health-gate verdict (a
    genuine, checked FAIL: new topic creates WILL fail). ``crossed_warn_threshold``
    is visibility-only and never flips ``overall`` on its own -- unrelated
    historical topic accumulation crossing 80% headroom usage should not
    retroactively fail an otherwise-healthy refresh; it is an early-warning
    signal for a separate topic-retirement/cap-raise action.
    """

    cap: int | None
    total_partitions: int | None
    usage_ratio: float | None
    warn_threshold: float
    at_or_over_cap: bool
    crossed_warn_threshold: bool
    detail: str
    error: str | None = None


@dataclass
class HealthGateReport:
    """Aggregate health-gate report. ``overall`` is the AND of every check.

    ``require_digest_change`` defaults to True (the normal post-refresh gate:
    prove the refresh actually changed the running image). Set False for the
    ROLLBACK re-verification pass -- after a rollback the running image is
    deliberately back to the PRE-refresh image, so "digest changed" is the
    wrong question; that pass instead asserts health/manifest/cluster/groups
    plus a revision match against the OLD (rolled-back-to) revision.
    """

    lane: str
    services: list[ServiceDigestCheck] = field(default_factory=list)
    manifest_count: int | None = None
    manifest_floor: int = DEFAULT_MIN_CONTRACTS
    manifest_ok: bool = False
    health_ok: bool = False
    health_detail: str | None = None
    # OMN-17563 AC-3: the receipt names the policy that produced ``health_ok``
    # and the raw status it saw, so a later reader can tell a strict PASS from
    # a lenient one without re-deriving the whole probe.
    health_status: str | None = None
    health_policy: str = HEALTH_POLICY_STATUS_ONLY_STRICT
    cluster_healthy: bool = False
    cluster_detail: str | None = None
    consumer_groups: list[ConsumerGroupCheck] = field(default_factory=list)
    partition_headroom: PartitionHeadroomCheck | None = None
    errors: list[str] = field(default_factory=list)
    require_digest_change: bool = True

    @property
    def digests_changed(self) -> bool:
        return bool(self.services) and all(s.digest_changed for s in self.services)

    @property
    def revisions_match(self) -> bool:
        return bool(self.services) and all(s.revision_match for s in self.services)

    @property
    def groups_stable(self) -> bool:
        return bool(self.consumer_groups) and all(
            g.stable for g in self.consumer_groups
        )

    @property
    def partition_headroom_ok(self) -> bool:
        """True unless the headroom check ran AND found the lane at/over cap.

        A crossed-warn-threshold-but-below-cap state is still "ok" here by
        design (visibility only, see ``PartitionHeadroomCheck`` docstring); a
        probe failure (``error is not None``) is surfaced via ``errors`` ->
        ``INFRA_ERROR`` instead, mirroring ``check_manifest_count``.
        """
        return (
            self.partition_headroom is None
            or not self.partition_headroom.at_or_over_cap
        )

    @property
    def overall(self) -> str:
        if self.errors:
            return "INFRA_ERROR"
        digest_ok = self.digests_changed if self.require_digest_change else True
        if (
            digest_ok
            and self.manifest_ok
            and self.health_ok
            and self.cluster_healthy
            and self.groups_stable
            and self.revisions_match
            and self.partition_headroom_ok
        ):
            return "PASS"
        return "FAIL"

    def to_dict(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "require_digest_change": self.require_digest_change,
            "digest_changed": self.digests_changed,
            "manifest_count": self.manifest_count,
            "manifest_floor": self.manifest_floor,
            "manifest_ok": self.manifest_ok,
            "health_ok": self.health_ok,
            "health_detail": self.health_detail,
            "health_status": self.health_status,
            "health_policy": self.health_policy,
            "cluster_healthy": self.cluster_healthy,
            "cluster_detail": self.cluster_detail,
            "consumer_groups": {g.group: g.state for g in self.consumer_groups},
            "consumer_groups_stable": self.groups_stable,
            "revision_readback_ok": self.revisions_match,
            "partition_headroom": (
                asdict(self.partition_headroom)
                if self.partition_headroom is not None
                else None
            ),
            "partition_headroom_ok": self.partition_headroom_ok,
            "services": [asdict(s) for s in self.services],
            "errors": self.errors,
            "overall": self.overall,
        }


# ─── Subprocess / HTTP helpers (mockable via `runner`) ──────────────────────


def _run(
    cmd: list[str], *, runner: object | None = None, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    run_fn = runner or subprocess.run
    return run_fn(  # type: ignore[operator, no-any-return]
        cmd, capture_output=True, text=True, timeout=timeout, check=False
    )


def get_image_id(
    container: str, *, runner: object | None = None
) -> tuple[str | None, str | None]:
    """Return the running container's image ID (``docker inspect .Image``)."""
    try:
        result = _run(
            ["docker", "inspect", container, "--format", "{{.Image}}"], runner=runner
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out inspecting {container}"
    except FileNotFoundError:
        return None, "docker command not found"
    if result.returncode != 0:
        return (
            None,
            f"docker inspect failed (exit {result.returncode}): {(result.stderr or '').strip()}",
        )
    image_id = (result.stdout or "").strip()
    if not image_id:
        return None, f"empty image id for {container}"
    return image_id, None


def get_revision_label(
    container: str, *, runner: object | None = None
) -> tuple[str | None, str | None]:
    """Return the ``org.opencontainers.image.revision`` label off *container*."""
    try:
        result = _run(
            [
                "docker",
                "inspect",
                container,
                "--format",
                f'{{{{index .Config.Labels "{_REVISION_LABEL}"}}}}',
            ],
            runner=runner,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out inspecting {container}"
    except FileNotFoundError:
        return None, "docker command not found"
    if result.returncode != 0:
        return (
            None,
            f"docker inspect failed (exit {result.returncode}): {(result.stderr or '').strip()}",
        )
    revision = (result.stdout or "").strip()
    if not revision or revision == "<no value>":
        return None, f"no {_REVISION_LABEL} label on {container}"
    return revision, None


def _revisions_match(actual: str, expected: str) -> bool:
    """Tolerate short/full SHA length differences (prefix match), never a
    genuinely different revision."""
    a, b = actual.strip().lower(), expected.strip().lower()
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a)


def check_service_digest(
    service: str,
    container: str,
    pre_image_id: str | None,
    expected_revision: str,
    *,
    runner: object | None = None,
) -> ServiceDigestCheck:
    post_image_id, image_err = get_image_id(container, runner=runner)
    revision, rev_err = get_revision_label(container, runner=runner)
    error = image_err or rev_err
    digest_changed = bool(
        pre_image_id and post_image_id and pre_image_id != post_image_id
    )
    revision_match = bool(revision) and _revisions_match(
        revision or "", expected_revision
    )
    return ServiceDigestCheck(
        service=service,
        container=container,
        pre_image_id=pre_image_id,
        post_image_id=post_image_id,
        digest_changed=digest_changed,
        revision_label=revision,
        expected_revision=expected_revision,
        revision_match=revision_match,
        error=error,
    )


def check_manifest_count(
    manifest_url: str, min_contracts: int, *, opener: object | None = None
) -> tuple[int | None, str | None]:
    """Fetch the introspection manifest and count contracts."""
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(manifest_url, timeout=10) as resp:  # type: ignore[operator]
            raw = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        return None, f"manifest fetch failed: {exc}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"manifest not valid JSON: {exc}"
    if isinstance(payload, list):
        contracts = payload
    elif isinstance(payload, dict):
        contracts = payload.get("contracts", [])
    else:
        return None, "manifest payload has unexpected shape"
    return len(contracts), None


def check_health(health_url: str, *, opener: object | None = None) -> HealthVerdict:
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(health_url, timeout=10) as resp:  # type: ignore[operator]
            status_code = getattr(resp, "status", 200)
            raw = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        return unreachable_verdict(f"health fetch failed: {exc}")
    if status_code and status_code != 200:
        return unreachable_verdict(f"health endpoint returned HTTP {status_code}")
    return evaluate_health_body(raw)


def check_cluster_health(
    broker_container: str, *, runner: object | None = None
) -> tuple[bool, str]:
    try:
        result = _run(
            [
                "docker",
                "exec",
                broker_container,
                "rpk",
                "cluster",
                "health",
                "-X",
                "brokers=redpanda:9092",
            ],
            runner=runner,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out probing cluster health via {broker_container}"
    except FileNotFoundError:
        return False, "docker command not found"
    if result.returncode != 0:
        return (
            False,
            f"rpk cluster health failed (exit {result.returncode}): {(result.stderr or '').strip()}",
        )
    stdout = result.stdout or ""
    # `rpk cluster health` prints a "Healthy:" line whose padding varies by rpk
    # version (observed: "Healthy:                          true" on the live
    # .201 broker) -- match on the label + boolean via regex, not a hardcoded
    # column width, so a whitespace difference never produces a false FAIL.
    healthy = bool(re.search(r"Healthy:\s*true\b", stdout, re.IGNORECASE)) or (
        "cluster is healthy" in stdout.lower()
    )
    return healthy, stdout.strip().splitlines()[0] if stdout.strip() else "no output"


def _get_partition_cap(
    broker_container: str, *, runner: object | None = None
) -> tuple[int | None, str | None]:
    """Read the live ``topic_partitions_per_shard`` cluster config value."""
    try:
        result = _run(
            [
                "docker",
                "exec",
                broker_container,
                "rpk",
                "cluster",
                "config",
                "get",
                "topic_partitions_per_shard",
            ],
            runner=runner,
        )
    except subprocess.TimeoutExpired:
        return (
            None,
            f"timed out reading topic_partitions_per_shard via {broker_container}",
        )
    except FileNotFoundError:
        return None, "docker command not found"
    if result.returncode != 0:
        return (
            None,
            f"rpk cluster config get failed (exit {result.returncode}): {(result.stderr or '').strip()}",
        )
    stdout = (result.stdout or "").strip()
    match = re.search(r"(\d+)", stdout)
    if not match:
        return (
            None,
            f"could not parse topic_partitions_per_shard from output: {stdout!r}",
        )
    return int(match.group(1)), None


def _get_total_partitions(
    broker_container: str, *, runner: object | None = None
) -> tuple[int | None, str | None]:
    """Sum the PARTITIONS column of `rpk topic list` across every topic.

    `rpk topic list` has no documented `-f json` mode verified against this
    lane's rpk version (mirrors the same absence already noted for `rpk group
    describe` above) -- parse the fixed-width plain-text table instead,
    locating the PARTITIONS column by its header label rather than a
    hardcoded position, since column position/order in rpk table output is
    not guaranteed stable across subcommands or rpk versions.
    """
    try:
        result = _run(
            ["docker", "exec", broker_container, "rpk", "topic", "list"],
            runner=runner,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out listing topics via {broker_container}"
    except FileNotFoundError:
        return None, "docker command not found"
    if result.returncode != 0:
        return (
            None,
            f"rpk topic list failed (exit {result.returncode}): {(result.stderr or '').strip()}",
        )
    lines = (result.stdout or "").splitlines()
    if not lines:
        return None, "rpk topic list produced no output"
    header = lines[0].split()
    try:
        partitions_idx = header.index("PARTITIONS")
    except ValueError:
        return None, f"rpk topic list header missing PARTITIONS column: {lines[0]!r}"
    total = 0
    for line in lines[1:]:
        fields = line.split()
        if len(fields) <= partitions_idx:
            continue
        try:
            total += int(fields[partitions_idx])
        except ValueError:
            continue
    return total, None


def check_partition_headroom(
    broker_container: str,
    *,
    runner: object | None = None,
    warn_threshold: float = DEFAULT_PARTITION_WARN_THRESHOLD,
) -> PartitionHeadroomCheck:
    """Live partition-allocation headroom vs. `topic_partitions_per_shard` (OMN-14013)."""
    cap, cap_err = _get_partition_cap(broker_container, runner=runner)
    total, total_err = _get_total_partitions(broker_container, runner=runner)
    error = cap_err or total_err
    if error is not None or cap is None or total is None or cap <= 0:
        detail = error or ("cap reported as <= 0" if cap is not None else "unknown")
        return PartitionHeadroomCheck(
            cap=cap,
            total_partitions=total,
            usage_ratio=None,
            warn_threshold=warn_threshold,
            at_or_over_cap=False,
            crossed_warn_threshold=False,
            detail=f"could not determine partition headroom: {detail}",
            error=error or detail,
        )
    usage_ratio = total / cap
    at_or_over_cap = total >= cap
    crossed_warn_threshold = usage_ratio >= warn_threshold
    if at_or_over_cap:
        detail = (
            f"{total}/{cap} partitions ({usage_ratio:.1%}) -- AT OR OVER CAP: "
            "new topic creates will fail"
        )
    elif crossed_warn_threshold:
        detail = (
            f"{total}/{cap} partitions ({usage_ratio:.1%}) -- crossed "
            f"{warn_threshold:.0%} warn threshold"
        )
    else:
        detail = f"{total}/{cap} partitions ({usage_ratio:.1%})"
    return PartitionHeadroomCheck(
        cap=cap,
        total_partitions=total,
        usage_ratio=usage_ratio,
        warn_threshold=warn_threshold,
        at_or_over_cap=at_or_over_cap,
        crossed_warn_threshold=crossed_warn_threshold,
        detail=detail,
        error=None,
    )


def check_consumer_group(
    broker_container: str, group: str, *, runner: object | None = None
) -> ConsumerGroupCheck:
    """Describe one consumer group and report its ``STATE``.

    ``rpk group describe`` has NO ``-f json`` output mode (unlike most other
    ``rpk`` subcommands) -- verified live against the .201 broker's rpk
    version, which rejects ``-f`` as an unknown flag. Output is a fixed-width
    plain-text key/value block:

        GROUP        <name>
        COORDINATOR  0
        STATE        Stable
        BALANCER
        MEMBERS      1
        TOTAL-LAG    0

    Parsed by matching the ``STATE`` line, not by JSON decoding.
    """
    try:
        result = _run(
            [
                "docker",
                "exec",
                broker_container,
                "rpk",
                "group",
                "describe",
                group,
                "-X",
                "brokers=redpanda:9092",
            ],
            runner=runner,
        )
    except subprocess.TimeoutExpired:
        return ConsumerGroupCheck(
            group=group, state=None, stable=False, error="timed out"
        )
    except FileNotFoundError:
        return ConsumerGroupCheck(
            group=group, state=None, stable=False, error="docker not found"
        )
    if result.returncode != 0:
        return ConsumerGroupCheck(
            group=group,
            state=None,
            stable=False,
            error=f"rpk group describe failed (exit {result.returncode}): {(result.stderr or '').strip()}",
        )
    state: str | None = None
    for line in (result.stdout or "").splitlines():
        match = re.match(r"^STATE\s+(\S+)", line)
        if match:
            state = match.group(1)
            break
    if state is None:
        return ConsumerGroupCheck(
            group=group,
            state=None,
            stable=False,
            error="no STATE line found in rpk group describe output",
        )
    # "Stable" (actively balanced, member(s) connected) and "Empty" (group is
    # REGISTERED with the broker -- the topic subscription is known -- but has
    # zero currently-connected members) are both healthy for a demand-driven
    # consumer that only joins when there is work to process. "Dead" (or any
    # other/absent state) means the group coordinator does not know this
    # group at all -- the actual "silent wiring death" signal
    # (reference_silent_wiring_death_mechanisms: a consumer that never
    # registered a group) -- and stays a hard failure.
    return ConsumerGroupCheck(
        group=group, state=state, stable=(state in ("Stable", "Empty"))
    )


def check_consumer_group_with_retry(
    broker_container: str,
    group: str,
    *,
    runner: object | None = None,
    attempts: int = 10,
    interval_seconds: float = 3.0,
    sleep_fn: object | None = None,
) -> ConsumerGroupCheck:
    """Retry ``check_consumer_group`` for a bounded window.

    Right after a `--force-recreate`, a consumer's Kafka client needs a few
    seconds to reconnect and rejoin its group -- querying immediately can
    observe a transient ``Empty``/``Dead`` state on an otherwise-healthy
    refresh. Mirrors the retry shape of `assert_broker_reachable()` in
    deploy-runtime.sh (bounded attempts + fixed interval, not an unbounded
    poll). Returns the LAST observed result; a group that never reaches
    Stable within the window is a genuine, accurately-reported finding.
    """
    sleep = sleep_fn or time.sleep
    result = ConsumerGroupCheck(group=group, state=None, stable=False)
    for attempt in range(1, attempts + 1):
        result = check_consumer_group(broker_container, group, runner=runner)
        if result.stable:
            return result
        if attempt < attempts:
            sleep(interval_seconds)  # type: ignore[operator]
    return result


def load_declared_consumer_groups(path: Path) -> list[str]:
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return [entry["name"] for entry in data.get("consumer_groups", [])]


# ─── Orchestration ───────────────────────────────────────────────────────────


CORE_SERVICES: dict[str, str] = {
    "omninode-runtime": "omninode-stability-test-runtime",
    "runtime-effects": "omninode-stability-test-runtime-effects",
    "runtime-worker": "omninode-stability-test-runtime-worker",
    "projection-api": "omnimarket-stability-test-projection-api",
}


def run_health_gate(
    *,
    lane: str,
    pre_image_ids: dict[str, str],
    expected_revision: str,
    manifest_url: str,
    health_url: str,
    broker_container: str,
    min_contracts: int,
    consumer_groups: list[str],
    runner: object | None = None,
    opener: object | None = None,
    require_digest_change: bool = True,
    sleep_fn: object | None = None,
    partition_warn_threshold: float = DEFAULT_PARTITION_WARN_THRESHOLD,
) -> HealthGateReport:
    report = HealthGateReport(
        lane=lane,
        manifest_floor=min_contracts,
        require_digest_change=require_digest_change,
    )

    for service, container in CORE_SERVICES.items():
        report.services.append(
            check_service_digest(
                service,
                container,
                pre_image_ids.get(service),
                expected_revision,
                runner=runner,
            )
        )

    count, err = check_manifest_count(manifest_url, min_contracts, opener=opener)
    report.manifest_count = count
    if err is not None:
        report.errors.append(err)
    else:
        report.manifest_ok = count is not None and count >= min_contracts

    health_verdict = check_health(health_url, opener=opener)
    report.health_ok = health_verdict.ok
    report.health_detail = health_verdict.detail
    report.health_status = health_verdict.status
    report.health_policy = health_verdict.policy

    cluster_healthy, cluster_detail = check_cluster_health(
        broker_container, runner=runner
    )
    report.cluster_healthy = cluster_healthy
    report.cluster_detail = cluster_detail

    partition_headroom = check_partition_headroom(
        broker_container, runner=runner, warn_threshold=partition_warn_threshold
    )
    report.partition_headroom = partition_headroom
    if partition_headroom.error is not None:
        report.errors.append(partition_headroom.error)

    for group in consumer_groups:
        report.consumer_groups.append(
            check_consumer_group_with_retry(
                broker_container, group, runner=runner, sleep_fn=sleep_fn
            )
        )

    return report


# ─── Receipt ─────────────────────────────────────────────────────────────────


def build_receipt(
    *,
    lane: str,
    prior_refs: dict[str, str],
    new_refs: dict[str, str],
    ancestry_ok: bool,
    ancestry_commands: list[str],
    build_scope: list[str],
    gate: HealthGateReport,
    rollback_triggered: bool,
    rollback_gate: HealthGateReport | None,
) -> dict[str, object]:
    result: str
    if gate.overall == "PASS":
        result = "SUCCESS"
    elif (
        rollback_triggered
        and rollback_gate is not None
        and rollback_gate.overall == "PASS"
    ):
        result = "FAILED_ROLLED_BACK"
    else:
        result = "FAILED"

    return {
        "ts_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lane": lane,
        "prior_refs": prior_refs,
        "new_refs": new_refs,
        "ancestry_proof": {
            "merge_base_is_ancestor": ancestry_ok,
            "commands": ancestry_commands,
        },
        "build_scope": build_scope,
        "health_gate": gate.to_dict(),
        "rollback": {
            "triggered": rollback_triggered,
            "gate": rollback_gate.to_dict() if rollback_gate is not None else None,
        },
        "result": result,
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", default="stability-test")
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument(
        "--pre-image-ids",
        required=True,
        help='JSON dict of {service: pre_refresh_image_id}, e.g. \'{"omninode-runtime": "sha256:..."}\'',
    )
    # fallback-ok: fixed stability-test lane ops-tooling defaults (port 18085); the
    # automated caller (refresh_stability_lane.sh) always passes these explicitly.
    # url-authority-ok: no contract/routing-authority exists for this standalone
    # ops script; the lane's port is a fixed, documented convention.
    manifest_url_default = "http://localhost:18085/v1/introspection/manifest"  # fallback-ok  # url-authority-ok: fixed lane port, no routing authority applies
    health_url_default = "http://localhost:18085/health"  # fallback-ok  # url-authority-ok: fixed lane port, no routing authority applies
    parser.add_argument("--manifest-url", default=manifest_url_default)
    parser.add_argument("--health-url", default=health_url_default)
    parser.add_argument(
        "--broker-container", default="omnibase-infra-stability-test-redpanda"
    )
    parser.add_argument("--min-contracts", type=int, default=DEFAULT_MIN_CONTRACTS)
    parser.add_argument(
        "--consumer-groups-file",
        default=str(Path(__file__).resolve().parent / "consumer_groups_stability.yaml"),
    )
    parser.add_argument(
        "--partition-warn-threshold",
        type=float,
        default=DEFAULT_PARTITION_WARN_THRESHOLD,
        help=(
            "Fraction of topic_partitions_per_shard in use that triggers a "
            "visible (non-blocking) partition-headroom WARN [OMN-14013]. "
            # argparse runs `help % params` when rendering, so a literal
            # percent must be escaped or `--help` dies with
            # "ValueError: incomplete format" (it did, on every invocation,
            # until OMN-17563 -- an operator could not read this gate's own
            # usage text).
            f"Default {DEFAULT_PARTITION_WARN_THRESHOLD * 100:.0f}%%."
        ),
    )
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--no-require-digest-change",
        action="store_false",
        dest="require_digest_change",
        default=True,
        help=(
            "Skip the digest-changed requirement (use for the POST-ROLLBACK "
            "re-verification pass, where the running image is deliberately "
            "back to the pre-refresh image)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        pre_image_ids = json.loads(args.pre_image_ids)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --pre-image-ids is not valid JSON: {exc}", file=sys.stderr)
        return 2

    consumer_groups = load_declared_consumer_groups(Path(args.consumer_groups_file))

    report = run_health_gate(
        lane=args.lane,
        pre_image_ids=pre_image_ids,
        expected_revision=args.expected_revision,
        manifest_url=args.manifest_url,
        health_url=args.health_url,
        broker_container=args.broker_container,
        min_contracts=args.min_contracts,
        consumer_groups=consumer_groups,
        require_digest_change=args.require_digest_change,
        partition_warn_threshold=args.partition_warn_threshold,
    )

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Health gate for lane={report.lane}: {report.overall}")
        print(f"  digest_changed={report.digests_changed}")
        print(
            f"  manifest_count={report.manifest_count} (floor={report.manifest_floor}) ok={report.manifest_ok}"
        )
        print(f"  health_ok={report.health_ok} ({report.health_detail})")
        print(f"  cluster_healthy={report.cluster_healthy} ({report.cluster_detail})")
        if report.partition_headroom is not None:
            ph = report.partition_headroom
            print(
                f"  partition_headroom: {ph.detail}"
                + (f" error={ph.error}" if ph.error else "")
            )
        for g in report.consumer_groups:
            print(
                f"  group {g.group}: state={g.state} stable={g.stable}"
                + (f" error={g.error}" if g.error else "")
            )
        for s in report.services:
            print(
                f"  service {s.service}: digest_changed={s.digest_changed} "
                f"revision_match={s.revision_match} (label={s.revision_label})"
                + (f" error={s.error}" if s.error else "")
            )
        if report.errors:
            print(f"  errors={report.errors}")

    if report.overall == "PASS":
        return 0
    if report.overall == "INFRA_ERROR":
        return 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
