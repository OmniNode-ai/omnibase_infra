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
  5. **Consumer groups** [derivation fixed in OMN-15837] -- the declared set
     is DERIVED at gate time, never hand-pinned: contract identities from the
     ``/v1/introspection/manifest`` payload the refreshed image itself serves
     (main + effects runtimes), plus the lane compose file's literal
     ``KAFKA_CONSUMER_GROUP`` values for the standalone projection writers.
     Reconciled against ONE ``rpk group list`` call, which is the only honest
     answer to "does this group exist" -- ``rpk group describe`` reports
     ``Dead`` for a name that was never a group at all. ``Stable``/``Empty``
     are healthy; a ``Dead`` group with no members and no lag is a RETIRED
     identity (logged, not a failure -- that is what a version bump or a
     re-home leaves behind); a group that lost its members while still holding
     lag FAILS; and the fraction of derived identities that are live must meet
     ``--min-derived-coverage``. Until OMN-15837 this list was a hand-copied,
     version-pinned YAML that went stale twice (fd4a84b1c, then OMN-16753) and
     rolled back a healthy refresh both times.
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
        not met, unhealthy, cluster unhealthy, a consumer group stalled with
        lag or a declared writer group missing, derived-group coverage below
        the floor, a revision mismatch, or partition usage at/over the cap)
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
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Sibling module in this same directory. These verifiers are executed as
# scripts (``python scripts/runtime_build/verify_*.py`` or ``uv run python
# <path>``), so ``sys.path[0]`` is this directory and the plain import
# resolves. Sharing the verdict rather than re-copying it is the point:
# OMN-17563 was one defect that had already been duplicated into both files.
from declared_consumer_groups import (
    ConsumerGroupAudit,
    DerivationError,
    DerivedGroupKey,
    GroupDescription,
    NonContractGroup,
    derive_compose_declared_groups,
    derive_expected_keys,
    load_non_contract_groups,
    parse_group_describe,
    parse_group_list,
    reconcile,
)
from health_payload import (
    DEFAULT_MAX_VERDICT_AGE,
    HEALTH_POLICY_STATUS_ONLY_STRICT,
    HealthVerdict,
    default_max_verdict_age,
    derive_verdict_wait_bound,
    evaluate_health_body,
    unreachable_verdict,
    wait_for_verdict,
)

# OCI label stamped from VCS_REF/GIT_SHA at build time (Dockerfile.runtime).
_REVISION_LABEL = "org.opencontainers.image.revision"

DEFAULT_MIN_CONTRACTS = 288
# OMN-15837: fraction of contract-derived consumer-group identities that must be
# live on the broker. A floor, never silently lowered -- the same discipline as
# ``DEFAULT_MIN_CONTRACTS``. Measured on the live stability lane 2026-09-08:
# 604/611 = 0.989. The seven absentees are contracts wired through the
# core-runtime single-owner path or whose topics are owned elsewhere; the floor
# exists to catch a wiring COLLAPSE (a runtime that subscribed to nothing),
# which is the failure mode the old hand-pinned list was reaching for and kept
# mis-reporting.
DEFAULT_MIN_DERIVED_COVERAGE = 0.90
# Bounded re-read window after a --force-recreate: a consumer's Kafka client
# needs a few seconds to rejoin its group, so an immediately-observed absence is
# not yet a finding. Same shape as ``assert_broker_reachable()`` in
# deploy-runtime.sh -- bounded attempts at a fixed interval, never a poll.
DEFAULT_GROUP_AUDIT_ATTEMPTS = 10
DEFAULT_GROUP_AUDIT_INTERVAL_SECONDS = 3.0
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
    # OMN-17624 AC-3: the bound that was waited, with the arithmetic behind it.
    # A PASS after a 60s wait and a PASS after 360s are different claims.
    verdict_wait: str | None = None
    cluster_healthy: bool = False
    cluster_detail: str | None = None
    group_audit: ConsumerGroupAudit | None = None
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
        """True when the derived declared set reconciled cleanly.

        ``None`` (the audit never ran) is deliberately NOT healthy: a gate that
        could not derive its own expectation has not passed.
        """
        return self.group_audit is not None and self.group_audit.ok

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
            "verdict_wait": self.verdict_wait,
            "cluster_healthy": self.cluster_healthy,
            "cluster_detail": self.cluster_detail,
            "consumer_groups": (
                self.group_audit.to_dict() if self.group_audit is not None else None
            ),
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


def fetch_manifest(
    manifest_url: str, *, opener: object | None = None
) -> tuple[dict[str, object] | None, str | None]:
    """Fetch one ``/v1/introspection/manifest`` payload.

    Split out of ``check_manifest_count`` by OMN-15837 so the SAME fetched
    payload feeds both the contract-count floor and the consumer-group
    derivation -- the declared set must describe the image that is actually
    running, and a second fetch could observe a different one.
    """
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(manifest_url, timeout=10) as resp:  # type: ignore[operator]
            raw = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        return None, f"manifest fetch failed ({manifest_url}): {exc}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"manifest not valid JSON ({manifest_url}): {exc}"
    if isinstance(payload, list):
        return {"contracts": payload}, None
    if isinstance(payload, dict):
        return payload, None
    return None, f"manifest payload has unexpected shape ({manifest_url})"


def count_manifest_contracts(payload: dict[str, object]) -> int:
    contracts = payload.get("contracts", [])
    return len(contracts) if isinstance(contracts, list) else 0


def check_manifest_count(
    manifest_url: str, min_contracts: int, *, opener: object | None = None
) -> tuple[int | None, str | None]:
    """Fetch the introspection manifest and count contracts."""
    _ = min_contracts
    payload, err = fetch_manifest(manifest_url, opener=opener)
    if err is not None or payload is None:
        return None, err
    return count_manifest_contracts(payload), None


def check_health(
    health_url: str,
    *,
    opener: object | None = None,
    require_verdict: bool = False,
    max_verdict_age_seconds: float | None = None,
) -> HealthVerdict:
    """Probe ``/health`` once and report what it said.

    OMN-17624 deliberately does NOT flip this default. This function answers
    "what does /health report", and callers asking only that keep getting it.
    The verdict REQUIREMENT belongs to the gate -- see
    ``check_health_with_retry`` -- because that is the path whose receipt
    underwrites the ``stability-proven`` premise of a prod-promotion grant.
    Moving the policy here would change the meaning of every caller to fix one.
    """
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(health_url, timeout=10) as resp:  # type: ignore[operator]
            status_code = getattr(resp, "status", 200)
            raw = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        return unreachable_verdict(f"health fetch failed: {exc}")
    if status_code and status_code != 200:
        return unreachable_verdict(f"health endpoint returned HTTP {status_code}")
    return evaluate_health_body(
        raw,
        require_verdict=require_verdict,
        max_verdict_age_seconds=max_verdict_age_seconds,
    )


def check_health_with_retry(
    health_url: str,
    *,
    opener: object | None = None,
    require_verdict: bool = True,
    max_verdict_age_seconds: float | None | object = DEFAULT_MAX_VERDICT_AGE,
    check_interval_seconds: float = 300.0,
    boot_grace_seconds: float = 120.0,
    sleep_fn: object | None = None,
) -> tuple[HealthVerdict, str]:
    """Probe until a monitor verdict exists, for a bounded window (OMN-17624).

    AC5: this cannot hang -- it fails after a stated window. A profile running
    no monitor opts out via ``require_verdict=False`` at the call site, and the
    returned description records that it did. Absence is never inferred as
    permission: ``service_kernel`` starts the monitor only ``if use_kafka`` and
    swallows a start failure, so a crashed monitor and a deliberately absent
    one are indistinguishable at this boundary.
    """
    if not require_verdict:
        verdict = check_health(health_url, opener=opener, require_verdict=False)
        return verdict, "verdict_wait: skipped (require_verdict=False, opted out)"

    bound = derive_verdict_wait_bound(
        check_interval_seconds=check_interval_seconds,
        boot_grace_seconds=boot_grace_seconds,
    )
    # OMN-17624 review (omnibase_infra#3208): freshness must NOT be opt-in.
    # A monitor that publishes one verdict and then crashes serves that same
    # verdict forever; a gate with no ceiling accepts it forever, which turns
    # this fix's blind window into "before first verdict, plus always".
    if max_verdict_age_seconds is DEFAULT_MAX_VERDICT_AGE:
        effective_max_age: float | None = default_max_verdict_age(
            check_interval_seconds
        )
    elif max_verdict_age_seconds is None or isinstance(
        max_verdict_age_seconds, int | float
    ):
        effective_max_age = max_verdict_age_seconds
    else:
        raise TypeError("max_verdict_age_seconds must be a number, None, or omitted")

    def _probe() -> HealthVerdict:
        return check_health(
            health_url,
            opener=opener,
            require_verdict=True,
            max_verdict_age_seconds=effective_max_age,
        )

    # Unpacked rather than returned directly: health_payload is imported by
    # plain sibling name, so its types resolve to Any here and a bare
    # passthrough trips no-any-return. An explicit tuple keeps the
    # annotation honest without a suppression.
    verdict, described = wait_for_verdict(_probe, bound=bound, sleep_fn=sleep_fn)
    return verdict, described


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


def _rpk(
    broker_container: str, args: list[str], *, runner: object | None = None
) -> tuple[str | None, str | None]:
    """Run one ``rpk`` subcommand inside the broker container."""
    try:
        result = _run(
            [
                "docker",
                "exec",
                broker_container,
                "rpk",
                *args,
                "-X",
                "brokers=redpanda:9092",
            ],
            runner=runner,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out running rpk {' '.join(args)}"
    except FileNotFoundError:
        return None, "docker command not found"
    if result.returncode != 0:
        return None, (
            f"rpk {' '.join(args)} failed (exit {result.returncode}): "
            f"{(result.stderr or '').strip()}"
        )
    return result.stdout or "", None


def list_consumer_groups(
    broker_container: str, *, runner: object | None = None
) -> tuple[dict[str, str] | None, str | None]:
    """``rpk group list`` -> ``{group: state}``.

    This is the ONLY honest existence probe. ``rpk group describe`` answers
    ``STATE Dead / MEMBERS 0 / TOTAL-LAG 0`` for a name that was never a group,
    which is byte-identical to a real group that died -- the ambiguity that made
    a stale hand-pinned name read as a hard failure twice (fd4a84b1c, OMN-16753).
    One list call also removes the per-group describe fan-out: 611 derived
    identities are reconciled from a single command.
    """
    stdout, err = _rpk(broker_container, ["group", "list"], runner=runner)
    if err is not None or stdout is None:
        return None, err
    return parse_group_list(stdout), None


def describe_consumer_group(
    broker_container: str, group: str, *, runner: object | None = None
) -> GroupDescription:
    """``rpk group describe`` -> ``STATE`` / ``MEMBERS`` / ``TOTAL-LAG``.

    ``rpk group describe`` has NO ``-f json`` output mode (unlike most other
    ``rpk`` subcommands) -- verified live against the .201 broker's rpk version,
    which rejects ``-f`` as an unknown flag. Output is a fixed-width plain-text
    key/value block followed by per-partition rows:

        GROUP        <name>
        COORDINATOR  0
        STATE        Stable
        BALANCER     roundrobin
        MEMBERS      1
        TOTAL-LAG    0

    Called only for a group the list already reported as NOT healthy, to decide
    retired (no members, no lag) vs stalled (members lost, lag retained).
    """
    stdout, err = _rpk(broker_container, ["group", "describe", group], runner=runner)
    if err is not None or stdout is None:
        return GroupDescription(state=None, members=None, total_lag=None, error=err)
    return parse_group_describe(stdout)


def build_declared_group_inputs(
    *,
    lane: str,
    manifest_payloads: Sequence[dict[str, object]],
    declared_groups_file: Path,
    compose_file: Path | None,
) -> tuple[tuple[DerivedGroupKey, ...], tuple[NonContractGroup, ...]]:
    """Derive ``(contract_keys, non_contract_groups)`` or raise ``DerivationError``.

    Fail-closed by construction: there is no fallback to a static list. If the
    manifests are unusable, the declared-groups file is malformed, or the lane
    compose file cannot be read, the caller turns the raised cause into an
    INFRA_ERROR naming it. A gate that cannot derive its own expectation has not
    passed -- it has not run.
    """
    if not manifest_payloads:
        raise DerivationError(
            "no introspection manifest could be fetched from the lane, so the "
            "declared consumer-group set cannot be derived from the running image"
        )
    contract_keys = derive_expected_keys(manifest_payloads, env=lane)
    non_contract = list(load_non_contract_groups(declared_groups_file))
    if compose_file is not None:
        non_contract.extend(derive_compose_declared_groups(compose_file, env=lane))
    deduped = {group.name: group for group in non_contract}
    return contract_keys, tuple(deduped[name] for name in sorted(deduped))


def run_consumer_group_audit(
    broker_container: str,
    *,
    lane: str,
    contract_keys: Sequence[DerivedGroupKey],
    non_contract: Sequence[NonContractGroup],
    min_coverage: float,
    runner: object | None = None,
    attempts: int = DEFAULT_GROUP_AUDIT_ATTEMPTS,
    interval_seconds: float = DEFAULT_GROUP_AUDIT_INTERVAL_SECONDS,
    sleep_fn: object | None = None,
) -> ConsumerGroupAudit:
    """Reconcile the derived declared set against the live broker, with retry.

    Retries the WHOLE reconciliation rather than a single group: right after a
    ``--force-recreate`` every consumer is mid-rejoin, so a first pass can see a
    dozen identities transiently absent on an otherwise-healthy refresh.
    Returns the LAST observed audit; an audit that never reconciles inside the
    window is a genuine, accurately-reported finding.
    """
    sleep = sleep_fn or time.sleep
    audit = ConsumerGroupAudit(env=lane, min_coverage=min_coverage)
    for attempt in range(1, attempts + 1):
        live, err = list_consumer_groups(broker_container, runner=runner)
        if err is not None or live is None:
            audit = ConsumerGroupAudit(env=lane, min_coverage=min_coverage)
            audit.errors.append(err or "rpk group list produced no output")
        else:
            audit = reconcile(
                env=lane,
                derived=contract_keys,
                non_contract=non_contract,
                live=live,
                describe=lambda group: describe_consumer_group(
                    broker_container, group, runner=runner
                ),
                min_coverage=min_coverage,
            )
        if audit.ok:
            return audit
        if attempt < attempts:
            sleep(interval_seconds)  # type: ignore[operator]
    return audit


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
    declared_groups_file: Path,
    effects_manifest_url: str | None = None,
    compose_file: Path | None = None,
    min_derived_coverage: float = DEFAULT_MIN_DERIVED_COVERAGE,
    runner: object | None = None,
    opener: object | None = None,
    require_digest_change: bool = True,
    sleep_fn: object | None = None,
    partition_warn_threshold: float = DEFAULT_PARTITION_WARN_THRESHOLD,
    require_verdict: bool = True,
    max_verdict_age_seconds: float | None = None,
    health_check_interval_seconds: float = 300.0,
    health_boot_grace_seconds: float = 120.0,
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

    manifest_payloads: list[dict[str, object]] = []
    main_manifest, err = fetch_manifest(manifest_url, opener=opener)
    if err is not None or main_manifest is None:
        report.errors.append(err or "manifest fetch returned no payload")
    else:
        manifest_payloads.append(main_manifest)
        count = count_manifest_contracts(main_manifest)
        report.manifest_count = count
        report.manifest_ok = count >= min_contracts

    # OMN-15837: the effects runtime serves its OWN profile-filtered manifest.
    # Its contracts mint real consumer groups (273 live on the stability lane)
    # that the main runtime's manifest does not describe at all, so deriving the
    # declared set from the main manifest alone would under-declare the effects
    # half of the lane. Not fetched -> fail closed, never silently narrowed.
    if effects_manifest_url:
        effects_manifest, effects_err = fetch_manifest(
            effects_manifest_url, opener=opener
        )
        if effects_err is not None or effects_manifest is None:
            report.errors.append(
                effects_err or "effects manifest fetch returned no payload"
            )
        else:
            manifest_payloads.append(effects_manifest)

    health_verdict, verdict_wait = check_health_with_retry(
        health_url,
        opener=opener,
        require_verdict=require_verdict,
        max_verdict_age_seconds=max_verdict_age_seconds,
        check_interval_seconds=health_check_interval_seconds,
        boot_grace_seconds=health_boot_grace_seconds,
        sleep_fn=sleep_fn,
    )
    report.health_ok = health_verdict.ok
    report.health_detail = health_verdict.detail
    report.health_status = health_verdict.status
    report.health_policy = health_verdict.policy
    report.verdict_wait = verdict_wait

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

    try:
        contract_keys, non_contract = build_declared_group_inputs(
            lane=lane,
            manifest_payloads=manifest_payloads,
            declared_groups_file=declared_groups_file,
            compose_file=compose_file,
        )
    except DerivationError as exc:
        # Fail CLOSED and name the cause. Falling back to the retired static
        # list is exactly the behaviour OMN-15837 removes.
        report.errors.append(f"declared consumer-group derivation failed: {exc}")
        return report

    report.group_audit = run_consumer_group_audit(
        broker_container,
        lane=lane,
        contract_keys=contract_keys,
        non_contract=non_contract,
        min_coverage=min_derived_coverage,
        runner=runner,
        sleep_fn=sleep_fn,
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
    effects_manifest_url_default = "http://localhost:18086/v1/introspection/manifest"  # fallback-ok  # url-authority-ok: fixed lane port, no routing authority applies
    parser.add_argument("--manifest-url", default=manifest_url_default)
    parser.add_argument("--health-url", default=health_url_default)
    parser.add_argument(
        "--effects-manifest-url",
        default=effects_manifest_url_default,
        help=(
            "Introspection manifest of the EFFECTS runtime [OMN-15837]. Its "
            "profile-filtered contracts mint consumer groups the main "
            "runtime's manifest never describes; without it the derived "
            "declared set silently omits the effects half of the lane. Pass an "
            "empty string only for a lane that runs no effects runtime."
        ),
    )
    parser.add_argument(
        "--broker-container", default="omnibase-infra-stability-test-redpanda"
    )
    parser.add_argument("--min-contracts", type=int, default=DEFAULT_MIN_CONTRACTS)
    parser.add_argument(
        "--consumer-groups-file",
        default=str(Path(__file__).resolve().parent / "consumer_groups_stability.yaml"),
        help=(
            "Declared-groups file [OMN-15837]. No longer a list of group names: "
            "it carries only `non_contract_groups`, the escape hatch for a "
            "load-bearing group neither derivation source can see."
        ),
    )
    parser.add_argument(
        "--lane-compose-file",
        default=str(
            Path(__file__).resolve().parents[2]
            / "docker"
            / "docker-compose.stability-test.yml"
        ),
        help=(
            "Lane compose file whose literal KAFKA_CONSUMER_GROUP values mint "
            "the standalone projection writers' groups [OMN-15837/OMN-17562]. "
            "Read as a derivation source so a re-homed writer changes one file, "
            "not two."
        ),
    )
    parser.add_argument(
        "--min-derived-coverage",
        type=float,
        default=DEFAULT_MIN_DERIVED_COVERAGE,
        help=(
            "Minimum fraction of contract-derived consumer-group identities "
            "that must be live on the broker. A floor, never silently lowered. "
            f"Default {DEFAULT_MIN_DERIVED_COVERAGE}."
        ),
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

    compose_file = Path(args.lane_compose_file) if args.lane_compose_file else None

    report = run_health_gate(
        lane=args.lane,
        pre_image_ids=pre_image_ids,
        expected_revision=args.expected_revision,
        manifest_url=args.manifest_url,
        health_url=args.health_url,
        broker_container=args.broker_container,
        min_contracts=args.min_contracts,
        declared_groups_file=Path(args.consumer_groups_file),
        effects_manifest_url=args.effects_manifest_url or None,
        compose_file=compose_file,
        min_derived_coverage=args.min_derived_coverage,
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
        audit = report.group_audit
        if audit is None:
            print("  consumer_groups: NOT AUDITED (derivation failed, see errors)")
        else:
            print(
                f"  consumer_groups: derived={audit.derived_total} "
                f"live={audit.derived_live} coverage={audit.coverage:.3f} "
                f"(floor={audit.min_coverage}) ok={audit.ok}"
            )
            if audit.retired_identities:
                print(
                    f"    retired identities (logged, not failures): "
                    f"{len(audit.retired_identities)}"
                )
                for group in audit.retired_identities:
                    print(f"      retired {group}")
            if audit.absent_identities:
                print(
                    f"    contract identities with no live group: "
                    f"{len(audit.absent_identities)}"
                )
            for finding in audit.failures:
                print(
                    f"    FAIL {finding.group}: {finding.classification} "
                    f"({finding.detail})"
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
