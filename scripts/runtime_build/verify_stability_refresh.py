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
  3. **Health endpoint** -- ``/health`` returns HTTP 200 with a JSON body
     whose ``status`` (or ``details.healthy``) indicates healthy.
  4. **Cluster health** -- ``rpk cluster health`` inside the broker container
     reports healthy.
  5. **Consumer groups Stable** -- every group declared in
     ``consumer_groups_stability.yaml`` reports ``state == "Stable"`` via
     ``rpk group describe``.
  6. **Revision readback** -- the ``org.opencontainers.image.revision`` label
     on each core container equals the intended new ref (or a prefix thereof,
     tolerating short/full SHA differences).

``refresh_stability_lane.sh`` calls this script both for the post-refresh gate
and, on a FAIL, again against the rolled-back state to confirm rollback
actually restored health.

Exit codes:
    0 - PASS (all requested checks succeeded)
    1 - FAIL (a genuine check failure -- digest not changed, manifest floor
        not met, unhealthy, cluster unhealthy, a group not Stable, or a
        revision mismatch)
    2 - INFRA_ERROR (could not run a check at all -- docker/curl/rpk
        unavailable, container missing, etc.) -- distinguished from a
        genuine FAIL so the caller does not conflate "couldn't check" with
        "checked and it's broken"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

# OCI label stamped from VCS_REF/GIT_SHA at build time (Dockerfile.runtime).
_REVISION_LABEL = "org.opencontainers.image.revision"

DEFAULT_MIN_CONTRACTS = 288


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
    cluster_healthy: bool = False
    cluster_detail: str | None = None
    consumer_groups: list[ConsumerGroupCheck] = field(default_factory=list)
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
            "cluster_healthy": self.cluster_healthy,
            "cluster_detail": self.cluster_detail,
            "consumer_groups": {g.group: g.state for g in self.consumer_groups},
            "consumer_groups_stable": self.groups_stable,
            "revision_readback_ok": self.revisions_match,
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


def check_health(health_url: str, *, opener: object | None = None) -> tuple[bool, str]:
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(health_url, timeout=10) as resp:  # type: ignore[operator]
            status_code = getattr(resp, "status", 200)
            raw = resp.read()
    except (urllib.error.URLError, OSError) as exc:
        return False, f"health fetch failed: {exc}"
    if status_code and status_code != 200:
        return False, f"health endpoint returned HTTP {status_code}"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return False, "health payload not valid JSON"
    status = str(payload.get("status", "")).lower()
    details_healthy = bool(payload.get("details", {}).get("healthy", False))
    if status == "healthy" or details_healthy:
        return True, f"status={status}"
    return False, f"status={status!r} details.healthy={details_healthy}"


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
    healthy = (
        "Healthy:             true" in stdout or "cluster is healthy" in stdout.lower()
    )
    return healthy, stdout.strip().splitlines()[0] if stdout.strip() else "no output"


def check_consumer_group(
    broker_container: str, group: str, *, runner: object | None = None
) -> ConsumerGroupCheck:
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
                "-f",
                "json",
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
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return ConsumerGroupCheck(
            group=group, state=None, stable=False, error=f"bad JSON: {exc}"
        )
    state = payload.get("state") or payload.get("STATE")
    return ConsumerGroupCheck(group=group, state=state, stable=(state == "Stable"))


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

    healthy, detail = check_health(health_url, opener=opener)
    report.health_ok = healthy
    report.health_detail = detail

    cluster_healthy, cluster_detail = check_cluster_health(
        broker_container, runner=runner
    )
    report.cluster_healthy = cluster_healthy
    report.cluster_detail = cluster_detail

    for group in consumer_groups:
        report.consumer_groups.append(
            check_consumer_group(broker_container, group, runner=runner)
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
