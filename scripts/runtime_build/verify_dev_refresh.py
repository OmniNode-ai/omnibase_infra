#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Dev-lane refresh health-gate + receipt emission [OMN-14889].

Dev-lane analog of ``verify_stability_refresh.py`` (OMN-14873), reused for
structure/shape but NOT for its hardcoded container-name map: the dev lane's
``docker-compose.infra.yml`` does not set an explicit ``container_name`` for
``runtime-worker`` (compose-assigned instead), and the dev lane is routinely
GC-reclaimed to zero containers (OMN-13414) unlike the always-warm
stability-test lane. Container IDs are therefore resolved by the CALLER
(``refresh_dev_lane.sh``, via ``docker compose ps -q``) and passed in as a
JSON map (``--container-ids``), never assumed here.

Every check is required (fail-closed AND-of-all-checks):

  1. **Digest changed** (warm branch only; skipped via
     ``--no-require-digest-change`` on the cold-aware branch, where there is
     no baseline image to diff against) -- each core service's running image
     ID differs from the pre-refresh snapshot.
  2. **Manifest count** -- ``/v1/introspection/manifest`` contract count is
     >= ``--min-contracts``.
  3. **Health endpoint** -- ``/health`` returns HTTP 200 with a healthy body.
  4. **Cluster health** -- ``rpk cluster health`` inside the broker container
     reports healthy.
  5. **Revision readback** -- the ``org.opencontainers.image.revision``
     label on each core container equals the intended new ref (prefix
     match tolerated).

Consumer-group Stable-state checking (present in the stability-test gate) is
intentionally NOT included here: this dev-lane gate was built against a lane
whose core services were not live at build time, so there was no way to
capture verified-real group names (the stability gate's own consumer_groups
file needed a live-lane correction during ITS build for exactly this
reason -- see that file's header). Declaring unverified group names here
would be a plausible-but-unverified check, worse than omitting it. Folding
dev-lane consumer-group coverage in is flagged as follow-up, not silently
faked.

Exit codes: 0 PASS, 1 FAIL, 2 INFRA_ERROR (mirrors verify_stability_refresh.py).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

_REVISION_LABEL = "org.opencontainers.image.revision"

DEFAULT_MIN_CONTRACTS = 288

CORE_SERVICE_NAMES = (
    "omninode-runtime",
    "runtime-effects",
    "runtime-worker",
    "projection-api",
)


@dataclass
class ServiceDigestCheck:
    service: str
    container: str | None
    pre_image_id: str | None
    post_image_id: str | None
    digest_changed: bool
    revision_label: str | None
    expected_revision: str
    revision_match: bool
    error: str | None = None


@dataclass
class HealthGateReport:
    lane: str
    services: list[ServiceDigestCheck] = field(default_factory=list)
    manifest_count: int | None = None
    manifest_floor: int = DEFAULT_MIN_CONTRACTS
    manifest_ok: bool = False
    health_ok: bool = False
    health_detail: str | None = None
    cluster_healthy: bool = False
    cluster_detail: str | None = None
    errors: list[str] = field(default_factory=list)
    require_digest_change: bool = True

    @property
    def digests_changed(self) -> bool:
        return bool(self.services) and all(s.digest_changed for s in self.services)

    @property
    def revisions_match(self) -> bool:
        return bool(self.services) and all(s.revision_match for s in self.services)

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
            "revision_readback_ok": self.revisions_match,
            "services": [asdict(s) for s in self.services],
            "errors": self.errors,
            "overall": self.overall,
        }


def _run(
    cmd: list[str], *, runner: object | None = None, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    run_fn = runner or subprocess.run
    return run_fn(cmd, capture_output=True, text=True, timeout=timeout, check=False)  # type: ignore[operator, no-any-return]


def get_image_id(
    container: str, *, runner: object | None = None
) -> tuple[str | None, str | None]:
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
    a, b = actual.strip().lower(), expected.strip().lower()
    if not a or not b:
        return False
    return a == b or a.startswith(b) or b.startswith(a)


def check_service_digest(
    service: str,
    container: str | None,
    pre_image_id: str | None,
    expected_revision: str,
    *,
    runner: object | None = None,
) -> ServiceDigestCheck:
    if not container:
        return ServiceDigestCheck(
            service=service,
            container=None,
            pre_image_id=pre_image_id,
            post_image_id=None,
            digest_changed=False,
            revision_label=None,
            expected_revision=expected_revision,
            revision_match=False,
            error="no running container resolved for this service",
        )
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
    healthy = bool(re.search(r"Healthy:\s*true\b", stdout, re.IGNORECASE)) or (
        "cluster is healthy" in stdout.lower()
    )
    return healthy, stdout.strip().splitlines()[0] if stdout.strip() else "no output"


def run_health_gate(
    *,
    lane: str,
    pre_image_ids: dict[str, str],
    container_ids: dict[str, str],
    expected_revision: str,
    manifest_url: str,
    health_url: str,
    broker_container: str,
    min_contracts: int,
    runner: object | None = None,
    opener: object | None = None,
    require_digest_change: bool = True,
) -> HealthGateReport:
    report = HealthGateReport(
        lane=lane,
        manifest_floor=min_contracts,
        require_digest_change=require_digest_change,
    )

    for service in CORE_SERVICE_NAMES:
        report.services.append(
            check_service_digest(
                service,
                container_ids.get(service) or None,
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

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", default="dev")
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument(
        "--pre-image-ids",
        default="{}",
        help="JSON dict of {service: pre_refresh_image_id}. Empty on the cold-aware branch.",
    )
    parser.add_argument(
        "--container-ids",
        required=True,
        help="JSON dict of {service: container_id}, resolved by the caller via `docker compose ps -q`.",
    )
    # fallback-ok: fixed dev lane ops-tooling defaults (port 8085); the
    # automated caller (refresh_dev_lane.sh) always passes these explicitly.
    # url-authority-ok: no contract/routing-authority exists for this standalone
    # ops script; the lane's port is a fixed, documented convention.
    manifest_url_default = "http://localhost:8085/v1/introspection/manifest"  # fallback-ok  # url-authority-ok: fixed lane port, no routing authority applies
    health_url_default = "http://localhost:8085/health"  # fallback-ok  # url-authority-ok: fixed lane port, no routing authority applies
    parser.add_argument("--manifest-url", default=manifest_url_default)
    parser.add_argument("--health-url", default=health_url_default)
    parser.add_argument("--broker-container", default="omnibase-infra-redpanda")
    parser.add_argument("--min-contracts", type=int, default=DEFAULT_MIN_CONTRACTS)
    parser.add_argument("--json", action="store_true", dest="json_output")
    parser.add_argument(
        "--no-require-digest-change",
        action="store_false",
        dest="require_digest_change",
        default=True,
        help="Skip the digest-changed requirement (cold-aware branch or post-rollback re-verification).",
    )
    args = parser.parse_args(argv)

    try:
        pre_image_ids = json.loads(args.pre_image_ids)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --pre-image-ids is not valid JSON: {exc}", file=sys.stderr)
        return 2
    try:
        container_ids = json.loads(args.container_ids)
    except json.JSONDecodeError as exc:
        print(f"ERROR: --container-ids is not valid JSON: {exc}", file=sys.stderr)
        return 2

    report = run_health_gate(
        lane=args.lane,
        pre_image_ids=pre_image_ids,
        container_ids=container_ids,
        expected_revision=args.expected_revision,
        manifest_url=args.manifest_url,
        health_url=args.health_url,
        broker_container=args.broker_container,
        min_contracts=args.min_contracts,
        require_digest_change=args.require_digest_change,
    )

    if args.json_output:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"Health gate for lane={report.lane}: {report.overall}")
        print(
            f"  digest_changed={report.digests_changed} (required={report.require_digest_change})"
        )
        print(
            f"  manifest_count={report.manifest_count} (floor={report.manifest_floor}) ok={report.manifest_ok}"
        )
        print(f"  health_ok={report.health_ok} ({report.health_detail})")
        print(f"  cluster_healthy={report.cluster_healthy} ({report.cluster_detail})")
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
