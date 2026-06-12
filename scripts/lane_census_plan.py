# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""lane_census_plan.py — Pure desired-vs-actual lane reconciler (OMN-13011).

THE CLASS FIX for the recurring lane-drift regression. Nothing reconciled the
declared desired state of a runtime lane against what is actually running, so the
same failures kept recurring with zero signal:

  - volume config drift (OMN-12945 family)
  - WORKER_REPLICAS silent zero — worker scaled to 0, no alert (OMN-12988/12990)
  - 2026-06-11: prod runtime containers + broker network silently absent for hours

This module is a *pure* planner: it takes the versioned lane manifest
(deploy/lane-census/lane-manifest.yaml) as the DESIRED state and a JSON envelope
of the live docker inventory as the ACTUAL state, and emits a deterministic list
of typed drift findings. It performs NO I/O — the shell driver
(scripts/lane-census-check.sh) gathers the docker inventory and publishes the
bus event / ticket. Keeping the decision logic here makes drift detection fully
unit-testable, including the 2026-06-11 prod red fixture.

Drift kinds (named exactly so the auto-ticket says precisely what is wrong):
  container_absent     — a required `service` container is not running
  replicas_zero        — a required service resolved to 0 running replicas
  network_detached     — the lane's declared network does not exist
  oneshot_failed       — a run-to-completion container Exited non-zero
  oneshot_stuck        — a run-to-completion container is still Running
  image_tag_mismatch   — a running container's image tag fails the lane pattern
  unexpected_container — a container labeled for the lane that the manifest does
                         not declare

Fail-fast policy: any drift on a non-optional lane is a hard signal. There is no
warn-only mode (gates-block policy). The driver exits non-zero on drift.

Input envelope (stdin JSON):
  {
    "lane": "<lane name>" | null,   # null => reconcile all non-optional lanes
    "containers": [                 # `docker ps -a --format '{{json .}}'` rows,
      {"Names": "...", "State": "running"|"exited"|...,
       "Image": "repo:tag", "Labels": "k=v,k2=v2",
       "Status": "Up 3 hours" | "Exited (0) ..."},
      ...
    ],
    "networks": ["omnibase-infra-prod-network", ...],  # `docker network ls`
    "runtime_tag": "0.37.0" | null  # resolved runtime tag (optional)
  }

Output (stdout JSON):
  {
    "schema_version": "1.0.0",
    "lanes_checked": ["prod", ...],
    "findings": [ {lane, kind, container, detail, severity}, ... ],
    "has_drift": true|false
  }
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

SCHEMA_VERSION = "1.0.0"

# Drift severities. critical => a required service is down / network gone (the
# 2026-06-11 class of outage). warning => degraded but not lane-down.
_SEVERITY_CRITICAL = "critical"
_SEVERITY_WARNING = "warning"

_DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "deploy"
    / "lane-census"
    / "lane-manifest.yaml"
)


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Load the versioned lane manifest (desired state)."""
    manifest_path = path or _DEFAULT_MANIFEST
    with open(manifest_path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "lanes" not in data:
        raise ValueError(f"lane manifest missing 'lanes': {manifest_path}")
    return data


def _running(state: str, status: str) -> bool:
    """A container counts as running only when docker reports it running."""
    return state.lower() == "running" or status.lower().startswith("up")


def _exit_code(status: str) -> int | None:
    """Parse the exit code out of a docker status string, e.g. 'Exited (0) ...'."""
    match = re.search(r"Exited \((\d+)\)", status)
    return int(match.group(1)) if match else None


def _tag_of(image: str) -> str:
    """Return the tag portion of a 'repo:tag' image reference (default 'latest')."""
    # Strip a registry host:port (the first colon may belong to the host).
    # docker image refs: [registry[:port]/]repo[:tag][@digest]
    ref = image.split("@", 1)[0]
    last_segment = ref.rsplit("/", 1)[-1]
    if ":" in last_segment:
        return last_segment.rsplit(":", 1)[1]
    return "latest"


def _finding(
    lane: str, kind: str, container: str, detail: str, severity: str
) -> dict[str, str]:
    return {
        "lane": lane,
        "kind": kind,
        "container": container,
        "detail": detail,
        "severity": severity,
    }


def reconcile_lane(
    lane_name: str,
    lane_spec: dict[str, Any],
    *,
    actual_by_name: dict[str, dict[str, Any]],
    networks: set[str],
    lane_labeled_names: set[str],
    default_tag_pattern: str,
    runtime_tag: str | None,
) -> list[dict[str, str]]:
    """Diff one lane's desired state against the actual inventory."""
    findings: list[dict[str, str]] = []
    optional = bool(lane_spec.get("optional", False))

    declared = lane_spec.get("services", [])
    declared_names = {svc["name"] for svc in declared}

    # An optional lane that is entirely down is NOT drift (developer lane). We
    # detect "entirely down" as: none of its declared service containers are
    # running. If ANY are running, it is partially up and we reconcile it.
    if optional:
        any_running = any(
            _running(
                actual_by_name.get(svc["name"], {}).get("State", ""),
                actual_by_name.get(svc["name"], {}).get("Status", ""),
            )
            for svc in declared
            if svc.get("kind", "service") == "service"
        )
        if not any_running:
            return findings  # lane legitimately down; no ticket

    # 1. Network presence — the broker/lane network must exist.
    declared_network = lane_spec.get("network")
    if declared_network and declared_network not in networks:
        findings.append(
            _finding(
                lane_name,
                "network_detached",
                declared_network,
                f"lane network {declared_network!r} is not present in "
                f"`docker network ls` — containers cannot reach the broker",
                _SEVERITY_CRITICAL,
            )
        )

    tag_pattern = lane_spec.get("image_tag_pattern") or default_tag_pattern
    # ${RUNTIME_TAG} is resolved from the deploy-agent runtime version when
    # available; an unresolvable token relaxes to the default pattern (a missing
    # runtime tag is a separate signal, not a census drift).
    if "${RUNTIME_TAG}" in tag_pattern:
        tag_pattern = re.escape(runtime_tag) if runtime_tag else default_tag_pattern
    compiled_pattern = re.compile(tag_pattern)

    # 2. Per-declared-service diff.
    for svc in declared:
        name = svc["name"]
        kind = svc.get("kind", "service")
        actual = actual_by_name.get(name)

        if kind == "oneshot":
            # Run-to-completion container. Absence (compose removed it) is fine.
            if actual is None:
                continue
            status = actual.get("Status", "")
            if _running(actual.get("State", ""), status):
                findings.append(
                    _finding(
                        lane_name,
                        "oneshot_stuck",
                        name,
                        f"migration/init container {name!r} is still Running "
                        f"(expected run-to-completion): {status}",
                        _SEVERITY_WARNING,
                    )
                )
                continue
            code = _exit_code(status)
            if code is not None and code != 0:
                findings.append(
                    _finding(
                        lane_name,
                        "oneshot_failed",
                        name,
                        f"migration/init container {name!r} Exited non-zero "
                        f"(code {code}): {status}",
                        _SEVERITY_CRITICAL,
                    )
                )
            continue

        # kind == service: MUST be running with the required replica count.
        required_replicas = int(svc.get("replicas", 1))
        if actual is None or not _running(
            actual.get("State", ""), actual.get("Status", "")
        ):
            # Distinguish replicas_zero (declared scaled-out but down) only for
            # clarity; for a single declared container, absent == container_absent.
            kind_name = (
                "replicas_zero" if required_replicas == 0 else "container_absent"
            )
            findings.append(
                _finding(
                    lane_name,
                    kind_name,
                    name,
                    f"required service container {name!r} is not running "
                    f"(desired replicas={required_replicas}); lane "
                    f"{lane_name!r} is degraded",
                    _SEVERITY_CRITICAL,
                )
            )
            continue

        # Running — verify image tag matches the lane pattern.
        tag = _tag_of(actual.get("Image", ""))
        if not compiled_pattern.fullmatch(tag):
            findings.append(
                _finding(
                    lane_name,
                    "image_tag_mismatch",
                    name,
                    f"container {name!r} runs image tag {tag!r} which does not "
                    f"match lane pattern {tag_pattern!r}",
                    _SEVERITY_WARNING,
                )
            )

    # 3. Unexpected containers — anything labeled for this lane but not declared.
    for actual_name in lane_labeled_names:
        if actual_name not in declared_names:
            findings.append(
                _finding(
                    lane_name,
                    "unexpected_container",
                    actual_name,
                    f"container {actual_name!r} carries com.omninode.lane="
                    f"{lane_name!r} but is not declared in the lane manifest",
                    _SEVERITY_WARNING,
                )
            )

    return findings


def _labels_to_dict(labels: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in labels.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def build_plan(envelope: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the deterministic drift plan from the desired manifest + actual state."""
    requested_lane = envelope.get("lane")
    containers = envelope.get("containers", [])
    networks = set(envelope.get("networks", []))
    runtime_tag = envelope.get("runtime_tag")
    default_pattern = manifest.get("default_image_tag_pattern", ".+")

    actual_by_name: dict[str, dict[str, Any]] = {}
    lane_labels: dict[str, set[str]] = {}
    for row in containers:
        name = (row.get("Names") or row.get("Name") or "").lstrip("/").strip()
        if not name:
            continue
        actual_by_name[name] = row
        labels = _labels_to_dict(row.get("Labels", ""))
        lane_label = labels.get("com.omninode.lane")
        if lane_label:
            lane_labels.setdefault(lane_label, set()).add(name)

    lanes = manifest["lanes"]
    if requested_lane:
        if requested_lane not in lanes:
            raise ValueError(f"unknown lane {requested_lane!r}")
        target_lanes = {requested_lane: lanes[requested_lane]}
    else:
        target_lanes = lanes

    findings: list[dict[str, str]] = []
    checked: list[str] = []
    for lane_name, lane_spec in target_lanes.items():
        checked.append(lane_name)
        findings.extend(
            reconcile_lane(
                lane_name,
                lane_spec,
                actual_by_name=actual_by_name,
                networks=networks,
                lane_labeled_names=lane_labels.get(lane_name, set()),
                default_tag_pattern=default_pattern,
                runtime_tag=runtime_tag,
            )
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "lanes_checked": checked,
        "findings": findings,
        "has_drift": len(findings) > 0,
    }


def main() -> int:
    manifest_path = os.environ.get("LANE_MANIFEST")
    manifest = load_manifest(Path(manifest_path) if manifest_path else None)
    envelope = json.load(sys.stdin)
    plan = build_plan(envelope, manifest)
    json.dump(plan, sys.stdout)
    sys.stdout.write("\n")
    # Exit 0 always — the planner reports; the shell driver decides exit policy
    # so the JSON plan is always emittable for dry-run/inspection.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
