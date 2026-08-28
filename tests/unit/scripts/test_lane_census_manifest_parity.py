# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Manifest <-> compose parity ratchet for the lane census (OMN-13011).

The lane manifest is the desired-state authority, but it can only police a lane
if it stays in lock-step with that lane's compose file. These tests assert that
every container_name declared in a lane's compose file appears in the manifest
(and vice-versa) so the manifest can never silently drift away from the lane it
is supposed to reconcile. A lane that gains/renames a container without updating
the manifest in the same PR fails CI here — that is the whole point.

This also closes the OMN-12988 census gap where
docker/catalog/services/runtime-worker.yaml carried container_name: null: the
manifest sources concrete names from the compose lane files, not the catalog
stub.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_MANIFEST_PATH = _REPO / "deploy" / "lane-census" / "lane-manifest.yaml"

# Lanes whose compose files have concrete container_name values we can diff.
# dev uses generated/non-prefixed names and is intentionally optional/loose, so
# it is parity-checked against its own service list only, not a compose scrape.
_COMPOSE_LANES = ("stability-test", "prod", "judge")


def _load_manifest() -> dict:
    with open(_MANIFEST_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _compose_container_names(lane: str) -> set[str]:
    compose_path = _REPO / "docker" / f"docker-compose.{lane}.yml"
    raw = compose_path.read_text(encoding="utf-8")
    # The compose files use custom !override / !!merge tags that safe_load cannot
    # parse, so scrape container_name lines directly.
    return set(re.findall(r"^\s*container_name:\s*(\S+)\s*$", raw, re.MULTILINE))


@pytest.mark.parametrize("lane", _COMPOSE_LANES)
def test_every_compose_container_is_declared(lane: str) -> None:
    """No compose container_name may be missing from the lane manifest."""
    manifest = _load_manifest()
    declared = {s["name"] for s in manifest["lanes"][lane]["services"]}
    compose_names = _compose_container_names(lane)
    missing = compose_names - declared
    assert not missing, (
        f"lane {lane!r}: compose declares containers absent from the lane "
        f"manifest (update deploy/lane-census/lane-manifest.yaml in the same PR): "
        f"{sorted(missing)}"
    )


@pytest.mark.parametrize("lane", _COMPOSE_LANES)
def test_no_manifest_phantom_containers(lane: str) -> None:
    """No manifest service may reference a container the compose file lacks."""
    manifest = _load_manifest()
    declared = {s["name"] for s in manifest["lanes"][lane]["services"]}
    compose_names = _compose_container_names(lane)
    phantom = declared - compose_names
    assert not phantom, (
        f"lane {lane!r}: lane manifest declares containers the compose file does "
        f"not define (stale manifest entry): {sorted(phantom)}"
    )


@pytest.mark.parametrize("lane", _COMPOSE_LANES)
def test_lane_network_matches_compose(lane: str) -> None:
    """The manifest's declared network must exist in the compose networks block."""
    manifest = _load_manifest()
    declared_network = manifest["lanes"][lane]["network"]
    compose_path = _REPO / "docker" / f"docker-compose.{lane}.yml"
    raw = compose_path.read_text(encoding="utf-8")
    network_names = set(
        re.findall(r"^\s*name:\s*(omnibase-infra-\S+-network)\s*$", raw, re.MULTILINE)
    )
    assert declared_network in network_names, (
        f"lane {lane!r}: manifest network {declared_network!r} not found among "
        f"compose networks {sorted(network_names)}"
    )


def _compose_disabled_container_names(lane: str) -> set[str]:
    """container_names whose lane overlay disables them via a profile override.

    OMN-16803. A service carrying ``profiles: !override ["<lane>-disabled"]`` in
    the lane overlay is not a member of the lane's ``runtime`` profile, so no
    sanctioned ``docker compose --profile runtime up`` can start it. Scraped
    rather than YAML-parsed for the same reason the container_name scrape above
    is: the compose files carry custom ``!override`` / ``!!merge`` tags that
    ``yaml.safe_load`` refuses.

    Walks the 2-space-indented service blocks under ``services:`` and pairs each
    block's ``container_name`` with the presence of a disabling profile override
    in that same block.
    """
    compose_path = _REPO / "docker" / f"docker-compose.{lane}.yml"
    raw = compose_path.read_text(encoding="utf-8")

    disabled: set[str] = set()
    current_container: str | None = None
    current_disabled = False
    in_services = False

    def _flush() -> None:
        if current_container and current_disabled:
            disabled.add(current_container)

    for line in raw.splitlines():
        if re.match(r"^services:\s*$", line):
            in_services = True
            continue
        if not in_services:
            continue
        # A new top-level key (column 0, non-comment) ends the services block.
        if re.match(r"^[A-Za-z_]", line):
            break
        # A new service block starts at exactly 2 spaces of indent.
        if re.match(r"^  [A-Za-z0-9_.-]+:\s*$", line):
            _flush()
            current_container = None
            current_disabled = False
            continue
        name_match = re.match(r"^\s+container_name:\s*(\S+)\s*$", line)
        if name_match:
            current_container = name_match.group(1)
            continue
        if re.search(r"^\s+profiles:.*-disabled", line):
            current_disabled = True
    _flush()
    return disabled


@pytest.mark.parametrize("lane", _COMPOSE_LANES)
def test_profile_disabled_services_are_declared_profile_gated(lane: str) -> None:
    """A compose-disabled service may not be declared a required service.

    OMN-16803 root cause. ``docker-compose.stability-test.yml`` disables
    agent-actions-consumer, skill-lifecycle-consumer, intelligence-api and
    omninode-contract-resolver via ``profiles: !override
    ["stability-test-disabled"]``, but the manifest still declared all four
    ``kind: service, replicas: 1``. The census therefore emitted four permanent
    false ``container_absent`` criticals, and a genuinely degraded lane
    (runtime-effects + runtime-worker + migration-gate actually missing) was
    indistinguishable from that standing noise for a month.

    The two declarations must move together: disable a service in the lane
    overlay and it must be ``kind: profile_gated`` in the manifest; re-enable it
    and it must go back to ``kind: service``.
    """
    manifest = _load_manifest()
    by_name = {s["name"]: s for s in manifest["lanes"][lane]["services"]}
    compose_disabled = _compose_disabled_container_names(lane)

    wrongly_required = sorted(
        name
        for name in compose_disabled
        if by_name.get(name, {}).get("kind", "service") == "service"
    )
    assert not wrongly_required, (
        f"lane {lane!r}: these containers are disabled by a compose profile "
        f"override yet are declared `kind: service` (a required, must-be-running "
        f"service) in deploy/lane-census/lane-manifest.yaml. No `--profile "
        f"runtime up` can ever start them, so the census will report them as "
        f"permanent false container_absent criticals. Declare them "
        f"`kind: profile_gated`, or remove the compose profile override in this "
        f"same PR: {wrongly_required}"
    )

    wrongly_gated = sorted(
        name
        for name, svc in by_name.items()
        if svc.get("kind") == "profile_gated" and name not in compose_disabled
    )
    assert not wrongly_gated, (
        f"lane {lane!r}: these containers are declared `kind: profile_gated` "
        f"(expected absent) but the lane's compose file does NOT disable them "
        f"via a profile override — so they are real, startable services whose "
        f"absence would now be silently tolerated. Declare them `kind: service`: "
        f"{wrongly_gated}"
    )


def test_no_service_declares_replicas_zero() -> None:
    """A service may never declare replicas: 0 — that is the silent-drop surface."""
    manifest = _load_manifest()
    for lane, spec in manifest["lanes"].items():
        for svc in spec["services"]:
            if svc.get("kind", "service") == "service":
                assert svc.get("replicas", 1) >= 1, (
                    f"lane {lane!r} service {svc['name']!r} declares replicas 0 — "
                    f"forbidden (the WORKER_REPLICAS silent-zero regression)"
                )


def test_runtime_worker_declared_in_every_runtime_lane() -> None:
    """The worker that silently dropped (OMN-12988) must be a required service."""
    manifest = _load_manifest()
    for lane in ("stability-test", "prod"):
        names = {s["name"] for s in manifest["lanes"][lane]["services"]}
        worker = next((n for n in names if n.endswith("runtime-worker")), None)
        assert worker is not None, f"lane {lane!r} missing a runtime-worker service"
