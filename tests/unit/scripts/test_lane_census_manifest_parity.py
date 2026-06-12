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
