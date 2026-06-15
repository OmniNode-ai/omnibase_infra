# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""End-to-end coverage for the volume-config drift + re-seed cycle (OMN-12958).

Reproduces the OMN-12945 defect mechanically: the deployed Bifrost delegation
contract lives on a Docker named volume that survives image rebuilds, so the
volume copy can silently diverge from the packaged source. This test wires the
real runtime modules together against a temp dir that stands in for the volume:

1. render the contract from packaged source into the "volume" target,
2. mutate the volume copy so it drifts from source,
3. compute provenance and assert drift is detected + logged + sidecared,
4. assert the health check degrades on drift (operator-visible signal),
5. re-render (the deploy re-seed) and assert the volume copy is back in-sync.

Unlike the per-module unit tests, this exercises render -> provenance -> sidecar
-> health as one flow against on-disk artifacts, which is the surface the deploy
procedure and drift sweep actually depend on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from omnibase_infra.runtime.config_provenance import (
    PROVENANCE_SIDECAR_NAME,
    build_config_provenance,
    write_provenance_sidecar,
)
from omnibase_infra.runtime.health.health_config_provenance import (
    check_config_provenance_health,
)
from omnibase_infra.runtime.render_bifrost_delegation_contract import (
    render_bifrost_delegation_contract,
)

_SOURCE_CONTRACT: dict[str, object] = {
    "backends": [
        {
            "backend_id": "local-primary",
            "model_name": "qwen3-coder",
            "endpoint_url": "",
            "endpoint_url_env": "OMN_INT_DRIFT_PRIMARY_URL",
            "required": True,
        }
    ],
    "routing_rules": [
        {"rule_id": "default", "backend_ids": ["local-primary"]},
    ],
    "default_backends": ["local-primary"],
}

_PRIMARY_URL = "http://192.0.2.10:8000/v1/chat/completions"


def _write_source(tmp_path: Path) -> Path:
    source = tmp_path / "packaged" / "bifrost_delegation.yaml"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(yaml.safe_dump(_SOURCE_CONTRACT, sort_keys=False), "utf-8")
    return source


async def test_volume_config_drift_detected_and_reseeded(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    volume_target = tmp_path / "volume" / "delegation" / "bifrost_delegation.yaml"
    env = {"OMN_INT_DRIFT_PRIMARY_URL": _PRIMARY_URL}

    # 1. Initial render seeds the volume copy from packaged source.
    rendered = render_bifrost_delegation_contract(
        source_path=source,
        target_path=volume_target,
        environ=env,
        verify_endpoints=False,
    )
    assert rendered == volume_target
    assert volume_target.exists()

    provenance = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=volume_target,
        source_path=source,
    )
    # Rendered output is derived from (not byte-identical to) source — both must
    # be present and the health check must not be unhealthy on a fresh seed.
    assert provenance.deployed_present is True
    assert provenance.source_present is True
    seed_health = check_config_provenance_health(provenance)
    assert seed_health.status in {"healthy", "degraded"}
    assert seed_health.status != "unhealthy"

    # 2. Drift the volume copy (simulate a hand-mutated stale binding that
    #    survives a rebuild — the exact OMN-12945 mechanism).
    drifted = yaml.safe_load(volume_target.read_text("utf-8"))
    drifted["backends"][0]["model_name"] = "stale-model-from-jun-9"
    volume_target.write_text(yaml.safe_dump(drifted, sort_keys=False), "utf-8")

    drift_prov = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=volume_target,
        source_path=source,
    )
    assert drift_prov.has_drifted is True
    assert drift_prov.deployed_sha256 != drift_prov.source_sha256
    assert "DRIFT" in drift_prov.provenance_line()

    # 3. The sidecar records drift so the sweep + proof packets read it directly.
    sidecar = write_provenance_sidecar(drift_prov, deployed_path=volume_target)
    assert sidecar.name == PROVENANCE_SIDECAR_NAME
    sidecar_payload = json.loads(sidecar.read_text("utf-8"))
    assert sidecar_payload["deployed_sha256"] == drift_prov.deployed_sha256
    assert sidecar_payload["source_sha256"] == drift_prov.source_sha256

    # 4. Health degrades on drift — operator-visible, not a silent failure.
    drift_health = check_config_provenance_health(drift_prov)
    assert drift_health.status == "degraded"
    assert drift_health.details["has_drifted"] is True
    assert "re-seed required" in (drift_health.error or "")

    # 5. Re-render is the deploy re-seed: the volume copy is rebuilt from source
    #    and drift is cleared with zero manual volume mutation.
    reseeded = render_bifrost_delegation_contract(
        source_path=source,
        target_path=volume_target,
        environ=env,
        verify_endpoints=False,
    )
    assert reseeded == volume_target
    post_reseed = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=volume_target,
        source_path=source,
    )
    post_data = yaml.safe_load(volume_target.read_text("utf-8"))
    assert post_data["backends"][0]["model_name"] == "qwen3-coder"
    assert check_config_provenance_health(post_reseed).status != "unhealthy"


async def test_absent_volume_config_is_unhealthy(tmp_path: Path) -> None:
    """A missing deployed copy is unhealthy: the runtime has no config to load."""
    source = _write_source(tmp_path)
    missing_volume = tmp_path / "volume" / "delegation" / "bifrost_delegation.yaml"

    provenance = build_config_provenance(
        config_name="bifrost_delegation",
        deployed_path=missing_volume,
        source_path=source,
    )
    assert provenance.deployed_present is False
    assert provenance.has_drifted is False  # absence is not drift

    health = check_config_provenance_health(provenance)
    assert health.status == "unhealthy"


def test_empty_bifrost_contract_path_disables_render_no_provenance(
    tmp_path: Path,
) -> None:
    """Empty BIFROST_CONTRACT_PATH disables rendering — the documented kill switch.

    Guards the second competing authority OMN-12958 calls out: an empty
    BIFROST_CONTRACT_PATH silently disables the packaged-source render entirely.
    The renderer must return None (no volume copy materialized) rather than
    writing to the default volume path.
    """
    source = _write_source(tmp_path)
    rendered = render_bifrost_delegation_contract(
        source_path=source,
        target_path=None,
        environ={"BIFROST_CONTRACT_PATH": "  "},
        verify_endpoints=False,
    )
    assert rendered is None
