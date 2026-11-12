# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-23513: the shipped runtime policy folds dev services end to end.

``scripts/lane_census_plan.py`` runs the way the lane-census-check.sh driver
runs it: envelope JSON on stdin, manifest and registry in temp files, and
``LANE_CENSUS_RUNTIME_POLICY`` on the SHIPPED docker/runtime-policy.env. The
three folded dev services are then expected absent; a running fold is
``profile_gated_present``; without the policy the three read as absent drift.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[3]
_PLAN_PATH = _ROOT / "scripts" / "lane_census_plan.py"
_SHIPPED_POLICY = _ROOT / "docker" / "runtime-policy.env"

_LAB_HOST = "lab-01"
_REGISTRY_ROWS: list[dict[str, Any]] = [
    {"name": "host-a", "aliases": [_LAB_HOST], "lanes": ["dev", "other"]}
]


def _container(name: str, *, lane: str) -> dict[str, Any]:
    return {
        "Names": name,
        "State": "running",
        "Status": "Up 3 hours",
        "Image": "omninode-runtime:0.37.0",
        "RestartCount": 0,
        "Labels": f"com.omninode.lane={lane},com.omninode.layer=runtime",
    }


def _dev_manifest() -> dict[str, Any]:
    return {
        "default_image_tag_pattern": ".+",
        "lanes": {
            "dev": {
                "compose_project": "omnibase-infra",
                "network": "omnibase-infra-dev-network",
                "services": [
                    {"name": "omnibase-infra-postgres", "kind": "service"},
                    {"name": "omninode-runtime", "kind": "service"},
                    {"name": "omninode-runtime-effects", "kind": "service"},
                    {"name": "omnimarket-projection-api", "kind": "service"},
                    {
                        "name": "omnimarket-projection-delegation-writer",
                        "kind": "service",
                    },
                ],
            },
            "other": {
                "compose_project": "other-project",
                "network": "omnibase-infra-other-network",
                "services": [
                    {"name": "omnibase-infra-postgres", "kind": "service"},
                    {"name": "omninode-runtime", "kind": "service"},
                ],
            },
        },
    }


def _dev_envelope(containers: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lane": "dev",
        "host": _LAB_HOST,
        "containers": containers,
        "networks": ["omnibase-infra-dev-network"],
        "runtime_tag": None,
    }


def _run_planner(
    tmp_path: Path, envelope: dict[str, Any], policy: Path
) -> dict[str, Any]:
    """The plan the script prints when driven the way the shell driver drives it."""
    manifest = tmp_path / "lane-manifest.yaml"
    manifest.write_text(yaml.safe_dump(_dev_manifest()), encoding="utf-8")
    registry = tmp_path / "overlay.yaml"
    registry.write_text(yaml.safe_dump({"registry": _REGISTRY_ROWS}), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != "LANE_CENSUS_DESIRED_STATE"}
    env["LANE_MANIFEST"] = str(manifest)
    env["LANE_CENSUS_REGISTRY"] = str(registry)
    env["LANE_CENSUS_RUNTIME_POLICY"] = str(policy)
    result = subprocess.run(
        [sys.executable, str(_PLAN_PATH)],
        input=json.dumps(envelope),
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _running() -> list[dict[str, Any]]:
    return [
        _container("omnibase-infra-postgres", lane="dev"),
        _container("omninode-runtime", lane="dev"),
    ]


def test_shipped_policy_makes_folded_dev_services_expected_absent(
    tmp_path: Path,
) -> None:
    plan = _run_planner(tmp_path, _dev_envelope(_running()), _SHIPPED_POLICY)
    kinds = {f["kind"] for f in plan["findings"]}
    assert "container_absent" not in kinds, plan["findings"]
    assert plan["has_drift"] is False


def test_running_folded_service_is_profile_gated_present(tmp_path: Path) -> None:
    containers = _running() + [_container("omnimarket-projection-api", lane="dev")]
    plan = _run_planner(tmp_path, _dev_envelope(containers), _SHIPPED_POLICY)
    gated = [
        f
        for f in plan["findings"]
        if f["kind"] == "profile_gated_present"
        and f["container"] == "omnimarket-projection-api"
    ]
    assert gated, plan["findings"]


def test_control_without_the_policy_reads_the_three_as_absent(
    tmp_path: Path,
) -> None:
    plan = _run_planner(tmp_path, _dev_envelope(_running()), tmp_path / "missing.env")
    absent = {
        f["container"] for f in plan["findings"] if f["kind"] == "container_absent"
    }
    assert absent == {
        "omninode-runtime-effects",
        "omnimarket-projection-api",
        "omnimarket-projection-delegation-writer",
    }, plan["findings"]
