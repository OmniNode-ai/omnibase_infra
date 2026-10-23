# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19513: raise only the dev lane's main runtime CPU cap."""

from pathlib import Path
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(compose_file: str) -> dict[str, Any]:
    text = (REPO_ROOT / "docker" / compose_file).read_text(encoding="utf-8")
    return yaml.safe_load(text.replace("!override", ""))


def test_dev_lane_raises_omninode_runtime_cpu_cap() -> None:
    base = _load("docker-compose.infra.yml")
    overlay = _load("docker-compose.dev-lane.yml")
    limits = base["services"]["omninode-runtime"]["deploy"]["resources"][
        "limits"
    ].copy()
    limits.update(
        overlay["services"]["omninode-runtime"]["deploy"]["resources"]["limits"]
    )

    assert float(limits["cpus"]) == 2.0
    assert limits["memory"] == "1536M"


def test_base_file_keeps_the_one_cpu_cap() -> None:
    base = _load("docker-compose.infra.yml")
    limits = base["services"]["omninode-runtime"]["deploy"]["resources"]["limits"]

    assert float(limits["cpus"]) == 1.0


def test_dev_lane_overlay_touches_no_other_runtime_cpu_cap() -> None:
    overlay = _load("docker-compose.dev-lane.yml")

    for service_name in ("runtime-effects", "runtime-worker"):
        cpus = (
            overlay["services"][service_name]
            .get("deploy", {})
            .get("resources", {})
            .get("limits", {})
            .get("cpus")
        )
        assert cpus is None, f"{service_name} must inherit its base CPU cap"
