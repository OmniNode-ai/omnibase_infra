# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17232: the HP DL560 Gen8 canary pool is declared as data and no workflow can select it.

Operator RULING 2026-10-29T14:00:26Z: make before break. The canary pool carries a label of its own and
neither production label, so no existing ``runs-on`` list reaches it, and its class is ``canary`` so no
required-label spread or capacity count includes it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
PREFIX = "hp051-ci-runner"
COMPOSE = REPO_ROOT / "docker" / f"docker-compose.runners-{PREFIX}.yml"
SLICE = (
    REPO_ROOT
    / "docker"
    / "runners"
    / "systemd"
    / "pools"
    / PREFIX
    / "omnirunners.slice"
)
FLEET = REPO_ROOT / "config" / "runner_fleet.yaml"
PRODUCTION_LABELS = {"omnibase-ci", "omnibase-ci-multiarch", "omnibase-verify"}


def _compose() -> dict[str, Any]:
    return cast("dict[str, Any]", yaml.safe_load(COMPOSE.read_text(encoding="utf-8")))


def test_every_canary_runner_carries_hp_canary_and_no_production_label() -> None:
    services = _compose()["services"]
    assert len(services) == 4
    for name, service in services.items():
        labels = set(str(service["environment"]["RUNNER_LABELS"]).split(","))
        assert "hp-canary" in labels, f"{name} lacks the canary label"
        assert not labels & PRODUCTION_LABELS, f"{name} carries a production label"


def test_the_canary_pool_is_its_own_cgroup_slice_with_declared_limits() -> None:
    services = _compose()["services"]
    assert {s["cgroup_parent"] for s in services.values()} == {"omnirunners.slice"}
    text = SLICE.read_text(encoding="utf-8")
    for key in ("MemoryHigh", "MemoryMax", "MemorySwapMax", "CPUWeight"):
        assert f"{key}=" in text, f"the slice must declare {key}"


def test_the_fleet_declares_the_pool_with_a_class_no_probe_counts() -> None:
    document = yaml.safe_load(FLEET.read_text(encoding="utf-8"))
    rows = [row for row in document["hosts"] if row["host"] == "h051"]
    assert len(rows) == 1
    assert rows[0]["runner_name_prefix"] == PREFIX
    assert rows[0]["expected_count"] == 4
    assert rows[0]["classes"] == ["canary"]


def test_the_canary_names_match_the_declared_count() -> None:
    names = sorted(_compose()["services"])
    assert names == [f"{PREFIX}-{n}" for n in range(1, 5)]
