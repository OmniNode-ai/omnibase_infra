# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The recorded read-only census of the dev lane, as the one-runtime plan's table reads it."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

RECORDED = Path(__file__).parent / "fixtures" / "dev_lane_census_recorded.json"
HANDLER_WRITERS = {
    "omnimarket-projection-delegation-writer",
    "omnimarket-projection-live-events-writer",
    "omnimarket-projection-registration-writer",
    "omnimarket-projection-savings-writer",
    "omnimarket-projection-tenant-credentials-writer",
    "omnimarket-projection-tenant-registry-writer",
}


def _containers() -> list[dict[str, str]]:
    containers: list[dict[str, str]] = json.loads(RECORDED.read_text(encoding="utf-8"))[
        "containers"
    ]
    return containers


def test_dev_lane_runs_sixteen_runtime_image_containers_four_on_onex_runtime() -> None:
    containers = _containers()
    assert len(containers) == 16
    assert sum(c["family"] == "onex-runtime" for c in containers) == 4
    assert {c["container"] for c in containers if c["verdict"] == "allowed"} == {
        c["container"] for c in containers if c["family"] == "onex-runtime"
    }


def test_the_six_handler_writers_are_the_python_dash_m_handler_containers() -> None:
    handler = {
        c["container"]
        for c in _containers()
        if c["family"] == "python -m omnimarket...handler_*"
    }
    assert handler == HANDLER_WRITERS


def test_other_entry_points_match_the_plan_table() -> None:
    kinds = Counter(
        c["family"]
        for c in _containers()
        if c["verdict"] == "remove" and c["container"] not in HANDLER_WRITERS
    )
    assert kinds == {
        "python -m ...consumer": 4,
        "uvicorn": 1,
        "omnimarket-projection-api": 1,
    }
