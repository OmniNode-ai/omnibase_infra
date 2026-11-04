# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Rebuild the dev table from the read-only inspection recorded for this change."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from omnibase_infra.handlers.handler_lane_entry_point_census import (
    build_census,
    load_policy,
    render_table,
)
from omnibase_infra.models.lane_entry_point_census import ModelLaneEntryPointCensus

pytestmark = pytest.mark.unit
RECORDED = Path(__file__).parent / "fixtures" / "dev_lane_census_recorded.json"
POLICY = Path(__file__).parents[4] / "deploy/lane-census/runtime-entry-points.yaml"


@pytest.fixture
def recorded_response() -> dict[str, object]:
    recording: dict[str, object] = json.loads(RECORDED.read_text(encoding="utf-8"))
    return recording


def _census(recording: dict[str, object] | None = None) -> ModelLaneEntryPointCensus:
    if recording is None:
        recording = json.loads(RECORDED.read_text(encoding="utf-8"))
    assert recording is not None
    inspection = TypeAdapter(list[dict[str, object]]).validate_python(
        recording["inspection"]
    )
    (census,) = build_census(inspection, load_policy(POLICY), {"omnibase-infra": "dev"})
    return census


@pytest.mark.live_contact(
    "tests/unit/nodes/node_lane_entry_point_census_effect/fixtures/dev_lane_census_recorded.json"
)
def test_dev_lane_runs_seventeen_runtime_image_containers_four_on_onex_runtime(
    recorded_response: dict[str, object],
) -> None:
    census = _census(recorded_response)
    assert census.container_count == len(census.containers) == 17
    assert census.allowed_count == 4
    assert census.remove_count == len(census.remove) == 13
    assert {c.container for c in census.containers if c.verdict == "allowed"} == {
        "omnibase-infra-runtime-worker-1",
        "omnimarket-projection-claude-hook-events-writer",
        "omninode-runtime",
        "omninode-runtime-effects",
    }
    assert all(
        c.command == " ".join((*c.entrypoint, *c.cmd)) for c in census.containers
    )


def test_families_are_derived_from_recorded_commands() -> None:
    assert Counter(c.family for c in _census().containers) == {
        "onex-runtime": 4,
        "python -m omnimarket...handler_*": 6,
        "python -m ...consumer": 4,
        "uvicorn": 2,
        "omnimarket-projection-api": 1,
    }


def test_table_is_regenerated_from_the_recorded_inspection() -> None:
    census = _census()
    table = render_table((census,))
    assert (
        "17 runtime-image containers, 4 on an allowed entry point, 13 to remove"
        in table
    )
    assert "omnibase-intelligence-api" in table
    assert "entry point to remove" in table
    assert all(c.container in table for c in census.containers)
