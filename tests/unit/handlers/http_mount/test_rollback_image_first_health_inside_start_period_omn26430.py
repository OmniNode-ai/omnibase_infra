# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-26430: the dev rollback image's first /health 200 reads inside the start period.

The recorded dev reading (``tests/fixtures/omn26430/rollback_image_boot_readings.json``)
has the boot-good rollback image answering its first ``/health`` 200 1032 s after
the container started. The dev lane compose gives its runtime services a
``start_period`` of 1200 s or 1800 s, so that first answer sits inside the start
period. This test pins the reading against the compose file and reads no value
into either side.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[4]
_FIXTURE = (
    _ROOT / "tests" / "fixtures" / "omn26430" / "rollback_image_boot_readings.json"
)
_COMPOSE = _ROOT / "docker" / "docker-compose.dev-lane.yml"


def _runtime_start_periods_seconds() -> list[int]:
    text = _COMPOSE.read_text(encoding="utf-8")
    blocks = re.findall(
        r"onex-container-healthcheck(?:.|\n){0,200}?start_period:\s*(\d+)s", text
    )
    return [int(value) for value in blocks]


def test_rollback_image_first_health_inside_start_period() -> None:
    reading = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    first_answer = reading["dev_lane_reading"]["first_health_200_seconds_after_start"]
    periods = _runtime_start_periods_seconds()
    assert periods, "no semantic-healthcheck start_period found in the dev lane compose"
    assert first_answer < min(periods), (first_answer, periods)


def test_rollback_image_fixture_names_the_held_frames() -> None:
    reading = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    sandbox = reading["sandbox_reading"]
    assert any("wire_from_manifest" in f for f in sandbox["rollback_image_held_frames"])
    assert any(
        "TopicProvisioner" in f for f in sandbox["rollback_image_second_held_frames"]
    )
    assert (
        sandbox["rollback_image_probe_unreachable_after_first_answer"]["elapsed_ms"]
        > 1000 * sandbox["probe_urlopen_timeout_seconds"]
    )
