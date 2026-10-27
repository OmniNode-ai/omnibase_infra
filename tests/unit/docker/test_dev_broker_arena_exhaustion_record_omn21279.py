# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The dev broker's arena-exhaustion record is internally consistent (OMN-21279).

``docs/incidents/OMN-21279/dev-broker-crash-cause.json`` records the allocation
failure aborts the dev broker took after the 3G arena landed (#5385). These
tests keep that record checkable: the abort list is ordered, begins after the
container that carried the cap was created, and matches the count the record
states; and the 12G floor it cites is still declared where it points.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any

import pytest

from tests.unit.docker.conftest import DOCKER_DIR

pytestmark = [pytest.mark.unit]

RECORD = (
    DOCKER_DIR.parent
    / "docs"
    / "incidents"
    / "OMN-21279"
    / "dev-broker-crash-cause.json"
)
CAP_CREATED = datetime(2026, 10, 26, 18, 57, 22, tzinfo=UTC)


def _stamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)


def _record() -> dict[str, Any]:
    loaded = json.loads(RECORD.read_text())
    assert isinstance(loaded, dict)
    section = loaded["arena_exhaustion_since_cap"]
    assert isinstance(section, dict)
    return section


def test_aborts_are_ordered_and_follow_the_cap() -> None:
    stamps = [_stamp(abort["at"]) for abort in _record()["aborts"]]
    assert stamps, "the record lists no abort"
    assert stamps == sorted(set(stamps))
    assert stamps[0] > CAP_CREATED


def test_every_abort_names_the_failed_allocation() -> None:
    for abort in _record()["aborts"]:
        assert abort["shard_task"] in {"fetch", "main"}
        assert abort["failed_bytes"] > 0


def test_stated_abort_count_matches_the_list() -> None:
    section = _record()
    stated = re.match(r"(\d+) seastar allocation-failure aborts", section["result"])
    assert stated, "result must open with the abort count"
    assert int(stated.group(1)) == len(section["aborts"])


def test_record_names_the_decision_it_leaves_open() -> None:
    assert "OPERATOR-CONSENT 2026-10-25T17:53:44Z" in _record()["decision_needed"]


def test_cited_twelve_gig_floor_is_still_declared() -> None:
    cited = _record()["load_on_the_arena"]["declared_floor"]
    assert "docker-compose.dev-bus-ha.yml" in cited
    declared = (DOCKER_DIR / "docker-compose.dev-bus-ha.yml").read_text()
    assert "12G is the floor OMN-20266 measured" in declared
