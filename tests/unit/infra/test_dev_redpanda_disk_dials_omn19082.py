# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-19082/OMN-19419: inherited disk dials retain DLQ and SASL safety.

The shared profile replaces the dev/dogfood command copies. These regressions
still assert the values, prior DLQ pinning and authenticated deploy ordering.
"""

from __future__ import annotations

import shlex
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import pytest

from tests.unit.infra.test_broker_profile_omn19419 import _render

_DOCKER = Path(__file__).resolve().parents[3] / "docker"
_DEV_LANE = _DOCKER / "docker-compose.dev-lane.yml"
_DOGFOOD = _DOCKER / "docker-compose.dogfood.yml"
_BASE = _DOCKER / "docker-compose.infra.yml"

_DIALS = ("segment_fallocation_step", "log_segment_ms", "retention_bytes")
_RETENTION_AFFECTING = ("log_segment_ms", "retention_bytes")
_BASE_KEYS = ("topic_partitions_per_shard", "topic_memory_per_partition")
# The cluster default segment age the DLQ topics inherit before this change.
_FOURTEEN_DAYS_MS = 14 * 24 * 3600 * 1000


@cache
def _service(path: Path, name: str) -> dict[str, Any]:
    lane = "dogfood" if path == _DOGFOOD else "infra" if path == _BASE else "dev"
    return _render(lane)["services"][name]


def _script_lines(path: Path) -> list[str]:
    command = _service(path, "redpanda-partition-cap")["command"]
    # Compose turns `$$` into a literal `$` before the shell sees it.
    script = "\n".join(str(c) for c in command).replace("$$", "$")
    for key, value in _service(path, "redpanda-partition-cap")["environment"].items():
        script = script.replace(f"${key}", str(value))
    return script.splitlines()


def _cluster_sets(lines: list[str]) -> dict[str, tuple[int, str]]:
    """Map each `rpk cluster config set KEY VALUE` to (line index, value)."""
    found: dict[str, tuple[int, str]] = {}
    for index, line in enumerate(lines):
        if (
            not line.strip().startswith("/usr/bin/rpk")
            or "cluster config set" not in line
        ):
            continue
        parts = shlex.split(line)
        if parts[:1] != ["/usr/bin/rpk"]:
            continue
        for start in range(1, len(parts) - 4):
            if parts[start : start + 3] == ["cluster", "config", "set"]:
                found[parts[start + 3]] = (index, parts[start + 4])
                break
    return found


def _first_index(lines: list[str], needle: str) -> int:
    for index, line in enumerate(lines):
        if needle in line:
            return index
    raise AssertionError(f"{needle!r} not found in the dev-lane partition-cap script")


@pytest.mark.unit
def test_the_override_is_parsed() -> None:
    """Rule 16 positive control: the dev-lane script is read and has cluster sets."""
    assert len(_cluster_sets(_script_lines(_DEV_LANE))) == 5


@pytest.mark.unit
@pytest.mark.parametrize("key", _DIALS)
def test_the_dev_broker_matches_the_dogfood_dial(key: str) -> None:
    dev = _cluster_sets(_script_lines(_DEV_LANE))
    dogfood = _cluster_sets(_script_lines(_DOGFOOD))
    assert key in dogfood, f"dogfood broker no longer sets {key}"
    assert dev[key][1] == dogfood[key][1]


@pytest.mark.unit
@pytest.mark.parametrize("key", _BASE_KEYS)
def test_the_override_keeps_the_base_value_it_replaces(key: str) -> None:
    dev = _cluster_sets(_script_lines(_DEV_LANE))
    base = _cluster_sets(_script_lines(_BASE))
    assert dev[key][1] == base[key][1]


@pytest.mark.unit
@pytest.mark.parametrize("key", _RETENTION_AFFECTING)
def test_dlq_topics_are_pinned_before_the_cluster_value_is_set(key: str) -> None:
    lines = _script_lines(_DEV_LANE)
    cluster_index = _cluster_sets(lines)[key][0]
    assert _first_index(lines, "--set retention.bytes=-1") < cluster_index
    assert _first_index(lines, f"--set segment.ms={_FOURTEEN_DAYS_MS}") < cluster_index
    assert (
        _first_index(lines, "still inherits retention.bytes or segment.ms")
        < cluster_index
    )


@pytest.mark.unit
def test_dlq_selection_covers_both_naming_forms() -> None:
    """Both `onex.dlq.*` and `*-dlq.v1` topics match the selector."""
    selector = next(
        line for line in _script_lines(_DEV_LANE) if line.startswith("DLQ=")
    )
    assert "tolower($1) ~ /dlq/" in selector


@pytest.mark.unit
def test_credentials_never_reach_a_command_line() -> None:
    script = "\n".join(_script_lines(_DEV_LANE))
    assert "-X user=" not in script
    assert "-X pass=" not in script
    assert (
        _service(_DEV_LANE, "redpanda-partition-cap")["environment"]["RPK_PASS"]
        == "render-only"
    )


@pytest.mark.unit
def test_the_override_runs_after_the_sasl_flip() -> None:
    depends_on = _service(_DEV_LANE, "redpanda-partition-cap")["depends_on"]
    assert (
        depends_on["redpanda-sasl-enable"]["condition"]
        == "service_completed_successfully"
    )


@pytest.mark.unit
@pytest.mark.parametrize("key", _DIALS)
def test_the_shared_service_carries_the_dial(key: str) -> None:
    assert key in _cluster_sets(_script_lines(_BASE))
