# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21717: the dedicated CI broker's Seastar memory budget must hold its recovery.

WHAT THIS PROTECTS. Seastar aborts the whole Redpanda process when its
``--memory`` budget is spent. On 2026-10-31 the dedicated CI broker (the one the
lab units and CI publishers share) was started with
``--memory 1G``; a few seconds after every start it logged
``Used memory: 1G / Free memory: 0B``, ``Failed to allocate 64 bytes`` and
``Aborting on shard 0`` (exit 139) while ``kafka::group_manager`` replayed a
383 MB ``__consumer_offsets`` log, and it crash-looped about every 15 s (restart
count 518). Client reconnects landed on each fresh start, so the loop fed itself.

MEASURED, NOT GUESSED. On a copy of the live 6.2 GB data volume, ``--memory 1G``
aborted at 20 s; ``--memory 4G`` was healthy at 20 s and held at 2.2 GiB
resident. The floor below is the measured working figure, not a round number.
Lowering it needs a replay of a copy of the live volume at the new value, not a
reading of this file.
"""

from __future__ import annotations

import re
import shlex
from typing import Any

import pytest

from tests.unit.docker.test_ci_bus_broker_isolation_omn18691 import (
    CI_BUS_COMPOSE,
    _load,
)

pytestmark = [pytest.mark.unit]

#: Smallest budget (GiB) known to recover the 2026-10-31 production volume.
RECOVERY_FLOOR_GIB = 4

_UNIT_GIB = {
    "B": 1 / 1024**3,
    "K": 1 / 1024**2,
    "M": 1 / 1024,
    "G": 1.0,
    "T": 1024.0,
}


def _gib(size: str) -> float:
    """Convert a Seastar size such as ``4G``, ``4096M`` or ``4.5G`` to GiB."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([BKMGT])?", size)
    assert match, f"unparseable Seastar memory size: {size!r}"
    return float(match.group(1)) * _UNIT_GIB[match.group(2) or "B"]


def _services() -> dict[str, Any]:
    """Read service boundaries rather than depending on Compose YAML formatting."""
    services = _load(CI_BUS_COMPOSE)["services"]
    assert isinstance(services, dict), "the CI bus compose must declare services"
    return services


def _broker_memory(services: dict[str, Any]) -> str:
    """Read only the broker command, accepting Compose string and list forms."""
    command = services["redpanda"]["command"]
    entries = [command] if isinstance(command, str) else command
    tokens = [token for entry in entries for token in shlex.split(str(entry))]
    values = []
    for index, token in enumerate(tokens):
        if token == "--memory":
            assert index + 1 < len(tokens), "the broker --memory flag needs a value"
            values.append(tokens[index + 1])
        elif token.startswith("--memory="):
            values.append(token.split("=", 1)[1])
    assert len(values) == 1, (
        "the dedicated CI broker must declare exactly one --memory dial"
    )
    return values[0]


def _setup_environment(services: dict[str, Any]) -> dict[str, str]:
    """Read only the partition-cap job, accepting mapping and list environments."""
    environment = services["redpanda-partition-cap"]["environment"]
    if isinstance(environment, dict):
        return {key: str(value) for key, value in environment.items()}
    return dict(str(entry).split("=", 1) for entry in environment)


def test_default_memory_budget_covers_recovery() -> None:
    """The broker's default budget is at least the figure that recovered the volume."""
    assert _gib(_broker_memory(_services())) >= RECOVERY_FLOOR_GIB


def test_partition_cap_profile_follows_the_same_dial() -> None:
    """The partition-cap job sizes against the broker's own budget, not its own."""
    services = _services()
    broker = _broker_memory(services)
    profile = _setup_environment(services)["REDPANDA_PROFILE_MEMORY"]
    assert _gib(profile) == _gib(broker)


#: Longest retention (seconds) that keeps the per-run group pile bounded.
MAX_GROUP_OFFSET_RETENTION_SEC = 86400


def test_group_offsets_expire_within_a_day() -> None:
    """Per-run lab-run groups must not pile up for the seven-day cluster default.

    ``lab_work_client.consume.v1.<12 hex>.cap.__t.<topic>`` is a new group per
    call; about 15,000 of them sat in ``__consumer_offsets`` on 2026-10-31 and
    the broker replays every one into heap at each start.
    """
    value = _setup_environment(_services())["REDPANDA_GROUP_OFFSET_RETENTION_SEC"]
    match = re.fullmatch(r"\$\{CI_BUS_GROUP_OFFSET_RETENTION_SEC:-(\d+)\}", value)
    assert match, f"the broker setup job must declare the retention default: {value!r}"
    assert 0 < int(match.group(1)) <= MAX_GROUP_OFFSET_RETENTION_SEC


def test_memory_budget_has_no_environment_override() -> None:
    """A stale .env cannot lower either startup or sizing below recovery."""
    services = _services()
    broker = _broker_memory(services)
    profile = _setup_environment(services)["REDPANDA_PROFILE_MEMORY"]
    for value in (broker, profile):
        assert "$" not in value, "the recovery memory budget must be a literal value"
        assert _gib(value) >= RECOVERY_FLOOR_GIB


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("4G", 4),
        ("4096M", 4),
        ("4.5G", 4.5),
        ("100M", 100 / 1024),
        ("0.1G", 0.1),
        ("4194304K", 4),
        ("0.00390625T", 4),
        ("4294967296", 4),
        ("4294967296B", 4),
        ("1", 1 / 1024**3),
    ],
)
def test_memory_size_units(size: str, expected: float) -> None:
    """Equivalent Seastar units compare by size, including bare byte values."""
    assert _gib(size) == pytest.approx(expected)


@pytest.mark.parametrize("size", ["", "bogus", "4X", "-4G", "4Gjunk"])
def test_memory_size_diagnostic(size: str) -> None:
    """Unsupported sizes name the value rather than an unrelated dial count."""
    with pytest.raises(AssertionError, match="unparseable Seastar memory size"):
        _gib(size)


@pytest.mark.parametrize(
    "command",
    [
        "redpanda start --memory 4G",
        ["redpanda", "start", "--memory", "4G"],
        ["redpanda start", "--memory 4G"],
        ["redpanda", "start", "--memory=4G"],
    ],
)
def test_broker_memory_is_scoped_to_service(command: str | list[str]) -> None:
    """Another service's memory flag must not affect the CI broker budget."""
    services = {
        "redpanda": {"command": command},
        "other": {"command": ["--memory 1G"]},
    }
    assert _broker_memory(services) == "4G"


@pytest.mark.parametrize(
    "command",
    ["redpanda start", "redpanda start --memory", "--memory 4G --memory 1G"],
)
def test_broker_memory_requires_one_value(command: str) -> None:
    """A missing or duplicate dial must fail rather than read another service."""
    with pytest.raises(AssertionError, match="--memory"):
        _broker_memory({"redpanda": {"command": command}})


@pytest.mark.parametrize(
    "environment",
    [
        {"REDPANDA_PROFILE_MEMORY": "4G"},
        ["REDPANDA_PROFILE_MEMORY=4G"],
    ],
)
def test_setup_environment_is_scoped_to_service(
    environment: dict[str, str] | list[str],
) -> None:
    """Mapping and list forms read the setup job's own sizing value."""
    services = {
        "redpanda-partition-cap": {"environment": environment},
        "other": {"environment": {"REDPANDA_PROFILE_MEMORY": "1G"}},
    }
    assert _setup_environment(services)["REDPANDA_PROFILE_MEMORY"] == "4G"
