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

import pytest

from tests.unit.docker.test_ci_bus_broker_isolation_omn18691 import CI_BUS_COMPOSE

pytestmark = [pytest.mark.unit]

#: Smallest budget (GiB) known to recover the 2026-10-31 production volume.
RECOVERY_FLOOR_GIB = 4

#: The broker's own flag line: ``- --memory ${CI_BUS_REDPANDA_MEMORY:-4G}``.
_BROKER_FLAG = re.compile(
    r"^\s*-\s+--memory\s+\$\{CI_BUS_REDPANDA_MEMORY:-(\d+)G\}\s*$", re.MULTILINE
)
#: The partition-cap job's sizing input.
_PROFILE = re.compile(
    r"^\s*REDPANDA_PROFILE_MEMORY:\s+\$\{CI_BUS_REDPANDA_MEMORY:-(\d+)G\}\s*$",
    re.MULTILINE,
)


def _text() -> str:
    return CI_BUS_COMPOSE.read_text(encoding="utf-8")


def test_default_memory_budget_covers_recovery() -> None:
    """The broker's default budget is at least the figure that recovered the volume."""
    matches = _BROKER_FLAG.findall(_text())
    assert len(matches) == 1, (
        "the dedicated CI broker must declare exactly one --memory dial"
    )
    assert int(matches[0]) >= RECOVERY_FLOOR_GIB


def test_partition_cap_profile_follows_the_same_dial() -> None:
    """The partition-cap job sizes against the broker's own budget, not its own."""
    broker = _BROKER_FLAG.findall(_text())
    profile = _PROFILE.findall(_text())
    assert len(profile) == 1
    assert profile == broker


#: Longest retention (seconds) that keeps the per-run group pile bounded.
MAX_GROUP_OFFSET_RETENTION_SEC = 86400

_RETENTION = re.compile(
    r"^\s*REDPANDA_GROUP_OFFSET_RETENTION_SEC:\s+"
    r"\$\{CI_BUS_GROUP_OFFSET_RETENTION_SEC:-(\d+)\}\s*$",
    re.MULTILINE,
)


def test_group_offsets_expire_within_a_day() -> None:
    """Per-run lab-run groups must not pile up for the seven-day cluster default.

    ``lab_work_client.consume.v1.<12 hex>.cap.__t.<topic>`` is a new group per
    call; about 15,000 of them sat in ``__consumer_offsets`` on 2026-10-31 and
    the broker replays every one into heap at each start.
    """
    matches = _RETENTION.findall(_text())
    assert len(matches) == 1, "the broker setup job must declare the retention"
    assert 0 < int(matches[0]) <= MAX_GROUP_OFFSET_RETENTION_SEC
