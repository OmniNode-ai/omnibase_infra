# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23024: the broker-lag alarm reads the runtime's delegation writer groups.

## The defect

``config/lab_alarm.json`` declared the standalone delegation writer's group
``local.omnimarket-projections.delegation-writer.consume.v1``. The writer was
folded into omninode-runtime (omnibase_infra#5743) and that group has had no
members since: read on the dev broker 2026-11-07 it was ``Empty`` with 0
members, so the alarm watched a group nobody consumes while the runtime's own
groups carried the lag (159 on delegate-skill-completed, 3 on
delegate-skill-failed).

## What replaces it

The runtime mints one group per subscribed topic from the node's contract
identity, ``{env}.{package}.{node}.consume.{version}.__i.{instance}.__t.{topic}``.
For ``node_projection_delegation`` that is the eight names pinned below, each read
``Stable`` with one member on the dev broker on 2026-11-07 through the alarm's own
in-process reader. The growth rule is not touched: a group that grows between two
consecutive samples alarms, and a flat lag, however high, does not.

The names are pinned literally rather than derived because the node's package is
not a dependency of this repository (a derivation test would have to skip where
it is absent). A contract version bump mints new names; this file is the thing
that goes red and names the repin.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.lab_alarm import (
    EnumConditionOutcome,
    GroupLagReader,
    ModelAlarmConfig,
    evaluate_consumer_group_lag,
)
from tests.helpers.lane_census_fixtures import use_neutral_lane_census

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "lab_alarm.json"

RETIRED_STANDALONE_GROUP = "local.omnimarket-projections.delegation-writer.consume.v1"

RUNTIME_BASE = "local.omnimarket.projection_delegation.consume.1.0.0"
RUNTIME_INSTANCE = "runtime-main"
DELEGATION_TOPICS = (
    "onex.evt.omnibase-infra.delegation-completed.v1",
    "onex.evt.omnibase-infra.delegation-failed.v1",
    "onex.evt.omnibase-infra.quality-gate-result.v1",
    "onex.evt.omniclaude.delegation-shadow-comparison.v1",
    "onex.evt.omnimarket.delegate-skill-completed.v1",
    "onex.evt.omnimarket.delegate-skill-failed.v1",
    "onex.evt.omnimarket.node-generation-completed.v1",
    "onex.evt.omnimarket.node-generation-failed.v1",
)
LIVE_DELEGATION_GROUPS = tuple(
    f"{RUNTIME_BASE}.__i.{RUNTIME_INSTANCE}.__t.{topic}" for topic in DELEGATION_TOPICS
)

#: Declared before this change and not touched by it.
UNCHANGED_STANDALONE_GROUPS = (
    "local.omnimarket-projections.live-events-writer.consume.v1",
    "local.omnimarket-projections.registration-writer.consume.v1",
    "local.omnimarket-projections.savings-writer.consume.v1",
    "local.omnimarket-projections.tenant-credentials-writer.consume.v1",
    "local.omnimarket-projections.tenant-registry-writer.consume.v1",
)

COMPLETED_GROUP = LIVE_DELEGATION_GROUPS[
    DELEGATION_TOPICS.index("onex.evt.omnimarket.delegate-skill-completed.v1")
]


@pytest.fixture(autouse=True)
def _neutral_lane_census(monkeypatch: pytest.MonkeyPatch) -> None:
    use_neutral_lane_census(monkeypatch)


@pytest.fixture
def declared_groups(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[str, ...]:
    monkeypatch.setenv("ONEX_INFRA_HOST", "lab.invalid")
    monkeypatch.setenv("ONEX_RUNTIME_SSH_HOST", "user@lab.invalid")
    monkeypatch.setenv("OMNI_HOME", str(tmp_path / "omni-home"))
    return ModelAlarmConfig.load(CONFIG).consumer_groups


def _reader(lags: dict[str, int]) -> GroupLagReader:
    def read(group: str) -> int:
        if group not in lags:
            raise ValueError(f"group {group} unreadable (unknown to this fixture)")
        return lags[group]

    return read


# ---------------------------------------------------------------------------
# The binding: the alarm names the runtime's groups, not the orphaned one.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_alarm_no_longer_reads_the_orphaned_standalone_group(
    declared_groups: tuple[str, ...],
) -> None:
    assert RETIRED_STANDALONE_GROUP not in declared_groups


@pytest.mark.unit
def test_the_alarm_reads_every_live_runtime_delegation_group(
    declared_groups: tuple[str, ...],
) -> None:
    missing = [g for g in LIVE_DELEGATION_GROUPS if g not in declared_groups]
    assert missing == [], f"declared but not bound to the runtime's groups: {missing}"
    assert len(LIVE_DELEGATION_GROUPS) == 8


@pytest.mark.unit
def test_every_runtime_delegation_group_shares_the_contract_identity() -> None:
    for group in LIVE_DELEGATION_GROUPS:
        base, _, rest = group.partition(".__i.")
        instance, _, topic = rest.partition(".__t.")
        assert base == RUNTIME_BASE
        assert instance == RUNTIME_INSTANCE
        assert topic in DELEGATION_TOPICS


@pytest.mark.unit
def test_the_other_declared_groups_are_untouched(
    declared_groups: tuple[str, ...],
) -> None:
    assert all(group in declared_groups for group in UNCHANGED_STANDALONE_GROUPS)
    assert len(declared_groups) == len(LIVE_DELEGATION_GROUPS) + len(
        UNCHANGED_STANDALONE_GROUPS
    )


@pytest.mark.unit
def test_the_shipped_json_names_the_groups_literally() -> None:
    declared = json.loads(CONFIG.read_text(encoding="utf-8"))["consumer_groups"]
    assert RETIRED_STANDALONE_GROUP not in declared
    assert COMPLETED_GROUP in declared


# ---------------------------------------------------------------------------
# The planted-lag control: the alarm fires on these groups, rule unchanged.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_planted_lag_growth_on_the_runtime_group_raises_an_alarm(
    declared_groups: tuple[str, ...],
) -> None:
    previous = dict.fromkeys(declared_groups, 0)
    planted = dict(previous)
    planted[COMPLETED_GROUP] = 159

    report, _ = evaluate_consumer_group_lag(
        _reader(planted), declared_groups, previous=previous
    )

    assert report.outcome is EnumConditionOutcome.ALARM
    assert [a.subject for a in report.alarms] == [COMPLETED_GROUP]


@pytest.mark.unit
def test_growth_by_one_still_alarms_and_flat_lag_still_does_not(
    declared_groups: tuple[str, ...],
) -> None:
    """The rule is untouched: no bound was added and none was widened."""
    previous = dict.fromkeys(declared_groups, 159)

    grown = dict(previous)
    grown[COMPLETED_GROUP] = 160
    report, _ = evaluate_consumer_group_lag(
        _reader(grown), declared_groups, previous=previous
    )
    assert report.outcome is EnumConditionOutcome.ALARM

    report, _ = evaluate_consumer_group_lag(
        _reader(dict(previous)), declared_groups, previous=previous
    )
    assert report.outcome is EnumConditionOutcome.OK


@pytest.mark.unit
def test_lag_on_the_retired_group_raises_nothing_because_it_is_no_longer_read(
    declared_groups: tuple[str, ...],
) -> None:
    """The control for the control: a dead group's lag is not this alarm's read."""
    previous = dict.fromkeys(declared_groups, 0)
    readings = dict(previous)
    readings[RETIRED_STANDALONE_GROUP] = 874

    report, sample = evaluate_consumer_group_lag(
        _reader(readings), declared_groups, previous=previous
    )

    assert RETIRED_STANDALONE_GROUP not in sample
    assert report.outcome is EnumConditionOutcome.OK
