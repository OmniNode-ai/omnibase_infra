# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-23024: the broker-lag alarm reads the runtime's live consumer groups.

## The defect

``config/lab_alarm.json`` declared six standalone projection-writer groups
(``local.omnimarket-projections.<name>-writer.consume.v1``). The writers were
folded into omninode-runtime (omnibase_infra#5743) and those groups have had no
members since: read on the dev broker 2026-11-07 the delegation one was
``Empty`` with 0 members, three others never held a commit, and the alarm
watched groups nobody consumes while the runtime's own groups carried the lag
(159 on delegate-skill-completed, 3 on delegate-skill-failed).

## What replaces them

The runtime mints one group per subscribed topic from each node's contract
identity, ``{env}.{package}.{node}.consume.{version}.__i.{instance}.__t.{topic}``.
The alarm now names those groups for all six writers. They are not typed by hand:
``CONTRACTS`` below is the contract data (name, version, subscribed topics) of the
six nodes as the running image served it on 2026-11-07, and the expected set is
DERIVED from it through the derivation the lab-pass receipt uses
(``scripts/runtime_build/declared_consumer_groups.py``). The shipped config must
equal the derived set, so a contract version bump that is recorded here turns this
file red until the config follows, and the failing control below shows it does.

The growth rule is not touched: a group that grows between two consecutive
samples alarms, and a flat lag, however high, does not. No bound was added or
widened.

The contract data is a fixture so this file runs everywhere and never skips.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.lab_alarm import (
    EnumConditionOutcome,
    GroupLagReader,
    ModelAlarmConfig,
    evaluate_consumer_group_lag,
)
from scripts.runtime_build.declared_consumer_groups import (
    INSTANCE_INFIX,
    TOPIC_INFIX,
    derive_expected_keys,
)
from tests.helpers.lane_census_fixtures import use_neutral_lane_census

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "config" / "lab_alarm.json"

ENV = "local"
INSTANCE = "runtime-main"

#: The standalone groups the alarm used to read; none holds a member.
RETIRED_STANDALONE_GROUPS = (
    "local.omnimarket-projections.delegation-writer.consume.v1",
    "local.omnimarket-projections.live-events-writer.consume.v1",
    "local.omnimarket-projections.registration-writer.consume.v1",
    "local.omnimarket-projections.savings-writer.consume.v1",
    "local.omnimarket-projections.tenant-credentials-writer.consume.v1",
    "local.omnimarket-projections.tenant-registry-writer.consume.v1",
)

#: node -> (contract version, subscribed topics), read from the dev runtime's
#: ``/v1/introspection/manifest`` on 2026-11-07.
CONTRACTS: dict[str, tuple[tuple[int, int, int], tuple[str, ...]]] = {
    "projection_delegation": (
        (1, 0, 0),
        (
            "onex.evt.omniclaude.delegation-shadow-comparison.v1",
            "onex.evt.omnimarket.node-generation-completed.v1",
            "onex.evt.omnimarket.node-generation-failed.v1",
            "onex.evt.omnimarket.delegate-skill-completed.v1",
            "onex.evt.omnimarket.delegate-skill-failed.v1",
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
            "onex.evt.omnibase-infra.quality-gate-result.v1",
        ),
    ),
    "projection_live_events": (
        (1, 3, 0),
        (
            "onex.evt.platform.log-entry.v1",
            "onex.evt.platform.node-heartbeat.v1",
            "onex.evt.platform.node-introspection.v1",
            "onex.evt.platform.node-state-change.v1",
            "onex.evt.platform.node-registration.v1",
            "onex.cmd.omnimarket.delegate-skill.v1",
            "onex.evt.omnimarket.delegate-skill-completed.v1",
            "onex.evt.omnimarket.delegate-skill-failed.v1",
            "onex.cmd.omnibase-infra.delegation-request.v1",
            "onex.evt.omnibase-infra.routing-decision.v1",
            "onex.evt.omnibase-infra.inference-response.v1",
            "onex.evt.omnibase-infra.quality-gate-result.v1",
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
            "onex.cmd.omnimarket.node-generation-requested.v1",
            "onex.evt.omnimarket.node-generation-completed.v1",
            "onex.evt.omnimarket.node-generation-failed.v1",
            "onex.cmd.omnimarket.node-deploy.v1",
            "onex.evt.omnimarket.generated-node-invoked.v1",
            "onex.evt.omnimarket.delegation-escalation-triggered.v1",
            "onex.cmd.omnimarket.tool-reuse-match-requested.v1",
            "onex.evt.omnimarket.tool-reuse-matched.v1",
            "onex.evt.omnimarket.tool-reuse-no-match.v1",
        ),
    ),
    "projection_registration": (
        (1, 0, 0),
        (
            "onex.evt.platform.node-introspection.v1",
            "onex.evt.platform.node-heartbeat.v1",
            "onex.evt.platform.node-state-change.v1",
        ),
    ),
    "projection_savings": (
        (1, 0, 0),
        (
            "onex.evt.omnibase-infra.savings-estimated.v1",
            "onex.evt.omnimarket.delegate-skill-completed.v1",
            "onex.evt.omnimarket.delegate-skill-failed.v1",
            "onex.evt.omnibase-infra.delegation-completed.v1",
            "onex.evt.omnibase-infra.delegation-failed.v1",
            "onex.evt.omniclaude.hook-event.v1",
        ),
    ),
    "projection_tenant_credentials": (
        (1, 0, 0),
        (
            "onex.evt.omnimarket.credential-registered.v1",
            "onex.evt.omnimarket.credential-revoked.v1",
            "onex.evt.omnimarket.tenant-credential-state.v1",
            "onex.evt.omnimarket.tenant-routing-overlay-state.v1",
            "onex.evt.omnimarket.delegate-skill-completed.v1",
            "onex.evt.omnimarket.delegate-skill-failed.v1",
        ),
    ),
    "projection_tenant_registry": ((1, 0, 0), ("onex.tenant.events",)),
}

COMPLETED_TOPIC = "onex.evt.omnimarket.delegate-skill-completed.v1"
COMPLETED_GROUP = (
    "local.omnimarket.projection_delegation.consume.1.0.0.__i.runtime-main."
    f"__t.{COMPLETED_TOPIC}"
)


def _derived_groups(
    bumped: dict[str, tuple[int, int, int]] | None = None,
) -> tuple[str, ...]:
    """Every group the runtime mints for CONTRACTS, derived, never typed."""
    bumped = bumped or {}
    contracts: list[dict[str, Any]] = []
    for name, (version, topics) in CONTRACTS.items():
        major, minor, patch = bumped.get(name, version)
        contracts.append(
            {
                "name": name,
                "package_name": "omnimarket",
                "contract_version": {"major": major, "minor": minor, "patch": patch},
                "event_bus": {
                    "plugin_managed": False,
                    "subscribe_topics": list(topics),
                },
            }
        )
    keys = derive_expected_keys([{"contracts": contracts}], env=ENV)
    return tuple(
        sorted(
            f"{k.base_group_id}{INSTANCE_INFIX}{INSTANCE}{TOPIC_INFIX}{k.topic}"
            for k in keys
        )
    )


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
# The binding: the alarm names the runtime's groups, derived from contracts.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_the_alarm_declares_exactly_the_groups_derived_from_the_contracts(
    declared_groups: tuple[str, ...],
) -> None:
    expected = _derived_groups()
    assert sorted(declared_groups) == list(expected), (
        "the shipped consumer_groups differ from the set derived from the "
        f"contracts: missing {sorted(set(expected) - set(declared_groups))}, "
        f"extra {sorted(set(declared_groups) - set(expected))}"
    )


@pytest.mark.unit
def test_no_standalone_writer_group_is_declared(
    declared_groups: tuple[str, ...],
) -> None:
    assert not set(declared_groups) & set(RETIRED_STANDALONE_GROUPS)
    assert all(INSTANCE_INFIX in g and TOPIC_INFIX in g for g in declared_groups)


@pytest.mark.unit
def test_every_one_of_the_six_writers_is_read_through_the_runtime(
    declared_groups: tuple[str, ...],
) -> None:
    for name in CONTRACTS:
        assert any(f".omnimarket.{name}.consume." in g for g in declared_groups), name
    assert (
        "local.omnimarket.projection_tenant_registry.consume.1.0.0."
        "__i.runtime-main.__t.onex.tenant.events" in declared_groups
    )
    assert COMPLETED_GROUP in declared_groups


@pytest.mark.unit
def test_the_shipped_json_has_no_retired_group() -> None:
    declared = json.loads(CONFIG.read_text(encoding="utf-8"))["consumer_groups"]
    assert not set(declared) & set(RETIRED_STANDALONE_GROUPS)
    assert len(declared) == len(set(declared)) == 47


@pytest.mark.unit
def test_a_recorded_version_bump_turns_the_binding_red(
    declared_groups: tuple[str, ...],
) -> None:
    """The failing control: the derivation follows a contract bump and the
    shipped config, which has not been repinned, no longer matches it."""
    bumped = _derived_groups({"projection_delegation": (1, 1, 0)})
    assert sorted(declared_groups) != list(bumped)
    assert COMPLETED_GROUP not in bumped
    assert (
        "local.omnimarket.projection_delegation.consume.1.1.0.__i.runtime-main."
        f"__t.{COMPLETED_TOPIC}" in bumped
    )


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
def test_planted_lag_on_a_formerly_standalone_writer_now_alarms(
    declared_groups: tuple[str, ...],
) -> None:
    """The five writers the first repair left on dead groups are read live."""
    savings = next(g for g in declared_groups if ".projection_savings." in g)
    previous = dict.fromkeys(declared_groups, 10)
    planted = dict(previous)
    planted[savings] = 11

    report, _ = evaluate_consumer_group_lag(
        _reader(planted), declared_groups, previous=previous
    )

    assert report.outcome is EnumConditionOutcome.ALARM
    assert [a.subject for a in report.alarms] == [savings]


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
def test_lag_on_a_retired_standalone_group_raises_nothing_because_it_is_not_read(
    declared_groups: tuple[str, ...],
) -> None:
    """The control for the control: a dead group's lag is not this alarm's read."""
    previous = dict.fromkeys(declared_groups, 0)
    readings = dict(previous)
    for retired in RETIRED_STANDALONE_GROUPS:
        readings[retired] = 874

    report, sample = evaluate_consumer_group_lag(
        _reader(readings), declared_groups, previous=previous
    )

    assert not set(sample) & set(RETIRED_STANDALONE_GROUPS)
    assert report.outcome is EnumConditionOutcome.OK
