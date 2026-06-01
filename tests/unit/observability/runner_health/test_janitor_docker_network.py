# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Unit tests for the bounded Docker network janitor + subnet-pool alerting.

Covers the four OMN-12566 acceptance criteria:

  1. leak -> reclaim          : an owned, idle, aged-out network is reclaimed.
  2. unknown-ownership-preserved: a network matching no rule is preserved.
  3. active-lane-never-pruned : a network with attached containers is preserved
                                even when it matches a rule and is old.
  4. alert threshold          : a simulated leak crossing the pre-exhaustion
                                threshold fires a pool-pressure alert.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

import omnibase_infra.observability.runner_health.janitor_docker_network as janitor_module
from omnibase_infra.observability.runner_health.enum_network_disposition import (
    EnumNetworkDisposition,
)
from omnibase_infra.observability.runner_health.janitor_docker_network import (
    JanitorDockerNetwork,
    classify_network,
)
from omnibase_infra.observability.runner_health.model_network_info import (
    ModelNetworkInfo,
)
from omnibase_infra.observability.runner_health.model_network_ownership_rule import (
    DEFAULT_OWNERSHIP_RULES,
    ModelNetworkOwnershipRule,
)
from omnibase_infra.observability.runner_health.model_network_pool_alert import (
    build_pool_alert_if_pressured,
)
from omnibase_infra.observability.runner_health.model_network_pool_status import (
    ModelNetworkPoolStatus,
)

_NOW = datetime(2026, 6, 1, 18, 0, 0, tzinfo=UTC)
_RULE = DEFAULT_OWNERSHIP_RULES[0]


def _boot_network(
    *,
    name: str = "omnibase-infra-boot-12345-1_default",
    age: timedelta = timedelta(hours=3),
    containers: int = 0,
    created_at: datetime | None = None,
    is_builtin: bool = False,
) -> ModelNetworkInfo:
    return ModelNetworkInfo(
        network_ref=f"net-{name}",
        name=name,
        created_at=_NOW - age if created_at is None else created_at,
        container_count=containers,
        is_builtin=is_builtin,
    )


# --------------------------------------------------------------------------- #
# Acceptance 1: leak -> reclaim
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_owned_idle_aged_network_is_reclaimed() -> None:
    decision = classify_network(_boot_network(), DEFAULT_OWNERSHIP_RULES, _NOW)
    assert decision.disposition is EnumNetworkDisposition.RECLAIM
    assert decision.matched_rule == _RULE.name
    assert decision.age_seconds is not None and decision.age_seconds >= 0


@pytest.mark.unit
def test_dash_network_suffix_variant_is_owned() -> None:
    # reusable-runtime-boot.yml also derives `<project>-network` names.
    decision = classify_network(
        _boot_network(name="omnibase-infra-boot-987-2-network"),
        DEFAULT_OWNERSHIP_RULES,
        _NOW,
    )
    assert decision.disposition is EnumNetworkDisposition.RECLAIM


# --------------------------------------------------------------------------- #
# Acceptance 2: unknown ownership -> preserve
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_unknown_ownership_is_preserved() -> None:
    decision = classify_network(
        _boot_network(name="some-unrelated-prod-network", age=timedelta(days=30)),
        DEFAULT_OWNERSHIP_RULES,
        _NOW,
    )
    assert decision.disposition is EnumNetworkDisposition.PRESERVE_UNKNOWN_OWNERSHIP
    assert decision.matched_rule == ""


@pytest.mark.unit
def test_substring_match_does_not_leak_ownership() -> None:
    # A name that merely CONTAINS the owned prefix but is not a full match
    # must NOT be treated as owned (fullmatch semantics).
    decision = classify_network(
        _boot_network(name="prod-omnibase-infra-boot-1-network"),
        DEFAULT_OWNERSHIP_RULES,
        _NOW,
    )
    assert decision.disposition is EnumNetworkDisposition.PRESERVE_UNKNOWN_OWNERSHIP


@pytest.mark.unit
def test_builtin_networks_are_preserved() -> None:
    for builtin in ("bridge", "host", "none"):
        decision = classify_network(
            _boot_network(name=builtin, is_builtin=True),
            DEFAULT_OWNERSHIP_RULES,
            _NOW,
        )
        assert decision.disposition is EnumNetworkDisposition.PRESERVE_BUILTIN


# --------------------------------------------------------------------------- #
# Acceptance 3: active lane -> never pruned
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_active_lane_with_containers_is_never_reclaimed() -> None:
    # Owned + old, but containers attached -> active lane, must preserve.
    decision = classify_network(
        _boot_network(age=timedelta(hours=10), containers=2),
        DEFAULT_OWNERSHIP_RULES,
        _NOW,
    )
    assert decision.disposition is EnumNetworkDisposition.PRESERVE_ACTIVE


@pytest.mark.unit
def test_owned_but_too_young_is_preserved() -> None:
    decision = classify_network(
        _boot_network(age=timedelta(minutes=5)),
        DEFAULT_OWNERSHIP_RULES,
        _NOW,
    )
    assert decision.disposition is EnumNetworkDisposition.PRESERVE_TOO_YOUNG


@pytest.mark.unit
def test_owned_age_unknown_is_preserved() -> None:
    # created_at=None means age-unknown -> preserve, never guess.
    net = ModelNetworkInfo(
        network_ref="net-x",
        name="omnibase-infra-boot-1-1_default",
        created_at=None,
        container_count=0,
    )
    decision = classify_network(net, DEFAULT_OWNERSHIP_RULES, _NOW)
    assert decision.disposition is EnumNetworkDisposition.PRESERVE_AGE_UNKNOWN


# --------------------------------------------------------------------------- #
# Janitor pass (end to end against injected network lists)
# --------------------------------------------------------------------------- #
@pytest.mark.unit
async def test_janitor_dry_run_reclaims_nothing_but_flags_candidate() -> None:
    janitor = JanitorDockerNetwork(runner_host="testhost")

    async def _fake_fetch() -> list[ModelNetworkInfo]:
        return [
            _boot_network(),  # reclaimable
            _boot_network(name="bridge", is_builtin=True),  # builtin
            _boot_network(name="unrelated-net", age=timedelta(days=5)),  # unknown
        ]

    janitor._fetch_networks = _fake_fetch  # type: ignore[method-assign]
    result = await janitor.run(correlation_id=uuid4(), dry_run=True, now=_NOW)

    assert result.dry_run is True
    assert result.reclaim_candidate_count == 1
    assert result.preserved_count == 2
    assert result.reclaimed == ()  # dry run removes nothing


@pytest.mark.unit
async def test_janitor_execute_removes_only_reclaim_candidates() -> None:
    janitor = JanitorDockerNetwork(runner_host="testhost")
    removed: list[str] = []

    async def _fake_fetch() -> list[ModelNetworkInfo]:
        return [
            _boot_network(name="omnibase-infra-boot-aaa-1_default"),  # reclaim
            _boot_network(
                name="omnibase-infra-boot-bbb-1_default",
                containers=1,
            ),  # active -> preserve
            _boot_network(name="prod-db-net", age=timedelta(days=2)),  # unknown
        ]

    async def _fake_remove(ids: list[str]) -> list[str]:
        removed.extend(ids)
        return []

    janitor._fetch_networks = _fake_fetch  # type: ignore[method-assign]
    janitor._remove_networks = _fake_remove  # type: ignore[method-assign]
    result = await janitor.run(correlation_id=uuid4(), dry_run=False, now=_NOW)

    assert removed == ["net-omnibase-infra-boot-aaa-1_default"]
    assert result.reclaimed == ("net-omnibase-infra-boot-aaa-1_default",)
    assert result.reclaim_errors == ()


@pytest.mark.unit
async def test_janitor_never_issues_blanket_prune_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Behavioral guard: every shell command the janitor sends over SSH for
    # both inspection and removal must use targeted `docker network rm`, never
    # a blanket `docker network prune`. Capture the actual subprocess argv.
    captured_commands: list[str] = []

    class _FakeProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"", b""

    async def _fake_exec(*args: str, **_kwargs: object) -> _FakeProc:
        # ssh invocations are ("ssh", host, command); capture the command body.
        if len(args) >= 3:
            captured_commands.append(args[2])
        return _FakeProc()

    monkeypatch.setattr(
        "omnibase_infra.observability.runner_health.janitor_docker_network."
        "asyncio.create_subprocess_exec",
        _fake_exec,
    )

    janitor = JanitorDockerNetwork(runner_host="testhost")
    # Drive the removal path directly with an explicit candidate set.
    await janitor._remove_networks(["net-a", "net-b"])

    assert captured_commands, "expected at least one SSH command to be captured"
    for cmd in captured_commands:
        assert "network prune" not in cmd, f"blanket prune found in: {cmd}"
    assert any("docker network rm" in cmd for cmd in captured_commands)


@pytest.mark.unit
async def test_fetch_networks_timeout_kills_process_and_logs(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    killed = False
    waited = False

    class _HangingProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(10)
            return b"", b""

        def kill(self) -> None:
            nonlocal killed
            killed = True

        async def wait(self) -> int:
            nonlocal waited
            waited = True
            return 0

    async def _fake_exec(*_args: str, **_kwargs: object) -> _HangingProc:
        return _HangingProc()

    monkeypatch.setattr(
        janitor_module.asyncio,
        "create_subprocess_exec",
        _fake_exec,
    )
    monkeypatch.setattr(janitor_module, "_SSH_FETCH_TIMEOUT_SECONDS", 0.01)

    janitor = JanitorDockerNetwork(runner_host="testhost")
    networks = await janitor._fetch_networks()

    assert networks == []
    assert killed is True
    assert waited is True
    assert "Docker network fetch timed out" in caplog.text


@pytest.mark.unit
async def test_remove_networks_timeout_marks_all_candidates_failed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    killed = False
    waited = False

    class _HangingProc:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(10)
            return b"", b""

        def kill(self) -> None:
            nonlocal killed
            killed = True

        async def wait(self) -> int:
            nonlocal waited
            waited = True
            return 0

    async def _fake_exec(*_args: str, **_kwargs: object) -> _HangingProc:
        return _HangingProc()

    monkeypatch.setattr(
        janitor_module.asyncio,
        "create_subprocess_exec",
        _fake_exec,
    )
    monkeypatch.setattr(janitor_module, "_SSH_REMOVE_TIMEOUT_SECONDS", 0.01)

    janitor = JanitorDockerNetwork(runner_host="testhost")
    failures = await janitor._remove_networks(["net-a", "net-b"])

    assert failures == ["rm_failed:net-a", "rm_failed:net-b"]
    assert killed is True
    assert waited is True
    assert "Docker network removal timed out" in caplog.text


# --------------------------------------------------------------------------- #
# Acceptance 4: alert before subnet-pool exhaustion
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_pool_alert_fires_when_threshold_crossed() -> None:
    status = ModelNetworkPoolStatus(
        host="testhost",
        network_count=26,
        pool_capacity=31,
        warn_threshold_ratio=0.8,
    )
    assert status.is_over_threshold is True
    alert = build_pool_alert_if_pressured(
        status, correlation_id=uuid4(), reclaim_candidate_count=7
    )
    assert alert is not None
    assert alert.remaining_capacity == 5
    assert "7" in alert.to_slack_message()


@pytest.mark.unit
def test_pool_alert_silent_below_threshold() -> None:
    status = ModelNetworkPoolStatus(
        host="testhost",
        network_count=10,
        pool_capacity=31,
        warn_threshold_ratio=0.8,
    )
    assert status.is_over_threshold is False
    assert build_pool_alert_if_pressured(status, correlation_id=uuid4()) is None


@pytest.mark.unit
def test_pool_alert_fires_strictly_before_exhaustion() -> None:
    # The whole point: alert must trigger with capacity STILL remaining, not
    # only once the pool is fully exhausted.
    status = ModelNetworkPoolStatus(host="testhost", network_count=25, pool_capacity=31)
    assert status.remaining_capacity > 0
    assert status.is_over_threshold is True


@pytest.mark.unit
def test_custom_rule_age_threshold_respected() -> None:
    strict_rule = ModelNetworkOwnershipRule(
        name="strict",
        name_pattern=r"ci-temp-.*",
        min_age_seconds=10,
    )
    young = ModelNetworkInfo(
        network_ref="n1",
        name="ci-temp-xyz",
        created_at=_NOW - timedelta(seconds=5),
        container_count=0,
    )
    old = ModelNetworkInfo(
        network_ref="n2",
        name="ci-temp-xyz",
        created_at=_NOW - timedelta(seconds=30),
        container_count=0,
    )
    assert (
        classify_network(young, (strict_rule,), _NOW).disposition
        is EnumNetworkDisposition.PRESERVE_TOO_YOUNG
    )
    assert (
        classify_network(old, (strict_rule,), _NOW).disposition
        is EnumNetworkDisposition.RECLAIM
    )


# --------------------------------------------------------------------------- #
# Subnet-pool capacity derivation
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_pool_capacity_derived_from_address_pool_json() -> None:
    from omnibase_infra.observability.runner_health.collector_network_pool import (
        _capacity_from_pool_json,
    )

    # One /16 base subnetted at /24 -> 256 subnets.
    blob = '[{"Base":"172.17.0.0/16","Size":24}]'
    assert _capacity_from_pool_json(blob) == 256
    # Two pools sum.
    blob2 = '[{"Base":"172.17.0.0/16","Size":24},{"Base":"10.0.0.0/8","Size":24}]'
    assert _capacity_from_pool_json(blob2) == 256 + 2 ** (24 - 8)


@pytest.mark.unit
def test_pool_capacity_returns_zero_on_garbage() -> None:
    from omnibase_infra.observability.runner_health.collector_network_pool import (
        _capacity_from_pool_json,
    )

    assert _capacity_from_pool_json("") == 0
    assert _capacity_from_pool_json("null") == 0
    assert _capacity_from_pool_json("not json") == 0
    assert _capacity_from_pool_json("{}") == 0


@pytest.mark.unit
async def test_collector_falls_back_to_capacity_when_pool_unavailable() -> None:
    from omnibase_infra.observability.runner_health.collector_network_pool import (
        CollectorNetworkPool,
    )

    collector = CollectorNetworkPool(runner_host="testhost", pool_capacity=31)

    async def _count() -> int:
        return 26

    async def _capacity() -> int:
        return 31  # fallback path returns configured capacity

    collector._fetch_network_count = _count  # type: ignore[method-assign]
    collector._fetch_pool_capacity = _capacity  # type: ignore[method-assign]
    status = await collector.collect()
    assert status.network_count == 26
    assert status.pool_capacity == 31
    assert status.is_over_threshold is True
