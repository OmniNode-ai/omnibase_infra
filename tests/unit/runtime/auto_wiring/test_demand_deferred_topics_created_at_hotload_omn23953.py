# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Demand-deferred contracts' topics are created when the contract starts (OMN-23953).

Boot provisioning (OMN-23951) creates the publish topics of every contract the
per-contract interleave never reached. A contract the demand split defers
(OMN-18087: none of its subscribe topics holds a record, so its consumer groups
join on its first request) is such a contract, yet it will go through the whole
provision -> ready -> attach interleave, publish topics included, when its first
record arrives. Creating its publish topics at boot as well created topics for
contracts that may never load.

The RED control is ``test_deferred_contract_publish_topic_is_not_created_at_boot``:
it fails on the base commit (the unattached pass created the deferred
contract's publish topic) and passes once the pass is told which contracts the
interleave owns, deferred ones included.
``test_publish_only_contract_topic_still_created_beside_a_deferred_contract``
is the control for the other side of the rule: dropping the publish-only side
from the rule fails it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import cast
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.event_bus.enum_contract_attach_exclusion_reason import (
    EnumContractAttachExclusionReason,
)
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.event_bus_kafka import EventBusKafka
from omnibase_infra.event_bus.model_contract_attach_exclusion import (
    ModelContractAttachExclusion,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_set_readiness import ModelTopicSetReadiness
from omnibase_infra.handlers.handler_kafka_topic_demand_source import (
    HandlerKafkaTopicDemandSource,
)
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
    ModelEventBusWiring,
    ModelHandlerRef,
    ModelHandlerRouting,
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

IN_USE_SUBSCRIBE_TOPIC = "onex.evt.platform.hotload-pass-in-use.v1"
IN_USE_PUBLISH_TOPIC = "onex.evt.platform.hotload-pass-in-use-emitted.v1"
DEFERRED_SUBSCRIBE_TOPIC = "onex.cmd.platform.hotload-pass-deferred.v1"
DEFERRED_PUBLISH_TOPIC = "onex.evt.platform.hotload-pass-deferred-emitted.v1"
PUBLISH_ONLY_TOPIC = "onex.evt.omnimarket.hotload-pass-floor-snapshot.v1"


class RecordingProvisioner:
    """Fake provisioner recording provision + readiness calls in order."""

    def __init__(self, *, failing: Mapping[str, Exception] | None = None) -> None:
        self.calls: list[tuple[str, str]] = []
        self._failing = dict(failing or {})

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: object | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        self.calls.append(("provision", topic_name))
        if topic_name in self._failing:
            raise self._failing[topic_name]
        return True

    async def confirm_topics_ready(
        self,
        topics: Sequence[str],
        *,
        expected_specs: Mapping[str, object] | None = None,
        config: ModelTopicReadinessConfig | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelTopicSetReadiness:
        for topic in topics:
            self.calls.append(("ready", topic))
        return ModelTopicSetReadiness(
            topics=tuple(topics),
            status=EnumTopicReadinessStatus.READY,
            ready_topics=tuple(topics),
            attempts=1,
        )


class RecordingBus(EventBusKafka):
    """Kafka-typed bus stand-in that records subscribe (attach) calls.

    A demand start reads its retained record from ``earliest``, which the
    wiring accepts only from an ``EventBusKafka``; the transport itself is
    never opened (``__init__`` is not run) and every call lands in the list.
    """

    def __init__(self, shared_calls: list[tuple[str, str]]) -> None:
        self._calls = shared_calls
        self._started = False

    async def publish_envelope(
        self, envelope: object, topic: str, *, key: bytes | None = None
    ) -> None:
        return None

    async def publish(self, topic: str, key: bytes | None, value: bytes) -> None:
        return None

    def get_consumer_groups(self) -> dict[tuple[str, str], str]:
        return {}

    async def subscribe(
        self,
        *,
        topic: str,
        node_identity: object,
        on_message: object,
        auto_offset_reset: str | None = None,
    ) -> object:
        self._calls.append(("attach", topic))

        async def _unsub() -> None:
            return None

        return _unsub


class FakeDemandSource:
    """The demand split's broker read: the idle set it reports is the fixture's."""

    def __init__(self, in_use: Collection[str]) -> None:
        self._in_use = frozenset(in_use)
        self.on_first_record: Callable[[str], Awaitable[None]] | None = None
        self.watched: frozenset[str] = frozenset()

    async def watch_idle_topics(
        self,
        topics: Collection[str],
        on_first_record: Callable[[str], Awaitable[None]],
    ) -> tuple[frozenset[str], Callable[[], Awaitable[None]]]:
        self.on_first_record = on_first_record
        self.watched = frozenset(topics)

        async def _stop() -> None:
            return None

        return self._in_use & frozenset(topics), _stop


def _fake_handler_cls() -> type:
    class FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    return FakeHandler


def _routing() -> ModelHandlerRouting:
    return ModelHandlerRouting(
        routing_strategy="payload_type_match",
        handlers=(
            ModelHandlerRoutingEntry(
                handler=ModelHandlerRef(name="FakeHandler", module="fake.module"),
                event_model=None,
                operation=None,
            ),
        ),
    )


def _wired_contract(name: str, subscribe: str, publish: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(subscribe,),
            publish_topics=(publish,),
        ),
        handler_routing=_routing(),
    )


def _in_use_contract() -> ModelDiscoveredContract:
    return _wired_contract("node_in_use", IN_USE_SUBSCRIBE_TOPIC, IN_USE_PUBLISH_TOPIC)


def _deferred_contract() -> ModelDiscoveredContract:
    return _wired_contract(
        "node_deferred", DEFERRED_SUBSCRIBE_TOPIC, DEFERRED_PUBLISH_TOPIC
    )


def _publish_only_contract() -> ModelDiscoveredContract:
    """A publisher with no subscribe_topics and no handler_routing."""
    return ModelDiscoveredContract(
        name="node_publish_only",
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/publish_only/contract.yaml"),
        entry_point_name="node_publish_only",
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(),
            publish_topics=(PUBLISH_ONLY_TOPIC,),
        ),
    )


async def _boot(
    contracts: tuple[ModelDiscoveredContract, ...],
    provisioner: RecordingProvisioner,
    demand_source: FakeDemandSource,
) -> tuple[
    list[tuple[str, str]],
    dict[str, tuple[str, ...]],
    list[ModelContractAttachExclusion],
]:
    manifest = ModelAutoWiringManifest(contracts=contracts)
    engine = MessageDispatchEngine()
    shared_calls: list[tuple[str, str]] = []
    provisioner.calls = shared_calls
    bus = RecordingBus(shared_calls)
    attach_out: list[ModelContractAttachResult] = []
    exclusions: list[ModelContractAttachExclusion] = []
    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=_fake_handler_cls(),
    ):
        report = await wire_from_manifest(
            manifest,
            engine,
            event_bus=bus,
            environment="local",
            subscribe_immediately=False,
        )
        subscriptions = await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="local",
            provisioner=provisioner,
            readiness_config=ModelTopicReadinessConfig(
                max_concurrent_contract_attach=1
            ),
            attach_results_out=attach_out,
            exclusions_out=exclusions,
            demand_source=cast("HandlerKafkaTopicDemandSource", demand_source),
        )
    return shared_calls, subscriptions, exclusions


async def _first_request(demand_source: FakeDemandSource, topic: str) -> None:
    """The first record reaches a watched topic; let the demand start run."""
    assert demand_source.on_first_record is not None
    await demand_source.on_first_record(topic)
    pending = [
        t for t in asyncio.all_tasks() if (t.get_name() or "").startswith("demand-")
    ]
    await asyncio.gather(*pending)


@pytest.fixture(autouse=True)
def _lane_declares_on_demand(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The lane's overlay row declares on_demand (OMN-21318); a control removes it."""
    root = tmp_path / "overlays"
    (root / "runtime_contract_hotload").mkdir(parents=True)
    (root / "runtime_contract_hotload" / "overlay.yaml").write_text(
        "schema_version: 1\nlanes:\n  dogfood:\n    enabled: true\n"
        "    mode: on_demand\n    allowed_environments: [local]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ONEX_SKILL_OVERLAY_ROOTS", str(root))
    monkeypatch.setenv("ONEX_RUNTIME_LANE", "dogfood")


@pytest.fixture(autouse=True)
def _no_retry_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """A demand start that fails is retried; the unit run never waits on it."""
    monkeypatch.setattr(
        "omnibase_infra.runtime.auto_wiring.handler_wiring."
        "DEFAULT_NOT_READY_RETRY_BACKOFF_SECONDS",
        0.0,
    )


class TestDemandDeferredContractTopicsAreCreatedWhenItStarts:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_deferred_contract_publish_topic_is_not_created_at_boot(
        self,
    ) -> None:
        """RED on base: the unattached pass created the deferred contract's topic."""
        provisioner = RecordingProvisioner()
        demand = FakeDemandSource(in_use={IN_USE_SUBSCRIBE_TOPIC})

        calls, subscriptions, exclusions = await _boot(
            (_in_use_contract(), _deferred_contract()), provisioner, demand
        )

        deferred = [
            e
            for e in exclusions
            if e.reason is EnumContractAttachExclusionReason.DEMAND_DEFERRED
        ]
        assert [e.contract_name for e in deferred] == ["node_deferred"]
        # Nothing of the deferred contract is created at boot, either side.
        assert ("provision", DEFERRED_PUBLISH_TOPIC) not in calls
        assert ("provision", DEFERRED_SUBSCRIBE_TOPIC) not in calls
        assert "node_deferred" not in subscriptions
        # The in-use contract is unchanged: provision -> ready -> attach.
        assert ("provision", IN_USE_PUBLISH_TOPIC) in calls
        assert subscriptions["node_in_use"] == (IN_USE_SUBSCRIBE_TOPIC,)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_deferred_contract_topics_are_created_when_its_first_request_arrives(
        self,
    ) -> None:
        """Golden chain: boot withholds, the first request creates then attaches."""
        provisioner = RecordingProvisioner()
        demand = FakeDemandSource(in_use={IN_USE_SUBSCRIBE_TOPIC})
        calls, _, _ = await _boot(
            (_in_use_contract(), _deferred_contract()), provisioner, demand
        )
        assert DEFERRED_SUBSCRIBE_TOPIC in demand.watched
        before = len(calls)

        await _first_request(demand, DEFERRED_SUBSCRIBE_TOPIC)

        started = calls[before:]
        assert ("provision", DEFERRED_PUBLISH_TOPIC) in started
        assert ("provision", DEFERRED_SUBSCRIBE_TOPIC) in started
        assert started.index(("provision", DEFERRED_SUBSCRIBE_TOPIC)) < started.index(
            ("ready", DEFERRED_SUBSCRIBE_TOPIC)
        )
        assert started.index(("ready", DEFERRED_SUBSCRIBE_TOPIC)) < started.index(
            ("attach", DEFERRED_SUBSCRIBE_TOPIC)
        )
        assert started.count(("provision", DEFERRED_PUBLISH_TOPIC)) == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_publish_only_contract_topic_still_created_beside_a_deferred_contract(
        self,
    ) -> None:
        """Falsifier for the publish side of the rule (OMN-23951 stays in force)."""
        provisioner = RecordingProvisioner()
        demand = FakeDemandSource(in_use={IN_USE_SUBSCRIBE_TOPIC})

        calls, subscriptions, _ = await _boot(
            (_in_use_contract(), _deferred_contract(), _publish_only_contract()),
            provisioner,
            demand,
        )

        assert ("provision", PUBLISH_ONLY_TOPIC) in calls
        assert ("ready", PUBLISH_ONLY_TOPIC) not in calls
        assert ("attach", PUBLISH_ONLY_TOPIC) not in calls
        assert "node_publish_only" not in subscriptions
        assert ("provision", DEFERRED_PUBLISH_TOPIC) not in calls

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_replication_policy_error_on_publish_only_topic_still_fails_closed(
        self,
    ) -> None:
        """Error chain: withholding a deferred contract never softens a policy error."""
        provisioner = RecordingProvisioner(
            failing={PUBLISH_ONLY_TOPIC: TopicReplicationPolicyError("rf1 on msk")}
        )
        demand = FakeDemandSource(in_use={IN_USE_SUBSCRIBE_TOPIC})

        with pytest.raises(TopicReplicationPolicyError):
            await _boot(
                (_in_use_contract(), _deferred_contract(), _publish_only_contract()),
                provisioner,
                demand,
            )

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_without_a_demand_source_every_unattached_topic_is_as_before(
        self,
    ) -> None:
        """No demand split, no deferral: the unattached pass is unchanged."""
        provisioner = RecordingProvisioner()
        manifest = ModelAutoWiringManifest(
            contracts=(_in_use_contract(), _publish_only_contract())
        )
        engine = MessageDispatchEngine()
        shared: list[tuple[str, str]] = []
        provisioner.calls = shared
        bus = RecordingBus(shared)
        with patch(
            "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
            return_value=_fake_handler_cls(),
        ):
            report = await wire_from_manifest(
                manifest,
                engine,
                event_bus=bus,
                environment="local",
                subscribe_immediately=False,
            )
            await subscribe_wired_contract_topics(
                manifest=manifest,
                report=report,
                dispatch_engine=engine,
                event_bus=bus,
                environment="local",
                provisioner=provisioner,
                readiness_config=ModelTopicReadinessConfig(
                    max_concurrent_contract_attach=1
                ),
            )

        assert ("provision", PUBLISH_ONLY_TOPIC) in shared
        assert ("provision", IN_USE_PUBLISH_TOPIC) in shared

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_a_lane_with_no_overlay_row_creates_the_deferred_contract_topics_at_boot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failing control (OMN-21318): no row for the lane, so boot is as OMN-23951 made it.

        Red if start-up ignores the lane's mode: the deferred contract's publish
        topic would be withheld on every lane, rows or none.
        """
        monkeypatch.delenv("ONEX_RUNTIME_LANE")
        provisioner = RecordingProvisioner()
        demand = FakeDemandSource(in_use={IN_USE_SUBSCRIBE_TOPIC})

        calls, subscriptions, exclusions = await _boot(
            (_in_use_contract(), _deferred_contract()), provisioner, demand
        )

        # The contract still waits for its first request: only the topic
        # creation differs, never the split.
        assert [
            e.contract_name
            for e in exclusions
            if e.reason is EnumContractAttachExclusionReason.DEMAND_DEFERRED
        ] == ["node_deferred"]
        assert "node_deferred" not in subscriptions
        assert ("provision", DEFERRED_PUBLISH_TOPIC) in calls
        assert ("provision", IN_USE_PUBLISH_TOPIC) in calls
