# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Boot provisioning creates the declared publish topics of unattached contracts (OMN-23951).

A node that publishes but declares no ``subscribe_topics`` and no
``handler_routing`` is SKIPPED by ``wire_from_manifest``; the per-contract
provision -> ready -> attach interleave (OMN-13237) only runs for WIRED
contracts, so the node's declared ``publish_topics`` were never created. With
the universe warm off (``ONEX_BOOT_UNIVERSE_PROVISION=0``, the standing dev
setting) NOTHING created them, and a reader of such a topic found it absent.

The RED control is ``test_publish_only_contract_topic_is_created_at_boot``:
it fails on the base commit (no ``ensure_topic_exists`` call for the
publish-only topic) and passes with ``_provision_unattached_publish_topics``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_set_readiness import ModelTopicSetReadiness
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

WIRED_SUBSCRIBE_TOPIC = "onex.evt.platform.publish-pass-subscribed.v1"
WIRED_PUBLISH_TOPIC = "onex.evt.platform.publish-pass-wired-emitted.v1"
PUBLISH_ONLY_TOPIC = "onex.evt.omnimarket.publish-pass-floor-snapshot.v1"
PLUGIN_MANAGED_PUBLISH_TOPIC = "onex.evt.omnimarket.publish-pass-plugin-emitted.v1"


class RecordingProvisioner:
    """Fake provisioner recording provision + readiness calls in order."""

    def __init__(
        self,
        *,
        failing: Mapping[str, Exception] | None = None,
        unconfirmed: frozenset[str] = frozenset(),
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self._failing = dict(failing or {})
        self._unconfirmed = unconfirmed

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: object | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        self.calls.append(("provision", topic_name))
        if topic_name in self._failing:
            raise self._failing[topic_name]
        return topic_name not in self._unconfirmed

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


class RecordingBus:
    """Fake event bus that records subscribe (attach) calls."""

    def __init__(self, shared_calls: list[tuple[str, str]]) -> None:
        self._calls = shared_calls

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
    ) -> object:
        self._calls.append(("attach", topic))

        async def _unsub() -> None:
            return None

        return _unsub


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


def _wired_contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_wired",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/wired/contract.yaml"),
        entry_point_name="node_wired",
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(WIRED_SUBSCRIBE_TOPIC,),
            publish_topics=(WIRED_PUBLISH_TOPIC,),
        ),
        handler_routing=_routing(),
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


def _plugin_managed_contract() -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_plugin_managed",
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/plugin_managed/contract.yaml"),
        entry_point_name="node_plugin_managed",
        package_name="test-package",
        event_bus=ModelEventBusWiring(
            subscribe_topics=("onex.evt.platform.publish-pass-plugin-in.v1",),
            publish_topics=(PLUGIN_MANAGED_PUBLISH_TOPIC,),
            plugin_managed=True,
        ),
        handler_routing=_routing(),
    )


async def _boot(
    contracts: tuple[ModelDiscoveredContract, ...],
    provisioner: RecordingProvisioner | None,
) -> tuple[list[tuple[str, str]], dict[str, tuple[str, ...]]]:
    manifest = ModelAutoWiringManifest(contracts=contracts)
    engine = MessageDispatchEngine()
    shared_calls: list[tuple[str, str]] = []
    if provisioner is not None:
        provisioner.calls = shared_calls
    bus = RecordingBus(shared_calls)
    attach_out: list[ModelContractAttachResult] = []
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
        )
    return shared_calls, subscriptions


class TestPublishOnlyContractTopicProvisioning:
    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_publish_only_contract_topic_is_created_at_boot(self) -> None:
        """RED on base: nothing created the publish-only contract's topic."""
        provisioner = RecordingProvisioner()
        calls, subscriptions = await _boot(
            (_wired_contract(), _publish_only_contract()), provisioner
        )

        assert ("provision", PUBLISH_ONLY_TOPIC) in calls
        # The publish-only contract is not a consumer: no readiness confirm, no attach.
        assert ("ready", PUBLISH_ONLY_TOPIC) not in calls
        assert ("attach", PUBLISH_ONLY_TOPIC) not in calls
        assert "node_publish_only" not in subscriptions
        # The wired contract is unchanged: provision -> ready -> attach.
        assert calls.index(("provision", WIRED_SUBSCRIBE_TOPIC)) < calls.index(
            ("ready", WIRED_SUBSCRIBE_TOPIC)
        )
        assert subscriptions["node_wired"] == (WIRED_SUBSCRIBE_TOPIC,)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_wired_contract_publish_topic_is_provisioned_once(self) -> None:
        provisioner = RecordingProvisioner()
        calls, _ = await _boot(
            (_wired_contract(), _publish_only_contract()), provisioner
        )

        assert calls.count(("provision", WIRED_PUBLISH_TOPIC)) == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_plugin_managed_contract_publish_topic_is_created(self) -> None:
        """A contract the eligibility filters drop still gets its publish topics."""
        provisioner = RecordingProvisioner()
        calls, subscriptions = await _boot((_plugin_managed_contract(),), provisioner)

        assert ("provision", PLUGIN_MANAGED_PUBLISH_TOPIC) in calls
        assert "node_plugin_managed" not in subscriptions

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_no_provisioner_creates_nothing_and_still_attaches(self) -> None:
        calls, subscriptions = await _boot(
            (_wired_contract(), _publish_only_contract()), None
        )

        assert not [c for c in calls if c[0] == "provision"]
        assert subscriptions["node_wired"] == (WIRED_SUBSCRIBE_TOPIC,)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_replication_policy_error_on_publish_only_topic_fails_closed(
        self,
    ) -> None:
        provisioner = RecordingProvisioner(
            failing={PUBLISH_ONLY_TOPIC: TopicReplicationPolicyError("rf1 on msk")}
        )

        with pytest.raises(TopicReplicationPolicyError):
            await _boot((_wired_contract(), _publish_only_contract()), provisioner)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_other_provision_error_on_publish_only_topic_is_non_fatal(
        self,
    ) -> None:
        provisioner = RecordingProvisioner(
            failing={PUBLISH_ONLY_TOPIC: RuntimeError("broker busy")}
        )

        calls, subscriptions = await _boot(
            (_wired_contract(), _publish_only_contract()), provisioner
        )

        assert ("provision", PUBLISH_ONLY_TOPIC) in calls
        assert subscriptions["node_wired"] == (WIRED_SUBSCRIBE_TOPIC,)

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_unconfirmed_publish_only_topic_is_logged_and_non_fatal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provisioner = RecordingProvisioner(unconfirmed=frozenset({PUBLISH_ONLY_TOPIC}))

        with caplog.at_level("WARNING"):
            calls, subscriptions = await _boot(
                (_wired_contract(), _publish_only_contract()), provisioner
            )

        assert ("provision", PUBLISH_ONLY_TOPIC) in calls
        assert any(PUBLISH_ONLY_TOPIC in r.getMessage() for r in caplog.records)
        assert subscriptions["node_wired"] == (WIRED_SUBSCRIBE_TOPIC,)
