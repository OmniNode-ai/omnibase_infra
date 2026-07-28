# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary seam test: contract-declared DLQ topics MUST be provisioned.

OMN-15330 (OMN-14208 seam rule -- ONE test driving the real seam, not two
independent unit suites).

The seam has two sides, joined by the *contract YAML string*:

* **Routing side** -- ``handler_wiring._read_dlq_topics(contract.contract_path)``
  reads ``event_bus.dlq_topics`` and hands it to the projection auto-wiring as
  ``ModelProjectionSinks.dlq_topics``; on a projection handler error
  ``_route_projection_error_to_dlq`` publishes to ``dlq_topics[0]``.
* **Provisioning side** -- ``_contract_provision_topics`` decides which topics
  the per-contract boot interleave (OMN-13237) creates + readiness-confirms
  before the consumer attaches.

Before OMN-15330 the provisioning side deliberately EXCLUDED ``dlq_topics``
("DLQ topics are handled by the best-effort universe warm"). On onex-dev the
universe warm is OFF (``ONEX_BOOT_UNIVERSE_PROVISION=0``, set after the
2026-07-27 >1000-topic near-meltdown), so nothing created them and the first
malformed event produced
``[ONEX_CORE_041_INVALID_CONFIGURATION] Topic '<dlq>' not found on broker``
with the record dropped. Live evidence: onex-dev pod
``omninode-runtime-595bf88868-s5qqn``, 2026-07-28T16:29Z, on
``onex.dlq.omnimarket.projection-delegation-inference-response-malformed.v1``
and siblings.

This test drives BOTH sides off ONE real on-disk contract and asserts the exact
string the router would publish to is the exact string the provisioner created.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest

from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_set_readiness import ModelTopicSetReadiness
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _contract_provision_topics,
    _read_dlq_topics,
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

SUBSCRIBE_TOPIC = "onex.evt.omnibase-infra.inference-response.v1"
PUBLISH_TOPIC = "onex.snapshot.projection.delegation.inference-response-text.v1"
# The live onex-dev name that threw ONEX_CORE_041 at 2026-07-28T16:29:24Z.
DLQ_TOPIC = "onex.dlq.omnimarket.projection-delegation-inference-response-malformed.v1"

CONTRACT_YAML = f"""\
contract_version:
  major: 1
  minor: 0
  patch: 0
node_type: REDUCER_GENERIC
event_bus:
  subscribe_topics:
    - {SUBSCRIBE_TOPIC}
  publish_topics:
    - {PUBLISH_TOPIC}
  dlq_topics:
    - {DLQ_TOPIC}
"""


class RecordingProvisioner:
    """Records every topic the boot interleave provisions / readiness-confirms."""

    def __init__(self) -> None:
        self.provisioned: list[str] = []
        self.readiness_confirmed: list[str] = []

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: object | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        self.provisioned.append(topic_name)
        return True

    async def confirm_topics_ready(
        self,
        topics: Sequence[str],
        *,
        expected_specs: Mapping[str, object] | None = None,
        config: ModelTopicReadinessConfig | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelTopicSetReadiness:
        self.readiness_confirmed.extend(topics)
        return ModelTopicSetReadiness(
            topics=tuple(topics),
            status=EnumTopicReadinessStatus.READY,
            ready_topics=tuple(topics),
            attempts=1,
        )


class RecordingBus:
    """Event-bus double satisfying ``ProtocolEventBusLike`` so attach succeeds."""

    def __init__(self) -> None:
        self.subscribed: list[str] = []

    async def subscribe(
        self,
        *,
        topic: str,
        node_identity: object,
        on_message: object,
    ) -> object:
        self.subscribed.append(topic)

        async def _unsub() -> None:
            return None

        return _unsub

    async def publish_envelope(self, *args: object, **kwargs: object) -> None:
        return None

    async def publish(self, *args: object, **kwargs: object) -> None:
        return None

    def get_consumer_groups(self) -> dict[tuple[str, str], str]:
        return {}


def _fake_handler_cls() -> type:
    class FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    return FakeHandler


def _write_contract(tmp_path: Path) -> Path:
    contract_dir = tmp_path / "node_projection_delegation_inference_response"
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract_dir / "contract.yaml"
    contract_path.write_text(CONTRACT_YAML)
    return contract_path


def _discovered(contract_path: Path) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name="node_projection_delegation_inference_response",
        node_type="REDUCER_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=contract_path,
        entry_point_name="node_projection_delegation_inference_response",
        package_name="omnimarket",
        event_bus=ModelEventBusWiring(
            subscribe_topics=(SUBSCRIBE_TOPIC,),
            publish_topics=(PUBLISH_TOPIC,),
        ),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(
                        name="HandlerProjectionDelegationInferenceResponse",
                        module="fake.module",
                    ),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


@pytest.mark.asyncio
async def test_contract_declared_dlq_topic_is_provisioned_at_boot(
    tmp_path: Path,
) -> None:
    """The topic the router publishes to MUST be one the provisioner created.

    RED before OMN-15330: ``_contract_provision_topics`` returned only
    subscribe+publish topics, so ``DLQ_TOPIC`` never reached the provisioner
    and the first dead-letter hit ONEX_CORE_041 on the live broker.
    """
    contract_path = _write_contract(tmp_path)
    contract = _discovered(contract_path)
    manifest = ModelAutoWiringManifest(contracts=(contract,))
    engine = MessageDispatchEngine()
    provisioner = RecordingProvisioner()
    bus = RecordingBus()

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

    # ---- ROUTING SIDE: the exact string the projection error path publishes to.
    # Same reader handler_wiring uses at the projection auto-wiring call site to
    # build ModelProjectionSinks.dlq_topics; _route_projection_error_to_dlq
    # sends to dlq_topics[0].
    routed_dlq_topics = _read_dlq_topics(contract_path)
    assert routed_dlq_topics, "routing side read no dlq_topics from the contract"
    router_target = routed_dlq_topics[0]
    assert router_target == DLQ_TOPIC

    # ---- SEAM ASSERTION: provisioning side must cover the routing target.
    assert router_target in provisioner.provisioned, (
        "contract-declared DLQ topic was never provisioned at boot -- the first "
        "dead-letter will raise ONEX_CORE_041 topic-not-found and drop the record "
        f"(provisioned={provisioner.provisioned})"
    )
    # Readiness must also cover it: attaching a consumer whose DLQ target is not
    # ready guarantees silent loss on the first malformed event (fail closed).
    assert router_target in provisioner.readiness_confirmed

    # Regression guard: subscribe/publish coverage is unchanged, and the
    # consumer really attached (a DLQ topic entering the readiness set must not
    # block the attach it protects).
    assert SUBSCRIBE_TOPIC in provisioner.provisioned
    assert PUBLISH_TOPIC in provisioner.provisioned
    assert bus.subscribed == [SUBSCRIBE_TOPIC]


def test_provision_set_includes_dlq_topics_without_duplicates(tmp_path: Path) -> None:
    """``_contract_provision_topics`` is the single source of the boot topic set."""
    contract_path = _write_contract(tmp_path)
    topics = _contract_provision_topics(_discovered(contract_path))

    assert topics.count(DLQ_TOPIC) == 1
    assert set(topics) == {SUBSCRIBE_TOPIC, PUBLISH_TOPIC, DLQ_TOPIC}
    # Declared order is preserved: subscribe, then publish, then DLQ.
    assert topics.index(SUBSCRIBE_TOPIC) < topics.index(DLQ_TOPIC)


def test_unreadable_contract_degrades_instead_of_aborting_boot(tmp_path: Path) -> None:
    """A broken contract YAML must not abort the whole boot subscribe pass.

    ``_interleave_contract`` runs under ``asyncio.gather(...)`` without
    ``return_exceptions=True``, so a raise out of ``_contract_provision_topics``
    would take down provisioning for EVERY contract, not just this one.
    """
    contract_dir = tmp_path / "node_broken"
    contract_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contract_dir / "contract.yaml"
    contract_path.write_text("event_bus: {subscribe_topics: [a\n  unbalanced: [\n")

    topics = _contract_provision_topics(_discovered(contract_path))

    # Degrades to pre-OMN-15330 behaviour for this contract only.
    assert set(topics) == {SUBSCRIBE_TOPIC, PUBLISH_TOPIC}


def test_missing_contract_file_is_not_an_error(tmp_path: Path) -> None:
    """Contracts discovered from a non-file source keep working (no DLQ topics)."""
    topics = _contract_provision_topics(
        _discovered(tmp_path / "absent" / "contract.yaml")
    )
    assert set(topics) == {SUBSCRIBE_TOPIC, PUBLISH_TOPIC}
