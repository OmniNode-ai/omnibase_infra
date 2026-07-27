# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""NOT_READY reconciliation regression for OMN-15215.

CONFIRMED root cause (not the ``handler_routing_loader`` "Unknown
routing_strategy 'topic_match'" hypothesis the ticket flagged as unverified):
``subscribe_wired_contract_topics`` makes exactly ONE provision->confirm->attach
attempt per contract via ``_interleave_contract``. A contract whose topic
metadata has not converged within the bounded readiness poll is recorded
NOT_READY and its consumer attach is skipped — PERMANENTLY, because nothing
in the pre-fix tree ever retries ``_interleave_contract`` for it again.

Live evidence (fresh omnibase-infra-stability-test boot, 2026-07-27):
``node_ledger_projection_compute`` (26 topic_match ``subscribe_topics``,
OMN-15006/OMN-15168) hit ``NOT-READY: topic metadata did not converge`` for 4
OCC-governance topics exactly once at boot, and then NEVER logged a single
"Auto-wired subscription ... node=node_ledger_projection_compute" line for
the rest of the container's observed lifetime (3 discovery passes, ~11
minutes) — zero Kafka consumer groups were ever created for the contract, not
even the 19 unrelated topics that had nothing to do with the race. The "7
platform topics have live consumer groups" the filing lane observed is STALE
Kafka group data left over from an older, pre-OMN-15006 deploy image (the
filing lane's own words: "14 stale groups") — not live proof of a partial
attach by the CURRENT boot.

This is deliberately NOT a fix to ``handler_routing_loader.VALID_ROUTING_STRATEGIES``
(see the loader-specific coverage under ``tests/unit/runtime/contract_loaders/``)
— a real, unmocked repro of the CURRENT ``discovery.py``/``handler_wiring.py``
path (topic-folded dispatcher-ID derivation, OMN-14580/OMN-13825) already
wires and attaches every topic_match entry once readiness passes; fixing the
loader alone would not have unblocked OMN-15169.

Uses a REAL topic_match contract.yaml on disk (the exact per-topic-entry,
shared-operation/shared-handler shape node_ledger_projection_compute uses),
parsed through the real ``discovery.py`` YAML loader. Only the Kafka boundary
(topic provisioner + event bus) is doubled — the same class of test double
``test_per_contract_boot_interleave.py`` (OMN-13237) already uses for this
exact machinery; there is no live broker in a unit test.
``_import_handler_class`` is patched to a trivial handler, matching that same
file's precedent, so wiring does not require a real importable handler
module on disk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
import yaml

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.enum_topic_readiness_failure_reason import (
    EnumTopicReadinessFailureReason,
)
from omnibase_infra.event_bus.enum_topic_readiness_status import (
    EnumTopicReadinessStatus,
)
from omnibase_infra.event_bus.model_contract_attach_result import (
    ModelContractAttachResult,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_readiness_failure import (
    ModelTopicReadinessFailure,
)
from omnibase_infra.event_bus.model_topic_set_readiness import (
    ModelTopicSetReadiness,
)
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    reattach_not_ready_contracts,
    run_not_ready_reconciliation_loop,
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.auto_wiring.report import ModelAutoWiringReport
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

CONTRACT_NAME = "node_topic_match_fanout_fixture"

# Mirrors node_ledger_projection_compute's real handler_routing shape: N
# per-topic entries sharing the SAME operation + handler class, disambiguated
# only by their own `topic` field (OMN-14594/OMN-14580 pattern) — the exact
# shape live-affected by OMN-15215.
TOPICS = (
    "onex.evt.omn15215.fixture-alpha.v1",
    "onex.evt.omn15215.fixture-beta.v1",
    "onex.evt.omn15215.fixture-gamma.v1",
)


class HandlerFixtureNotReady:
    """Trivial def-B handler for the fixture contract (import target only)."""

    async def handle(self, envelope: object) -> None:
        return None


def _write_contract(tmp_path: Path) -> Path:
    """Write a REAL topic_match contract.yaml, parsed by the real YAML loader."""
    contract_dir = tmp_path / CONTRACT_NAME
    contract_dir.mkdir()
    contract_path = contract_dir / "contract.yaml"
    contract_dict = {
        "name": CONTRACT_NAME,
        "node_type": "COMPUTE_GENERIC",
        "contract_version": {"major": 1, "minor": 0, "patch": 0},
        "event_bus": {"subscribe_topics": list(TOPICS)},
        "handler_routing": {
            "routing_strategy": "topic_match",
            "handlers": [
                {
                    "topic": topic,
                    "operation": "fixture.project",
                    "message_category": "event",
                    "event_model": {
                        "name": "ModelEventMessage",
                        "module": "omnibase_infra.event_bus.models.model_event_message",
                    },
                    "handler": {
                        "name": "HandlerFixtureNotReady",
                        "module": (
                            "tests.unit.runtime.auto_wiring."
                            "test_not_ready_reconciliation_omn15215"
                        ),
                    },
                    "supported_operations": ["fixture.project"],
                }
                for topic in TOPICS
            ],
        },
    }
    contract_path.write_text(yaml.safe_dump(contract_dict, sort_keys=False))
    return contract_path


class FlakyProvisioner:
    """Fake provisioner: NOT_READY for `fail_first_n_calls` confirms, then READY.

    Models the exact live symptom: broker topic metadata that has not yet
    converged at boot-time confirm, but WOULD converge shortly after if
    anything ever asked again.
    """

    def __init__(self, *, fail_first_n_calls: int) -> None:
        self._remaining_failures = fail_first_n_calls
        self.confirm_calls: int = 0
        self.provision_calls: list[str] = []

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: object | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
        self.provision_calls.append(topic_name)
        return True

    async def confirm_topics_ready(
        self,
        topics: Sequence[str],
        *,
        expected_specs: Mapping[str, object] | None = None,
        config: ModelTopicReadinessConfig | None = None,
        correlation_id: UUID | None = None,
    ) -> ModelTopicSetReadiness:
        self.confirm_calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            return ModelTopicSetReadiness(
                topics=tuple(topics),
                status=EnumTopicReadinessStatus.NOT_READY,
                ready_topics=(),
                failures=tuple(
                    ModelTopicReadinessFailure(
                        topic=t,
                        reason=EnumTopicReadinessFailureReason.TOPIC_ABSENT,
                    )
                    for t in topics
                ),
                attempts=1,
            )
        return ModelTopicSetReadiness(
            topics=tuple(topics),
            status=EnumTopicReadinessStatus.READY,
            ready_topics=tuple(topics),
            attempts=1,
        )


class AlwaysNotReadyProvisioner(FlakyProvisioner):
    """Never converges — models a genuinely stuck (not just slow) topic."""

    def __init__(self) -> None:
        super().__init__(fail_first_n_calls=10_000_000)


class RecordingBus:
    """Fake event bus recording every attach (the real Kafka consumer-group-join)."""

    def __init__(self) -> None:
        self.attached_topics: list[str] = []

    async def subscribe(
        self, *, topic: str, node_identity: object, on_message: object
    ) -> object:
        self.attached_topics.append(topic)

        async def _unsub() -> None:
            return None

        return _unsub


async def _wire_fixture_contract(
    contract_path: Path,
) -> tuple[ModelAutoWiringManifest, MessageDispatchEngine, ModelAutoWiringReport]:
    """Real discovery + real wire_from_manifest over the on-disk fixture."""
    discovered = discover_contracts_from_paths([contract_path])
    contracts = getattr(discovered, "contracts", discovered)
    manifest = ModelAutoWiringManifest(contracts=tuple(contracts))
    engine = MessageDispatchEngine()

    with patch(
        "omnibase_infra.runtime.auto_wiring.handler_wiring._import_handler_class",
        return_value=HandlerFixtureNotReady,
    ):
        report = await wire_from_manifest(
            manifest,
            engine,
            event_bus=None,
            environment="test",
            subscribe_immediately=False,
        )
    result = next(r for r in report.results if r.contract_name == CONTRACT_NAME)
    assert result.wirings, (
        f"fixture contract must wire all {len(TOPICS)} topic_match entries "
        f"through the REAL loader before the attach-retry seam can be tested "
        f"(outcome={result.outcome}, reason={result.reason!r})"
    )
    assert len(result.wirings) == len(TOPICS), (
        f"expected one dispatcher per topic_match entry ({len(TOPICS)}), got "
        f"{len(result.wirings)} — topic-folded dispatcher-ID disambiguation "
        "(OMN-14580) regressed"
    )
    return manifest, engine, report


class TestSeededRedNotReadyPermanentlyStarvesConsumer:
    """RED under the pre-fix code path: NOT_READY once == starved forever."""

    @pytest.mark.asyncio
    async def test_old_code_path_never_retries_and_zero_topics_attach(
        self, tmp_path: Path
    ) -> None:
        """Seeded-RED: exactly what OMN-15215 observed live.

        The provisioner is flaky for exactly ONE confirm call, then would
        succeed on any subsequent call. The pre-fix code (a single
        ``subscribe_wired_contract_topics`` call, no retry) has no mechanism
        to ever make that subsequent call — so even though the underlying
        resource becomes ready almost immediately, the contract's consumer
        group is NEVER created. This is the exact "runtime stays live" gap:
        NOT_READY is recorded but nothing ever revisits it.
        """
        contract_path = _write_contract(tmp_path)
        manifest, engine, report = await _wire_fixture_contract(contract_path)

        provisioner = FlakyProvisioner(fail_first_n_calls=1)
        bus = RecordingBus()
        attach_out: list[ModelContractAttachResult] = []

        # This IS the entire pre-fix code path: the boot-time interleave,
        # called exactly once, with nothing downstream ever re-attempting it.
        subscriptions = await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            provisioner=provisioner,
            readiness_config=ModelTopicReadinessConfig(),
            attach_results_out=attach_out,
        )

        # RED: the contract never attaches. Zero consumer groups — exactly
        # the live symptom (a contract with N topics stuck at zero forever).
        assert subscriptions == {}, (
            "pre-fix code path attached topics it should not have been able "
            f"to on a single NOT_READY confirm: {subscriptions}"
        )
        assert bus.attached_topics == []
        assert provisioner.confirm_calls == 1, (
            "pre-fix code path must make exactly ONE confirm call — proving "
            "there is no retry mechanism in subscribe_wired_contract_topics "
            "alone"
        )
        by_name = {r.contract_name: r for r in attach_out}
        assert by_name[CONTRACT_NAME].status is EnumContractAttachStatus.NOT_READY


class TestFixReattachesOnceReady:
    """GREEN under the fix: the reconciliation loop actually attaches."""

    @pytest.mark.asyncio
    async def test_reattach_not_ready_contracts_attaches_all_topics_on_retry(
        self, tmp_path: Path
    ) -> None:
        """The fix: reattach_not_ready_contracts() drives the SAME real
        _interleave_contract path a second time, and the (now-ready)
        provisioner lets every topic attach — real event_bus.subscribe()
        calls, exactly what creates a live Kafka consumer group.
        """
        contract_path = _write_contract(tmp_path)
        manifest, engine, report = await _wire_fixture_contract(contract_path)

        provisioner = FlakyProvisioner(fail_first_n_calls=1)
        bus = RecordingBus()
        attach_out: list[ModelContractAttachResult] = []
        subscriptions = await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            provisioner=provisioner,
            readiness_config=ModelTopicReadinessConfig(),
            attach_results_out=attach_out,
        )
        assert subscriptions == {}  # still starting from the RED state

        newly_subscribed, results = await reattach_not_ready_contracts(
            manifest,
            attach_out,
            engine,
            bus,
            "test",
            provisioner=provisioner,
            readiness_config=ModelTopicReadinessConfig(),
        )

        assert set(newly_subscribed.get(CONTRACT_NAME, ())) == set(TOPICS)
        assert set(bus.attached_topics) == set(TOPICS)
        assert provisioner.confirm_calls == 2  # 1 failed + 1 succeeded
        assert len(results) == 1
        assert results[0].status is EnumContractAttachStatus.ATTACHED
        assert set(results[0].topics_subscribed) == set(TOPICS)

    @pytest.mark.asyncio
    async def test_reattach_is_a_noop_when_nothing_is_not_ready(
        self, tmp_path: Path
    ) -> None:
        """Sanity: with no NOT_READY input, no retry work happens at all."""
        contract_path = _write_contract(tmp_path)
        manifest, engine, _report = await _wire_fixture_contract(contract_path)
        provisioner = FlakyProvisioner(fail_first_n_calls=0)
        bus = RecordingBus()

        newly_subscribed, results = await reattach_not_ready_contracts(
            manifest, (), engine, bus, "test", provisioner=provisioner
        )

        assert newly_subscribed == {}
        assert results == ()
        assert provisioner.confirm_calls == 0
        assert bus.attached_topics == []

    @pytest.mark.asyncio
    async def test_reconciliation_loop_attaches_after_backoff_wait(
        self, tmp_path: Path
    ) -> None:
        """End-to-end fix proof: run_not_ready_reconciliation_loop (the actual
        function service_kernel.py schedules as a background task) resolves
        the contract to ATTACHED without real wall-clock delay (injected
        sleep), the same seam a boot-time background task exercises live.
        """
        contract_path = _write_contract(tmp_path)
        manifest, engine, report = await _wire_fixture_contract(contract_path)

        provisioner = FlakyProvisioner(fail_first_n_calls=2)
        bus = RecordingBus()
        attach_out: list[ModelContractAttachResult] = []
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            provisioner=provisioner,
            readiness_config=ModelTopicReadinessConfig(),
            attach_results_out=attach_out,
        )
        assert bus.attached_topics == []

        sleep_calls: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        final_results = await run_not_ready_reconciliation_loop(
            manifest,
            attach_out,
            engine,
            bus,
            "test",
            provisioner=provisioner,
            readiness_config=ModelTopicReadinessConfig(),
            max_attempts=5,
            sleep=_fake_sleep,
        )

        assert set(bus.attached_topics) == set(TOPICS), (
            "the reconciliation loop must eventually make the real "
            "event_bus.subscribe() calls that create the Kafka consumer "
            "group — this is the actual live-symptom fix"
        )
        assert len(final_results) == 1
        assert final_results[0].status is EnumContractAttachStatus.ATTACHED
        # initial delay + 1 backoff before the 2nd (successful) attempt.
        assert len(sleep_calls) == 2


class TestReconciliationLoopIsBounded:
    """A contract that never converges degrades — it must not retry forever."""

    @pytest.mark.asyncio
    async def test_loop_gives_up_after_max_attempts_never_raises(
        self, tmp_path: Path
    ) -> None:
        contract_path = _write_contract(tmp_path)
        manifest, engine, report = await _wire_fixture_contract(contract_path)

        provisioner = AlwaysNotReadyProvisioner()
        bus = RecordingBus()
        attach_out: list[ModelContractAttachResult] = []
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            provisioner=provisioner,
            readiness_config=ModelTopicReadinessConfig(),
            attach_results_out=attach_out,
        )

        sleep_calls: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        final_results = await run_not_ready_reconciliation_loop(
            manifest,
            attach_out,
            engine,
            bus,
            "test",
            provisioner=provisioner,
            readiness_config=ModelTopicReadinessConfig(),
            max_attempts=3,
            sleep=_fake_sleep,
        )

        assert bus.attached_topics == []
        assert len(final_results) == 1
        assert final_results[0].status is EnumContractAttachStatus.NOT_READY
        # initial delay + 2 backoffs between 3 attempts = 3 sleep calls total.
        assert len(sleep_calls) == 3
        # 1 initial confirm (the pre-loop subscribe_wired_contract_topics
        # call, already NOT_READY) + 3 loop attempts (max_attempts=3) = 4.
        assert provisioner.confirm_calls == 4


class TestReattachIgnoresAlreadyAttachedContracts:
    @pytest.mark.asyncio
    async def test_attached_status_entries_are_not_retried(
        self, tmp_path: Path
    ) -> None:
        contract_path = _write_contract(tmp_path)
        manifest, engine, _report = await _wire_fixture_contract(contract_path)
        provisioner = FlakyProvisioner(fail_first_n_calls=0)
        bus = RecordingBus()

        already_attached = ModelContractAttachResult(
            contract_name=CONTRACT_NAME,
            status=EnumContractAttachStatus.ATTACHED,
            topics_subscribed=TOPICS,
        )
        newly_subscribed, results = await reattach_not_ready_contracts(
            manifest,
            (already_attached,),
            engine,
            bus,
            "test",
            provisioner=provisioner,
        )

        assert newly_subscribed == {}
        assert results == ()
        assert provisioner.confirm_calls == 0
