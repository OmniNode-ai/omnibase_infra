# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A FAILED contract attach must be retried, not stranded (OMN-18110).

The gap OMN-15215 left open
---------------------------
``_interleave_contract`` records two different non-attached outcomes:

* ``NOT_READY`` — topic provisioning / the readiness confirm did not converge,
  so the consumer attach was never attempted.
* ``FAILED`` — readiness PASSED and the consumer attach itself raised.

OMN-15215 built the bounded reconciliation loop that makes "runtime stays live"
actually recoverable, but wired only the first of those two into it: both
``service_kernel`` (which selects the loop's input) and
``handler_wiring._validate_unattached_contract_identities`` (which re-validates
it) filtered on ``NOT_READY`` alone. A contract whose Kafka consumer
group-join timed out was therefore recorded, logged once, and never revisited
for the life of the process.

Live evidence, ``.201`` dev lane (compose project ``omnibase-infra``, port
8085), read off the runtime's own boot aggregate on
``onex.evt.omnibase-infra.runtime-manifest-published.v1``. The ``main``-profile
boot at ``2026-09-10T00:03:45.913617Z`` published ``state=degraded
required_contracts=215 attached_contracts=211`` with four blocker rows —
``node_occ_evidence_draft_orchestrator``,
``node_occ_evidence_validator_compute``, ``node_omnigate_projection`` and
``node_omnigate_receipt_generator`` — each of them:

    status="failed"  detail="InfraTimeoutError"  topics_subscribed=[]
    readiness.status="ready"  readiness.failures=[]

So the topics existed, metadata had converged, and the group-join alone timed
out. The eleven preceding boots on the same lane (2026-09-08T13:42:58Z through
2026-09-09T21:02:16Z) all published ``required=215 attached=215``, which is
what makes this a TRANSIENT fault turned permanent by a missing retry rather
than a deterministic wiring defect. The visible symptom was the runtime
holding ``projection_attachment`` DEGRADED — "1/22 declared projection(s) have
no attached consumer and persist nothing: node_omnigate_projection" — with the
container healthcheck failing continuously since bring-up.

Shape of this file
------------------
The same harness OMN-15215's regression uses: a REAL ``contract.yaml`` on disk
parsed by the real discovery loader, the real ``wire_from_manifest`` and the
real ``_interleave_contract`` path. Only the Kafka boundary is doubled — the
provisioner always reports READY (this defect is downstream of readiness) and
the bus raises a real ``InfraTimeoutError`` from ``subscribe`` for a
configurable number of calls, which is precisely the live shape above.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import pytest
import yaml

from omnibase_infra.enums import EnumInfraTransportType
from omnibase_infra.errors import InfraTimeoutError
from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
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
from omnibase_infra.event_bus.model_topic_set_readiness import (
    ModelTopicSetReadiness,
)
from omnibase_infra.models.errors.model_timeout_error_context import (
    ModelTimeoutErrorContext,
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

pytestmark = pytest.mark.unit

CONTRACT_NAME = "node_failed_attach_fixture"

TOPICS = (
    "onex.evt.omn18110.fixture-alpha.v1",
    "onex.evt.omn18110.fixture-beta.v1",
)


class HandlerFixtureFailedAttach:
    """Trivial def-B handler for the fixture contract (import target only)."""

    async def handle(self, envelope: object) -> None:
        return None


def _write_contract(tmp_path: Path) -> Path:
    """Write a REAL contract.yaml, parsed by the real discovery YAML loader."""
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
                    "operation": "fixture.consume",
                    "message_category": "event",
                    "event_model": {
                        "name": "ModelEventMessage",
                        "module": "omnibase_infra.event_bus.models.model_event_message",
                    },
                    "handler": {
                        "name": "HandlerFixtureFailedAttach",
                        "module": (
                            "tests.unit.runtime.auto_wiring."
                            "test_failed_attach_reconciliation_omn18110"
                        ),
                    },
                    "supported_operations": ["fixture.consume"],
                }
                for topic in TOPICS
            ],
        },
    }
    contract_path.write_text(yaml.safe_dump(contract_dict, sort_keys=False))
    return contract_path


class AlwaysReadyProvisioner:
    """Topics always exist and metadata always converges.

    This defect lives strictly DOWNSTREAM of readiness, so the fixture must
    never be able to produce a NOT_READY result — otherwise a green test could
    be passing through the pre-existing OMN-15215 path instead of the one
    under test.
    """

    def __init__(self) -> None:
        self.confirm_calls: int = 0

    async def ensure_topic_exists(
        self,
        topic_name: str,
        spec: object | None = None,
        correlation_id: UUID | None = None,
    ) -> bool:
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
        return ModelTopicSetReadiness(
            topics=tuple(topics),
            status=EnumTopicReadinessStatus.READY,
            ready_topics=tuple(topics),
            attempts=1,
        )


def _group_join_timeout(topic: str) -> InfraTimeoutError:
    """The real error the live boot recorded, with the real context model."""
    return InfraTimeoutError(
        f"Timeout starting consumer for topic {topic} after 30s",
        context=ModelTimeoutErrorContext(
            transport_type=EnumInfraTransportType.KAFKA,
            operation="subscribe",
            target_name=topic,
            timeout_seconds=30.0,
        ),
    )


class TimingOutBus:
    """Fake bus whose consumer group-join times out for the first N attaches.

    ``fail_first_n_calls`` counts SUBSCRIBE CALLS, not contracts:
    ``_subscribe_contract_topics`` gathers a contract's topics together, so one
    raising call is enough to fail the whole contract's attach — which is
    exactly the all-or-nothing unit the live incident hit.
    """

    def __init__(self, *, fail_first_n_calls: int) -> None:
        self._remaining_failures = fail_first_n_calls
        self.attached_topics: list[str] = []
        self.subscribe_calls: int = 0

    async def subscribe(
        self, *, topic: str, node_identity: object, on_message: object
    ) -> object:
        self.subscribe_calls += 1
        if self._remaining_failures > 0:
            self._remaining_failures -= 1
            raise _group_join_timeout(topic)
        self.attached_topics.append(topic)

        async def _unsub() -> None:
            return None

        return _unsub


class AlwaysTimingOutBus(TimingOutBus):
    """Every group-join times out — a genuinely stuck broker, not a blip."""

    def __init__(self) -> None:
        super().__init__(fail_first_n_calls=10_000_000)


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
        return_value=HandlerFixtureFailedAttach,
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
        "fixture contract must wire through the REAL loader before the "
        f"attach-retry seam can be tested (outcome={result.outcome}, "
        f"reason={result.reason!r})"
    )
    return manifest, engine, report


async def _boot_with_timeout(
    tmp_path: Path, *, fail_first_n_calls: int
) -> tuple[
    ModelAutoWiringManifest,
    MessageDispatchEngine,
    TimingOutBus,
    list[ModelContractAttachResult],
]:
    """Drive the real boot interleave against a bus that times out."""
    contract_path = _write_contract(tmp_path)
    manifest, engine, report = await _wire_fixture_contract(contract_path)

    bus = TimingOutBus(fail_first_n_calls=fail_first_n_calls)
    attach_out: list[ModelContractAttachResult] = []
    subscriptions = await subscribe_wired_contract_topics(
        manifest=manifest,
        report=report,
        dispatch_engine=engine,
        event_bus=bus,
        environment="test",
        provisioner=AlwaysReadyProvisioner(),
        readiness_config=ModelTopicReadinessConfig(),
        attach_results_out=attach_out,
    )
    assert subscriptions == {}, (
        "the fixture boot must strand the contract before the retry seam is "
        f"exercised, got {subscriptions}"
    )
    return manifest, engine, bus, attach_out


class TestBootRecordsTheLiveFailedShape:
    """The boot interleave reproduces the shape the live lane published."""

    @pytest.mark.asyncio
    async def test_group_join_timeout_is_failed_with_readiness_ready(
        self, tmp_path: Path
    ) -> None:
        """status=FAILED, readiness=READY, no failures, nothing subscribed.

        This is the row-for-row shape of the four live blockers. If a change
        ever reclassifies a group-join timeout as NOT_READY, this fails and
        the reclassification has to be deliberate.
        """
        _manifest, _engine, bus, attach_out = await _boot_with_timeout(
            tmp_path, fail_first_n_calls=len(TOPICS)
        )

        result = next(r for r in attach_out if r.contract_name == CONTRACT_NAME)
        assert result.status is EnumContractAttachStatus.FAILED
        assert result.detail == "InfraTimeoutError"
        assert result.topics_subscribed == ()
        assert result.readiness is not None
        assert result.readiness.status is EnumTopicReadinessStatus.READY
        assert result.readiness.failures == ()
        assert bus.attached_topics == []


class TestFailedAttachIsRetried:
    """The fix: a FAILED attach re-enters the reconciliation, and converges."""

    @pytest.mark.asyncio
    async def test_reattach_retries_a_failed_attach_and_attaches_every_topic(
        self, tmp_path: Path
    ) -> None:
        """RED before the fix: the FAILED row is filtered out of the retry.

        Pre-fix, the wiring seam's identity validator admitted only
        ``NOT_READY``, so this call returns ``({}, ())`` and the contract's
        consumer group is never created — the live defect exactly.
        """
        manifest, engine, bus, attach_out = await _boot_with_timeout(
            tmp_path, fail_first_n_calls=len(TOPICS)
        )

        newly_subscribed, results = await reattach_not_ready_contracts(
            manifest,
            attach_out,
            engine,
            bus,
            "test",
            provisioner=AlwaysReadyProvisioner(),
            readiness_config=ModelTopicReadinessConfig(),
        )

        assert set(newly_subscribed.get(CONTRACT_NAME, ())) == set(TOPICS)
        assert set(bus.attached_topics) == set(TOPICS)
        assert len(results) == 1
        assert results[0].status is EnumContractAttachStatus.ATTACHED
        assert set(results[0].topics_subscribed) == set(TOPICS)

    @pytest.mark.asyncio
    async def test_retry_keeps_the_contract_dispatcher_scope(
        self, tmp_path: Path
    ) -> None:
        """The retry reuses the boot dispatcher scope, never process-global.

        ``_interleave_contract`` is handed ``previous_result.dispatcher_ids``;
        a FAILED row that lost them would fall back to fan-out dispatch, which
        is the failure mode OMN-15474 closed. Assert the scope survives the
        FAILED classification too.
        """
        manifest, engine, bus, attach_out = await _boot_with_timeout(
            tmp_path, fail_first_n_calls=len(TOPICS)
        )
        boot_result = next(r for r in attach_out if r.contract_name == CONTRACT_NAME)
        assert boot_result.dispatcher_ids, (
            "a FAILED boot row must still carry the contract's dispatcher scope"
        )

        _newly_subscribed, results = await reattach_not_ready_contracts(
            manifest,
            attach_out,
            engine,
            bus,
            "test",
            provisioner=AlwaysReadyProvisioner(),
            readiness_config=ModelTopicReadinessConfig(),
        )

        assert results[0].dispatcher_ids == boot_result.dispatcher_ids

    @pytest.mark.asyncio
    async def test_reconciliation_loop_resolves_a_failed_attach(
        self, tmp_path: Path
    ) -> None:
        """End-to-end: the function ``service_kernel`` schedules converges.

        The bus times out for the boot attempt AND the first retry, so the
        loop has to take a second pass — proving the backoff path, not just a
        single opportunistic retry. Sleep is injected, so no wall clock.
        """
        manifest, engine, bus, attach_out = await _boot_with_timeout(
            tmp_path, fail_first_n_calls=len(TOPICS) + 1
        )
        slept: list[float] = []

        async def _sleep(seconds: float) -> None:
            slept.append(seconds)

        attempts: list[tuple[dict[str, tuple[str, ...]], int]] = []

        results = await run_not_ready_reconciliation_loop(
            manifest,
            attach_out,
            engine,
            bus,
            "test",
            provisioner=AlwaysReadyProvisioner(),
            readiness_config=ModelTopicReadinessConfig(),
            on_attempt=lambda subscribed, rows: attempts.append(
                (subscribed, len(rows))
            ),
            sleep=_sleep,
        )

        by_name = {r.contract_name: r for r in results}
        assert by_name[CONTRACT_NAME].status is EnumContractAttachStatus.ATTACHED
        assert set(bus.attached_topics) == set(TOPICS)
        assert slept, "the loop must honour its initial delay before retrying"
        assert attempts, "every retry outcome must reach the readiness-gate fold"

    @pytest.mark.asyncio
    async def test_retry_is_bounded_and_never_raises_when_it_cannot_converge(
        self, tmp_path: Path
    ) -> None:
        """A permanently stuck attach exhausts the budget, quietly and finitely.

        Retrying a FAILED attach must not turn a degraded runtime into a
        crash-looping one, and must not retry forever: the contract is still
        reported FAILED after the bounded budget.
        """
        contract_path = _write_contract(tmp_path)
        manifest, engine, report = await _wire_fixture_contract(contract_path)
        bus = AlwaysTimingOutBus()
        attach_out: list[ModelContractAttachResult] = []
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            provisioner=AlwaysReadyProvisioner(),
            readiness_config=ModelTopicReadinessConfig(),
            attach_results_out=attach_out,
        )

        async def _sleep(seconds: float) -> None:
            return None

        max_attempts = 3
        results = await run_not_ready_reconciliation_loop(
            manifest,
            attach_out,
            engine,
            bus,
            "test",
            provisioner=AlwaysReadyProvisioner(),
            readiness_config=ModelTopicReadinessConfig(),
            max_attempts=max_attempts,
            sleep=_sleep,
        )

        by_name = {r.contract_name: r for r in results}
        assert by_name[CONTRACT_NAME].status is EnumContractAttachStatus.FAILED
        assert bus.attached_topics == []
        # boot attempt + exactly max_attempts retries, one subscribe call per
        # topic per attempt: bounded, and provably not an unbounded loop.
        assert bus.subscribe_calls <= len(TOPICS) * (max_attempts + 1)

    @pytest.mark.asyncio
    async def test_exhaustion_warning_names_the_attach_failure_not_an_empty_list(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A FAILED exhaustion must say WHY, and readiness cannot say it.

        ``readiness.failures`` is empty on a FAILED row by construction —
        readiness is the half that passed — so the pre-existing
        readiness-keyed rendering printed ``[]`` here. The attach status and
        detail are carried alongside it, and the OMN-15578 ``readiness_failures``
        field keeps its shape.
        """
        contract_path = _write_contract(tmp_path)
        manifest, engine, report = await _wire_fixture_contract(contract_path)
        bus = AlwaysTimingOutBus()
        attach_out: list[ModelContractAttachResult] = []
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            provisioner=AlwaysReadyProvisioner(),
            readiness_config=ModelTopicReadinessConfig(),
            attach_results_out=attach_out,
        )

        async def _sleep(seconds: float) -> None:
            return None

        with caplog.at_level(
            "WARNING", logger="omnibase_infra.runtime.auto_wiring.handler_wiring"
        ):
            await run_not_ready_reconciliation_loop(
                manifest,
                attach_out,
                engine,
                bus,
                "test",
                provisioner=AlwaysReadyProvisioner(),
                readiness_config=ModelTopicReadinessConfig(),
                max_attempts=2,
                sleep=_sleep,
            )

        exhausted = [
            r for r in caplog.records if "reconciliation exhausted" in r.getMessage()
        ]
        assert exhausted, "an exhausted reconciliation must warn"
        record = exhausted[0]
        attach_failures = record.attach_failures
        assert attach_failures[CONTRACT_NAME]["status"] == "failed"
        assert attach_failures[CONTRACT_NAME]["detail"] == "InfraTimeoutError"
        # The OMN-15578 field survives, and is legitimately empty here.
        assert record.readiness_failures[CONTRACT_NAME] == []

    @pytest.mark.asyncio
    async def test_attached_contracts_are_never_re_attached(
        self, tmp_path: Path
    ) -> None:
        """Widening to FAILED must not widen to ATTACHED.

        A retry of a live consumer would re-join a group that already has a
        member. The predicate is "did NOT attach", never "is not NOT_READY".
        """
        contract_path = _write_contract(tmp_path)
        manifest, engine, report = await _wire_fixture_contract(contract_path)
        bus = TimingOutBus(fail_first_n_calls=0)
        attach_out: list[ModelContractAttachResult] = []
        subscriptions = await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            provisioner=AlwaysReadyProvisioner(),
            readiness_config=ModelTopicReadinessConfig(),
            attach_results_out=attach_out,
        )
        assert set(subscriptions.get(CONTRACT_NAME, ())) == set(TOPICS)
        subscribe_calls_after_boot = bus.subscribe_calls

        newly_subscribed, results = await reattach_not_ready_contracts(
            manifest,
            attach_out,
            engine,
            bus,
            "test",
            provisioner=AlwaysReadyProvisioner(),
            readiness_config=ModelTopicReadinessConfig(),
        )

        assert newly_subscribed == {}
        assert results == ()
        assert bus.subscribe_calls == subscribe_calls_after_boot


class TestOneDefinitionOfUnattached:
    """The selector and the validator read the SAME predicate (AC2)."""

    def test_needs_reattach_is_true_for_every_non_attached_status(self) -> None:
        """Exhaustive over the enum, so a new status cannot be forgotten."""
        for status in EnumContractAttachStatus:
            result = ModelContractAttachResult(
                contract_name=CONTRACT_NAME, status=status
            )
            assert result.needs_reattach is (
                status is not EnumContractAttachStatus.ATTACHED
            )

    def test_service_kernel_selects_the_loop_input_via_that_predicate(self) -> None:
        """``service_kernel`` must not re-implement the status filter.

        A source assertion rather than a behavioural one because the selection
        happens inside the kernel's boot coroutine, which a unit test cannot
        drive. Two independent implementations of "which contracts get
        retried" is the exact drift this ticket is repairing, so the ratchet
        is that there is only one.
        """
        from omnibase_infra.runtime import service_kernel

        source = Path(service_kernel.__file__).read_text()
        assert "needs_reattach" in source, (
            "service_kernel must select the reconciliation input with "
            "ModelContractAttachResult.needs_reattach, not a local status filter"
        )
        assert "is EnumContractAttachStatus.NOT_READY" not in source, (
            "service_kernel still filters the reconciliation input on "
            "NOT_READY alone — a FAILED attach would go unretried again"
        )
