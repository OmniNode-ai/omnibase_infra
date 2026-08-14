# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-boundary regression: the NOT-READY blocker set must LEAVE the runtime.

OMN-15512. The defect this guards is NOT "``ModelRuntimeAttachReadiness`` computes
the wrong thing" — it computes the right thing and always did (OMN-13237). The
defect is that the aggregate never left ``service_kernel``: it was a local
variable consumed by exactly one ``logger.info``, so the only way to read the
blocker set was ``ssh`` + ``docker logs omninode-runtime | grep NOT-READY``,
which is literally how the parent OMN-15508 had to be diagnosed.

So a unit test on ``ModelRuntimeAttachReadiness.from_results`` would be vacuous
here — it would pass both before and after the fix. These tests drive the whole
chain end to end with a contract whose required topic is ABSENT:

    real wire_from_manifest
      -> real subscribe_wired_contract_topics  (provision -> confirm-ready -> attach)
      -> real publish_runtime_manifest         (the function the kernel calls)
      -> the envelope captured off a recording bus
      -> real ModelPayloadInsertRuntimeManifest coercion of that wire payload
      -> real HandlerPostgresRuntimeManifestInsert SQL arguments

and assert the absent topic is NAMED at the far end. Everything in that chain is
the artifact that runs; the only doubles are the event bus, the topic
provisioner, and the asyncpg pool — the three genuine external boundaries.

RED-before proof lives in ``TestGuardActuallyGuards``: the same chain run
WITHOUT threading the aggregate reproduces the pre-fix shape (payload
``attach_readiness is None``, SQL writes ``'unknown'`` / ``[]``) and fails every
assertion the fixed path makes. That discriminates against exists-but-wrong
rather than against a surrogate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from omnibase_infra.event_bus.enum_contract_attach_status import (
    EnumContractAttachStatus,
)
from omnibase_infra.event_bus.enum_runtime_readiness_state import (
    EnumRuntimeReadinessState,
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
from omnibase_infra.event_bus.model_runtime_attach_readiness import (
    ModelRuntimeAttachReadiness,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.model_topic_readiness_failure import (
    ModelTopicReadinessFailure,
)
from omnibase_infra.event_bus.model_topic_set_readiness import ModelTopicSetReadiness
from omnibase_infra.nodes.node_runtime_manifest_reducer.handlers.handler_postgres_runtime_manifest_insert import (
    SQL_INSERT_RUNTIME_MANIFEST,
    HandlerPostgresRuntimeManifestInsert,
)
from omnibase_infra.nodes.node_runtime_manifest_reducer.models.model_payload_insert_runtime_manifest import (
    ModelPayloadInsertRuntimeManifest,
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
from omnibase_infra.runtime.manifest_builder import publish_runtime_manifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.models.model_runtime_manifest_published import (
    ModelRuntimeManifestPublished,
)

if TYPE_CHECKING:
    from omnibase_infra.runtime.auto_wiring.report import ModelAutoWiringReport

pytestmark = pytest.mark.integration

# The contract that CAN attach, and the one whose required topic is absent.
ATTACHING_CONTRACT = "node_alpha"
ATTACHING_TOPIC = "onex.evt.omnibase-infra.seam-alpha.v1"
BLOCKED_CONTRACT = "node_beta"
BLOCKED_TOPIC = "onex.evt.omnibase-infra.seam-beta-absent.v1"

MANIFEST_TOPIC = "onex.evt.omnibase-infra.runtime-manifest-published.v1"


# ---------------------------------------------------------------------------
# Boundary doubles: bus, provisioner, asyncpg pool. Nothing else is faked.
# ---------------------------------------------------------------------------


class _AbsentTopicProvisioner:
    """Topic provisioner where ``absent_topics`` never converge.

    Mirrors the live failure: the runtime creates the topic, then the metadata
    readiness confirm does not converge for it, so the consumer attach is
    skipped and the contract is recorded NOT_READY (OMN-13237).
    """

    def __init__(self, *, absent_topics: frozenset[str]) -> None:
        self._absent = absent_topics

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
        unready = [t for t in topics if t in self._absent]
        if not unready:
            return ModelTopicSetReadiness(
                topics=tuple(topics),
                status=EnumTopicReadinessStatus.READY,
                ready_topics=tuple(topics),
                attempts=1,
            )
        return ModelTopicSetReadiness(
            topics=tuple(topics),
            status=EnumTopicReadinessStatus.NOT_READY,
            ready_topics=tuple(t for t in topics if t not in unready),
            failures=tuple(
                ModelTopicReadinessFailure(
                    topic=t,
                    reason=EnumTopicReadinessFailureReason.TOPIC_ABSENT,
                )
                for t in unready
            ),
            attempts=1,
        )


class _RecordingBus:
    """Event bus double that captures every published envelope."""

    def __init__(self) -> None:
        self.published: list[tuple[str, object]] = []

    async def subscribe(
        self,
        *,
        topic: str,
        node_identity: object,
        on_message: object,
    ) -> object:
        async def _unsub() -> None:
            return None

        return _unsub

    async def publish_envelope(
        self,
        envelope: object,
        topic: str,
        *,
        key: bytes | None = None,
    ) -> None:
        self.published.append((topic, envelope))


def _make_pool() -> MagicMock:
    pool = MagicMock()
    conn = AsyncMock()
    record = MagicMock()
    record.__getitem__ = MagicMock(side_effect=lambda k: 1 if k == "id" else None)
    conn.fetchrow = AsyncMock(return_value=record)
    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=conn)
    ctx.__aexit__ = AsyncMock(return_value=False)
    pool.acquire = MagicMock(return_value=ctx)
    pool._test_conn = conn
    return pool


# ---------------------------------------------------------------------------
# Chain drivers
# ---------------------------------------------------------------------------


def _contract(name: str, topic: str) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="ORCHESTRATOR_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path("/fake/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        event_bus=ModelEventBusWiring(subscribe_topics=(topic,), publish_topics=()),
        handler_routing=ModelHandlerRouting(
            routing_strategy="payload_type_match",
            handlers=(
                ModelHandlerRoutingEntry(
                    handler=ModelHandlerRef(name="FakeHandler", module="fake.module"),
                    event_model=None,
                    operation=None,
                ),
            ),
        ),
    )


def _fake_handler_cls() -> type:
    class FakeHandler:
        async def handle(self, envelope: object) -> None:
            return None

    return FakeHandler


async def _run_boot_interleave() -> tuple[
    ModelAutoWiringManifest,
    ModelAutoWiringReport,
    _RecordingBus,
    ModelRuntimeAttachReadiness,
]:
    """Run the REAL provision -> confirm-ready -> attach interleave.

    ``BLOCKED_TOPIC`` never converges, so ``BLOCKED_CONTRACT`` is recorded
    NOT_READY exactly as it is on a cold lane.
    """
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract(ATTACHING_CONTRACT, ATTACHING_TOPIC),
            _contract(BLOCKED_CONTRACT, BLOCKED_TOPIC),
        )
    )
    engine = MessageDispatchEngine()
    bus = _RecordingBus()
    attach_results: list[ModelContractAttachResult] = []

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
            provisioner=_AbsentTopicProvisioner(
                absent_topics=frozenset({BLOCKED_TOPIC})
            ),
            readiness_config=ModelTopicReadinessConfig(
                max_concurrent_contract_attach=1
            ),
            attach_results_out=attach_results,
        )

    readiness = ModelRuntimeAttachReadiness.from_results(tuple(attach_results))
    return manifest, report, bus, readiness


async def _publish(
    *,
    thread_readiness: bool,
) -> tuple[_RecordingBus, ModelRuntimeManifestPublished]:
    """Drive the boot interleave then the REAL kernel publish function.

    ``thread_readiness=False`` reproduces the pre-OMN-15512 shape, where the
    aggregate was computed and then dropped.
    """
    manifest, report, bus, readiness = await _run_boot_interleave()
    published = await publish_runtime_manifest(
        event_bus=bus,
        report=report,
        manifest=manifest,
        runtime_profile="seam-test",
        topic=MANIFEST_TOPIC,
        correlation_id=uuid4(),
        image_digest=None,
        attach_readiness=readiness if thread_readiness else None,
    )
    return bus, published


def _captured_manifest_payload(bus: _RecordingBus) -> dict[str, object]:
    """The wire-shape payload as a consumer off the topic would see it."""
    manifest_events = [e for t, e in bus.published if t == MANIFEST_TOPIC]
    assert len(manifest_events) == 1, (
        f"expected exactly one runtime-manifest publish, got {len(manifest_events)}"
    )
    envelope = manifest_events[0]
    dumped = envelope.model_dump(mode="json")  # type: ignore[attr-defined]
    payload = dumped["payload"]
    assert isinstance(payload, dict)
    return payload


async def _sql_args_for(payload: dict[str, object]) -> tuple[object, ...]:
    """Coerce the wire payload into the intent model and run the real handler.

    ``ModelPayloadInsertRuntimeManifest`` is ``extra="forbid"``: if the producer
    key name and the consumer field name ever drift apart, this coercion raises
    instead of silently dropping the blocker set.
    """
    intent_payload = ModelPayloadInsertRuntimeManifest(
        runtime_profile=str(payload["runtime_profile"]),
        contract_hash=str(payload["contract_hash"]),
        topology_hash=str(payload["topology_hash"]),
        manifest_hash=str(payload["topology_hash"]),
        contracts=payload["contracts"],  # type: ignore[arg-type]
        owned_command_topics=payload["owned_command_topics"],  # type: ignore[arg-type]
        subscribed_event_topics=payload["subscribed_event_topics"],  # type: ignore[arg-type]
        handlers=payload["handlers"],  # type: ignore[arg-type]
        skipped_contracts=payload["skipped_contracts"],  # type: ignore[arg-type]
        failed_contracts=payload["failed_contracts"],  # type: ignore[arg-type]
        ownership_violations=payload["ownership_violations"],  # type: ignore[arg-type]
        image_digest=None,
        started_at=payload["started_at"],  # type: ignore[arg-type]
        attach_readiness=payload["attach_readiness"],  # type: ignore[arg-type]
    )
    pool = _make_pool()
    result = await HandlerPostgresRuntimeManifestInsert(pool).handle(
        intent_payload, uuid4()
    )
    assert result.success is True, result.error
    args: tuple[object, ...] = pool._test_conn.fetchrow.call_args[0]
    assert args[0] == SQL_INSERT_RUNTIME_MANIFEST
    return args


# ---------------------------------------------------------------------------
# The published envelope carries the blocker set (OMN-15512 AC1 / AC4)
# ---------------------------------------------------------------------------


class TestPublishedEnvelopeNamesTheAbsentTopic:
    @pytest.mark.asyncio
    async def test_blocked_contract_is_not_ready_in_published_payload(self) -> None:
        """AC4: the contract whose required topic is absent appears, by name."""
        bus, _published = await _publish(thread_readiness=True)
        payload = _captured_manifest_payload(bus)

        readiness = payload["attach_readiness"]
        assert readiness is not None, (
            "attach_readiness missing from the published payload — the aggregate "
            "did not leave service_kernel, which IS the OMN-15512 defect"
        )
        assert isinstance(readiness, dict)

        blocked = [
            r for r in readiness["results"] if r["contract_name"] == BLOCKED_CONTRACT
        ]
        assert len(blocked) == 1, (
            f"{BLOCKED_CONTRACT} absent from the published NOT-READY set: "
            f"{readiness['results']}"
        )
        assert blocked[0]["status"] == EnumContractAttachStatus.NOT_READY.value

    @pytest.mark.asyncio
    async def test_published_blocker_names_the_failing_topic(self) -> None:
        """AC1: the failing TOPIC survives, not just the contract name.

        A blocker set that says "node_beta did not attach" without naming the
        topic is not a replacement for the log grep — the operator still has to
        go read logs to find out which topic. The topic string is the payload.
        """
        bus, _published = await _publish(thread_readiness=True)
        payload = _captured_manifest_payload(bus)
        readiness = payload["attach_readiness"]
        assert isinstance(readiness, dict)

        blocked = next(
            r for r in readiness["results"] if r["contract_name"] == BLOCKED_CONTRACT
        )
        assert blocked["readiness"] is not None, (
            "readiness detail was dropped in serialization — the topic-level "
            "failure is exactly what makes this queryable"
        )
        failing_topics = {f["topic"] for f in blocked["readiness"]["failures"]}
        assert BLOCKED_TOPIC in failing_topics, (
            f"absent topic {BLOCKED_TOPIC!r} not named in the published blocker; "
            f"got {failing_topics}"
        )
        assert (
            blocked["readiness"]["status"] == EnumTopicReadinessStatus.NOT_READY.value
        )

    @pytest.mark.asyncio
    async def test_attaching_contract_is_not_in_the_blocker_set(self) -> None:
        """The blocker set is the NON-attached subset, not every contract."""
        bus, _published = await _publish(thread_readiness=True)
        payload = _captured_manifest_payload(bus)
        readiness = payload["attach_readiness"]
        assert isinstance(readiness, dict)

        names = {r["contract_name"] for r in readiness["results"]}
        assert BLOCKED_CONTRACT in names
        assert ATTACHING_CONTRACT not in names

    @pytest.mark.asyncio
    async def test_counts_are_whole_walk_not_blocker_length(self) -> None:
        """Narrowing ``results`` must not corrupt the counts.

        ``required - attached == len(results)`` is the invariant that lets a
        reader reconstruct the full picture from the narrowed copy.
        """
        bus, published = await _publish(thread_readiness=True)
        payload = _captured_manifest_payload(bus)
        readiness = payload["attach_readiness"]
        assert isinstance(readiness, dict)

        assert readiness["required_contracts"] == 2
        assert readiness["attached_contracts"] == 1
        assert readiness["state"] == EnumRuntimeReadinessState.DEGRADED.value
        assert (
            readiness["required_contracts"] - readiness["attached_contracts"]
            == len(readiness["results"])
            == 1
        )
        assert published.attach_readiness is not None
        assert published.attach_readiness.attached_contracts == 1

    @pytest.mark.asyncio
    async def test_payload_stays_wire_compatible_with_the_base_manifest(self) -> None:
        """Additive only: every pre-existing manifest key is still present.

        ``node_redeploy_orchestrator`` (omnimarket) also subscribes to this
        topic. Adding a key must not remove or rename one.
        """
        bus, _published = await _publish(thread_readiness=True)
        payload = _captured_manifest_payload(bus)

        for key in (
            "runtime_profile",
            "contracts",
            "owned_command_topics",
            "subscribed_event_topics",
            "handlers",
            "skipped_contracts",
            "failed_contracts",
            "ownership_violations",
            "image_digest",
            "started_at",
            "contract_hash",
            "topology_hash",
        ):
            assert key in payload, f"published payload lost base manifest key {key!r}"

    @pytest.mark.asyncio
    async def test_attach_readiness_does_not_move_the_topology_hash(self) -> None:
        """The dedup/drift hashes must not shift because of this field.

        ``runtime_manifests`` dedups on ``topology_hash``; if the new field fed
        the hash, every historical comparison would break.
        """
        _bus_with, published_with = await _publish(thread_readiness=True)
        _bus_without, published_without = await _publish(thread_readiness=False)
        assert published_with.topology_hash == published_without.topology_hash
        assert published_with.contract_hash == published_without.contract_hash


# ---------------------------------------------------------------------------
# The blocker set survives into the projection write (OMN-15512 AC2)
# ---------------------------------------------------------------------------


class TestProjectionWriteCarriesTheBlockerSet:
    @pytest.mark.asyncio
    async def test_sql_args_carry_state_and_counts(self) -> None:
        bus, _published = await _publish(thread_readiness=True)
        args = await _sql_args_for(_captured_manifest_payload(bus))

        assert args[14] == EnumRuntimeReadinessState.DEGRADED.value  # attach_state
        assert args[15] == 2  # attach_required_contracts
        assert args[16] == 1  # attach_attached_contracts

    @pytest.mark.asyncio
    async def test_sql_args_carry_the_named_topic(self) -> None:
        """The end of the chain: the absent topic reaches the DB write."""
        bus, _published = await _publish(thread_readiness=True)
        args = await _sql_args_for(_captured_manifest_payload(bus))

        blockers = json.loads(str(args[17]))  # attach_not_ready_contracts
        assert len(blockers) == 1
        assert blockers[0]["contract_name"] == BLOCKED_CONTRACT
        failing_topics = {f["topic"] for f in blockers[0]["readiness"]["failures"]}
        assert BLOCKED_TOPIC in failing_topics

    @pytest.mark.asyncio
    async def test_migration_declares_every_column_the_sql_writes(self) -> None:
        """Guard the SQL <-> DDL seam; a missing column fails only in prod."""
        repo_root = Path(__file__).resolve().parents[3]
        migration = (
            repo_root
            / "docker"
            / "migrations"
            / "forward"
            / "095_add_attach_readiness_to_runtime_manifests.sql"
        )
        assert migration.is_file(), f"migration not found: {migration}"
        ddl = migration.read_text(encoding="utf-8").lower()
        sql = SQL_INSERT_RUNTIME_MANIFEST.lower()
        for column in (
            "attach_state",
            "attach_required_contracts",
            "attach_attached_contracts",
            "attach_not_ready_contracts",
        ):
            assert column in sql, f"INSERT does not write {column!r}"
            assert f"add column if not exists {column}" in ddl, (
                f"migration 095 does not add {column!r}"
            )


# ---------------------------------------------------------------------------
# RED-before: the same chain without the fix fails every assertion above
# ---------------------------------------------------------------------------


class TestGuardActuallyGuards:
    """Prove the tests discriminate rather than passing on anything.

    ``thread_readiness=False`` is the pre-OMN-15512 code path: the aggregate is
    computed by the interleave and then not handed to the publisher.
    """

    @pytest.mark.asyncio
    async def test_unthreaded_publish_has_no_attach_readiness(self) -> None:
        bus, published = await _publish(thread_readiness=False)
        payload = _captured_manifest_payload(bus)
        assert payload["attach_readiness"] is None
        assert published.attach_readiness is None

    @pytest.mark.asyncio
    async def test_unthreaded_projection_write_names_nothing(self) -> None:
        bus, _published = await _publish(thread_readiness=False)
        args = await _sql_args_for(_captured_manifest_payload(bus))

        # 'unknown', NOT 'ready': a boot that reported nothing must never read
        # as a boot where everything attached.
        assert args[14] == "unknown"
        assert args[15] == 0
        assert args[16] == 0
        assert json.loads(str(args[17])) == []
        assert BLOCKED_TOPIC not in str(args[17])

    @pytest.mark.asyncio
    async def test_the_interleave_itself_was_never_the_defect(self) -> None:
        """In-memory computation was already correct — that is the point.

        This is the assertion a naive unit test would have made, and it passes
        both before and after the fix. It is kept only to document why it is
        NOT sufficient evidence for this ticket.
        """
        _manifest, _report, _bus, readiness = await _run_boot_interleave()
        by_name = {r.contract_name: r for r in readiness.results}
        assert by_name[BLOCKED_CONTRACT].status is EnumContractAttachStatus.NOT_READY
        assert by_name[ATTACHING_CONTRACT].status is EnumContractAttachStatus.ATTACHED
