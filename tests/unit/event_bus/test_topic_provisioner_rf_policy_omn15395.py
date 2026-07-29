# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Contract-driven replication + diff-before-create tests for the REAL provisioner.

These drive ``TopicProvisioner`` — the class the runtime kernel instantiates at
boot (``service_kernel`` §3.5 and the per-contract interleave) — from a real
contract YAML through the real ``ContractTopicExtractor`` to the real
``aiokafka.admin.NewTopic`` object handed to ``create_topics``. Nothing here is
a surrogate helper: the only substitution is the admin client itself, which is
the network boundary.

Each test's environment is selected the way production selects it — MSK IAM auth
in the Kafka config means "managed cluster" — not by injecting a policy object,
so the suite exercises the same discrimination the runtime performs.

RED-before / GREEN-after (OMN-15395 f), against two distinct baselines.

Against ``dev`` (the pre-fix runtime):

* ``TestManagedStagingRejectsRf1`` — the old provisioner created the RF1 topic
  and raised nothing;
* ``test_managed_staging_resolves_undeclared_rf_to_the_floor_not_one`` and
  ``test_ensure_topic_exists_uses_contract_declared_replication_factor`` — the
  old paths silently applied ``DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR = 1``;
* the five ``TestDiffBeforeCreate`` cases — the old provisioner issued one
  ``CreateTopics`` per known topic on every pass and used
  ``TopicAlreadyExistsError`` as flow control (~1,280 blind authorizations).

Against the FIRST revision of this fix (the refuse-on-undeclared policy that
adversarial review rejected) — these are the remediation guards:

* ``TestRealContractUniverseStaysProvisionable`` — that revision resolved 0 of
  168 production topics on managed staging, i.e. provisioning was a total
  no-op. The managed case fails hard there.
* ``TestDerivedTopicsWithNoContractSpec`` — derived DLQ topics are absent from
  the contract-derived registry, so ``kernel_glue._provision_dlq_topics`` (which
  has no ``try``/``except``) raised out of ``build_and_start_core_runtime`` and
  refused to start the S6 dispatch loop.
* ``TestPolicyErrorsEscapeBestEffortBoundaries`` — every external call site
  caught bare ``Exception``, so the fail-closed signal died at the module
  boundary; the static guard enumerated four offenders.
* ``test_self_hosted_reduces_declared_rf2_to_broker_capacity`` — without the
  capacity ceiling, a contract-declared RF2 fails ``CreateTopics`` on every
  single-broker broker.

Against the SECOND revision (the hardcoded self-hosted capacity ceiling) —
these are this round's remediation guards, each proven RED by reverting its
fix hunk and re-running:

* ``TestCapacityCeilingIsMeasuredNotAssumed`` — that revision set
  ``capacity_replication_factor = 1`` for every cluster whose ``sasl_mechanism``
  was not ``AWS_MSK_IAM`` and silently reduced every declared RF down to it.
  Broker count was probed nowhere in ``src/``, so the ceiling was an assumption:
  a 3-node SCRAM cluster had its contract-declared RF2/RF3 clamped to RF1.
* ``TestPolicyErrorsEscapeBestEffortBoundaries.test_guard_sees_every_receiver_shape``
  — the static guard's own regex could not match ``self._provisioner.…``, the
  dominant call shape and one already in the tree, so it certified a property
  it could not see.
* ``TestSnapshotConfigCreationPath`` — the ``config=`` creation branch shipped
  with zero coverage; reverting both halves of its fix left the whole
  event_bus/topics/runtime selection green.
* ``TestDriftIsReportedAgainstTheResolvedSpec`` — the drift feed compared the
  broker against the UNRESOLVED contract spec, so every single-node lane
  reported the eleven RF2 topics as replication drift on every pass and seeded
  the operator-gated reassignment queue with unhostable targets.

The remaining cases are deliberate regression guards on behaviour that was
already correct (a declared RF2 reaching the broker unmutated, self-hosted RF1
still working) and are labelled as such rather than claimed as RED.

Related:
    - OMN-15395: managed-staging provisioner must be contract-driven, reject RF1
    - OMN-13238: contract-declared per-topic config (the seam being made load-bearing)
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from aiokafka.admin import NewTopic

from omnibase_infra.errors import TopicReplicationPolicyError
from omnibase_infra.event_bus.enum_topic_readiness_failure_reason import (
    EnumTopicReadinessFailureReason,
)
from omnibase_infra.event_bus.model_topic_readiness_config import (
    ModelTopicReadinessConfig,
)
from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.topics.model_topic_provisioning_policy import (
    MANAGED_MINIMUM_REPLICATION_FACTOR,
    ModelTopicProvisioningPolicy,
)

# ``asyncio_mode = "auto"`` (pyproject) marks the async cases; an explicit
# module-level asyncio mark would warn on the synchronous guards below.
pytestmark = [pytest.mark.unit]

#: The real contract tree the runtime kernel provisions from at boot.
PRODUCTION_CONTRACTS_ROOT = (
    Path(__file__).resolve().parents[3] / "src" / "omnibase_infra" / "nodes"
)

TOPIC = "onex.evt.test-producer.example-event.v1"  # onex-topic-allow: unit fixture
OTHER_TOPIC = "onex.evt.test-producer.other-event.v1"  # onex-topic-allow: unit fixture


class _TopicAlreadyExistsError(Exception):
    """Stand-in for ``aiokafka.errors.TopicAlreadyExistsError``."""


@dataclass
class _AdminRecorder:
    """Records every admin call the provisioner makes."""

    existing_topics: tuple[str, ...] = ()
    describe_calls: int = 0
    #: Nodes the fake cluster reports from ``describe_cluster``. This is what
    #: the capacity ceiling is measured from — a single-node broker by default
    #: (local Redpanda / CI / the ``.201`` lanes).
    broker_count: int = 1
    #: ``describe_cluster`` is absent from the fake admin entirely when this is
    #: False, which is the "capacity could not be measured" path.
    supports_describe_cluster: bool = True
    describe_cluster_calls: int = 0
    #: Replica count the fake broker reports per partition on metadata reads.
    reported_replicas: int = 2
    #: Partition count the fake broker reports per topic on metadata reads.
    reported_partitions: int = 1
    created: list[NewTopic] = field(default_factory=list)
    #: Every name passed to ``create_topics``, INCLUDING calls the broker
    #: rejects with TopicAlreadyExistsError. This is the load-bearing counter
    #: for "issues zero CreateTopics": counting only successes cannot tell a
    #: diff-first provisioner apart from one that blind-creates and swallows
    #: the already-exists error (~1,280 wasted authorizations per pass).
    attempted: list[str] = field(default_factory=list)

    #: What the fake broker reports per topic on a metadata read.
    def metadata(self) -> list[dict[str, object]]:
        replicas = list(range(1, self.reported_replicas + 1))
        return [
            {
                "topic": name,
                "error_code": 0,
                "partitions": [
                    {"partition": index, "leader": 1, "replicas": replicas}
                    for index in range(self.reported_partitions)
                ],
            }
            for name in self.existing_topics
        ]

    def cluster(self) -> dict[str, object]:
        """The ``describe_cluster`` shape the capacity probe reads."""
        return {
            "cluster_id": "fake-cluster",
            "controller_id": 1,
            "brokers": [
                {"node_id": index, "host": f"broker-{index}", "port": 9092}
                for index in range(1, self.broker_count + 1)
            ],
        }

    @property
    def created_names(self) -> list[str]:
        return [topic.name for topic in self.created]

    def created_spec(self, name: str) -> NewTopic:
        """The NewTopic issued for ``name`` (fails loudly if none was)."""
        matches = [topic for topic in self.created if topic.name == name]
        assert matches, f"no CreateTopics was issued for {name!r}"
        return matches[0]

    def created_under_test(self) -> list[str]:
        """Only the fixture topics, ignoring installed-package contract topics.

        ``TopicProvisioner`` extracts from installed packages as well as the
        contracts_root, which is production behaviour; the fixture assertions
        scope to the topics this test declares.
        """
        return [name for name in self.created_names if name in (TOPIC, OTHER_TOPIC)]

    def attempted_under_test(self) -> list[str]:
        """Fixture topics a ``CreateTopics`` request was issued for."""
        return [name for name in self.attempted if name in (TOPIC, OTHER_TOPIC)]


@contextmanager
def _patched_admin(recorder: _AdminRecorder) -> Iterator[None]:
    """Substitute only the network boundary: the aiokafka admin client."""

    class _FakeAdmin:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def describe_cluster(self) -> dict[str, object]:
            recorder.describe_cluster_calls += 1
            return recorder.cluster()

        async def describe_topics(
            self, topics: Sequence[str] | None = None
        ) -> list[dict[str, object]]:
            recorder.describe_calls += 1
            return recorder.metadata()

        async def create_topics(self, new_topics: Sequence[NewTopic]) -> None:
            for new_topic in new_topics:
                recorder.attempted.append(new_topic.name)
                if new_topic.name in recorder.existing_topics:
                    raise _TopicAlreadyExistsError(new_topic.name)
                recorder.created.append(new_topic)
                # The broker now has it: subsequent metadata reads must see it,
                # which is what makes the readiness confirm meaningful.
                recorder.existing_topics = recorder.existing_topics + (new_topic.name,)

    if not recorder.supports_describe_cluster:
        del _FakeAdmin.describe_cluster

    with patch.dict(
        "sys.modules",
        {
            "aiokafka": MagicMock(),
            "aiokafka.admin": MagicMock(
                AIOKafkaAdminClient=_FakeAdmin,
                NewTopic=NewTopic,
            ),
            "aiokafka.errors": MagicMock(
                TopicAlreadyExistsError=_TopicAlreadyExistsError,
            ),
        },
    ):
        yield


def _write_contract(
    root: Path,
    *,
    topic: str = TOPIC,
    replication_factor: int | None = None,
    partitions: int = 3,
) -> Path:
    """Write a real node contract the extractor can read."""
    node_dir = root / "node_example"
    node_dir.mkdir(exist_ok=True)
    lines = [
        "name: node_example",
        "version: 1.0.0",
        "namespace: onex.stamped",
        "event_bus:",
        "  publish_topics:",
        f"    - {topic}",
        "published_events:",
        f'  - topic: "{topic}"',
        '    event_type: "ExampleEvent"',
        "    topic_config:",
        f"      partitions: {partitions}",
    ]
    if replication_factor is not None:
        lines.append(f"      replication_factor: {replication_factor}")
    (node_dir / "contract.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def _use_managed_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the Kafka config at MSK — the managed-cluster discriminator."""
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "b-1.msk.example:9098")
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "AWS_MSK_IAM")
    monkeypatch.setenv("KAFKA_MSK_REGION", "us-east-1")


def _use_self_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the Kafka config at a self-hosted single-broker Redpanda."""
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
    monkeypatch.delenv("KAFKA_SECURITY_PROTOCOL", raising=False)
    monkeypatch.delenv("KAFKA_SASL_MECHANISM", raising=False)


def _use_scram_multi_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the Kafka config at a multi-broker cluster reached over SCRAM.

    Not MSK IAM, therefore classified SELF_HOSTED by the profile discriminator —
    and that classification must NOT imply a node count.
    """
    monkeypatch.setenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        "b-1.example:9096,b-2.example:9096,b-3.example:9096",
    )
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "SCRAM-SHA-512")


def _provisioner(contracts_root: Path) -> TopicProvisioner:
    return TopicProvisioner(
        bootstrap_servers="broker:9092",
        contracts_root=contracts_root,
    )


#: Any receiver, awaited or not, calling either provisioning entrypoint.
#:
#: Deliberately NOT anchored on a receiver name. The previous pattern
#: (``await\s+_?\w*provisioner\w*\.``) could not match ``self._provisioner.``
#: because ``\w`` does not span the dot, and matched nothing at all for a
#: receiver that simply is not called "provisioner" (``mgr``,
#: ``self.topic_manager``). Matching the METHOD is the invariant; the receiver
#: is incidental.
_PROVISIONING_CALL_RE = re.compile(
    r"\.(?:ensure_topic_exists|ensure_provisioned_topics_exist)\s*\("
)


def _provisioning_swallow_offenders(src_root: Path) -> list[str]:
    """Return ``path:line`` for provisioning calls behind a bare ``except``.

    A call site is an offender when the first ``except`` clause that follows it
    inside the same enclosing block is not ``except TopicReplicationPolicyError``
    — i.e. a durability violation would be degraded to a best-effort warning
    before it ever reaches the caller.
    """
    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        if path.name == "service_topic_manager.py":
            continue  # the module that raises; its own handlers are tested above
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if not _PROVISIONING_CALL_RE.search(line):
                continue
            if line.lstrip().startswith(("#", "*", '"', "'")):
                continue  # prose/docstring mention, not a call site
            # Walk forward to the first except clause of the enclosing try.
            window = lines[index : index + 40]
            excepts = [
                entry.strip() for entry in window if entry.strip().startswith("except ")
            ]
            if not excepts:
                continue  # no boundary here; the error propagates by default
            if not excepts[0].startswith("except TopicReplicationPolicyError"):
                offenders.append(f"{path.relative_to(src_root)}:{index + 1}")
    return offenders


class TestManagedStagingRejectsRf1:
    """(b) RF1 against managed staging is refused before any CreateTopics."""

    async def test_managed_staging_rejects_contract_declared_rf1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A contract declaring replication_factor: 1 aborts the pass, creates nothing."""
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=1)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder()

        with _patched_admin(recorder):
            with pytest.raises(TopicReplicationPolicyError) as excinfo:
                await provisioner.ensure_provisioned_topics_exist()

        assert TOPIC in str(excinfo.value)
        assert "replication_factor=1" in str(excinfo.value)
        # Fail-closed: not a warning, not a clamp-and-continue.
        assert recorder.created == []

    async def test_managed_staging_resolves_undeclared_rf_to_the_floor_not_one(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An undeclared RF resolves to the managed floor — never to 1.

        This is the RED-before assertion for the module-level default: the
        pre-fix provisioner applied ``DEFAULT_EVENT_TOPIC_REPLICATION_FACTOR = 1``
        here, which is how 519 RF1 topics reached MSK. The post-fix value is the
        managed durability floor (the cluster's own broker default), so the
        topic is still created and it is created durably.
        """
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=None)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder()

        with _patched_admin(recorder):
            result = await provisioner.ensure_provisioned_topics_exist()

        assert TOPIC in result["created"]
        created = recorder.created_spec(TOPIC)
        assert created.replication_factor == MANAGED_MINIMUM_REPLICATION_FACTOR
        assert created.replication_factor != 1

    async def test_single_topic_path_rejects_rf1_in_managed_staging(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The per-contract ensure path is fail-closed too, not best-effort False."""
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=1)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder()

        with _patched_admin(recorder):
            with pytest.raises(TopicReplicationPolicyError):
                await provisioner.ensure_topic_exists(topic_name=TOPIC)

        assert recorder.created == []


class TestExplicitReplicationPassesThroughUnmutated:
    """(a)/(c) A declared RF reaches CreateTopics exactly as declared."""

    async def test_declared_rf2_reaches_create_topics_unmutated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RF2 + declared partitions arrive on the NewTopic object unchanged."""
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2, partitions=3)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder()

        with _patched_admin(recorder):
            result = await provisioner.ensure_provisioned_topics_exist()

        assert recorder.created_under_test() == [TOPIC]
        created = recorder.created_spec(TOPIC)
        assert created.replication_factor == 2
        assert created.num_partitions == 3
        assert result["status"] == "success"

    async def test_declared_rf_above_the_floor_is_not_clamped_down(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A contract asking for MORE durability than the floor keeps it.

        The managed profile has no capacity ceiling, so the resolver is a floor
        check, not a normaliser: RF3 reaches ``CreateTopics`` as RF3.
        """
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=3)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder()

        with _patched_admin(recorder):
            await provisioner.ensure_provisioned_topics_exist()

        assert recorder.created_spec(TOPIC).replication_factor == 3

    async def test_ensure_topic_exists_uses_contract_declared_replication_factor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A bare topic NAME still creates to the owning contract's spec.

        This is the per-contract boot interleave's call shape
        (``ensure_topic_exists(topic_name=topic)`` with no spec), which used to
        land on a hardcoded RF1 regardless of what the contract declared.
        """
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2, partitions=3)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder()

        with _patched_admin(recorder):
            created = await provisioner.ensure_topic_exists(topic_name=TOPIC)

        assert created is True
        assert recorder.created_names == [TOPIC]
        assert recorder.created_spec(TOPIC).replication_factor == 2
        assert recorder.created_spec(TOPIC).num_partitions == 3

    async def test_self_hosted_resolves_undeclared_rf_to_declared_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Single-broker self-hosted brokers keep working on RF1."""
        _use_self_hosted(monkeypatch)
        _write_contract(tmp_path, replication_factor=None)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder()

        with _patched_admin(recorder):
            result = await provisioner.ensure_provisioned_topics_exist()

        assert result["status"] == "success"
        assert recorder.created_spec(TOPIC).replication_factor == 1

    async def test_self_hosted_reduces_declared_rf2_to_broker_capacity(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A contract-declared RF2 still provisions on a single-broker Redpanda.

        This is what makes contract-declared RF2 landable at all: without the
        capacity ceiling, every one of the eleven RF2 declarations restored to
        the contract tree would fail ``CreateTopics`` with
        ``INVALID_REPLICATION_FACTOR`` on local dev, CI, and the ``.201`` lanes.
        The reduction is one-way — capacity never raises a value, and the
        validator forbids a ceiling below the profile's durability floor. Note
        that the ceiling comes from the fake broker's ``describe_cluster``
        node count, not from its auth mechanism.
        """
        _use_self_hosted(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(broker_count=1)

        with _patched_admin(recorder):
            result = await provisioner.ensure_provisioned_topics_exist()

        assert result["status"] == "success"
        assert recorder.describe_cluster_calls == 1
        assert recorder.created_spec(TOPIC).replication_factor == 1
        assert provisioner.policy.broker_count == 1


class TestCapacityCeilingIsMeasuredNotAssumed:
    """The declared RF is only reduced by a MEASURED node count (OMN-15395).

    RED-before, against the previous revision of this fix: ``self_hosted()``
    hardcoded ``capacity_replication_factor = 1`` for every cluster whose
    ``sasl_mechanism`` was not ``AWS_MSK_IAM``. Since ``ModelKafkaEventBusConfig``
    also accepts PLAIN / SCRAM / OAUTHBEARER, any multi-broker cluster reached
    over one of those had its contract-declared RF2/RF3 silently clamped to
    RF1 — reintroducing ``AWS_KAFKA_HIGH_RISK_CONFIG_RF_EQUALS_ONE`` through the
    mechanism meant to prevent it. Broker count was never probed anywhere in
    ``src/``.
    """

    async def test_three_broker_scram_cluster_keeps_declared_rf2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 3-node SCRAM cluster provisions the contract's RF2 as RF2."""
        _use_scram_multi_broker(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(broker_count=3)

        with _patched_admin(recorder):
            await provisioner.ensure_provisioned_topics_exist()

        assert recorder.created_spec(TOPIC).replication_factor == 2
        assert provisioner.policy.broker_count == 3
        assert provisioner.policy.capacity_replication_factor == 3

    async def test_three_broker_scram_cluster_keeps_declared_rf3(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The deleted ``test_declared_rf3_is_passed_through`` property, restored."""
        _use_scram_multi_broker(monkeypatch)
        _write_contract(tmp_path, replication_factor=3)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(broker_count=3)

        with _patched_admin(recorder):
            await provisioner.ensure_provisioned_topics_exist()

        assert recorder.created_spec(TOPIC).replication_factor == 3

    async def test_multi_broker_undeclared_rf_is_durable_not_one(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 3-node self-hosted cluster has no business minting RF1 either."""
        _use_scram_multi_broker(monkeypatch)
        _write_contract(tmp_path, replication_factor=None)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(broker_count=3)

        with _patched_admin(recorder):
            await provisioner.ensure_provisioned_topics_exist()

        assert recorder.created_spec(TOPIC).replication_factor == (
            MANAGED_MINIMUM_REPLICATION_FACTOR
        )

    async def test_unmeasurable_cluster_does_not_guess_a_ceiling(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No ``describe_cluster`` means no ceiling — never an assumed RF1.

        A declared RF the broker cannot host then fails loudly at
        ``CreateTopics`` instead of being quietly downgraded.
        """
        _use_self_hosted(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(supports_describe_cluster=False)

        with _patched_admin(recorder):
            await provisioner.ensure_provisioned_topics_exist()

        assert recorder.created_spec(TOPIC).replication_factor == 2
        assert provisioner.policy.capacity_replication_factor is None

    async def test_capacity_is_measured_once_per_provisioner(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The probe is one metadata request per instance, not per topic.

        The whole point of (d) is that a provisioning pass stops issuing a
        per-topic authorization sweep; a per-topic capacity probe would put the
        same fan-out straight back.
        """
        _use_self_hosted(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(broker_count=1)

        with _patched_admin(recorder):
            await provisioner.ensure_provisioned_topics_exist()
            await provisioner.ensure_topic_exists(topic_name=OTHER_TOPIC)
            await provisioner.ensure_provisioned_topics_exist()

        assert recorder.describe_cluster_calls == 1


class TestDiffBeforeCreate:
    """(d) List/diff first; only genuinely missing topics are created."""

    async def test_fully_provisioned_cluster_issues_zero_creates(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pass over an already-provisioned cluster issues no CreateTopics."""
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(existing_topics=(TOPIC,))

        with _patched_admin(recorder):
            result = await provisioner.ensure_provisioned_topics_exist()

        # Zero CreateTopics REQUESTS — not merely zero successful creations.
        # Scoped to the fixture topic: the provisioner also extracts the real
        # installed-package contract universe, which is production behaviour.
        assert recorder.attempted_under_test() == []
        assert recorder.describe_calls == 1
        assert TOPIC not in result["created"]
        assert TOPIC in result["existing"]

    async def test_only_missing_topics_are_created(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A partially-provisioned cluster creates only the absent topic."""
        _use_managed_staging(monkeypatch)
        node_dir = tmp_path / "node_example"
        node_dir.mkdir()
        (node_dir / "contract.yaml").write_text(
            "name: node_example\n"
            "event_bus:\n"
            "  publish_topics:\n"
            f"    - {TOPIC}\n"
            f"    - {OTHER_TOPIC}\n"
            "published_events:\n"
            f'  - topic: "{TOPIC}"\n'
            '    event_type: "ExampleEvent"\n'
            "    topic_config:\n"
            "      partitions: 3\n"
            "      replication_factor: 2\n"
            f'  - topic: "{OTHER_TOPIC}"\n'
            '    event_type: "OtherEvent"\n'
            "    topic_config:\n"
            "      partitions: 1\n"
            "      replication_factor: 2\n",
            encoding="utf-8",
        )
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(existing_topics=(TOPIC,))

        with _patched_admin(recorder):
            result = await provisioner.ensure_provisioned_topics_exist()

        # The already-present topic gets no CreateTopics request at all.
        assert recorder.attempted_under_test() == [OTHER_TOPIC]
        assert recorder.created_under_test() == [OTHER_TOPIC]
        assert OTHER_TOPIC in result["created"]
        assert TOPIC in result["existing"]

    async def test_existing_topic_skips_create_on_single_topic_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``ensure_topic_exists`` no longer blind-creates an existing topic."""
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(existing_topics=(TOPIC,))

        with _patched_admin(recorder):
            created = await provisioner.ensure_topic_exists(topic_name=TOPIC)

        assert created is True
        assert recorder.attempted == []

    async def test_broker_snapshot_is_fetched_once_per_provisioner(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Repeated per-contract ensures reuse one metadata request, not N creates."""
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(existing_topics=(TOPIC,))

        with _patched_admin(recorder):
            for _ in range(5):
                await provisioner.ensure_topic_exists(topic_name=TOPIC)

        assert recorder.describe_calls == 1
        assert recorder.attempted == []

    async def test_existing_topic_spec_drift_is_reported_not_recreated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Partition drift on a live topic is reported, never re-created/mutated."""
        _use_managed_staging(monkeypatch)
        # Contract wants 3 partitions; the broker snapshot reports 1.
        _write_contract(tmp_path, replication_factor=2, partitions=3)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(existing_topics=(TOPIC,))

        with _patched_admin(recorder):
            result = await provisioner.ensure_provisioned_topics_exist()

        assert recorder.attempted_under_test() == []
        drift = result["drift"]
        assert isinstance(drift, list)
        assert any("partition" in entry.lower() for entry in drift)


class TestRealContractUniverseStaysProvisionable:
    """The whole point: the policy must not make provisioning a no-op.

    A durability policy that refuses every topic is fail-closed in the same
    sense that unplugging the cluster is fail-closed. These drive the REAL
    production contract tree — the same ``contracts_root`` the kernel passes at
    boot (``service_kernel`` §3.5) — through the REAL policy, and assert the
    resolver produces a usable plan rather than an empty one.
    """

    def test_every_production_topic_resolves_under_the_managed_policy(self) -> None:
        """Zero topics may be unprovisionable on managed staging.

        RED-before: at the previous revision the managed policy had no default,
        no contract in the tree declared a replication factor, and this resolved
        0 of 168 topics — provisioning against MSK was a 100% no-op, which is
        strictly worse than the RF1 bug it replaced.
        """
        from uuid import uuid4

        provisioner = TopicProvisioner(
            bootstrap_servers="broker:9092",
            contracts_root=PRODUCTION_CONTRACTS_ROOT,
            policy=ModelTopicProvisioningPolicy.managed(),
        )
        specs = provisioner._topic_specs
        assert len(specs) > 100, (
            f"expected the full production topic universe, extracted {len(specs)} "
            "— an empty-ish extraction would make this guard vacuous"
        )

        resolved = provisioner._resolve_specs_for_creation(specs, uuid4())

        assert len(resolved) == len(specs), (
            f"{len(specs) - len(resolved)} of {len(specs)} production topics are "
            "unprovisionable under the managed policy"
        )
        under_replicated = [
            spec.suffix
            for spec in resolved
            if spec.replication_factor is None
            or spec.replication_factor < MANAGED_MINIMUM_REPLICATION_FACTOR
        ]
        assert not under_replicated, (
            "resolved specs must all carry an explicit RF at or above the "
            f"managed floor; offenders: {under_replicated[:10]}"
        )

    def test_every_production_topic_resolves_on_a_measured_single_broker(
        self,
    ) -> None:
        """The same tree still provisions at RF1 on a broker measured at one node."""
        from uuid import uuid4

        provisioner = TopicProvisioner(
            bootstrap_servers="broker:9092",
            contracts_root=PRODUCTION_CONTRACTS_ROOT,
            policy=ModelTopicProvisioningPolicy.self_hosted(broker_count=1),
        )
        specs = provisioner._topic_specs

        resolved = provisioner._resolve_specs_for_creation(specs, uuid4())

        assert len(resolved) == len(specs)
        # Every declared RF2 is reduced to what one broker can host, so a
        # contract-declared RF2 never breaks local/CI provisioning.
        assert {spec.replication_factor for spec in resolved} == {1}

    def test_every_production_topic_stays_durable_on_a_measured_3_broker_cluster(
        self,
    ) -> None:
        """On a cluster that CAN host RF2, nothing is downgraded to RF1.

        RED-before: with the hardcoded self-hosted ceiling, this set was ``{1}``
        for a 3-broker non-IAM cluster — every contract-declared RF2 clamped.
        """
        from uuid import uuid4

        provisioner = TopicProvisioner(
            bootstrap_servers="broker:9092",
            contracts_root=PRODUCTION_CONTRACTS_ROOT,
            policy=ModelTopicProvisioningPolicy.self_hosted(broker_count=3),
        )
        specs = provisioner._topic_specs

        resolved = provisioner._resolve_specs_for_creation(specs, uuid4())

        assert len(resolved) == len(specs)
        downgraded = [
            spec.suffix
            for spec in resolved
            if spec.replication_factor is None
            or spec.replication_factor < MANAGED_MINIMUM_REPLICATION_FACTOR
        ]
        assert not downgraded, (
            "a cluster measured at 3 nodes must not mint under-replicated "
            f"topics; offenders: {downgraded[:10]}"
        )


class TestDerivedTopicsWithNoContractSpec:
    """Topics the provisioner creates that no contract declares (DLQ family).

    ``kernel_glue._provision_dlq_topics`` calls ``ensure_topic_exists`` for
    every resolved dead-letter target with NO try/except, and derived DLQ names
    (``derive_canonical_dlq_topic``) are frequently absent from the
    contract-derived spec registry. A managed policy without a default therefore
    raised out of ``build_and_start_core_runtime`` and refused to start the S6
    dispatch loop for any DLQ topic not already on the broker.
    """

    async def test_derived_dlq_topic_is_created_not_refused(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """RED-before: this raised TopicReplicationPolicyError and wedged boot."""
        from omnibase_infra.runtime.core_runtime.dlq_resolver import (
            derive_canonical_dlq_topic,
        )

        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        dlq_topic = derive_canonical_dlq_topic(TOPIC)
        provisioner = _provisioner(tmp_path)
        assert dlq_topic not in provisioner._spec_by_name, (
            "fixture invalid: the derived DLQ topic must be absent from the "
            "contract-derived registry for this to exercise the real gap"
        )
        recorder = _AdminRecorder()

        with _patched_admin(recorder):
            created = await provisioner.ensure_topic_exists(topic_name=dlq_topic)

        assert created is True
        assert recorder.created_spec(dlq_topic).replication_factor == (
            MANAGED_MINIMUM_REPLICATION_FACTOR
        )

    async def test_dlq_boot_gate_starts_the_loop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Drive the real boot helper, not a surrogate: it must not raise."""
        from omnibase_infra.runtime.core_runtime.dlq_resolver import (
            derive_canonical_dlq_topic,
        )
        from omnibase_infra.runtime.core_runtime.kernel_glue import (
            _provision_dlq_topics,
        )

        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        dlq_topic = derive_canonical_dlq_topic(TOPIC)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder()

        with _patched_admin(recorder):
            await _provision_dlq_topics(
                frozenset({dlq_topic}),
                provisioner=provisioner,
                correlation_id=None,
            )

        assert dlq_topic in recorder.created_names


class TestPolicyErrorsEscapeBestEffortBoundaries:
    """(b) The fail-closed signal must survive the call sites' ``except Exception``.

    The distinct error class only buys anything if the boot call sites re-raise
    it. Previously every external call site caught bare ``Exception`` and
    degraded a durability violation to a warning, so the fail-closed property
    stopped at the module boundary.
    """

    async def test_per_contract_interleave_reraises_policy_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``_interleave_contract`` is the per-contract boot call shape."""
        from omnibase_infra.protocols.protocol_event_bus_like import (
            ProtocolEventBusLike,
        )
        from omnibase_infra.runtime.auto_wiring.handler_wiring import (
            _interleave_contract,
        )
        from omnibase_spi.protocols.runtime import ProtocolDispatchEngine

        _use_managed_staging(monkeypatch)
        # A contract declaring RF1 against managed staging: the violation the
        # policy exists to stop.
        _write_contract(tmp_path, replication_factor=1)
        provisioner = _provisioner(tmp_path)

        class _EventBus:
            subscribe_topics = (TOPIC,)
            publish_topics: tuple[str, ...] = ()

        class _Contract:
            name = "node_example"
            contract_path = tmp_path / "node_example" / "contract.yaml"
            event_bus = _EventBus()

        recorder = _AdminRecorder()
        with _patched_admin(recorder):
            with pytest.raises(TopicReplicationPolicyError):
                await _interleave_contract(
                    name="node_example",
                    contract=_Contract(),  # type: ignore[arg-type]
                    dispatch_engine=MagicMock(spec=ProtocolDispatchEngine),
                    event_bus=MagicMock(spec=ProtocolEventBusLike),
                    environment="test",
                    result_applier=None,
                    provisioner=provisioner,
                    readiness_config=ModelTopicReadinessConfig(),
                )

        assert recorder.created == []

    def test_every_provisioning_call_site_reraises_the_policy_error(self) -> None:
        """Static guard: no ``ensure_topic_exists``/``ensure_provisioned`` call
        site may sit behind a bare ``except Exception`` without first
        re-raising ``TopicReplicationPolicyError``.

        A prose docstring promising this is not a mechanism; this is. It reads
        the shipped source so a NEW best-effort call site cannot silently
        reintroduce the swallow.
        """
        src_root = Path(__file__).resolve().parents[3] / "src" / "omnibase_infra"
        offenders = _provisioning_swallow_offenders(src_root)
        assert not offenders, (
            "provisioning call sites that swallow a durability violation into "
            f"a best-effort boundary: {offenders}. Add "
            "`except TopicReplicationPolicyError: raise` ahead of the bare "
            "`except Exception`."
        )

    @pytest.mark.parametrize(
        "receiver",
        [
            "self._provisioner",
            "provisioner",
            "_provisioner",
            "mgr",
            "self.topic_manager",
            "container.provisioner",
        ],
    )
    def test_guard_sees_every_receiver_shape(
        self, tmp_path: Path, receiver: str
    ) -> None:
        """The guard is receiver-agnostic (RED-before for its own blind spot).

        The previous regex was ``await\\s+_?\\w*provisioner\\w*\\.`` — the ``.``
        in ``self._provisioner`` breaks ``\\w*``, so the guard could not see the
        dominant instance-attribute call shape, which is ALREADY in the repo at
        ``node_topic_migration_executor_effect/handlers/``. Proven blind by
        planting that shape behind a bare ``except Exception`` and watching the
        guard pass. A guard that cannot see the shape it is guarding is not a
        mechanism.
        """
        planted = tmp_path / "planted_swallow.py"
        planted.write_text(
            "async def go() -> None:\n"
            "    try:\n"
            f"        await {receiver}.ensure_topic_exists(topic_name='t')\n"
            "    except Exception:\n"
            "        pass\n",
            encoding="utf-8",
        )
        assert _provisioning_swallow_offenders(tmp_path) == ["planted_swallow.py:3"]

    def test_guard_accepts_a_call_site_that_reraises_first(
        self, tmp_path: Path
    ) -> None:
        """Negative control: the guard is not simply flagging everything."""
        compliant = tmp_path / "compliant.py"
        compliant.write_text(
            "async def go() -> None:\n"
            "    try:\n"
            "        await self._provisioner.ensure_provisioned_topics_exist()\n"
            "    except TopicReplicationPolicyError:\n"
            "        raise\n"
            "    except Exception:\n"
            "        pass\n",
            encoding="utf-8",
        )
        assert _provisioning_swallow_offenders(tmp_path) == []


class TestSnapshotConfigCreationPath:
    """(c) The ``config=`` branch of ``ensure_topic_exists`` is a creation path too.

    ``ModelSnapshotTopicConfig`` carries its own replication factor, and this
    branch used to hand that RAW value to ``NewTopic`` — bypassing the resolver
    entirely — and recorded no created spec for the readiness path. Both halves
    shipped untested: reverting them left the whole event_bus/topics/runtime
    selection at 1252 passed.
    """

    @staticmethod
    def _snapshot_config(
        *, replication_factor: int, partition_count: int = 1
    ) -> object:
        from omnibase_infra.models.projection.model_snapshot_topic_config import (
            ModelSnapshotTopicConfig,
        )

        return ModelSnapshotTopicConfig(
            topic=TOPIC,
            partition_count=partition_count,
            replication_factor=replication_factor,
        )

    async def test_config_rf1_is_rejected_in_managed_staging(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A snapshot config declaring RF1 on MSK is refused before CreateTopics."""
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(broker_count=3)

        with _patched_admin(recorder):
            with pytest.raises(TopicReplicationPolicyError):
                await provisioner.ensure_topic_exists(
                    topic_name=TOPIC,
                    config=self._snapshot_config(replication_factor=1),  # type: ignore[arg-type]
                )

        assert recorder.attempted == []

    async def test_resolver_output_not_the_raw_config_reaches_new_topic(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A config RF above the measured capacity is reduced on the way to NewTopic.

        Mutation-killing: the pre-fix line passed ``config.replication_factor``
        (2) straight through, so this asserts a value the unresolved path
        cannot produce.
        """
        _use_self_hosted(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(broker_count=1)

        with _patched_admin(recorder):
            created = await provisioner.ensure_topic_exists(
                topic_name=TOPIC,
                config=self._snapshot_config(replication_factor=2),  # type: ignore[arg-type]
            )

        assert created is True
        assert recorder.created_spec(TOPIC).replication_factor == 1

    async def test_config_created_topic_is_readiness_checked_against_its_spec(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The config path records a created spec, so readiness can assert it.

        Mutation-killing for the ``created_spec`` half: with no recorded spec
        the readiness poll has no expectation and reports READY against a
        broker that is serving one partition where three were asked for.
        """
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2)
        provisioner = _provisioner(tmp_path)
        # Fake broker under-serves the request: 1 partition, not the 3 created.
        recorder = _AdminRecorder(
            broker_count=3, reported_partitions=1, reported_replicas=2
        )

        with _patched_admin(recorder):
            await provisioner.ensure_topic_exists(
                topic_name=TOPIC,
                config=self._snapshot_config(replication_factor=2, partition_count=3),  # type: ignore[arg-type]
            )
            readiness = await provisioner.confirm_topics_ready(
                [TOPIC], config=ModelTopicReadinessConfig(max_attempts=1)
            )

        assert not readiness.is_ready
        assert any(
            failure.reason is EnumTopicReadinessFailureReason.PARTITION_MISMATCH
            for failure in readiness.failures
        ), f"expected a partition mismatch against the created spec: {readiness}"


class TestDriftIsReportedAgainstTheResolvedSpec:
    """(d) The drift feed uses the resolver's output, like the creation site does.

    RED-before: ``_report_spec_drift`` built its expectation from the UNRESOLVED
    contract spec, so on every single-node lane (local Redpanda, CI, the
    ``.201`` stability/prod/judge lanes) each contract-declared RF2 topic was
    reported as ``replication_mismatch`` on every provisioning pass — even
    though the provisioner deliberately and correctly created it at RF1 there.
    That feed is what the operator-gated reassignment lane consumes, so it was
    seeding the repair queue with RF2 targets a one-node cluster cannot host.
    """

    async def test_single_broker_lane_reports_no_replication_drift_for_rf2(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _use_self_hosted(monkeypatch)
        _write_contract(tmp_path, replication_factor=2, partitions=1)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(
            existing_topics=(TOPIC,),
            broker_count=1,
            reported_partitions=1,
            reported_replicas=1,
        )

        with _patched_admin(recorder):
            result = await provisioner.ensure_provisioned_topics_exist()

        drift = [entry for entry in result["drift"] if TOPIC in entry]
        assert drift == [], (
            "a topic the capacity ceiling correctly created at RF1 must not be "
            f"reported as replication drift on a one-node cluster: {drift}"
        )

    async def test_genuine_under_replication_is_still_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Negative control: a cluster that CAN host RF2 and doesn't is drift."""
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2, partitions=1)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(
            existing_topics=(TOPIC,),
            broker_count=3,
            reported_partitions=1,
            reported_replicas=1,
        )

        with _patched_admin(recorder):
            result = await provisioner.ensure_provisioned_topics_exist()

        assert any(
            TOPIC in entry and "replication" in entry for entry in result["drift"]
        ), f"expected RF drift on a 3-node cluster serving 1 replica: {result['drift']}"


class TestReadinessSpecPassThrough:
    """(c) The resolved spec reaches the readiness path for topics we created."""

    async def test_created_topic_readiness_asserts_created_spec(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Readiness confirms a freshly created topic against its resolved spec."""
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2, partitions=1)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder()

        with _patched_admin(recorder):
            await provisioner.ensure_provisioned_topics_exist()
            readiness = await provisioner.confirm_topics_ready([TOPIC])

        # The fake broker reports 1 partition with 2 replicas — matching the
        # contract-declared spec the topic was created with.
        assert readiness.is_ready

    async def test_preexisting_topic_readiness_is_not_spec_gated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pre-existing topic is not flipped NOT-READY by contract spec drift.

        Deliberate: on the cluster carrying the legacy RF1 topics, asserting the
        contract's RF against topics this process did not create would block
        consumer attach. Drift is reported by the provisioning pass instead.
        """
        _use_managed_staging(monkeypatch)
        _write_contract(tmp_path, replication_factor=2, partitions=3)
        provisioner = _provisioner(tmp_path)
        recorder = _AdminRecorder(existing_topics=(TOPIC,))

        with _patched_admin(recorder):
            readiness = await provisioner.confirm_topics_ready([TOPIC])

        assert readiness.is_ready
