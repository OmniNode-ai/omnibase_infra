# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""The boot interleave's admin-client cost must be O(1), not O(topics) (OMN-17372).

Measured cause. ``subscribe_wired_contract_topics`` → ``_interleave_contract``
calls ``ensure_topic_exists(topic_name=topic)`` once per provision topic, in
declared order, for every wired contract. Before this fix each of those calls
constructed, ``start()``-ed and ``close()``-d a **fresh** ``AIOKafkaAdminClient``
— its own docstring said so — purely to reach two values that are already
memoized on the instance: the capacity-bound policy (``_measured_policy``,
memoized since OMN-15395 D4) and the live topic snapshot
(``_existing_topics``, cached since OMN-15395 d).

So the connection was O(n) while the information it fetched was O(1).

Live cost, measured inside the onex-dev cluster against MSK over IAM SASL
(2026-09-06, ``omnimarket-projection-hook-ledger-writer`` pod, n=10 serial
open/close cycles): first 2.528 s, then a steady-state **median of 0.099 s** per
open+close. The runtime's own boot log for the same window reports
``topic provisioning (contract-first) — total: 1219`` topics. One TCP connect +
TLS handshake + SigV4/OAUTHBEARER exchange per topic, for a snapshot lookup.

The invariant this file pins is a COUNT, not a duration: N calls to
``ensure_topic_exists`` over topics the broker already has must construct at
most ONE admin client. A duration assertion would be a flake on a loaded host;
a construction count is deterministic and is the thing that actually scales.

RED before the fix: ``test_repeated_ensure_topic_exists_opens_one_admin_client``
records 12 constructions for 12 topics. GREEN after: 1.

Related Tickets:
    - OMN-17372: runtime boot wiring latency (the ticket the latency was flagged on).
    - OMN-15395: the memoized policy + cached broker snapshot this relies on.
    - OMN-13237: the per-contract provision → confirm → attach interleave.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

pytestmark = pytest.mark.unit


#: Topics the fake broker already has. The boot interleave's dominant case:
#: a lane that has been up before, where every declared topic already exists.
_EXISTING = tuple(f"onex.evt.omnibase-infra.fanout-probe-{i}.v1" for i in range(12))


class _AdminConstructionRecorder:
    """Counts admin-client constructions, starts and metadata reads."""

    def __init__(self) -> None:
        self.constructed = 0
        self.started = 0
        self.closed = 0
        self.describe_topics_calls = 0
        self.describe_cluster_calls = 0


class _FakeTopicAlreadyExistsError(Exception):
    pass


class _NewTopic:
    """Stand-in for ``aiokafka.admin.NewTopic``.

    A ``MagicMock`` cannot serve here: ``name=`` is ``Mock``'s own constructor
    keyword, so the created topic's ``.name`` comes back as a child mock rather
    than the topic name, and the create-path assertion silently compares mocks.
    """

    def __init__(
        self,
        *,
        name: str,
        num_partitions: int,
        replication_factor: int,
        topic_configs: dict[str, str] | None = None,
    ) -> None:
        self.name = name
        self.num_partitions = num_partitions
        self.replication_factor = replication_factor
        self.topic_configs = topic_configs


@contextmanager
def _patched_admin(recorder: _AdminConstructionRecorder) -> Iterator[None]:
    """Substitute only the network boundary, and count every construction."""

    class _FakeAdmin:
        def __init__(self, **_kwargs: object) -> None:
            recorder.constructed += 1

        async def start(self) -> None:
            recorder.started += 1

        async def close(self) -> None:
            recorder.closed += 1

        async def describe_cluster(self) -> dict[str, object]:
            recorder.describe_cluster_calls += 1
            return {"brokers": [{"node_id": 1}, {"node_id": 2}, {"node_id": 3}]}

        async def describe_topics(
            self, topics: Sequence[str] | None = None
        ) -> list[dict[str, object]]:
            recorder.describe_topics_calls += 1
            return [
                {
                    "topic": name,
                    "partitions": [
                        {"partition": 0, "replicas": [1, 2]},
                        {"partition": 1, "replicas": [1, 2]},
                        {"partition": 2, "replicas": [1, 2]},
                    ],
                }
                for name in _EXISTING
            ]

        async def create_topics(self, new_topics: Sequence[object]) -> None:
            raise AssertionError(
                "create_topics must not be issued for topics the broker already has"
            )

    with patch.dict(
        "sys.modules",
        {
            "aiokafka": MagicMock(),
            "aiokafka.admin": MagicMock(
                AIOKafkaAdminClient=_FakeAdmin, NewTopic=_NewTopic
            ),
            "aiokafka.errors": MagicMock(
                TopicAlreadyExistsError=_FakeTopicAlreadyExistsError
            ),
        },
    ):
        yield


def _write_contract(root: Path) -> None:
    """A real contract declaring the fixture topics, read by the real extractor."""
    node_dir = root / "node_fanout_probe"
    node_dir.mkdir(exist_ok=True)
    lines = [
        "name: node_fanout_probe",
        "version: 1.0.0",
        "namespace: onex.stamped",
        "event_bus:",
        "  publish_topics:",
    ]
    lines += [f"    - {name}" for name in _EXISTING]
    lines.append("published_events:")
    for index, name in enumerate(_EXISTING):
        lines += [
            f'  - topic: "{name}"',
            f'    event_type: "FanoutProbeEvent{index}"',
            "    topic_config:",
            "      partitions: 3",
            "      replication_factor: 2",
        ]
    (node_dir / "contract.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def contracts_root(tmp_path: Path) -> Path:
    root = tmp_path / "contracts"
    root.mkdir()
    _write_contract(root)
    return root


@pytest.mark.asyncio
async def test_repeated_ensure_topic_exists_opens_one_admin_client(
    contracts_root: Path,
) -> None:
    """12 topics the broker already has must cost ONE admin connection, not 12.

    This is the boot interleave's shape: ``_interleave_contract`` awaits
    ``ensure_topic_exists`` once per provision topic, serially, per contract.
    """
    recorder = _AdminConstructionRecorder()
    provisioner = TopicProvisioner(
        bootstrap_servers="broker:9092", contracts_root=contracts_root
    )

    with _patched_admin(recorder):
        for name in _EXISTING:
            assert await provisioner.ensure_topic_exists(topic_name=name) is True

    assert recorder.constructed == 1, (
        f"expected ONE admin client for {len(_EXISTING)} already-existing topics, "
        f"got {recorder.constructed} — the per-call connection is back"
    )
    assert recorder.started == 1
    # Every client that is opened is also closed: no leaked connection.
    assert recorder.closed == recorder.constructed
    # And the information it fetched is fetched once, not once per topic.
    assert recorder.describe_topics_calls == 1
    assert recorder.describe_cluster_calls == 1


@pytest.mark.asyncio
async def test_admin_client_count_does_not_grow_with_topic_count(
    contracts_root: Path,
) -> None:
    """The count is flat in n — the property, stated as a comparison.

    A single-count assertion can be satisfied by an off-by-one; asserting that
    the first topic and the twelfth cost the same is what pins O(1).
    """
    recorder = _AdminConstructionRecorder()
    provisioner = TopicProvisioner(
        bootstrap_servers="broker:9092", contracts_root=contracts_root
    )

    with _patched_admin(recorder):
        await provisioner.ensure_topic_exists(topic_name=_EXISTING[0])
        after_first = recorder.constructed
        for name in _EXISTING[1:]:
            await provisioner.ensure_topic_exists(topic_name=name)
        after_all = recorder.constructed

    assert after_first == 1
    assert after_all == after_first, (
        f"admin constructions grew from {after_first} to {after_all} across "
        f"{len(_EXISTING) - 1} further topics — the cost is O(n), not O(1)"
    )


@pytest.mark.asyncio
async def test_a_topic_the_broker_lacks_still_reaches_the_broker(
    contracts_root: Path, tmp_path: Path
) -> None:
    """POSITIVE CONTROL for the two zero-growth assertions above.

    A cache that answers everything from memory would satisfy them trivially and
    would also be a silent regression: a genuinely missing topic must still open
    a connection and attempt a create. This proves the fast path is a cache hit,
    not a blanket short-circuit.
    """
    recorder = _AdminConstructionRecorder()
    provisioner = TopicProvisioner(
        bootstrap_servers="broker:9092", contracts_root=contracts_root
    )

    created: list[str] = []

    class _CreatingAdmin:
        def __init__(self, **_kwargs: object) -> None:
            recorder.constructed += 1

        async def start(self) -> None:
            recorder.started += 1

        async def close(self) -> None:
            recorder.closed += 1

        async def describe_cluster(self) -> dict[str, object]:
            recorder.describe_cluster_calls += 1
            return {"brokers": [{"node_id": 1}, {"node_id": 2}, {"node_id": 3}]}

        async def describe_topics(
            self, topics: Sequence[str] | None = None
        ) -> list[dict[str, object]]:
            recorder.describe_topics_calls += 1
            return [
                {
                    "topic": name,
                    "partitions": [
                        {"partition": 0, "replicas": [1, 2]},
                        {"partition": 1, "replicas": [1, 2]},
                        {"partition": 2, "replicas": [1, 2]},
                    ],
                }
                for name in _EXISTING
            ]

        async def create_topics(self, new_topics: Sequence[_NewTopic]) -> None:
            for new_topic in new_topics:
                created.append(new_topic.name)

    with patch.dict(
        "sys.modules",
        {
            "aiokafka": MagicMock(),
            "aiokafka.admin": MagicMock(
                AIOKafkaAdminClient=_CreatingAdmin,
                NewTopic=_NewTopic,
            ),
            "aiokafka.errors": MagicMock(
                TopicAlreadyExistsError=_FakeTopicAlreadyExistsError
            ),
        },
    ):
        # Warm the snapshot with an existing topic (one connection), then ask
        # for one the broker does not have.
        await provisioner.ensure_topic_exists(topic_name=_EXISTING[0])
        warmed = recorder.constructed
        await provisioner.ensure_topic_exists(topic_name=_EXISTING[0])
        assert recorder.constructed == warmed, "a cache hit must not reconnect"

        from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

        await provisioner.ensure_topic_exists(
            topic_name="onex.evt.omnibase-infra.fanout-probe-absent.v1",
            spec=ModelTopicSpec(
                suffix="onex.evt.omnibase-infra.fanout-probe-absent.v1",
                partitions=3,
                replication_factor=2,
            ),
        )

    # ">=" not "==": the create path also runs the readiness confirm
    # (``confirm_topics_ready``), which opens its own client. That second
    # connection is scoped to topics genuinely missing — it is not the O(n)
    # this fix removes — so the control asserts the boundary was crossed, not
    # an exact count it would have to be updated for.
    assert recorder.constructed >= warmed + 1, (
        "a topic missing from the cached snapshot must still open a connection"
    )
    assert created == ["onex.evt.omnibase-infra.fanout-probe-absent.v1"]
