# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

# Copyright (c) 2026 OmniNode Team
"""Unit tests for create_kafka_topics.py multi-package discovery [OMN-5371]."""

from __future__ import annotations

import importlib.metadata as ilm
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from omnibase_infra.tools.contract_topic_extractor import ModelContractTopicEntry

# ---------------------------------------------------------------------------
# Load the script as a module
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "scripts"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "create_kafka_topics",
        _SCRIPTS_DIR / "create_kafka_topics.py",
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["create_kafka_topics"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_script()
_discover_all_packages = _mod._discover_all_packages


# ---------------------------------------------------------------------------
# Tests for _discover_all_packages
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_discover_skips_unimportable_entry_point(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Entry point that raises on load() is skipped with a warning in lenient mode."""
    bad_ep = MagicMock()
    bad_ep.name = "bad_pkg"
    bad_ep.load.side_effect = ImportError("missing module")
    monkeypatch.setattr(ilm, "entry_points", lambda *, group: [bad_ep])

    result = _discover_all_packages(lenient=True)
    assert result == []
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.unit
def test_discover_skips_nonexistent_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Entry point whose __path__ does not exist is skipped with a warning."""
    ep = MagicMock()
    ep.name = "missing_path_pkg"
    pkg = MagicMock()
    pkg.__path__ = [str(tmp_path / "does_not_exist")]
    ep.load.return_value = pkg
    monkeypatch.setattr(ilm, "entry_points", lambda *, group: [ep])

    result = _discover_all_packages(lenient=True)
    assert result == []
    assert "WARNING" in capsys.readouterr().err


@pytest.mark.unit
def test_discover_returns_valid_packages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Two valid entry points both appear in the result."""
    pkg_a_dir = tmp_path / "pkg_a_nodes"
    pkg_a_dir.mkdir()
    pkg_b_dir = tmp_path / "pkg_b_nodes"
    pkg_b_dir.mkdir()

    ep_a = MagicMock()
    ep_a.name = "alpha"
    pkg_a = MagicMock()
    pkg_a.__path__ = [str(pkg_a_dir)]
    ep_a.load.return_value = pkg_a

    ep_b = MagicMock()
    ep_b.name = "beta"
    pkg_b = MagicMock()
    pkg_b.__path__ = [str(pkg_b_dir)]
    ep_b.load.return_value = pkg_b

    monkeypatch.setattr(ilm, "entry_points", lambda *, group: [ep_b, ep_a])

    result = _discover_all_packages(lenient=True)
    assert len(result) == 2
    names = [name for name, _ in result]
    assert names == ["alpha", "beta"]  # sorted by name
    assert result[0][1] == pkg_a_dir
    assert result[1][1] == pkg_b_dir


@pytest.mark.unit
def test_discover_filters_by_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """--packages filter restricts which entry points are returned."""
    pkg_dir = tmp_path / "nodes"
    pkg_dir.mkdir()

    ep_a = MagicMock()
    ep_a.name = "alpha"
    pkg_a = MagicMock()
    pkg_a.__path__ = [str(pkg_dir)]
    ep_a.load.return_value = pkg_a

    ep_b = MagicMock()
    ep_b.name = "beta"
    # Should not even be loaded
    ep_b.load.side_effect = AssertionError("should not be called")

    monkeypatch.setattr(ilm, "entry_points", lambda *, group: [ep_a, ep_b])

    result = _discover_all_packages(filter_names=["alpha"], lenient=True)
    assert len(result) == 1
    assert result[0][0] == "alpha"


@pytest.mark.unit
def test_discover_exits_on_error_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In strict mode (lenient=False), unloadable entry points cause sys.exit."""
    bad_ep = MagicMock()
    bad_ep.name = "broken_pkg"
    bad_ep.load.side_effect = ImportError("missing")
    monkeypatch.setattr(ilm, "entry_points", lambda *, group: [bad_ep])

    with pytest.raises(SystemExit) as exc_info:
        _discover_all_packages(lenient=False)
    assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# OMN-15395 (D2): the operator CLI is the SECOND live CreateTopics path
#
# ``docs/operations/README.md`` tells operators to run this script, and
# ``scripts/compare_environments.py`` names it in the ``fix_hint`` for the
# kafka_topic_parity finding — i.e. it is the documented way to reconcile
# topics against the cloud broker. It created every topic with a flat
# ``--replication-factor`` whose default was 1, discarding each contract's
# declared ``topic_config.replication_factor`` and never consulting the
# fail-closed managed-staging policy. An operator following the runbook against
# MSK would recreate AWS_KAFKA_HIGH_RISK_CONFIG_RF_EQUALS_ONE by hand.
#
# These drive the script's real ``_run_live`` against a fake broker; the only
# substitution is the network boundary (confluent_kafka's AdminClient).
# ---------------------------------------------------------------------------

TOPIC = "onex.evt.test-producer.example-event.v1"  # onex-topic-allow: unit fixture


class _KafkaError(Exception):
    """Stand-in for ``confluent_kafka.KafkaException``.

    Named ``_KafkaError`` rather than ``_KafkaException`` only to satisfy the
    N818 naming rule; the script matches it structurally via ``isinstance``
    against whatever ``confluent_kafka.KafkaException`` resolves to.
    """


class _InvalidReplicationFactorError(Exception):
    """A broker rejection of the requested replica count (error code 38)."""

    errno = 38


@dataclass
class _FakeNewTopic:
    """Stand-in for ``confluent_kafka.admin.NewTopic``."""

    topic: str
    num_partitions: int = -1
    replication_factor: int = -1
    replica_assignment: object | None = None
    config: dict[str, str] | None = None


@dataclass
class _FakeClusterMetadata:
    topics: dict[str, object] = field(default_factory=dict)
    brokers: dict[int, object] = field(default_factory=dict)


class _FakeFuture:
    def __init__(self, exc: BaseException | None = None) -> None:
        self._exc = exc

    def exception(self) -> BaseException | None:
        return self._exc


@dataclass
class _BrokerRecorder:
    existing: tuple[str, ...] = ()
    broker_count: int = 1
    #: Reject any NewTopic asking for more replicas than this. ``None``
    #: disables the check.
    max_hostable_replication_factor: int | None = None
    requested: list[_FakeNewTopic] = field(default_factory=list)
    create_calls: int = 0


def _fake_admin_module(recorder: _BrokerRecorder) -> Any:
    class _FakeAdminClient:
        def __init__(self, _config: dict[str, str]) -> None:
            pass

        def list_topics(self, timeout: int = 10) -> _FakeClusterMetadata:
            return _FakeClusterMetadata(
                topics=dict.fromkeys(recorder.existing, object()),
                brokers={
                    index: object() for index in range(1, recorder.broker_count + 1)
                },
            )

        def create_topics(
            self, new_topics: Sequence[_FakeNewTopic]
        ) -> dict[str, _FakeFuture]:
            recorder.create_calls += 1
            futures: dict[str, _FakeFuture] = {}
            ceiling = recorder.max_hostable_replication_factor
            for new_topic in new_topics:
                recorder.requested.append(new_topic)
                if ceiling is not None and new_topic.replication_factor > ceiling:
                    futures[new_topic.topic] = _FakeFuture(
                        _InvalidReplicationFactorError(
                            "[Error 38] INVALID_REPLICATION_FACTOR"
                        )
                    )
                    continue
                recorder.existing = recorder.existing + (new_topic.topic,)
                futures[new_topic.topic] = _FakeFuture(None)
            return futures

    return MagicMock(AdminClient=_FakeAdminClient, NewTopic=_FakeNewTopic)


def _patched_confluent(recorder: _BrokerRecorder) -> Any:
    return patch.dict(
        "sys.modules",
        {
            "confluent_kafka": MagicMock(KafkaException=_KafkaError),
            "confluent_kafka.admin": _fake_admin_module(recorder),
        },
    )


def _entry(
    *,
    topic: str = TOPIC,
    partitions: int | None = None,
    replication_factor: int | None = None,
) -> ModelContractTopicEntry:
    return ModelContractTopicEntry(
        topic=topic,
        kind="evt",
        producer="test-producer",
        event_name="example-event",
        version="v1",
        source_contracts=(Path("contract.yaml"),),
        partitions=partitions,
        replication_factor=replication_factor,
    )


def _use_managed_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the Kafka config at MSK — the managed-cluster discriminator."""
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "b-1.msk.example:9098")
    monkeypatch.setenv("KAFKA_SECURITY_PROTOCOL", "SASL_SSL")
    monkeypatch.setenv("KAFKA_SASL_MECHANISM", "AWS_MSK_IAM")
    monkeypatch.setenv("KAFKA_MSK_REGION", "us-east-1")


def _use_self_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "redpanda:9092")
    monkeypatch.delenv("KAFKA_SECURITY_PROTOCOL", raising=False)
    monkeypatch.delenv("KAFKA_SASL_MECHANISM", raising=False)


def _run_live(
    entries: Sequence[ModelContractTopicEntry],
    recorder: _BrokerRecorder,
    *,
    partitions_fallback: int | None = None,
) -> int:
    specs = _mod._build_specs(entries, partitions_fallback=partitions_fallback)
    with _patched_confluent(recorder):
        return _mod._run_live(
            specs,
            bootstrap_servers="broker:9092",
            contracts_root=Path(),
        )


@pytest.mark.unit
def test_replication_factor_is_no_longer_a_cli_flag() -> None:
    """The flat ``--replication-factor`` default of 1 is GONE.

    A CLI flag whose default silently overrides every contract's declared
    durability is the exact mechanism OMN-15395 exists to remove; keeping the
    flag "for compatibility" keeps the defect one keystroke away.
    """
    parser = _mod._build_parser()
    flags = {option for action in parser._actions for option in action.option_strings}
    assert "--replication-factor" not in flags
    assert "--partitions" in flags
    # And the surviving partitions flag no longer hardcodes a value either.
    assert parser.parse_args([]).partitions is None


@pytest.mark.unit
def test_contract_declared_replication_factor_reaches_create_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RED-before: the CLI created this topic at RF1 regardless of the contract."""
    _use_self_hosted(monkeypatch)
    recorder = _BrokerRecorder(broker_count=3)

    assert _run_live([_entry(replication_factor=3)], recorder) == 0

    assert [t.replication_factor for t in recorder.requested] == [3]


@pytest.mark.unit
def test_managed_staging_rf1_is_rejected_before_any_create(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """(b) on the operator path: fail-closed, with ZERO CreateTopics issued.

    RED-before: the script cheerfully created the RF1 topic on MSK — the
    documented runbook command reproducing the AWS Health finding by hand.
    """
    _use_managed_staging(monkeypatch)
    recorder = _BrokerRecorder(broker_count=3)

    assert _run_live([_entry(replication_factor=1)], recorder) == 1

    assert recorder.create_calls == 0
    assert recorder.requested == []
    stderr = capsys.readouterr().err
    assert TOPIC in stderr
    assert "replication_factor=1" in stderr


@pytest.mark.unit
def test_undeclared_replication_resolves_to_the_managed_floor_not_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An undeclared RF gets the managed floor, never the old flat 1."""
    _use_managed_staging(monkeypatch)
    recorder = _BrokerRecorder(broker_count=3)

    assert _run_live([_entry(replication_factor=None)], recorder) == 0

    assert [t.replication_factor for t in recorder.requested] == [2]


@pytest.mark.unit
def test_capacity_ceiling_is_measured_from_the_same_list_topics_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-node broker reduces a declared RF3 — from a MEASUREMENT.

    The broker count comes off the ``list_topics()`` response the diff already
    needs, so the ceiling costs no extra round trip and is never inferred from
    the auth mechanism.
    """
    _use_self_hosted(monkeypatch)
    recorder = _BrokerRecorder(broker_count=1)

    assert _run_live([_entry(replication_factor=3)], recorder) == 0

    assert [t.replication_factor for t in recorder.requested] == [1]


@pytest.mark.unit
def test_lane_partition_cap_applies_on_the_operator_path_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI honours ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS.

    Otherwise the two live creation paths disagree about a topic's shape and the
    runtime provisioner reports permanent partition drift against topics the
    operator just created.
    """
    _use_self_hosted(monkeypatch)
    monkeypatch.setenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", "1")
    recorder = _BrokerRecorder(broker_count=1)

    assert _run_live([_entry(partitions=6, replication_factor=1)], recorder) == 0

    assert [t.num_partitions for t in recorder.requested] == [1]


@pytest.mark.unit
def test_contract_declared_partitions_beat_the_cli_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--partitions`` is a fallback for undeclared topics, not an override."""
    _use_self_hosted(monkeypatch)
    monkeypatch.delenv("ONEX_TOPIC_PROVISIONER_MAX_PARTITIONS", raising=False)
    recorder = _BrokerRecorder(broker_count=1)

    assert (
        _run_live(
            [
                _entry(partitions=6, replication_factor=1),
                _entry(
                    topic="onex.evt.test-producer.other-event.v1",  # onex-topic-allow: unit fixture
                    partitions=None,
                    replication_factor=1,
                ),
            ],
            recorder,
            partitions_fallback=2,
        )
        == 0
    )

    by_topic = {t.topic: t.num_partitions for t in recorder.requested}
    assert by_topic[TOPIC] == 6
    assert (
        by_topic["onex.evt.test-producer.other-event.v1"] == 2
    )  # onex-topic-allow: unit fixture


@pytest.mark.unit
def test_unhostable_replication_factor_is_an_error_not_a_warning(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """(D5) A broker-refused replica count is loud on the operator path too.

    The old code funnelled every create error into
    ``WARNING: Some topics failed to create``, so a durability refusal read
    exactly like a transient blip. Asserting merely that "ERROR" appears is
    NOT discriminating — the pre-existing ``still missing after creation``
    block prints ERROR either way and the exception's own text already carries
    the string ``INVALID_REPLICATION_FACTOR``. The load-bearing assertions are
    that the refusal is classified (``REFUSED by the broker``) and that it does
    NOT land in the generic best-effort warning bucket.
    """
    _use_self_hosted(monkeypatch)
    recorder = _BrokerRecorder(broker_count=3, max_hostable_replication_factor=1)

    assert _run_live([_entry(replication_factor=3)], recorder) == 1

    stderr = capsys.readouterr().err
    assert "REFUSED by the broker" in stderr
    assert "INVALID_REPLICATION_FACTOR" in stderr
    assert "Some topics failed to create" not in stderr


@pytest.mark.unit
def test_build_specs_keeps_an_undeclared_replication_factor_none() -> None:
    """``None`` means "the contract declared nothing" — never a coerced 1."""
    specs = _mod._build_specs(
        [_entry(replication_factor=None)], partitions_fallback=None
    )
    assert len(specs) == 1
    assert specs[0].replication_factor is None
    assert specs[0].partitions == 6  # the canonical default, not the old CLI 1
