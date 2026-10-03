# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Public locus gates reject malformed contracts and unproven consumers."""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from omnibase_infra.backends.backend_probe import (
    ConsumerGroupDescribeDeniedError,
    ConsumerGroupLivenessUnknownError,
    ConsumerGroupSaslRefusedError,
)
from omnibase_infra.backends.model_consumer_group_owner import ModelConsumerGroupOwner
from omnibase_infra.cli import delegate_locus
from omnibase_infra.cli.model_delegate_locus_decision import ModelDelegateLocusDecision
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus

pytestmark = pytest.mark.unit

_COMMAND = "onex.cmd.synthetic.delegate.v1"
_REQUEST = "onex.cmd.synthetic.request.v1"
_COMPLETED = "onex.evt.synthetic.completed.v1"
_BROKER = "broker.invalid:19092"
_FIRST = ("first-hop-group",)
_CHAIN = ("downstream-group",)


def _write(path: Path, raw: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


@pytest.fixture
def contract(tmp_path: Path) -> Path:
    return _write(
        tmp_path / "nodes" / "first" / "contract.yaml",
        {
            "name": "first",
            "event_bus": {"subscribe_topics": [_COMMAND]},
            "delegation_runtime_dispatch": {
                "topics": {"command": _REQUEST, "completed": _COMPLETED}
            },
        },
    )


def _consumer(
    contract: Path, *, name: object = "consumer", extra: object = "extra"
) -> Path:
    return _write(
        contract.parent.parent / "consumer" / "contract.yaml",
        {
            "name": name,
            "event_bus": {
                "subscribe_topics": [_REQUEST, extra],
                "publish_topics": [_COMPLETED],
            },
        },
    )


def _resolve(
    contract: Path,
    *,
    requested: EnumDelegateLocus = EnumDelegateLocus.AUTO,
    bus: str = "kafka",
    broker: str | None = _BROKER,
) -> ModelDelegateLocusDecision:
    return delegate_locus.resolve_delegate_locus(
        requested=requested,
        bus=bus,
        kafka_bootstrap=broker,
        contract_path=contract,
        shared_bus_value="kafka",
    )


@pytest.fixture
def probes(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, object]]]:
    """Fail closed offline: every liveness query is recorded and faked."""
    calls: list[tuple[str, dict[str, object]]] = []

    def first(**kwargs: object) -> tuple[str, ...]:
        calls.append(("first", kwargs))
        return _FIRST

    def chain(**kwargs: object) -> tuple[str, ...]:
        calls.append(("chain", kwargs))
        return _CHAIN

    monkeypatch.setattr(delegate_locus, "live_consumer_groups", first)
    monkeypatch.setattr(delegate_locus, "live_chain_consumer_groups", chain)
    return calls


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    now = [0.0]
    calls: list[float] = []

    def sleep(seconds: float) -> None:
        calls.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(delegate_locus, "_sleep", sleep)
    monkeypatch.setattr(delegate_locus, "_monotonic", lambda: now[0])
    monkeypatch.setattr(delegate_locus, "_REBIND_WAIT_SECONDS", 10.0)
    return calls


@pytest.mark.parametrize(
    "reader",
    [
        delegate_locus.contract_command_topic,
        delegate_locus.contract_consumer_owner,
        delegate_locus.contract_downstream_chain,
    ],
)
@pytest.mark.parametrize("contents", [None, "event_bus: ["])
def test_contract_read_failures_preserve_cause(
    tmp_path: Path, reader: Callable[[Path], object], contents: str | None
) -> None:
    path = tmp_path / "contract.yaml"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    with pytest.raises(delegate_locus.DelegateLocusRefusedError) as caught:
        reader(path)
    assert f"cannot read the delegate contract at {path}:" in str(caught.value)
    assert isinstance(caught.value.__cause__, (OSError, yaml.YAMLError))


@pytest.mark.parametrize(
    "contents", [None, "[", "[]", "terminal_event: 7", "terminal_event: done"]
)
def test_terminal_reader_is_safe_while_reporting_failures(
    tmp_path: Path, contents: str | None
) -> None:
    path = tmp_path / "contract.yaml"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    expected = "done" if contents == "terminal_event: done" else ""
    assert delegate_locus.contract_terminal_topic(path) == expected


@pytest.mark.parametrize("raw", [[], {}, {"name": ""}, {"name": 1}])
def test_owner_requires_a_nonempty_string_name(tmp_path: Path, raw: object) -> None:
    path = _write(tmp_path / "contract.yaml", raw)
    with pytest.raises(
        delegate_locus.DelegateLocusRefusedError, match="declares no string name"
    ):
        delegate_locus.contract_consumer_owner(path)


@pytest.mark.parametrize(
    "raw", [[], {}, {"event_bus": []}, {"event_bus": {"subscribe_topics": [1]}}]
)
def test_command_requires_a_topic_list(tmp_path: Path, raw: object) -> None:
    path = _write(tmp_path / "contract.yaml", raw)
    with pytest.raises(
        delegate_locus.DelegateLocusRefusedError,
        match=r"declares no event_bus\.subscribe_topics",
    ):
        delegate_locus.contract_command_topic(path)


@pytest.mark.parametrize("raw", [[], {"name": "first"}])
def test_absent_downstream_declaration_is_optional(tmp_path: Path, raw: object) -> None:
    path = _write(tmp_path / "contract.yaml", raw)
    assert delegate_locus.contract_downstream_chain(path) is None


@pytest.mark.parametrize(
    ("dispatch", "field"),
    [
        (None, "command"),
        ({"topics": []}, "command"),
        ({"topics": {"command": ""}}, "command"),
        ({"topics": {"command": 2}}, "command"),
        ({"topics": {"command": _REQUEST}}, "completed"),
        ({"topics": {"command": _REQUEST, "completed": ""}}, "completed"),
        ({"topics": {"command": _REQUEST, "completed": 2}}, "completed"),
    ],
)
def test_malformed_dispatch_refuses_before_network(
    contract: Path,
    probes: list[tuple[str, dict[str, object]]],
    dispatch: object,
    field: str,
) -> None:
    _write(
        contract,
        {
            "name": "first",
            "event_bus": {"subscribe_topics": [_COMMAND]},
            "delegation_runtime_dispatch": dispatch,
        },
    )
    with pytest.raises(delegate_locus.DelegateLocusRefusedError) as caught:
        _resolve(contract)
    assert f"no non-empty string delegation_runtime_dispatch.topics.{field}" in str(
        caught.value
    )
    assert probes == []


@pytest.mark.parametrize(
    "contents",
    [
        "name: unrelated",
        f"[ {_REQUEST}",
        f"- {_REQUEST}",
        f"note: {_REQUEST}\nevent_bus: []",
        f"event_bus:\n  subscribe_topics: {_REQUEST}\n  publish_topics: [{_COMPLETED}]",
        f"note: {_REQUEST}\nevent_bus:\n  subscribe_topics: [other]\n  publish_topics: [{_COMPLETED}]",
        f"event_bus:\n  subscribe_topics: [{_REQUEST}]\n  publish_topics: {_COMPLETED}",
        f"event_bus:\n  subscribe_topics: [{_REQUEST}]\n  publish_topics: [other]",
    ],
)
def test_nonmatching_siblings_do_not_supply_chain_evidence(
    contract: Path, probes: list[tuple[str, dict[str, object]]], contents: str
) -> None:
    sibling = contract.parent.parent / "unrelated" / "contract.yaml"
    sibling.parent.mkdir()
    sibling.write_text(contents, encoding="utf-8")
    with pytest.raises(delegate_locus.DelegateLocusRefusedError) as caught:
        _resolve(contract)
    assert (
        f"downstream command topic '{_REQUEST}' has 0 matching consumer contracts"
        in str(caught.value)
    )
    assert "no single installed contract declares" in str(caught.value)
    assert probes == []


@pytest.mark.parametrize(
    ("name", "extra", "message"),
    [
        (None, "extra", "declares no string name"),
        ("", "extra", "declares no string name"),
        (42, "extra", "declares no string name"),
        ("consumer", "", r"invalid event_bus\.subscribe_topics"),
        ("consumer", 42, r"invalid event_bus\.subscribe_topics"),
    ],
)
def test_matching_consumer_must_have_valid_identity_and_footprint(
    contract: Path,
    probes: list[tuple[str, dict[str, object]]],
    name: object,
    extra: object,
    message: str,
) -> None:
    _consumer(contract, name=name, extra=extra)
    with pytest.raises(delegate_locus.DelegateLocusRefusedError, match=message):
        _resolve(contract)
    assert probes == []


def test_multiple_consumers_refuse_with_matching_names(
    contract: Path, probes: list[tuple[str, dict[str, object]]]
) -> None:
    _consumer(contract)
    _write(
        contract.parent.parent / "second" / "contract.yaml",
        {
            "name": "second",
            "event_bus": {
                "subscribe_topics": [_REQUEST],
                "publish_topics": [_COMPLETED],
            },
        },
    )
    with pytest.raises(delegate_locus.DelegateLocusRefusedError) as caught:
        _resolve(contract)
    assert "2 matching consumer contracts: consumer, second" in str(caught.value)
    assert probes == []


def test_resolved_chain_records_both_hops_and_exact_probe_inputs(
    contract: Path,
    probes: list[tuple[str, dict[str, object]]],
    sleeps: list[float],
) -> None:
    consumer = _consumer(contract)
    chain = delegate_locus.contract_downstream_chain(contract)
    assert chain is not None
    assert chain.consumer_contract_path == str(consumer)
    decision = _resolve(contract)
    assert isinstance(decision, ModelDelegateLocusDecision)
    assert decision.locus is EnumDelegateLocus.DEPLOYED_LANE
    assert "resolved from transport=kafka" in decision.resolved_from
    assert decision.orchestrator_contract == str(contract)
    assert decision.broker == _BROKER
    assert decision.command_topic == _COMMAND
    assert decision.lane_consumer_groups == _FIRST
    assert decision.downstream_command_topic == _REQUEST
    assert decision.downstream_consumer_groups == _CHAIN
    assert decision.consumer_bind_wait_seconds == 0.0
    assert sleeps == []
    assert probes == [
        (
            "first",
            {
                "topic": _COMMAND,
                "bootstrap_servers": _BROKER,
                "timeout": 5.0,
                "owner": ModelConsumerGroupOwner(service="omnimarket", node="first"),
            },
        ),
        (
            "chain",
            {
                "command_topic": _REQUEST,
                "subscribe_topics": (_REQUEST, "extra"),
                "bootstrap_servers": _BROKER,
                "timeout": 5.0,
            },
        ),
    ]


@pytest.mark.parametrize("downstream", [False, True])
def test_rebinding_records_wait_and_then_admits(
    contract: Path,
    probes: list[tuple[str, dict[str, object]]],
    sleeps: list[float],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    downstream: bool,
) -> None:
    if downstream:
        _consumer(contract)
        target = "live_chain_consumer_groups"
        groups = _CHAIN
    else:
        _write(
            contract, {"name": "first", "event_bus": {"subscribe_topics": [_COMMAND]}}
        )
        target = "live_consumer_groups"
        groups = _FIRST
    answers = iter([(), groups])
    monkeypatch.setattr(delegate_locus, target, lambda **_: next(answers))
    decision = _resolve(contract)
    assert isinstance(decision, ModelDelegateLocusDecision)
    assert decision.lane_consumer_groups == _FIRST
    assert decision.downstream_consumer_groups == (_CHAIN if downstream else ())
    assert decision.consumer_bind_wait_seconds == 5.0
    assert sleeps == [5.0]
    assert "rebinding, not down" in caplog.text
    assert delegate_locus.REBIND_WINDOW_FAILURE_CLASS in caplog.text


@pytest.mark.parametrize("downstream", [False, True])
def test_unbound_chain_refuses_at_shared_wait_bound(
    contract: Path,
    probes: list[tuple[str, dict[str, object]]],
    sleeps: list[float],
    monkeypatch: pytest.MonkeyPatch,
    downstream: bool,
) -> None:
    _consumer(contract)
    target = "live_chain_consumer_groups" if downstream else "live_consumer_groups"
    monkeypatch.setattr(delegate_locus, target, lambda **_: ())
    expected = (
        delegate_locus.DelegateDownstreamChainRefusedError
        if downstream
        else delegate_locus.DelegateLocusRefusedError
    )
    with pytest.raises(expected) as caught:
        _resolve(contract)
    assert "Waited 10.0 s over 3 probes" in str(caught.value)
    assert "--bus inmemory --locus in-process" in str(caught.value)
    assert delegate_locus.REBIND_WINDOW_FAILURE_CLASS in str(caught.value)
    assert sleeps == [5.0, 5.0]
    if downstream:
        assert _FIRST[0] in str(caught.value)
        assert "execution budget" in str(caught.value)
    else:
        assert all(kind == "first" for kind, _ in probes)


@pytest.mark.parametrize("downstream", [False, True])
@pytest.mark.parametrize("kind", ["acl", "unknown", "sasl"])
def test_probe_failures_refuse_immediately_with_typed_context(
    contract: Path,
    probes: list[tuple[str, dict[str, object]]],
    sleeps: list[float],
    monkeypatch: pytest.MonkeyPatch,
    downstream: bool,
    kind: str,
) -> None:
    _consumer(contract)
    error: ConsumerGroupLivenessUnknownError
    if kind == "acl":
        error = ConsumerGroupDescribeDeniedError(
            group_ids=("denied-group",), bootstrap_servers=_BROKER, principal="tester"
        )
        expected: type[delegate_locus.DelegateLocusRefusedError] = (
            delegate_locus.DelegateLocusAclRefusedError
        )
    elif kind == "sasl":
        error = ConsumerGroupSaslRefusedError(
            bootstrap_servers=_BROKER, principal="tester"
        )
        expected = (
            delegate_locus.DelegateDownstreamChainRefusedError
            if downstream
            else delegate_locus.DelegateLocusSaslRefusedError
        )
    else:
        error = ConsumerGroupLivenessUnknownError("probe unavailable")
        expected = (
            delegate_locus.DelegateDownstreamChainRefusedError
            if downstream
            else delegate_locus.DelegateLocusRefusedError
        )

    def fail(**_: object) -> tuple[str, ...]:
        raise error

    target = "live_chain_consumer_groups" if downstream else "live_consumer_groups"
    monkeypatch.setattr(delegate_locus, target, fail)
    with pytest.raises(expected) as caught:
        _resolve(contract)
    assert caught.value.__cause__ is error
    assert "cannot confirm" in str(caught.value)
    assert (_REQUEST if downstream else _COMMAND) in str(caught.value)
    assert sleeps == []
    if isinstance(caught.value, delegate_locus.DelegateLocusAclRefusedError):
        assert caught.value.group_ids == ("denied-group",)
        assert "ALLOW User:tester DESCRIBE on GROUP 'denied-group'" in str(caught.value)
    if isinstance(caught.value, delegate_locus.DelegateLocusSaslRefusedError):
        assert caught.value.broker == _BROKER
        assert caught.value.principal == "tester"
    if downstream:
        assert delegate_locus.DOWNSTREAM_CHAIN_STAGE in str(caught.value)


def test_deployed_requires_explicit_broker_before_probing(
    contract: Path, probes: list[tuple[str, dict[str, object]]]
) -> None:
    _consumer(contract)
    with pytest.raises(
        delegate_locus.DelegateLocusRefusedError, match="no broker address was resolved"
    ) as caught:
        _resolve(contract, broker=None)
    assert "--lane <lane id>" in str(caught.value)
    assert probes == []


def test_deployed_cannot_use_inmemory_transport(
    contract: Path, probes: list[tuple[str, dict[str, object]]]
) -> None:
    with pytest.raises(
        delegate_locus.DelegateLocusRefusedError,
        match="incoherent with transport 'inmemory'",
    ):
        _resolve(contract, requested=EnumDelegateLocus.DEPLOYED_LANE, bus="inmemory")
    assert probes == []


@pytest.mark.parametrize("bus", ["inmemory", "kafka"])
def test_in_process_tolerates_missing_informational_topic(
    tmp_path: Path, probes: list[tuple[str, dict[str, object]]], bus: str
) -> None:
    decision = _resolve(
        tmp_path / "missing.yaml", requested=EnumDelegateLocus.IN_PROCESS, bus=bus
    )
    assert isinstance(decision, ModelDelegateLocusDecision)
    assert decision.locus is EnumDelegateLocus.IN_PROCESS
    assert "OVERRIDES" in decision.resolved_from
    assert decision.command_topic == ""
    assert decision.broker == ""
    assert decision.lane_consumer_groups == ()
    assert decision.downstream_consumer_groups == ()
    assert probes == []


def test_in_process_shared_command_refuses_double_dispatch(
    contract: Path, probes: list[tuple[str, dict[str, object]]]
) -> None:
    with pytest.raises(
        delegate_locus.DelegateLocusRefusedError, match="double dispatch"
    ) as caught:
        _resolve(contract, requested=EnumDelegateLocus.IN_PROCESS)
    assert _COMMAND in str(caught.value)
    assert "second copy overwrites the first" in str(caught.value)
    assert probes == []


@pytest.mark.parametrize(
    ("installed", "locations"),
    [(False, None), (True, []), (True, ["/synthetic/omnimarket"])],
)
def test_distribution_labels_version_and_location(
    monkeypatch: pytest.MonkeyPatch, installed: bool, locations: list[str] | None
) -> None:
    def version(name: str) -> str:
        assert name == "omnimarket"
        if not installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return "1.2.3"

    spec = importlib.machinery.ModuleSpec("omnimarket", loader=None)
    spec.submodule_search_locations = locations
    monkeypatch.setattr(delegate_locus.importlib.metadata, "version", version)
    monkeypatch.setattr(
        delegate_locus.importlib.util,
        "find_spec",
        lambda _: spec if installed else None,
    )
    expected_version = "1.2.3" if installed else "(not installed)"
    expected_location = locations[0] if locations else "(unresolved)"
    assert (
        delegate_locus.orchestrator_distribution()
        == f"omnimarket {expected_version} ({expected_location})"
    )
