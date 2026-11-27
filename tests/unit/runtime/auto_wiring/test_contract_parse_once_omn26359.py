# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A runtime start parses each contract file once (OMN-26359).

One start read and parsed every contract file eight to twelve times: discovery,
handler preparation, the subscribe interleave and the per-topic subscribe reads all
re-read the same file on the event loop before ``/health`` could be answered. The
tests count real parses (every YAML constructor run) and the repeated reads, with
nothing mocked on the read path.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from omnibase_infra.contracts.contract_yaml import read_contract_yaml
from omnibase_infra.models.event_bus.model_contract_topic_scan import (
    ModelContractTopicScan,
)
from omnibase_infra.runtime.auto_wiring import handler_wiring
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts_from_paths
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    subscribe_wired_contract_topics,
    wire_from_manifest,
)
from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.util_contract_content_hash import contract_content_hash

from .test_not_ready_reconciliation_omn15215 import FlakyProvisioner, RecordingBus

_HERE = "tests.unit.runtime.auto_wiring.test_contract_parse_once_omn26359"
_EXECUTOR_MODULE = (
    "omnibase_infra.nodes.node_topic_migration_executor_effect.handlers."
    "handler_topic_migration_executor"
)
_CONTRACT_COUNT = 10


class HandlerNoPublisher:
    """A def-B handler whose constructor takes nothing."""

    async def handle(self, envelope: object) -> None:
        return None


class HandlerWithPublisher:
    """A def-B handler whose constructor takes an ``event_publisher``."""

    def __init__(self, event_publisher: object) -> None:
        self._event_publisher = event_publisher

    async def handle(self, envelope: object) -> None:
        return None


class _Bus(RecordingBus):
    """A recording bus that also satisfies ``ProtocolEventBusLike``."""

    async def publish(self, *args: object, **kwargs: object) -> None:
        return None

    async def publish_envelope(self, *args: object, **kwargs: object) -> None:
        return None

    async def get_consumer_groups(self, *args: object, **kwargs: object) -> list[str]:
        return []


@pytest.fixture(autouse=True)
def _fresh_read_cache() -> Iterator[None]:
    from omnibase_infra.contracts.contract_read_cache import ContractReadCache

    ContractReadCache.clear()
    yield
    ContractReadCache.clear()


class _ParseCounter:
    """Counts every YAML constructor run, keyed by the contract ``name`` it read."""

    def __init__(self) -> None:
        self.by_name: Counter[str] = Counter()
        self.total = 0

    @staticmethod
    def install(counter: _ParseCounter) -> Callable[[], None]:
        real = yaml.constructor.BaseConstructor.get_single_data

        def counted(self: yaml.constructor.BaseConstructor) -> object:
            data = real(self)
            counter.total += 1
            if isinstance(data, Mapping) and "name" in data:
                counter.by_name[str(data["name"])] += 1
            return data

        patcher = patch.object(
            yaml.constructor.BaseConstructor, "get_single_data", counted
        )
        patcher.start()
        return patcher.stop


@pytest.fixture
def parse_counter() -> Iterator[_ParseCounter]:
    counter = _ParseCounter()
    stop = _ParseCounter.install(counter)
    try:
        yield counter
    finally:
        stop()


def _write_contract(
    root: Path,
    name: str,
    *,
    handler: str,
    topics: int = 2,
    publish: bool = False,
    entries: int = 1,
    module: str = _HERE,
) -> Path:
    directory = root / name
    directory.mkdir()
    subscribe = [f"onex.evt.omn26359.{name}-{i}.v1" for i in range(topics)]
    contract = {
        "name": name,
        "node_type": "COMPUTE_GENERIC",
        "contract_version": {"major": 1, "minor": 0, "patch": 0},
        "event_bus": {
            "subscribe_topics": subscribe,
            "publish_topics": [f"onex.evt.omn26359.{name}-out.v1"] if publish else [],
        },
        "handler_routing": {
            "routing_strategy": "topic_match",
            "handlers": [
                {
                    "topic": subscribe[i % len(subscribe)],
                    "operation": f"fixture.op{i}",
                    "message_category": "event",
                    "event_model": {
                        "name": "ModelEventMessage",
                        "module": "omnibase_infra.event_bus.models.model_event_message",
                    },
                    "handler": {"name": handler, "module": module},
                    "supported_operations": [f"fixture.op{i}"],
                }
                for i in range(entries)
            ],
        },
    }
    path = directory / "contract.yaml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    return path


async def _wire_and_subscribe(
    paths: list[Path],
    handler_cls: type,
    *,
    event_bus: object | None = None,
    between: Callable[[], None] | None = None,
) -> RecordingBus:
    manifest = ModelAutoWiringManifest(
        contracts=tuple(discover_contracts_from_paths(paths).contracts)
    )
    assert len(manifest.contracts) == len(paths)
    bus = _Bus()
    engine = MessageDispatchEngine()
    with patch.object(
        handler_wiring, "_import_handler_class", return_value=handler_cls
    ):
        report = await wire_from_manifest(
            manifest,
            engine,
            event_bus=event_bus,
            environment="test",
            subscribe_immediately=False,
        )
        assert report.total_failed == 0
        if between is not None:
            between()
        await subscribe_wired_contract_topics(
            manifest=manifest,
            report=report,
            dispatch_engine=engine,
            event_bus=bus,
            environment="test",
            provisioner=FlakyProvisioner(fail_first_n_calls=0),
        )
    return bus


def test_contract_cache_isolation_and_invalidation(
    tmp_path: Path, parse_counter: _ParseCounter
) -> None:
    """AC1: one parse per file change, a private copy per read, a changed file re-read."""
    path = tmp_path / "contract.yaml"
    path.write_text("name: node_cache_probe\nitems: [a, b]\n", encoding="utf-8")

    first = read_contract_yaml(path)
    assert isinstance(first, dict)
    first["items"].append("mutated")
    first["injected"] = True
    second = read_contract_yaml(path)
    third = read_contract_yaml(path)

    assert second == {"name": "node_cache_probe", "items": ["a", "b"]}, (
        "a caller's mutation reached a later read"
    )
    assert third == second and third is not second
    digest = contract_content_hash(path)
    assert contract_content_hash(path) == digest
    assert parse_counter.by_name["node_cache_probe"] == 1, (
        f"three reads of an unchanged file parsed it "
        f"{parse_counter.by_name['node_cache_probe']} times"
    )

    stat = path.stat()
    path.write_text("name: node_cache_probe\nitems: [a, b, c]\n", encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert read_contract_yaml(path) == {
        "name": "node_cache_probe",
        "items": ["a", "b", "c"],
    }, "a changed file was served from the memo"
    assert contract_content_hash(path) != digest
    assert parse_counter.by_name["node_cache_probe"] == 2


def test_contract_cache_does_not_store_failures(tmp_path: Path) -> None:
    """A malformed or absent file raises on every read and nothing is stored."""
    path = tmp_path / "contract.yaml"
    with pytest.raises(FileNotFoundError):
        read_contract_yaml(path)
    path.write_text("name: [unclosed\n", encoding="utf-8")
    for _ in range(2):
        with pytest.raises(yaml.YAMLError):
            read_contract_yaml(path)
    path.write_text("name: fixed\n", encoding="utf-8")
    assert read_contract_yaml(path) == {"name": "fixed"}


@pytest.mark.asyncio
async def test_contract_parse_once_across_discover_wire_and_subscribe(
    tmp_path: Path, parse_counter: _ParseCounter
) -> None:
    """AC2: on 10 contracts each path is parsed at most once for the whole start."""
    paths = [
        _write_contract(tmp_path, f"node_parse_once_{i}", handler="HandlerNoPublisher")
        for i in range(_CONTRACT_COUNT)
    ]
    bus = await _wire_and_subscribe(paths, HandlerNoPublisher)
    assert len(bus.attached_topics) == 2 * _CONTRACT_COUNT, "subscribe did not attach"

    counts = {name: parse_counter.by_name[name] for name in _names(paths)}
    assert all(count >= 1 for count in counts.values()), counts
    assert max(counts.values()) <= 1, (
        f"a contract was parsed more than once in one start: {counts}"
    )


def test_contract_parse_once_across_topic_scans(
    tmp_path: Path, parse_counter: _ParseCounter
) -> None:
    """Two topic scans of one contracts root parse each contract file once."""
    from omnibase_infra.tools.contract_topic_extractor import ContractTopicExtractor

    paths = [
        _write_contract(tmp_path, f"node_scan_{i}", handler="HandlerNoPublisher")
        for i in range(3)
    ]
    extractor = ContractTopicExtractor(include_installed_packages=False)
    first = extractor.extract_all(contracts_root=tmp_path)
    second = extractor.extract_all(contracts_root=tmp_path)

    assert first == second and first
    counts = {name: parse_counter.by_name[name] for name in _names(paths)}
    assert counts == dict.fromkeys(_names(paths), 1), counts


def _names(paths: list[Path]) -> list[str]:
    return [p.parent.name for p in paths]


@pytest.mark.asyncio
async def test_state_io_read_once_for_a_five_topic_contract(tmp_path: Path) -> None:
    """AC3: state_io and the failure terminals are read once, not once per topic."""
    path = _write_contract(
        tmp_path,
        "node_five_topics",
        handler="HandlerNoPublisher",
        topics=5,
        publish=True,
    )
    state_io_calls: list[Path] = []
    terminal_calls: list[Path] = []
    real_state_io = handler_wiring._read_state_io
    real_terminals = handler_wiring.declared_failure_terminal_topics

    def counting_state_io(contract_path: Path) -> dict[str, object]:
        state_io_calls.append(contract_path)
        return real_state_io(contract_path)

    def counting_terminals(
        contract_path: Path, *, success_topic: str, publishable_topics: Sequence[str]
    ) -> tuple[str, ...]:
        terminal_calls.append(contract_path)
        return real_terminals(
            contract_path,
            success_topic=success_topic,
            publishable_topics=publishable_topics,
        )

    def start_counting() -> None:
        state_io_calls.clear()
        terminal_calls.clear()

    with (
        patch.object(handler_wiring, "_read_state_io", counting_state_io),
        patch.object(
            handler_wiring, "declared_failure_terminal_topics", counting_terminals
        ),
    ):
        bus = await _wire_and_subscribe(
            [path], HandlerNoPublisher, between=start_counting
        )

    assert len(bus.attached_topics) == 5
    assert len(state_io_calls) == 1, (
        f"state_io was read {len(state_io_calls)} times for 5 subscribe topics"
    )
    assert len(terminal_calls) == 1, (
        f"failure terminal topics were read {len(terminal_calls)} times "
        "for 5 subscribe topics"
    )


@pytest.mark.asyncio
async def test_terminal_topics_computed_once_and_only_for_an_event_publisher(
    tmp_path: Path,
) -> None:
    """AC4: zero reads without an event_publisher, one per contract with one."""
    calls: list[Path | None] = []
    real = handler_wiring.load_terminal_event_topics

    def counting(contract_path: Path | None) -> frozenset[str]:
        calls.append(contract_path)
        return real(contract_path)

    plain = _write_contract(
        tmp_path, "node_no_publisher", handler="HandlerNoPublisher", entries=3
    )
    with patch.object(handler_wiring, "load_terminal_event_topics", counting):
        await _wire_and_subscribe([plain], HandlerNoPublisher, event_bus=_Bus())
    assert calls == [], f"a handler without event_publisher caused {len(calls)} reads"

    publishing = _write_contract(
        tmp_path, "node_with_publisher", handler="HandlerWithPublisher", entries=3
    )
    with patch.object(handler_wiring, "load_terminal_event_topics", counting):
        await _wire_and_subscribe([publishing], HandlerWithPublisher, event_bus=_Bus())
    assert len(calls) == 1, (
        f"3 handler entries of one contract read its terminal topics {len(calls)} times"
    )


@pytest.mark.asyncio
async def test_topic_provisioner_built_once_and_reused_by_the_migration_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC5: the executor is handed the kernel's provisioner; no second one is built."""
    from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
    from omnibase_infra.runtime.service_kernel import (
        _build_runtime_handler_dependencies,
    )

    root = tmp_path / "contracts"
    node = root / "node_demo"
    node.mkdir(parents=True)
    (node / "contract.yaml").write_text("name: node_demo\n", encoding="utf-8")
    monkeypatch.setenv("ONEX_CONTRACTS_DIR", str(root))
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:1")
    ModelContractTopicScan.clear()
    scans_before = ModelContractTopicScan.runs()

    kernel_provisioner = MagicMock(name="kernel_topic_provisioner")
    dependencies = _build_runtime_handler_dependencies(
        None, None, topic_provisioner=kernel_provisioner
    )
    assert dependencies is not None
    assert dependencies["HandlerTopicMigrationExecutor"] == {
        "provisioner": kernel_provisioner
    }

    contract_path = _write_contract(
        tmp_path,
        "node_topic_migration_executor_effect",
        handler="HandlerTopicMigrationExecutor",
        module=_EXECUTOR_MODULE,
    )
    manifest = ModelAutoWiringManifest(
        contracts=tuple(discover_contracts_from_paths([contract_path]).contracts)
    )
    built: list[object] = []
    real_init = TopicProvisioner.__init__

    def counting_init(self: TopicProvisioner, **kwargs: Any) -> None:
        built.append(self)
        real_init(self, **kwargs)

    with (
        patch.object(TopicProvisioner, "__init__", counting_init),
        patch("aiokafka.admin.AIOKafkaAdminClient"),
        patch("aiokafka.AIOKafkaConsumer"),
    ):
        report = await wire_from_manifest(
            manifest,
            MessageDispatchEngine(),
            event_bus=None,
            environment="test",
            subscribe_immediately=False,
            materialized_explicit_dependencies=dependencies,
        )
    assert report.total_failed == 0
    assert built == [], "the migration executor built a second TopicProvisioner"
    assert ModelContractTopicScan.runs() == scans_before, "a second contract scan ran"


class _EnsureRecorder:
    """The kernel's provisioner: records each pass of the auto-create."""

    def __init__(self) -> None:
        self.passes = 0

    async def ensure_provisioned_topics_exist(
        self, correlation_id: object = None
    ) -> dict[str, list[str] | str]:
        self.passes += 1
        return {"status": "success", "created": ["t.created"], "existing": []}


class _RecordingValidator:
    """Fails the first validation and passes the second, as after a create."""

    def __init__(self) -> None:
        self.log_missing: list[bool] = []

    async def validate(
        self, correlation_id: object = None, *, log_missing: bool = True
    ) -> Any:
        from omnibase_infra.event_bus.model_topic_validation_result import (
            ModelTopicValidationResult,
        )

        self.log_missing.append(log_missing)
        return ModelTopicValidationResult()


@pytest.mark.asyncio
async def test_topic_provisioner_built_once_the_auto_create_pass_reuses_the_kernels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC5: the auto-create pass runs on the kernel's provisioner and builds none."""
    from omnibase_infra.event_bus.model_topic_validation_result import (
        ModelTopicValidationResult,
    )
    from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
    from omnibase_infra.handlers.boot_topic_provisioning import (
        auto_create_missing_topics,
    )

    built: list[object] = []

    def counting_init(self: TopicProvisioner, **kwargs: Any) -> None:
        built.append(self)

    monkeypatch.setattr(TopicProvisioner, "__init__", counting_init)
    provisioner = _EnsureRecorder()
    validator = _RecordingValidator()
    missing = ModelTopicValidationResult(
        missing_topics=("t.created",), is_valid=False, status="degraded"
    )

    result = await auto_create_missing_topics(
        provisioner, validator, missing, correlation_id=None, strict_validation=True
    )

    assert built == [], "the auto-create pass built a second TopicProvisioner"
    assert provisioner.passes == 1
    assert validator.log_missing == [True]
    assert result.is_valid


@pytest.mark.asyncio
async def test_topic_provisioner_built_once_without_a_kernel_provisioner_nothing_runs() -> (
    None
):
    """AC5: no kernel provisioner means no pass and no second construction."""
    from omnibase_infra.event_bus.model_topic_validation_result import (
        ModelTopicValidationResult,
    )
    from omnibase_infra.handlers.boot_topic_provisioning import (
        auto_create_missing_topics,
    )

    validator = _RecordingValidator()
    missing = ModelTopicValidationResult(
        missing_topics=("t.created",), is_valid=False, status="degraded"
    )
    result = await auto_create_missing_topics(
        None, validator, missing, correlation_id=None, strict_validation=False
    )
    assert result is missing
    assert validator.log_missing == []


def test_topic_provisioner_built_once_the_kernel_source_constructs_none_in_a_call() -> (
    None
):
    """AC5: service_kernel names ``TopicProvisioner`` only to hand it to a thread once."""
    import ast

    import omnibase_infra.runtime.service_kernel as kernel

    tree = ast.parse(Path(kernel.__file__).read_text(encoding="utf-8"))
    direct_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "TopicProvisioner"
    ]
    handed_to_thread = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "to_thread"
        and n.args
        and isinstance(n.args[0], ast.Name)
        and n.args[0].id == "TopicProvisioner"
    ]
    assert direct_calls == [], "the kernel builds a TopicProvisioner in a direct call"
    assert len(handed_to_thread) == 1, "the kernel builds one TopicProvisioner per boot"


_REPO_ROOT = Path(__file__).resolve().parents[4]
_BOOT_RECORD = _REPO_ROOT / "docs" / "readbacks" / "OMN-26359_measured_boot.yaml"
_UTC = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


def _boot_record() -> dict[str, Any]:
    record = yaml.safe_load(_BOOT_RECORD.read_text(encoding="utf-8"))
    assert isinstance(record, dict)
    return record


def test_measured_boot_record_holds_before_and_after() -> None:
    """AC6: the committed record has host, refs, UTC read times, seconds and parse counts."""
    import re

    record = _boot_record()
    assert record["host"] and record["before_ref"] and record["after_ref"]
    assert record["before_ref"] != record["after_ref"]
    passes = record["passes"]
    assert len(passes) >= 1
    for one in passes:
        assert re.match(_UTC, one["before_utc"]), one
        assert re.match(_UTC, one["after_utc"]), one
        assert one["before_seconds_to_health"] > 0
        assert one["after_seconds_to_health"] > 0
        assert 0 < one["after_yaml_parses"] < one["before_yaml_parses"], (
            f"pass {one['pass']}: the parse count did not fall"
        )
        if "before_cpu_seconds_at_health" in one:
            assert (
                0
                < one["after_cpu_seconds_at_health"]
                < one["before_cpu_seconds_at_health"]
            ), f"pass {one['pass']}: the CPU seconds did not fall"


def test_measured_boot_record_names_the_pushed_branch_and_its_base() -> None:
    """AC8: the record names the pushed branch and the base the measurements stack on."""
    record = _boot_record()
    assert record["ticket_id"] == "OMN-26359"
    assert record["branch"] == "omn-26359-parse-once"
    assert record["base"] == "omn-26338-startup-loop-holds-2"
    assert all(
        "before_yaml_parses" in one and "after_yaml_parses" in one
        for one in record["passes"]
    )


def test_file_shape_the_cache_module_has_a_canonical_home_and_no_baseline_entry() -> (
    None
):
    """AC7: the new module is in a canonical location and adds no baseline line."""
    module = Path("src/omnibase_infra/contracts/contract_read_cache.py")
    assert (_REPO_ROOT / module).is_file()
    baseline = (
        _REPO_ROOT / ".onex_ratchets" / "canonical_file_shape_baseline.txt"
    ).read_text(encoding="utf-8")
    assert module.as_posix() not in baseline.splitlines()
    assert "handlers/boot_topic_provisioning.py" not in baseline
