# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-22607: a runtime start scans the contract topics once.

``ContractTopicExtractor.extract_all`` re-parses every contract. Each
``TopicProvisioner`` constructor and the ``TopicStartupValidator`` ran it, so one
runtime start scanned the same contracts once per construction. These tests count
the real extractor's calls: provisioners and a validator reading the same tree run
it once, a contract written after the first scan makes the next reader scan again,
and a different request is its own scan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.event_bus.service_topic_startup_validator import (
    TopicStartupValidator,
)
from omnibase_infra.models.event_bus.model_contract_topic_scan import (
    ModelContractTopicScan,
)
from omnibase_infra.tools.contract_topic_extractor import (
    ContractTopicExtractor,
    ModelContractTopicEntry,
)

pytestmark = pytest.mark.unit

_BOOTSTRAP = "broker.example.invalid:9092"
_TOPIC = "onex.evt.platform.scan-once-probe.v1"


@pytest.fixture
def extract_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Count the real extractor's ``extract_all`` calls; one entry per call."""
    calls: list[Path] = []
    entry = ModelContractTopicEntry(
        topic=_TOPIC,
        kind="evt",
        producer="platform",
        event_name="scan-once-probe",
        version="v1",
        source_contracts=(Path("scan_once/contract.yaml"),),
    )

    def counting_extract_all(self: ContractTopicExtractor, **kwargs: object) -> list:
        root = kwargs["contracts_root"]
        assert isinstance(root, Path)
        calls.append(root)
        return [entry]

    monkeypatch.setattr(ContractTopicExtractor, "extract_all", counting_extract_all)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", _BOOTSTRAP)
    monkeypatch.delenv("ONEX_RUNTIME_NODE_SELECTION", raising=False)
    ModelContractTopicScan.clear()
    return calls


def _provisioner(root: Path) -> TopicProvisioner:
    return TopicProvisioner(bootstrap_servers=_BOOTSTRAP, contracts_root=root)


def test_four_provisioners_and_a_validator_scan_once(
    tmp_path: Path, extract_calls: list[Path]
) -> None:
    (tmp_path / "contract.yaml").write_text("name: probe\n")
    specs = [_provisioner(tmp_path)._topic_specs for _ in range(4)]
    validator = TopicStartupValidator(
        bootstrap_servers=_BOOTSTRAP, contracts_root=tmp_path
    )
    assert validator._required_topics() == (_TOPIC,)
    assert len(extract_calls) == 1
    assert all([spec.suffix for spec in each] == [_TOPIC] for each in specs)


def test_a_contract_written_after_the_first_scan_is_scanned_again(
    tmp_path: Path, extract_calls: list[Path]
) -> None:
    (tmp_path / "contract.yaml").write_text("name: probe\n")
    _provisioner(tmp_path)
    _provisioner(tmp_path)
    assert len(extract_calls) == 1
    live = tmp_path / "node_live" / "contract.yaml"
    live.parent.mkdir()
    live.write_text("name: node_live\n")
    _provisioner(tmp_path)
    _provisioner(tmp_path)
    assert len(extract_calls) == 2


def test_an_edited_contract_is_scanned_again(
    tmp_path: Path, extract_calls: list[Path]
) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text("name: probe\n")
    _provisioner(tmp_path)
    contract.write_text("name: probe\nextra: edited so the size differs\n")
    _provisioner(tmp_path)
    assert len(extract_calls) == 2


def test_a_different_request_is_its_own_scan(
    tmp_path: Path, extract_calls: list[Path]
) -> None:
    other = tmp_path / "other"
    other.mkdir()
    (other / "contract.yaml").write_text("name: other\n")
    ModelContractTopicScan.read(contracts_root=tmp_path)
    ModelContractTopicScan.read(contracts_root=other)
    ModelContractTopicScan.read(contracts_root=tmp_path)
    ModelContractTopicScan.read(contracts_root=tmp_path, skill_manifests_roots=[other])
    assert extract_calls == [tmp_path, other, tmp_path]
    assert ModelContractTopicScan.runs() >= 3


def test_a_caller_filtering_its_result_does_not_change_the_next_read(
    tmp_path: Path, extract_calls: list[Path]
) -> None:
    first = ModelContractTopicScan.read(contracts_root=tmp_path)
    first.clear()
    assert len(ModelContractTopicScan.read(contracts_root=tmp_path)) == 1
    assert len(extract_calls) == 1


def test_clearing_forgets_every_stored_scan(
    tmp_path: Path, extract_calls: list[Path]
) -> None:
    ModelContractTopicScan.read(contracts_root=tmp_path)
    ModelContractTopicScan.clear()
    ModelContractTopicScan.read(contracts_root=tmp_path)
    assert len(extract_calls) == 2


def test_bytecode_written_by_an_import_is_not_a_change(
    tmp_path: Path, extract_calls: list[Path]
) -> None:
    (tmp_path / "contract.yaml").write_text("name: probe\n")
    ModelContractTopicScan.read(contracts_root=tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "handler.cpython-312.pyc").write_bytes(b"\0")
    ModelContractTopicScan.read(contracts_root=tmp_path)
    assert len(extract_calls) == 1
