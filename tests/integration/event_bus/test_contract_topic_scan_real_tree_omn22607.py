# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""A kafka runtime start reads the real contract tree fewer times (OMN-22607).

The unit tests count a stubbed extractor. This one builds the five objects a
kafka start builds (the kernel provisioner, the startup validator, the auto-create
provisioner, the host-process provisioner and the auto-wiring provisioner), with
the roots the kernel, the host process and auto-wiring pass, against the shipped
``nodes`` tree and the real extractor. No broker is contacted: the scan runs in
each constructor, before any admin client exists.

Two things are checked. The real extractor runs once for each distinct request
(two, not five), and what the stored scan hands every reader is what a fresh
extractor finds in the same tree, so sharing a scan never changes the topic set a
provisioner or the validator sees.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import omnibase_infra
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

pytestmark = [pytest.mark.integration]

_BOOTSTRAP = "broker.example.invalid:9092"


@pytest.fixture
def real_extract_calls(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Count calls of the real ``extract_all``; each call still does the real scan."""
    calls: list[Path] = []
    real = ContractTopicExtractor.extract_all

    def counting(
        self: ContractTopicExtractor,
        *,
        contracts_root: Path,
        skill_manifests_root: Path | None = None,
        skill_manifests_roots: list[Path] | None = None,
    ) -> list[ModelContractTopicEntry]:
        calls.append(contracts_root)
        return real(
            self,
            contracts_root=contracts_root,
            skill_manifests_root=skill_manifests_root,
            skill_manifests_roots=skill_manifests_roots,
        )

    monkeypatch.setattr(ContractTopicExtractor, "extract_all", counting)
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", _BOOTSTRAP)
    monkeypatch.delenv("ONEX_RUNTIME_NODE_SELECTION", raising=False)
    ModelContractTopicScan.clear()
    return calls


def test_a_kafka_start_scans_the_real_tree_once_per_distinct_request(
    real_extract_calls: list[Path],
) -> None:
    source = Path(omnibase_infra.__file__).resolve().parent
    nodes = source / "nodes"
    kernel_roots = [
        path for path in (source / "cli", source / "services") if path.is_dir()
    ]
    assert nodes.is_dir()
    assert kernel_roots, "the kernel's extra manifest roots must exist in the package"

    kernel = TopicProvisioner(
        bootstrap_servers=_BOOTSTRAP,
        contracts_root=nodes,
        skill_manifests_roots=list(kernel_roots),
    )
    validator = TopicStartupValidator(
        bootstrap_servers=_BOOTSTRAP,
        contracts_root=nodes,
        skill_manifests_roots=list(kernel_roots),
    )
    required = validator._required_topics()
    auto_create = TopicProvisioner(
        bootstrap_servers=_BOOTSTRAP,
        contracts_root=nodes,
        skill_manifests_roots=list(kernel_roots),
    )
    host_process = TopicProvisioner(bootstrap_servers=_BOOTSTRAP, contracts_root=nodes)
    auto_wiring = TopicProvisioner(bootstrap_servers=_BOOTSTRAP, contracts_root=nodes)

    assert len(real_extract_calls) == 2, (
        "five constructions with two distinct requests must run the real extractor "
        f"twice, ran it {len(real_extract_calls)} times"
    )

    kernel_topics = {spec.suffix for spec in kernel._topic_specs}
    assert kernel_topics, "the real tree must yield topics"
    assert {spec.suffix for spec in auto_create._topic_specs} == kernel_topics
    assert {spec.suffix for spec in host_process._topic_specs} == {
        spec.suffix for spec in auto_wiring._topic_specs
    }

    fresh = ContractTopicExtractor(include_installed_packages=True).extract_all(
        contracts_root=nodes,
        skill_manifests_roots=list(kernel_roots),
    )
    assert required == tuple(sorted({entry.topic for entry in fresh})), (
        "the stored scan must hand the validator exactly the topics a fresh scan finds"
    )
