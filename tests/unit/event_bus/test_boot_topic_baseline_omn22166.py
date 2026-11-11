# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-22166 (one-runtime plan T29): the boot topic provisioning list is held.

``TopicProvisioner`` builds the list of topic specs a runtime creates at boot.
This test builds it through the real boot path, from this repository's contracts
and its cli/services manifests with installed packages excluded so the count does
not depend on the environment, and fails when it grows past the figure recorded in
``boot_topic_baseline_omn22166.yaml``. A planted extra spec proves the check bites.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml

import omnibase_infra
from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner
from omnibase_infra.tools.contract_topic_extractor import (
    ContractTopicExtractor,
    ModelContractTopicEntry,
)
from omnibase_infra.topics.model_topic_spec import ModelTopicSpec

_BASELINE_PATH = Path(__file__).parent / "boot_topic_baseline_omn22166.yaml"
_PLANTED_TOPIC = "onex.evt.platform.planted-boot-topic.v1"


def _max_specs() -> int:
    baseline = yaml.safe_load(_BASELINE_PATH.read_text())
    return int(baseline["repo_slice"]["max_specs"])


def _boot_specs(
    monkeypatch: pytest.MonkeyPatch,
    installed: Sequence[ModelContractTopicEntry] = (),
) -> tuple[ModelTopicSpec, ...]:
    """Build the boot spec list the way the kernel does, minus installed packages."""
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.delenv("ONEX_RUNTIME_NODE_SELECTION", raising=False)
    monkeypatch.setattr(
        ContractTopicExtractor,
        "extract_from_installed_packages",
        lambda self: list(installed),
    )
    src = Path(omnibase_infra.__file__).resolve().parent
    provisioner = TopicProvisioner(
        bootstrap_servers="localhost:9092",
        contracts_root=src / "nodes",
        skill_manifests_roots=[
            root for root in (src / "cli", src / "services") if root.is_dir()
        ],
    )
    return provisioner._topic_specs


def _assert_within_baseline(specs: Sequence[ModelTopicSpec], max_specs: int) -> None:
    assert len(specs) <= max_specs, (
        f"boot provisions {len(specs)} topic specs, the recorded ceiling is "
        f"{max_specs} (boot_topic_baseline_omn22166.yaml)"
    )


@pytest.mark.unit
def test_boot_topic_specs_within_recorded_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = _boot_specs(monkeypatch)
    assert specs, "boot list is empty; the baseline check would pass vacuously"
    _assert_within_baseline(specs, _max_specs())


@pytest.mark.unit
def test_planted_extra_topic_spec_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planted = ModelContractTopicEntry(
        topic=_PLANTED_TOPIC,
        kind="evt",
        producer="platform",
        event_name="planted-boot-topic",
        version="v1",
        source_contracts=(Path("planted/contract.yaml"),),
    )
    specs = _boot_specs(monkeypatch, installed=[planted])
    assert _PLANTED_TOPIC in {spec.suffix for spec in specs}
    with pytest.raises(AssertionError, match="recorded ceiling"):
        _assert_within_baseline(specs, _max_specs())
