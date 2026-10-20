# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-17427: a deployment loads, subscribes and provisions only the nodes it names.

``ONEX_RUNTIME_NODE_SELECTION`` is a deployment-overlay list of contract names.
Unset, nothing changes: every discovered contract the profile owns is attached
and every contract topic is provisioned, which is the lab default. Set, the
auto-wiring ownership filter attaches only the named contracts and the topic
provisioner builds specs only for the topics those contracts declare. A named
contract that no discovered contract is becomes a discovery error, never a
silent zero.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from omnibase_infra.models.runtime.model_runtime_node_selection import (
    ENV_RUNTIME_NODE_SELECTION,
    ModelRuntimeNodeSelection,
)
from omnibase_infra.runtime.auto_wiring import filter_manifest_for_runtime_profile
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    runtime_profile_owns_contract,
)

pytestmark = pytest.mark.unit

SELECTED = "node_selected_omn17427"
UNSELECTED = "node_unselected_omn17427"
OTHER_PROFILE = "node_other_profile_omn17427"
SELECTED_TOPIC = "onex.evt.platform.selected-omn17427.v1"
UNSELECTED_TOPIC = "onex.evt.platform.unselected-omn17427.v1"


def _contract(name: str, profiles: tuple[str, ...] = ()) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="EFFECT_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        runtime_profiles=profiles,
    )


MANIFEST = ModelAutoWiringManifest(
    contracts=(
        _contract(SELECTED),
        _contract(UNSELECTED),
        _contract(OTHER_PROFILE, ("effects",)),
    )
)


def _names(contracts: tuple[ModelDiscoveredContract, ...]) -> set[str]:
    return {contract.name for contract in contracts}


def test_unset_selection_attaches_every_owned_contract() -> None:
    result = filter_manifest_for_runtime_profile(MANIFEST, "main", environ={})

    assert _names(result.manifest.contracts) == {SELECTED, UNSELECTED}
    assert result.node_selection_excluded_contracts == ()
    assert result.manifest.errors == ()


def test_selection_attaches_only_named_contracts() -> None:
    environ = {ENV_RUNTIME_NODE_SELECTION: f" {SELECTED} ,{OTHER_PROFILE}"}

    result = filter_manifest_for_runtime_profile(MANIFEST, "main", environ=environ)

    assert _names(result.manifest.contracts) == {SELECTED}
    assert result.node_selection_excluded_contracts == (UNSELECTED,)
    # Profile ownership still applies to a selected contract.
    assert OTHER_PROFILE in result.skipped_contracts
    assert result.manifest.errors == ()


def test_selection_naming_an_undiscovered_contract_is_a_discovery_error() -> None:
    environ = {ENV_RUNTIME_NODE_SELECTION: f"{SELECTED},node_missing_omn17427"}

    result = filter_manifest_for_runtime_profile(MANIFEST, "main", environ=environ)

    assert _names(result.manifest.contracts) == {SELECTED}
    errors = [error.error for error in result.manifest.errors]
    assert len(errors) == 1
    assert "node_missing_omn17427" in errors[0]
    assert ENV_RUNTIME_NODE_SELECTION in errors[0]


def test_raw_ownership_honours_selection() -> None:
    selected = {"name": SELECTED}
    unselected = {"name": UNSELECTED}
    environ = {ENV_RUNTIME_NODE_SELECTION: SELECTED}

    assert runtime_profile_owns_contract(selected, "main", environ=environ)
    assert not runtime_profile_owns_contract(unselected, "main", environ=environ)
    assert runtime_profile_owns_contract(unselected, "main", environ={})


def test_selection_model_parses_and_admits() -> None:
    unset = ModelRuntimeNodeSelection.from_env({})
    assert not unset.is_active
    assert unset.admits("anything")

    selection = ModelRuntimeNodeSelection.from_env(
        {ENV_RUNTIME_NODE_SELECTION: f"{SELECTED}, ,{SELECTED},{UNSELECTED}"}
    )
    assert selection.is_active
    assert selection.contracts == (SELECTED, UNSELECTED)
    assert selection.admits(SELECTED)
    assert not selection.admits(OTHER_PROFILE)


def _write_contracts(root: Path) -> Path:
    for name, topic in ((SELECTED, SELECTED_TOPIC), (UNSELECTED, UNSELECTED_TOPIC)):
        node_dir = root / name
        node_dir.mkdir(parents=True)
        (node_dir / "contract.yaml").write_text(
            textwrap.dedent(
                f"""
                name: {name}
                published_events:
                  - topic: "{topic}"
                    event_type: "Event"
                """
            )
        )
    return root


def test_provisioner_unset_selection_keeps_every_contract_topic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.delenv(ENV_RUNTIME_NODE_SELECTION, raising=False)
    from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

    provisioner = TopicProvisioner(contracts_root=_write_contracts(tmp_path))
    suffixes = {spec.suffix for spec in provisioner._topic_specs}

    assert {SELECTED_TOPIC, UNSELECTED_TOPIC} <= suffixes


def test_provisioner_selection_keeps_only_selected_contract_topics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    monkeypatch.setenv(ENV_RUNTIME_NODE_SELECTION, SELECTED)
    from omnibase_infra.event_bus.service_topic_manager import TopicProvisioner

    provisioner = TopicProvisioner(contracts_root=_write_contracts(tmp_path))
    suffixes = {spec.suffix for spec in provisioner._topic_specs}

    assert suffixes == {SELECTED_TOPIC}
