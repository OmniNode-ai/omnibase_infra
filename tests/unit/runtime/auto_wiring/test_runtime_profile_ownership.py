# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Runtime-profile ownership tests for auto-wiring contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    filter_manifest_for_runtime_profile,
)
from omnibase_infra.runtime.auto_wiring.models import (
    ModelAutoWiringManifest,
    ModelContractVersion,
    ModelDiscoveredContract,
)
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    extract_runtime_profiles_from_contract,
    runtime_profile_owns_contract,
)


def _contract(
    name: str,
    runtime_profiles: tuple[str, ...] = (),
) -> ModelDiscoveredContract:
    return ModelDiscoveredContract(
        name=name,
        node_type="COMPUTE_GENERIC",
        contract_version=ModelContractVersion(major=1, minor=0, patch=0),
        contract_path=Path(f"/fake/{name}/contract.yaml"),
        entry_point_name=name,
        package_name="test-package",
        runtime_profiles=runtime_profiles,
    )


def test_unscoped_contracts_default_to_main_runtime_profile() -> None:
    manifest = ModelAutoWiringManifest(contracts=(_contract("legacy_node"),))

    main_result = filter_manifest_for_runtime_profile(manifest, "main")
    effects_result = filter_manifest_for_runtime_profile(manifest, "effects")

    assert [contract.name for contract in main_result.manifest.contracts] == [
        "legacy_node"
    ]
    assert main_result.skipped_contracts == ()
    assert effects_result.manifest.contracts == ()
    assert effects_result.skipped_contracts == ("legacy_node",)


def test_contract_runtime_profiles_select_single_owner() -> None:
    manifest = ModelAutoWiringManifest(
        contracts=(
            _contract("effects_only", ("effects",)),
            _contract("main_only", ("main",)),
        )
    )

    effects_result = filter_manifest_for_runtime_profile(manifest, "effects")
    main_result = filter_manifest_for_runtime_profile(manifest, "main")

    assert [contract.name for contract in effects_result.manifest.contracts] == [
        "effects_only"
    ]
    assert effects_result.skipped_contracts == ("main_only",)
    assert [contract.name for contract in main_result.manifest.contracts] == [
        "main_only"
    ]
    assert main_result.skipped_contracts == ("effects_only",)


def test_raw_contract_runtime_profile_ownership_defaults_to_main() -> None:
    raw_contract = {
        "name": "node_delegation_orchestrator",
        "event_bus": {
            "subscribe_topics": [
                "onex.cmd.omnibase-infra.delegation-request.v1",
            ],
        },
    }

    assert extract_runtime_profiles_from_contract(raw_contract) == ()
    assert runtime_profile_owns_contract(raw_contract, "main") is True
    assert runtime_profile_owns_contract(raw_contract, "effects") is False


def test_raw_contract_runtime_profile_ownership_reads_descriptor_metadata() -> None:
    raw_contract = {
        "name": "node_delegation_quality_gate_effect",
        "descriptor": {"runtime_profiles": [" Effects ", "effects"]},
        "event_bus": {
            "subscribe_topics": [
                "onex.cmd.omnibase-infra.delegation-quality-gate-request.v1",
            ],
        },
    }

    assert extract_runtime_profiles_from_contract(raw_contract) == ("effects",)
    assert runtime_profile_owns_contract(raw_contract, "effects") is True
    assert runtime_profile_owns_contract(raw_contract, "main") is False


def test_runtime_profiles_are_normalized_and_deduplicated() -> None:
    contract = _contract("normalized", (" Effects ", "effects", "MAIN"))

    assert contract.runtime_profiles == ("effects", "main")


def test_blank_runtime_profile_entry_is_invalid() -> None:
    with pytest.raises(ValidationError):
        _contract("bad", ("effects", " "))


def test_discovery_reads_descriptor_runtime_profiles(tmp_path: Path) -> None:
    contract_dir = tmp_path / "node_build_loop_orchestrator"
    contract_dir.mkdir()
    contract_path = contract_dir / "contract.yaml"
    contract_path.write_text(
        """
name: build_loop_orchestrator
contract_version: {major: 1, minor: 0, patch: 0}
node_type: orchestrator
descriptor:
  node_archetype: orchestrator
  purity: effectful
  runtime_profiles:
    - effects
handler_routing:
  routing_strategy: operation_match
  handlers: []
event_bus:
  subscribe_topics:
    - onex.cmd.omnimarket.build-loop-orchestrator-start.v1
terminal_event: onex.evt.omnimarket.build-loop-orchestrator-completed.v1
""".strip(),
        encoding="utf-8",
    )

    manifest = discover_contracts_from_paths([contract_path])

    assert manifest.total_errors == 0
    assert manifest.contracts[0].runtime_profiles == ("effects",)
    assert (
        manifest.contracts[0].terminal_event
        == "onex.evt.omnimarket.build-loop-orchestrator-completed.v1"
    )


def test_github_pr_poller_is_owned_by_effects_runtime_profile() -> None:
    contract_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "omnibase_infra"
        / "nodes"
        / "node_github_pr_poller_effect"
        / "contract.yaml"
    )

    manifest = discover_contracts_from_paths([contract_path])

    assert manifest.total_errors == 0
    contract = manifest.contracts[0]
    assert contract.name == "node_github_pr_poller_effect"
    assert contract.runtime_profiles == ("effects",)
    assert contract.event_bus is not None
    assert contract.event_bus.subscribe_topics == (
        "onex.intent.platform.runtime-tick.v1",
    )

    effects_result = filter_manifest_for_runtime_profile(manifest, "effects")
    main_result = filter_manifest_for_runtime_profile(manifest, "main")

    assert [item.name for item in effects_result.manifest.contracts] == [
        "node_github_pr_poller_effect"
    ]
    assert main_result.manifest.contracts == ()
    assert main_result.skipped_contracts == ("node_github_pr_poller_effect",)


def test_llm_inference_effect_is_owned_by_effects_runtime_profile() -> None:
    contract_path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "omnibase_infra"
        / "nodes"
        / "node_llm_inference_effect"
        / "contract.yaml"
    )

    manifest = discover_contracts_from_paths([contract_path])

    assert manifest.total_errors == 0
    contract = manifest.contracts[0]
    assert contract.name == "node_llm_inference_effect"
    assert contract.runtime_profiles == ("effects",)
    assert contract.event_bus is not None
    assert contract.event_bus.subscribe_topics == (
        "onex.cmd.omnibase-infra.llm-inference-request.v1",
    )

    effects_result = filter_manifest_for_runtime_profile(manifest, "effects")
    main_result = filter_manifest_for_runtime_profile(manifest, "main")

    assert [item.name for item in effects_result.manifest.contracts] == [
        "node_llm_inference_effect"
    ]
    assert main_result.manifest.contracts == ()
    assert main_result.skipped_contracts == ("node_llm_inference_effect",)
