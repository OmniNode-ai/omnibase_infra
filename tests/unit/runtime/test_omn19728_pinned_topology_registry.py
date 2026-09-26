# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Graph topology versions resolve only immutable, verified snapshots."""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pytest

from omnibase_infra.enums.generated.enum_omnimarket_topic import EnumOmnimarketTopic
from omnibase_infra.runtime.execution_graph_topology_registry import (
    ExecutionGraphTopologySnapshotCandidate,
    PackagedExecutionGraphTopologyContract,
    PinnedExecutionGraphTopology,
    _resolve_payload,
    build_snapshot_from_chain_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    REPO_ROOT
    / "src/omnibase_infra/nodes/node_delegation_chain_ledger_effect/contract.yaml"
)


@pytest.mark.unit
def test_packaged_snapshot_matches_current_contract_and_resolves() -> None:
    built = build_snapshot_from_chain_contract(
        CONTRACT_PATH, EnumOmnimarketTopic.EVT_DOD_VERIFY_COMPLETED_V1.value
    )
    registry = PackagedExecutionGraphTopologyContract()
    resolved = registry.resolve(built.version)

    assert type(built) is ExecutionGraphTopologySnapshotCandidate
    assert type(resolved) is PinnedExecutionGraphTopology
    assert resolved.version == built.version
    assert resolved.source_contract_sha256 == built.source_contract_sha256
    assert resolved.declared_chain == built.declared_chain
    assert resolved.read_set.topics == built.read_set.topics
    assert resolved.read_set.head_topic == built.declared_chain[0].topic
    assert resolved.read_set.verdict_topic == (
        EnumOmnimarketTopic.EVT_DOD_VERIFY_COMPLETED_V1.value
    )
    assert len(resolved.declared_chain) == 5
    assert len(resolved.read_set.topics) == 9
    with pytest.raises(TypeError, match="verified artifact"):
        PinnedExecutionGraphTopology(
            version=resolved.version,
            source_contract_sha256=resolved.source_contract_sha256,
            declared_chain=resolved.declared_chain,
            read_set=resolved.read_set,
        )


@pytest.mark.unit
def test_registry_is_independent_of_mutated_live_contract(tmp_path: Path) -> None:
    built = build_snapshot_from_chain_contract(
        CONTRACT_PATH, EnumOmnimarketTopic.EVT_DOD_VERIFY_COMPLETED_V1.value
    )
    contract_copy = tmp_path / "contract.yaml"
    contract_copy.write_text("chain_topology: []\n", encoding="utf-8")

    with pytest.raises(ValueError):
        build_snapshot_from_chain_contract(
            contract_copy, EnumOmnimarketTopic.EVT_DOD_VERIFY_COMPLETED_V1.value
        )
    assert PackagedExecutionGraphTopologyContract().resolve(built.version).version == (
        built.version
    )


@pytest.mark.unit
def test_registry_refuses_missing_or_tampered_snapshot(tmp_path: Path) -> None:
    built = build_snapshot_from_chain_contract(
        CONTRACT_PATH, EnumOmnimarketTopic.EVT_DOD_VERIFY_COMPLETED_V1.value
    )
    empty_registry = PackagedExecutionGraphTopologyContract(snapshot_dir=tmp_path)
    with pytest.raises(LookupError):
        empty_registry.resolve(built.version)

    packaged = PackagedExecutionGraphTopologyContract().snapshot_path(built.version)
    altered = json.loads(packaged.read_text(encoding="utf-8"))
    altered["source_contract_sha256"] = "0" * 64
    (tmp_path / packaged.name).write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest"):
        empty_registry.resolve(built.version)


@pytest.mark.unit
def test_registry_refuses_content_addressed_disconnected_cycle() -> None:
    packaged = PackagedExecutionGraphTopologyContract().snapshot_path(
        build_snapshot_from_chain_contract(
            CONTRACT_PATH, EnumOmnimarketTopic.EVT_DOD_VERIFY_COMPLETED_V1.value
        ).version
    )
    payload = json.loads(packaged.read_text(encoding="utf-8"))
    payload["chain_topology"] = [
        {"topic": "head", "parent": None},
        {"topic": "left", "parent": "right"},
        {"topic": "right", "parent": "left"},
    ]
    identity = {
        key: value for key, value in payload.items() if key != "topology_sha256"
    }
    payload["topology_sha256"] = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="parent cycle"):
        _resolve_payload(payload)
