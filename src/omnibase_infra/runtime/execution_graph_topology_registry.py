# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Content-addressed chain topology snapshots for deterministic graph replay.

The live chain-writer YAML is read only when creating/checking a packaged
snapshot. A replay resolves its recorded version from a retained artifact,
never from whichever YAML happens to be deployed today.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from omnibase_core.models.execution_graph_replay.model_execution_graph_topology_version import (
    ModelExecutionGraphTopologyVersion,
)
from omnibase_core.models.primitives.model_semver import ModelSemVer
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.handlers.handler_delegation_chain_ledger import (
    _parse_declared_topology,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_declared_chain_hop import (
    ModelDeclaredChainHop,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.topology_validation import (
    validate_declared_topology,
)
from omnibase_infra.runtime.db.execution_graph_read_adapters import (
    _PINNED_READ_SET_MINT,
    PinnedExecutionGraphReadSet,
)

_SNAPSHOT_DIR = Path(__file__).resolve().parent / "execution_graph_topologies"
_PINNED_TOPOLOGY_MINT = object()


@dataclass(frozen=True, slots=True)
class ExecutionGraphTopologySnapshotCandidate:
    """Build-time preview; not accepted as a packaged replay authority."""

    version: ModelExecutionGraphTopologyVersion
    source_contract_sha256: str
    declared_chain: tuple[ModelDeclaredChainHop, ...]
    read_set: PinnedExecutionGraphReadSet


@dataclass(frozen=True, slots=True, init=False)
class PinnedExecutionGraphTopology:
    """Sealed verified artifact: one authority for grade, read scope, and stamp."""

    version: ModelExecutionGraphTopologyVersion
    source_contract_sha256: str
    declared_chain: tuple[ModelDeclaredChainHop, ...]
    read_set: PinnedExecutionGraphReadSet

    def __init__(
        self,
        *,
        version: ModelExecutionGraphTopologyVersion,
        source_contract_sha256: str,
        declared_chain: tuple[ModelDeclaredChainHop, ...],
        read_set: PinnedExecutionGraphReadSet,
        _mint: object | None = None,
    ) -> None:
        if _mint is not _PINNED_TOPOLOGY_MINT:
            raise TypeError("Pinned topology requires a verified artifact")
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "source_contract_sha256", source_contract_sha256)
        object.__setattr__(self, "declared_chain", declared_chain)
        object.__setattr__(self, "read_set", read_set)


def _digest_snapshot(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _contract_version(raw: object) -> ModelSemVer:
    if not isinstance(raw, Mapping):
        raise ValueError("chain-writer contract version is missing")
    try:
        components = (raw["major"], raw["minor"], raw["patch"])
        if any(type(component) is not int for component in components):
            raise ValueError("chain-writer version components must be integers")
        return ModelSemVer(
            major=components[0],
            minor=components[1],
            patch=components[2],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid chain-writer contract version") from exc


def _resolve_payload(payload: object) -> PinnedExecutionGraphTopology:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("invalid pinned topology artifact schema")
    chain_raw = payload.get("chain_topology")
    verdict_topic = payload.get("verdict_topic")
    source_sha = payload.get("source_contract_sha256")
    recorded_digest = payload.get("topology_sha256")
    if (
        not isinstance(verdict_topic, str)
        or not verdict_topic
        or not isinstance(source_sha, str)
        or len(source_sha) != 64
        or not isinstance(recorded_digest, str)
        or len(recorded_digest) != 64
    ):
        raise ValueError("invalid pinned topology artifact identity")
    identity_payload = {
        key: value for key, value in payload.items() if key != "topology_sha256"
    }
    if _digest_snapshot(identity_payload) != recorded_digest:
        raise ValueError("pinned topology artifact digest mismatch")
    try:
        declared = _parse_declared_topology(chain_raw)
        validate_declared_topology(declared)
        contract_version = _contract_version(payload.get("contract_version"))
    except RuntimeError as exc:
        raise ValueError("invalid pinned chain topology") from exc
    heads = [hop.topic for hop in declared if hop.parent is None]
    topics = frozenset(
        (
            *(topic for hop in declared for topic in hop.topics),
            *(topic for hop in declared for topic in hop.reroute_parents),
            verdict_topic,
        )
    )
    version = ModelExecutionGraphTopologyVersion(
        contract_version=contract_version, topology_sha256=recorded_digest
    )
    read_set = PinnedExecutionGraphReadSet(
        topology_version=f"{contract_version}@sha256:{recorded_digest}",
        topics=topics,
        head_topic=heads[0],
        verdict_topic=verdict_topic,
        _mint=_PINNED_READ_SET_MINT,
    )
    return PinnedExecutionGraphTopology(
        version=version,
        source_contract_sha256=source_sha,
        declared_chain=declared,
        read_set=read_set,
        _mint=_PINNED_TOPOLOGY_MINT,
    )


def build_snapshot_from_chain_contract(
    contract_path: Path, verdict_topic: str
) -> ExecutionGraphTopologySnapshotCandidate:
    """Build/check a version at release time, never during request handling."""
    contract_bytes = contract_path.read_bytes()
    raw: object = yaml.safe_load(contract_bytes)  # yaml-safe-load-ok: package contract
    if not isinstance(raw, dict):
        raise ValueError("chain-writer contract root must be a mapping")
    chain_raw = raw.get("chain_topology")
    try:
        _parse_declared_topology(chain_raw)
    except RuntimeError as exc:
        raise ValueError("invalid chain-writer topology") from exc
    payload: dict[str, object] = {
        "schema_version": 1,
        "contract_version": raw.get("contract_version"),
        "source_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "chain_topology": chain_raw,
        "verdict_topic": verdict_topic,
    }
    payload["topology_sha256"] = _digest_snapshot(payload)
    built = _resolve_payload(payload)
    return ExecutionGraphTopologySnapshotCandidate(
        version=built.version,
        source_contract_sha256=built.source_contract_sha256,
        declared_chain=built.declared_chain,
        read_set=built.read_set,
    )


class PackagedExecutionGraphTopologyContract:
    """Read retained hash-named artifacts; unknown versions fail closed."""

    def __init__(self, snapshot_dir: Path = _SNAPSHOT_DIR) -> None:
        self._snapshot_dir = snapshot_dir

    def snapshot_path(self, version: ModelExecutionGraphTopologyVersion) -> Path:
        return self._snapshot_dir / f"{version.topology_sha256}.json"

    def resolve(
        self, version: ModelExecutionGraphTopologyVersion
    ) -> PinnedExecutionGraphTopology:
        path = self.snapshot_path(version)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise LookupError("pinned execution graph topology is unavailable") from exc
        resolved = _resolve_payload(payload)
        if resolved.version != version:
            raise ValueError("pinned topology artifact version mismatch")
        return resolved


__all__ = [
    "ExecutionGraphTopologySnapshotCandidate",
    "PackagedExecutionGraphTopologyContract",
    "PinnedExecutionGraphTopology",
    "build_snapshot_from_chain_contract",
]
