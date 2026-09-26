# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Content-addressed delegation topology snapshot (OMN-19729)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models.model_declared_chain_hop import (
    ModelDeclaredChainHop,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.topology_validation import (
    validate_declared_topology,
)


def _canonical_topology_json(
    *,
    schema_version: int,
    replay_algorithm_version: str,
    topology_contract_ref: str,
    hops: Sequence[ModelDeclaredChainHop],
) -> str:
    return json.dumps(
        {
            "schema_version": schema_version,
            "replay_algorithm_version": replay_algorithm_version,
            "topology_contract_ref": topology_contract_ref,
            "hops": [hop.model_dump(mode="json") for hop in hops],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class ModelPinnedChainTopology(BaseModel):
    """Immutable snapshot used instead of mutable package contract YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    replay_algorithm_version: str = Field(min_length=1)
    topology_contract_ref: str = Field(min_length=1)
    hops: tuple[ModelDeclaredChainHop, ...]
    topology_sha256: str = Field(min_length=64, max_length=64)

    @classmethod
    def create(
        cls,
        *,
        topology_contract_ref: str,
        replay_algorithm_version: str,
        hops: Sequence[ModelDeclaredChainHop],
        schema_version: Literal[1] = 1,
    ) -> ModelPinnedChainTopology:
        """Validate then content-address an immutable topology snapshot."""
        frozen_hops = tuple(hops)
        validate_declared_topology(frozen_hops)
        canonical = _canonical_topology_json(
            schema_version=schema_version,
            replay_algorithm_version=replay_algorithm_version,
            topology_contract_ref=topology_contract_ref,
            hops=frozen_hops,
        )
        return cls(
            schema_version=schema_version,
            replay_algorithm_version=replay_algorithm_version,
            topology_contract_ref=topology_contract_ref,
            hops=frozen_hops,
            topology_sha256=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    @model_validator(mode="after")  # type: ignore[untyped-decorator]
    def _validate_digest_and_topology(self) -> ModelPinnedChainTopology:
        validate_declared_topology(self.hops)
        canonical = _canonical_topology_json(
            schema_version=self.schema_version,
            replay_algorithm_version=self.replay_algorithm_version,
            topology_contract_ref=self.topology_contract_ref,
            hops=self.hops,
        )
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.topology_sha256 != actual:
            raise ValueError("pinned topology digest does not match canonical topology")
        return self


__all__ = ["ModelPinnedChainTopology"]
