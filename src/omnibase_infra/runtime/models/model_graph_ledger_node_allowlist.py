# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Explicit disposable evidence-only runtime node boundary."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

GRAPH_LEDGER_NODES = (
    "node_ledger_projection_compute",
    "node_ledger_write_effect",
    "node_delegation_chain_ledger_effect",
    "node_execution_graph_read_effect",
)


class ModelGraphLedgerNodeAllowlist(BaseModel):
    """An explicit four-node selection, never an implicit package filter."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    runtime_lane: Literal["sim-202"]
    nodes: tuple[str, ...]

    @model_validator(mode="after")
    def exact_evidence_nodes(self) -> Self:
        """Require the reviewed dependency-complete set without duplicates."""
        if len(self.nodes) != len(GRAPH_LEDGER_NODES) or set(self.nodes) != set(
            GRAPH_LEDGER_NODES
        ):
            raise ValueError(
                "graph ledger allowlist requires exactly the four evidence nodes"
            )
        return self
