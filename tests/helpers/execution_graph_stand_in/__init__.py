# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Stand-in nodes for the ``onex node`` execution-graph hosting tests (OMN-21071).

Two orchestrators whose ``execution_graph`` names stage nodes, and three stage
nodes, each a package carrying its own ``contract.yaml``. They are advertised
in ``onex.nodes`` through the CLI's one registry seam,
``omnibase_infra.cli.contract_registry._entry_points``, so ``onex node <name>``
resolves the orchestrator and the runtime resolves its stages exactly as it
resolves an installed node.
"""

from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from omnibase_infra.cli import contract_registry

__all__ = ["STAND_IN_NODES", "install_execution_graph_stand_ins"]

#: Every stand-in node name, each a package directly under this one.
STAND_IN_NODES = (
    "node_stand_in_sweep_orchestrator",
    "node_stand_in_stalled_sweep_orchestrator",
    "node_stand_in_stage_alpha",
    "node_stand_in_stage_beta",
    "node_stand_in_stage_silent",
)


def install_execution_graph_stand_ins(
    monkeypatch: pytest.MonkeyPatch, *, withhold: tuple[str, ...] = ()
) -> None:
    """Advertise the stand-in nodes in ``onex.nodes`` for this test.

    A name in ``withhold`` is left unadvertised, as a node that is not
    installed is.
    """
    real = contract_registry._entry_points
    stand_ins = tuple(
        EntryPoint(
            name=name, value=f"{__name__}.{name}", group=contract_registry.NODE_GROUP
        )
        for name in STAND_IN_NODES
        if name not in withhold
    )

    def _with_stand_ins(group: str) -> tuple[EntryPoint, ...]:
        if group != contract_registry.NODE_GROUP:
            return real(group)
        return tuple(e for e in real(group) if e.name not in STAND_IN_NODES) + stand_ins

    monkeypatch.setattr(contract_registry, "_entry_points", _with_stand_ins)
