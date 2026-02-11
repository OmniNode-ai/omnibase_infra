# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Declarative COMPUTE node for RRH validation.

Pure validation — no I/O.  Receives collected environment data, a
profile, and contract governance fields, then evaluates 13 rules
(RRH-1001 through RRH-1701).

All behavior is defined in contract.yaml and delegated to
``HandlerRRHValidate``.  This node contains no custom logic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from omnibase_core.nodes.node_compute import NodeCompute

if TYPE_CHECKING:
    from omnibase_core.models.container import ModelONEXContainer


class NodeRRHValidateCompute(NodeCompute):
    """Declarative COMPUTE node for RRH validation.

    Rule catalog: RRH-1001 through RRH-1701 covering repo, environment,
    kafka, kubernetes, toolchain, cross-checks, and repo-boundary.

    Profile precedence: PROFILE sets baseline -> CONTRACT can only TIGHTEN.

    All behavior is defined in contract.yaml — no custom logic here.
    """

    def __init__(self, container: ModelONEXContainer) -> None:
        super().__init__(container)
