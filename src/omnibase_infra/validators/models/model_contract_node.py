# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One contract.yaml reduced to its contract-graph surface (OMN-18013)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelContractNode(BaseModel):
    """One contract.yaml, reduced to its graph-relevant surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    package: str
    path: str
    publish_topics: tuple[str, ...] = ()
    subscribe_topics: tuple[str, ...] = ()
    command_topic: str | None = None
    terminal_topics: tuple[str, ...] = ()
    externally_consumed: tuple[str, ...] = ()
    has_dispatch_wiring: bool = False
    runtime_loaded: bool = False
    # OMN-14591: a REAL, positive, machine-checkable signal that this node is
    # deliberately invoked from outside the Kafka contract graph (cron/on-demand
    # poller, per contracts/integrations/catalog.yaml, or an explicit
    # runtime_dispatch.external_trigger annotation) -- see
    # _find_disconnected_subgraphs. Bare structural shape (subscribe_topics ==
    # () + has publish_topics) is NOT sufficient on its own: verify round 1
    # caught it silently rescuing 2 genuinely dead nodes (node_baseline_capture,
    # node_pattern_lifecycle_effect) that share the shape but have no
    # invocation path at all.
    declared_ingress_root: bool = False


__all__ = ["ModelContractNode"]
