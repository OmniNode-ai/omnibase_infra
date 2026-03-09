# SPDX-License-Identifier: MIT
# Copyright (c) 2025 OmniNode Team
"""Node Artifact Change Detector Effect — declarative EFFECT node for change detection.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Three handlers for different change detection surfaces:
        1. HandlerPRWebhookIngestion — GitHub PR webhook events
        2. HandlerContractFileWatcher — watchdog-based filesystem watcher
        3. HandlerManualTrigger — CLI-initiated manual reconcile commands
    - Publishes ModelUpdateTrigger to onex.evt.artifact.change-detected.v1
    - Lightweight shell — all logic in handlers

Related Tickets:
    - OMN-3940: Task 5 — Change Detector EFFECT Node
    - OMN-3925: Epic — Artifact Reconciliation + Update Planning MVP
"""

from __future__ import annotations

from omnibase_core.nodes.node_effect import NodeEffect


class NodeArtifactChangeDetectorEffect(NodeEffect):
    """Declarative effect node that detects artifact-relevant changes.

    Three change detection surfaces (defined in contract.yaml handler_routing):
        - ``artifact.ingest_pr_webhook``: Ingest GitHub PR webhook events
        - ``artifact.watch_contracts``: Filesystem-based contract change detection
        - ``artifact.manual_trigger``: CLI manual reconcile command ingestion

    All routing and execution logic is driven by contract.yaml.
    NO custom routing code.
    """

    # Pure declarative shell — all behavior defined in contract.yaml


__all__ = ["NodeArtifactChangeDetectorEffect"]
