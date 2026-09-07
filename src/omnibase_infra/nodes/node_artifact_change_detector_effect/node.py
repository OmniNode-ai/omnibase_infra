# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Node Artifact Change Detector Effect — declarative EFFECT node for change detection.

This node follows the ONEX declarative pattern:
    - DECLARATIVE effect driven by contract.yaml
    - Two BUS handlers, one per subscribe topic (OMN-18013):
        1. HandlerPRWebhookIngestion — onex.evt.github.pr-webhook.v1
        2. HandlerManualTrigger — onex.cmd.artifact.reconcile.v1
    - One non-bus service: ContractFileWatcher (watchdog filesystem
      watcher, services/contract_file_watcher.py). It has no dispatch
      entrypoint and therefore no handler_routing entry.
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

    Two bus change-detection surfaces (defined in contract.yaml handler_routing):
        - ``artifact.ingest_pr_webhook``: Ingest GitHub PR webhook events
        - ``artifact.manual_trigger``: CLI manual reconcile command ingestion

    Filesystem-based contract change detection is NOT a routing surface: it is
    ``ContractFileWatcher`` under ``services/``.

    All routing and execution logic is driven by contract.yaml.
    NO custom routing code.
    """

    # Pure declarative shell — all behavior defined in contract.yaml


__all__ = ["NodeArtifactChangeDetectorEffect"]
