# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Integration proof that runtime contract discovery no longer finds the GitHub
source-control nodes in omnibase_infra (OMN-20915).

The poller's runtime-profile test lived here while the node did. The nodes
moved to omnimarket, so the replacement runs the real auto-wiring discovery and
profile filter over every contract this package ships and requires that none of
the moved or deleted nodes is discovered, on any runtime profile.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring import (
    discover_contracts_from_paths,
    filter_manifest_for_runtime_profile,
)

pytestmark = pytest.mark.integration

_NODES = Path(__file__).resolve().parents[2] / "src" / "omnibase_infra" / "nodes"

REMOVED = frozenset(
    {
        "node_github_pr_poller_effect",
        "node_github_webhook_ingress_effect",
        "node_pr_state_projection_compute",
        "node_pr_state_write_effect",
        "node_merge_sweep_pr_list_effect",
        "node_merge_sweep_classify_compute",
        "node_merge_sweep_workflow_orchestrator",
        "node_merge_sweep_workflow_reducer",
        "node_merge_sweep_auto_merge_effect",
    }
)


def test_discovery_finds_no_moved_or_deleted_github_node() -> None:
    contract_paths = sorted(_NODES.glob("*/contract.yaml"))
    assert contract_paths, "the discovery positive control found no contracts at all"

    manifest = discover_contracts_from_paths(contract_paths)
    discovered = {contract.name for contract in manifest.contracts}

    assert discovered, "the manifest discovered nothing from a non-empty path list"
    assert discovered.isdisjoint(REMOVED)

    for profile in ("effects", "main"):
        filtered = filter_manifest_for_runtime_profile(manifest, profile)
        names = {contract.name for contract in filtered.manifest.contracts}
        assert names.isdisjoint(REMOVED)
