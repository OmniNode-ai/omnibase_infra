# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Fail-closed selection before any node lifecycle or wiring side effect."""

from collections import Counter
from collections.abc import Mapping

from omnibase_infra.runtime.auto_wiring.models import ModelAutoWiringManifest
from omnibase_infra.runtime.models.model_graph_ledger_node_allowlist import (
    ModelGraphLedgerNodeAllowlist,
)


def validate_graph_ledger_boot(
    selection: ModelGraphLedgerNodeAllowlist,
    runtime_profile: str,
    environ: Mapping[str, str],
) -> None:
    """The opt-in selection is only valid on its declared disposable lane."""
    if environ.get("ONEX_RUNTIME_LANE") != selection.runtime_lane:
        raise ValueError(
            "graph ledger allowlist requires its declared sim-202 runtime lane"
        )
    if runtime_profile not in ("main", "effects"):
        raise ValueError(
            "graph ledger allowlist requires main or effects runtime profile"
        )
    if environ.get("ONEX_CORE_RUNTIME_TOPICS", "").strip():
        raise ValueError(
            "graph ledger allowlist forbids the parallel core runtime loop"
        )


def select_graph_ledger_manifest(
    manifest: ModelAutoWiringManifest,
    selection: ModelGraphLedgerNodeAllowlist,
) -> ModelAutoWiringManifest:
    """Require the complete declared set, then remove every other node."""
    if manifest.errors:
        raise ValueError(
            "graph ledger allowlist requires error-free contract discovery"
        )
    counts = Counter(c.name for c in manifest.contracts)
    if any(counts[name] != 1 for name in selection.nodes):
        raise ValueError(
            "graph ledger allowlist requires each declared node exactly once"
        )
    if any(
        c.package_name.replace("-", "_") != "omnibase_infra"
        or c.entry_point_name != c.name
        for c in manifest.contracts
        if c.name in selection.nodes
    ):
        raise ValueError(
            "graph ledger allowlist requires canonical Infra node identities"
        )
    return manifest.model_copy(
        update={
            "contracts": tuple(
                c for c in manifest.contracts if c.name in selection.nodes
            )
        }
    )
