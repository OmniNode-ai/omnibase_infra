# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Cross-profile intent-routing regression for cold boot (OMN-14517).

``node_pr_state_projection_compute`` (no ``runtime_profiles`` declared, so it
defaults to ``main`` ownership per ``filter_manifest_for_runtime_profile``)
declares an ``intent_consumption.intent_routing_table`` naming
``node_pr_state_write_effect`` as its effect target. That effect node
correctly declares ``runtime_profiles: [effects]`` (it is a real,
independently-consuming EFFECT node in the ``effects`` runtime lane).

The kernel's OMN-14516 derivation resolved the effect node from the
runtime-profile-FILTERED manifest (the same manifest used to decide which
contracts THIS process owns for Kafka subscription). Since the write-effect
belongs to a different profile, it was silently excluded from that filtered
set, and the resolver raised ``RuntimeHostError`` /
``ONEX_CORE_081_OPERATION_FAILED`` claiming the node was "absent from the
manifest" -- crash-looping a genuinely cold ``--profile runtime`` bring-up of
the main container (a warm ``--restart`` of a targeted service subset never
exercises this code path, which is why it went undetected on already-running
lanes; see ops-deploy's comment on OMN-14517).

The fix resolves effect-node targets from the FULL (unfiltered) discovery
manifest, since deriving an in-process result applier only needs the effect
contract's static ``handler_routing`` metadata to import + construct its
handler -- it is not claiming ownership of that contract's Kafka
subscription, so runtime-profile ownership is the wrong filter to apply.

These tests drive the REAL seam -- the actual contract files through the
actual discovery + profile-filtering path -- rather than a hand-built
fixture, mirroring ``test_ledger_projection_dispatch_reachability.py``
(OMN-14516).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import _import_handler_class
from omnibase_infra.runtime.auto_wiring.profile_ownership import (
    filter_manifest_for_runtime_profile,
)
from omnibase_infra.runtime.service_intent_routing_loader import (
    load_intent_routing_table,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NODES_ROOT = _REPO_ROOT / "src" / "omnibase_infra" / "nodes"

COMPUTE_CONTRACT = _NODES_ROOT / "node_pr_state_projection_compute" / "contract.yaml"
EFFECT_NODE_NAME = "node_pr_state_write_effect"
MAIN_RUNTIME_PROFILE = "main"


def _resolve_effect_node(contracts_by_name: Mapping[str, object]) -> tuple[type, ...]:
    """Mirror the kernel's routing-table -> handler-class resolution."""
    routing = load_intent_routing_table(COMPUTE_CONTRACT)
    assert routing, "node_pr_state_projection_compute must declare a routing table"
    resolved: list[type] = []
    for intent_type, effect_node in routing.items():
        effect_contract = contracts_by_name.get(effect_node)
        if effect_contract is None:
            raise LookupError(
                f"intent_routing_table names effect node {effect_node!r} "
                "absent from the given manifest"
            )
        handler_routing = getattr(effect_contract, "handler_routing", None)
        if handler_routing is None:
            raise LookupError(
                f"effect node {effect_node!r} declares no handler_routing"
            )
        handler_ref = next(
            (
                entry.handler
                for entry in handler_routing.handlers
                if entry.operation == intent_type
            ),
            None,
        )
        assert handler_ref is not None, (
            f"effect node {effect_node!r} has no handler for operation {intent_type!r}"
        )
        resolved.append(_import_handler_class(handler_ref.module, handler_ref.name))
    return tuple(resolved)


def test_pr_state_write_effect_declares_a_different_profile_than_the_compute_node() -> (
    None
):
    """Ground truth: the compute/effect pair span two different runtime profiles.

    node_pr_state_projection_compute declares no runtime_profiles (defaults to
    "main"); node_pr_state_write_effect declares runtime_profiles: [effects].
    This is exactly the shape that broke the profile-filtered lookup.
    """
    manifest = discover_contracts()
    contracts_by_name = {c.name: c for c in manifest.contracts}
    compute_contract = contracts_by_name["node_pr_state_projection_compute"]
    effect_contract = contracts_by_name[EFFECT_NODE_NAME]

    assert not compute_contract.runtime_profiles
    assert effect_contract.runtime_profiles == ("effects",)


def test_main_profile_filtered_manifest_excludes_the_effect_node() -> None:
    """RED-on-pre-fix proof: the main-filtered manifest does NOT contain the
    effect node the compute node's routing table names.

    This is the root cause: a lookup keyed off the profile-filtered manifest
    (what the kernel used before the OMN-14517 fix) cannot find
    node_pr_state_write_effect from the main container, so it looks "absent
    from the manifest" even though the contract is perfectly valid.
    """
    manifest = discover_contracts()
    ownership = filter_manifest_for_runtime_profile(
        manifest=manifest, runtime_profile=MAIN_RUNTIME_PROFILE
    )
    main_filtered_names = {c.name for c in ownership.manifest.contracts}

    assert "node_pr_state_projection_compute" in main_filtered_names
    assert EFFECT_NODE_NAME not in main_filtered_names, (
        f"{EFFECT_NODE_NAME} unexpectedly present in the main-profile-filtered "
        "manifest -- if this now passes, the contract's runtime_profiles "
        "declaration changed and this test's premise must be revisited."
    )

    with pytest.raises(LookupError):
        _resolve_effect_node({c.name: c for c in ownership.manifest.contracts})


def test_pr_state_routing_resolves_from_the_full_discovery_manifest() -> None:
    """GREEN-on-fix proof: resolving from the FULL (unfiltered) manifest works.

    This is the fix the kernel now applies: `_contracts_by_name` for
    effect-node resolution must be built from the full discovery manifest
    (`auto_wiring_manifest_discovered` in service_kernel.py), not the
    runtime-profile-filtered one. Confirms the cross-profile route resolves
    to a real, importable HandlerPrStateUpsert.
    """
    manifest = discover_contracts()
    contracts_by_name = {c.name: c for c in manifest.contracts}

    resolved = _resolve_effect_node(contracts_by_name)

    assert len(resolved) == 1
    assert resolved[0].__name__ == "HandlerPrStateUpsert"
