# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Replacement integration test for delegation consolidation (OMN-10865).

Replaces tests/integration/test_delegation_event_chain.py which tested the
infra-side delegation orchestrator chain. Delegation now lives in omnimarket
(PR OmniNode-ai/omnimarket#607), so this asserts that the consolidated
import surface still resolves and the runtime can wire it without the
duplicate-local-ingress-route ValueError that previously crashed boot.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration]


def test_omnimarket_plugin_delegation_import_resolves() -> None:
    """PluginDelegation now lives in omnimarket — its import path must be stable.

    service_kernel.py imports this in its bootstrap registration block.
    A broken import would degrade silently (caught by BLE001) and the
    delegation pipeline would never register, so we assert at import-time here.
    """
    from omnimarket.nodes.node_delegation_orchestrator.plugin import (
        PluginDelegation,
    )

    assert callable(PluginDelegation)


def test_omnimarket_delegation_intent_bridge_import_resolves() -> None:
    """DelegationIntentBridge and LlmCallerDelegation are imported from
    omnimarket by plugin_llm.py when the delegation bridge is registered
    in the DI container."""
    from omnimarket.adapters.llm.adapter_llm_caller_delegation import (
        LlmCallerDelegation,
    )
    from omnimarket.nodes.node_delegation_orchestrator.delegation_intent_bridge import (
        DelegationIntentBridge,
        ProtocolLlmCaller,
    )

    assert callable(LlmCallerDelegation)
    assert callable(DelegationIntentBridge)
    assert ProtocolLlmCaller is not None


def test_infra_protocol_delegation_intent_bridge_sources_from_omnimarket() -> None:
    """The DI protocol key stays in infra (so infra owns the binding) but it
    must source ProtocolLlmCaller from omnimarket where the implementation
    lives. This guards against a regression where the protocol re-imports
    from the deleted infra delegation module."""
    import inspect

    from omnibase_infra.nodes.node_delegation_orchestrator import (
        protocol_delegation_intent_bridge,
    )

    source = inspect.getsource(protocol_delegation_intent_bridge)
    assert "from omnimarket.nodes.node_delegation_orchestrator" in source, (
        "ProtocolDelegationIntentBridge must source ProtocolLlmCaller from omnimarket, "
        "not from a stale omnibase_infra.nodes.node_delegation_orchestrator import."
    )
    assert (
        "from omnibase_infra.nodes.node_delegation_orchestrator.delegation_intent_bridge"
        not in source
    ), (
        "Stale infra import found in protocol_delegation_intent_bridge.py — "
        "the delegation_intent_bridge module was deleted from infra."
    )


def test_no_infra_delegation_contract_yaml_present() -> None:
    """Boot regression: a delegation contract.yaml in omnibase_infra would
    collide with omnimarket's and crash runtime boot with
    `ValueError: Duplicate local ingress route alias 'node_delegation_orchestrator'`.
    """
    from pathlib import Path

    infra_contract = (
        Path(__file__).parent.parent.parent
        / "src"
        / "omnibase_infra"
        / "nodes"
        / "node_delegation_orchestrator"
        / "contract.yaml"
    )
    assert not infra_contract.exists(), (
        f"Delegation contract must not exist in infra (lives in omnimarket): "
        f"{infra_contract}"
    )
