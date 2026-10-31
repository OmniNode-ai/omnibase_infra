# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""OMN-21536 — chain-orchestrator and registry-effect handlers take one typed request.

Drives the real contracts: every event of ``node_chain_orchestrator`` dispatches
through the engine built from the discovered contract, and every handler class the
``node_registry_effect`` contract names resolves a single request model from its
``handle`` signature, which is how the auto-wiring runtime dispatches it.
"""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest

from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.runtime.auto_wiring.discovery import discover_contracts
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    _resolve_def_b_input_model_type,
)
from tests.fixtures.dispatch_parity import harness

pytestmark = pytest.mark.integration

_CHAIN_EVENTS: dict[str, tuple[str, str, str, str]] = {
    "omnibase-infra.chain-learn": (
        "onex.cmd.omnibase-infra.chain-learn.v1",
        "omnibase_infra.nodes.node_chain_orchestrator.models.model_chain_learn_command",
        "ModelChainLearnCommand",
        "omnibase-infra.chain-learn",
    ),
    "omnibase-infra.chain-retrieval-result": (
        "onex.evt.omnibase-infra.chain-retrieval-result.v1",
        "omnibase_infra.nodes.node_chain_orchestrator.models.model_chain_retrieval_result",
        "ModelChainRetrievalResult",
        "omnibase-infra.chain-retrieval-result",
    ),
    "omnibase-infra.chain-stored": (
        "onex.evt.omnibase-infra.chain-stored.v1",
        "omnibase_infra.nodes.node_chain_orchestrator.models.model_chain_store_result",
        "ModelChainStoreResult",
        "omnibase-infra.chain-stored",
    ),
}


@pytest.fixture(scope="module")
def contracts_by_name() -> dict[str, Any]:
    manifest = discover_contracts()
    return {c.name: c for c in manifest.contracts}


@pytest.mark.parametrize("event_type", sorted(_CHAIN_EVENTS))
def test_chain_orchestrator_event_dispatches_to_its_single_request_handler(
    contracts_by_name: dict[str, Any], event_type: str
) -> None:
    topic, module, model_name, envelope_event_type = _CHAIN_EVENTS[event_type]
    engine, _meta, _routes, _topics = harness._build_engine(
        [contracts_by_name["node_chain_orchestrator"]]
    )
    instance, err = harness._model_construct_instance(module, model_name)
    assert err is None and instance is not None, f"could not construct payload: {err}"

    async def _run() -> ModelDispatchResult:
        envelope = harness._envelope(event_type=envelope_event_type, payload=instance)
        return await engine.dispatch(topic=topic, envelope=envelope)

    result = asyncio.run(_run())

    assert result.status == EnumDispatchStatus.SUCCESS, (
        f"{event_type} did not dispatch: {result}"
    )


def test_registry_effect_handlers_resolve_one_request_model_from_the_contract(
    contracts_by_name: dict[str, Any],
) -> None:
    contract = contracts_by_name["node_registry_effect"]
    handlers = contract.handler_routing.handlers
    resolved = {}
    for entry in handlers:
        module = importlib.import_module(entry.handler.module)
        handler_cls = getattr(module, entry.handler.name)
        resolved[entry.handler.name] = _resolve_def_b_input_model_type(
            handler_cls.handle
        )

    assert {name for name, model in resolved.items() if model is None} == {
        "HandlerPartialRetry"
    }, f"unexpected request-model resolution: {resolved}"
