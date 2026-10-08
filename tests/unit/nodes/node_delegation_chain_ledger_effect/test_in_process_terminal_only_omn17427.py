# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Producer evidence labels terminal-only chains without hiding bus faults."""

from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import yaml

from omnibase_core.container import ModelONEXContainer
from omnibase_core.models.dispatch import ModelHandlerOutput
from omnibase_infra.enums import EnumResponseStatus
from omnibase_infra.handlers.models.model_db_query_payload import ModelDbQueryPayload
from omnibase_infra.handlers.models.model_db_query_response import ModelDbQueryResponse
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.chain_replay import (
    assemble_replay_and_verify,
    derive_chain_state,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.handlers import (
    handler_delegation_chain_ledger as module,
)
from omnibase_infra.nodes.node_delegation_chain_ledger_effect.models import (
    EnumLedgerChainState,
    EnumTierTwoVerdict,
    ModelDelegationTerminalPayload,
    ModelObservedHop,
)

pytestmark = pytest.mark.unit


def _chain():
    return module._load_contract_settings()[0]


def _observed():
    chain = _chain()
    ids = {hop.topic: uuid4() for hop in chain}
    correlation = uuid4()
    return correlation, tuple(
        ModelObservedHop(
            topic=hop.topic,
            envelope_id=ids[hop.topic],
            parent_envelope_id=ids.get(hop.parent),
            correlation_id=correlation,
            source="local",
        )
        for hop in chain
    )


@pytest.mark.parametrize("failed", [False, True])
def test_in_process_terminal_only(failed):
    correlation, observed = _observed()
    terminal = observed[-1].model_copy(
        update={
            "source": "node_event_emit_effect",
            "parent_envelope_id": None,
            "topic": _chain()[-1].alternatives[0] if failed else observed[-1].topic,
        }
    )
    rows = assemble_replay_and_verify(
        correlation, (terminal,), _chain(), ("node_event_emit_effect",)
    )
    assert len(rows) == 1
    assert rows[0].chain_state is EnumLedgerChainState.IN_PROCESS_TERMINAL_ONLY
    assert not rows[0].replay_green
    assert "in-process terminal-only" in rows[0].replay_detail
    assert "not a pass and not a fault" in rows[0].replay_detail
    assert "causal edge cannot be re-derived" not in rows[0].replay_detail


@pytest.mark.parametrize(
    "case", ["missing", "local_rootless", "producer_parented", "upstream"]
)
def test_other_partial_shapes_stay_incomplete(case):
    correlation, observed = _observed()
    terminal = observed[-1]
    if case == "missing":
        partial = observed[:3] + observed[-1:]
    elif case == "local_rootless":
        partial = (terminal.model_copy(update={"parent_envelope_id": None}),)
    elif case == "producer_parented":
        partial = (terminal.model_copy(update={"source": "node_event_emit_effect"}),)
    else:
        partial = observed[:1] + (
            terminal.model_copy(
                update={"source": "node_event_emit_effect", "parent_envelope_id": None}
            ),
        )
    rows = assemble_replay_and_verify(
        correlation, partial, _chain(), ("node_event_emit_effect",)
    )
    assert rows and all(
        row.chain_state is EnumLedgerChainState.INCOMPLETE for row in rows
    )


def test_complete_bus_chain():
    correlation, observed = _observed()
    rows = assemble_replay_and_verify(
        correlation, observed, _chain(), ("node_event_emit_effect",)
    )
    assert len(rows) == 5
    assert all(
        row.chain_state is EnumLedgerChainState.COMPLETE
        and row.replay_green
        and row.verifier_verdict is EnumTierTwoVerdict.PASS
        for row in rows
    )


def test_contract_and_sql():
    raw = yaml.safe_load(module._CONTRACT_PATH.read_text())
    assert set(raw["chain_states"]) == {state.value for state in EnumLedgerChainState}
    assert raw["in_process_terminal_only"]["terminal_sources"] == [
        "node_event_emit_effect"
    ]
    assert module._load_contract_settings()[3] == ("node_event_emit_effect",)
    assert (
        "COALESCE(source, onex_headers ->> 'source', '') AS source"
        in module._SQL_READ_OBSERVED
    )
    assert "chain_state = EXCLUDED.chain_state" in module._SQL_UPSERT_ROW
    assert "$11" in module._SQL_UPSERT_ROW


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "parented", "state"),
    [
        ("node_event_emit_effect", False, "in_process_terminal_only"),
        ("local", True, "incomplete"),
    ],
)
async def test_handler_persists_source_label(source, parented, state, monkeypatch):
    correlation = uuid4()
    handler = module.HandlerDelegationChainLedger(
        cast("ModelONEXContainer", object()), settle_attempts=1, settle_delay_seconds=0
    )
    monkeypatch.setattr(handler, "_ensure_db_ready", AsyncMock())

    def response(rows, count):
        return ModelHandlerOutput.for_compute(
            input_envelope_id=uuid4(),
            correlation_id=correlation,
            handler_id="fake-db",
            result=ModelDbQueryResponse(
                status=EnumResponseStatus.SUCCESS,
                payload=ModelDbQueryPayload(rows=rows, row_count=count),
                correlation_id=correlation,
            ),
        )

    execute = AsyncMock(
        side_effect=[
            response(
                [
                    {
                        "topic": _chain()[-1].topic,
                        "envelope_id": str(uuid4()),
                        "parent_envelope_id": str(uuid4()) if parented else None,
                        "correlation_id": str(correlation),
                        "source": source,
                    }
                ],
                1,
            ),
            response([], 1),
        ]
    )
    monkeypatch.setattr(handler._db_handler, "execute", execute)
    await handler.handle(
        ModelDelegationTerminalPayload(correlation_id=correlation, status="completed")
    )
    assert execute.await_args_list[-1].args[0]["payload"]["parameters"][10] == state


@pytest.mark.parametrize(
    ("states", "sources"),
    [
        (["complete"], ["node_event_emit_effect"]),
        (["complete", "incomplete", "in_process_terminal_only"], None),
        (["complete", "incomplete", "in_process_terminal_only"], []),
        (["complete", "incomplete", "in_process_terminal_only"], [""]),
        (["complete", "incomplete", "in_process_terminal_only"], [42]),
    ],
)
def test_invalid_contract_refuses(states, sources, tmp_path, monkeypatch):
    raw = yaml.safe_load(module._CONTRACT_PATH.read_text())
    raw["chain_states"] = states
    raw["in_process_terminal_only"] = {"terminal_sources": sources}
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(module, "_CONTRACT_PATH", path)
    with pytest.raises(RuntimeError):
        module._load_contract_settings()


def test_state_uses_deduped_graded_hops():
    correlation, observed = _observed()
    terminal = observed[-1].model_copy(
        update={"source": "node_event_emit_effect", "parent_envelope_id": None}
    )
    evidence_topic = _chain()[2].reroute_parents[0]
    evidence = terminal.model_copy(
        update={"topic": evidence_topic, "envelope_id": uuid4(), "source": "local"}
    )
    assert (
        derive_chain_state(
            (terminal, terminal, evidence), _chain(), ("node_event_emit_effect",)
        )
        is EnumLedgerChainState.IN_PROCESS_TERMINAL_ONLY
    )
    assert (
        len(
            assemble_replay_and_verify(
                correlation,
                (terminal, terminal, evidence),
                _chain(),
                ("node_event_emit_effect",),
            )
        )
        == 1
    )


@pytest.mark.asyncio
async def test_real_contract_wires_required_handlers():
    """The changed constructor must resolve through the runtime boot seam."""
    from omnibase_infra.runtime.auto_wiring.discovery import (
        discover_contracts_from_paths,
    )
    from omnibase_infra.runtime.auto_wiring.handler_wiring import wire_from_manifest
    from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

    container = ModelONEXContainer()
    manifest = discover_contracts_from_paths([module._CONTRACT_PATH])
    assert not manifest.errors
    assert len(manifest.contracts) == 1
    report = await wire_from_manifest(
        manifest=manifest,
        dispatch_engine=MessageDispatchEngine(),
        event_bus=None,
        environment="local",
        container=container,
        subscribe_immediately=False,
        materialized_explicit_dependencies={
            "HandlerDelegationChainLedger": {"container": container}
        },
    )
    assert report.total_failed == 0
    assert report.total_wired == 1
    assert not report.quarantined_handlers
    assert len(report.results[0].wirings) == 2
    assert not report.results[0].skipped_handlers
