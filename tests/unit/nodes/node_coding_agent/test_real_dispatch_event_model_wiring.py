# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-dispatch-path regression for the coding-agent event_model wiring (OMN-13247).

The e2e proof surfaced a wiring bug the handler-isolation suite could not see:
the four coding-agent node contracts declared their PER-HANDLER model under the
YAML key ``input_model:`` (inside ``handler_routing.handlers[]``), but the
auto-wiring parser (``discovery.py::_parse_handler_routing``) reads ONLY the
per-handler key ``event_model:``. With ``event_model`` parsed as ``None`` the
auto-wired dispatch callback (``handler_wiring._make_dispatch_callback``) hands
the handler the RAW materialized dispatch dict instead of re-hydrating it into a
typed ``ModelEventEnvelope`` — and the dispatch engine ALWAYS materializes the
envelope to a dict before calling the dispatcher
(``message_dispatch_engine._execute_dispatcher`` -> ``envelope_for_handler:
dict``). The orchestrator's ``handle`` then does ``envelope.event_type`` on a
dict and crashes, so the workflow never advances past hop 1.

This test drives the REAL dispatch surface end to end:

  1. the orchestrator contract is parsed by the SAME parser the runtime uses
     (``_parse_handler_routing``), so a contract carrying the wrong per-handler
     key yields ``event_model=None`` and this test fails;
  2. the auto-wired dispatch callback + payload-type matcher are built exactly
     as ``handler_wiring`` builds them in production
     (``_make_dispatch_callback`` / ``_make_payload_type_matcher``);
  3. the callback is registered on a real ``MessageDispatchEngine`` and a
     ``ModelEventEnvelope`` carrying ``ModelCodingAgentInvokeCommand`` is
     dispatched through ``engine.dispatch`` — the engine materializes the
     envelope to a dict and the callback must re-hydrate it via ``event_model``.

It asserts (a) the orchestrator handler receives a typed ``ModelEventEnvelope``
(NOT a dict) and (b) the workflow advances past hop 1 by emitting the
workspace-validate command. It MUST fail on the pre-fix contract (per-handler
``input_model:`` -> ``event_model`` absent -> raw dict -> crash) and pass after
the rename to ``event_model:``. No real claude/codex subprocess is ever run.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast
from uuid import uuid4

import pytest
import yaml

from omnibase_core.enums import EnumMessageCategory
from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums.enum_dispatch_status import EnumDispatchStatus
from omnibase_infra.models.coding_agent import (
    EnumAgentSandbox,
    EnumCodingAgent,
    ModelCodingAgentInvokeCommand,
)
from omnibase_infra.nodes.node_coding_agent_orchestrator.handlers.handler_coding_agent_orchestrator import (
    HandlerCodingAgentOrchestrator,
)
from omnibase_infra.runtime.auto_wiring.discovery import _parse_handler_routing
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProtocolHandleable,
    _make_dispatch_callback,
    _make_payload_type_matcher,
)
from omnibase_infra.runtime.auto_wiring.models.model_handler_routing_entry import (
    ModelHandlerRoutingEntry,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine

_NODES_ROOT = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra" / "nodes"
_ORCH_CONTRACT = _NODES_ROOT / "node_coding_agent_orchestrator" / "contract.yaml"

# The orchestrator's contract-declared workflow entrypoint + hop-1 output topic.
_INVOKE_TOPIC = "onex.cmd.omnibase-infra.coding-agent-invoke.v1"
_WORKSPACE_VALIDATE_SUFFIX = "coding-agent-workspace-validate.v1"


def _orchestrator_entry() -> ModelHandlerRoutingEntry:
    """Parse the live orchestrator contract via the production parser.

    Returns the single ``handler_routing.handlers[]`` entry. ``entry.event_model``
    is the value the runtime feeds into ``_make_dispatch_callback``; it is
    ``None`` whenever the contract carries the wrong per-handler key.
    """
    raw = yaml.safe_load(_ORCH_CONTRACT.read_text(encoding="utf-8"))
    routing = _parse_handler_routing(raw["handler_routing"])
    assert len(routing.handlers) == 1, "orchestrator contract has one handler entry"
    return routing.handlers[0]


def _invoke_command(workspace_path: str) -> ModelCodingAgentInvokeCommand:
    return ModelCodingAgentInvokeCommand(
        correlation_id=uuid4(),
        agent=EnumCodingAgent.CLAUDE,
        prompt="add a docstring to foo.py",
        workspace_path=workspace_path,
        sandbox=EnumAgentSandbox.READ_ONLY,
        timeout_ms=60000,
    )


@pytest.mark.unit
class TestContractParsesEventModel:
    """The per-handler model must be visible to the auto-wiring parser."""

    def test_parser_reads_per_handler_event_model(self) -> None:
        """The orchestrator contract entry exposes a non-None ``event_model``.

        This is the load-bearing assertion: the parser reads ONLY
        ``handler_routing.handlers[].event_model``, so a contract declaring the
        model under ``input_model:`` parses to ``event_model=None`` and fails
        here. (Pre-fix the four coding-agent contracts all used ``input_model:``.)
        """
        entry = _orchestrator_entry()
        assert entry.event_model is not None, (
            "orchestrator handler entry must declare a per-handler 'event_model:' "
            "so the dispatch callback re-hydrates the typed envelope; found None "
            "(contract likely still uses the per-handler 'input_model:' key the "
            "parser ignores)"
        )
        assert entry.event_model.name == "ModelCodingAgentInvokeCommand"
        assert entry.event_model.module == "omnibase_infra.models.coding_agent"


@pytest.mark.unit
class TestRealDispatchRehydratesTypedEnvelope:
    """Drive the actual dispatch engine + auto-wired callback, not handler.handle()."""

    @pytest.mark.asyncio
    async def test_invoke_command_dispatch_advances_to_workspace_validate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dispatched invoke command re-hydrates to a typed envelope + emits hop 1.

        The dispatch engine ALWAYS materializes the envelope to a dict before
        calling the dispatcher, so the callback built from the contract's
        ``event_model`` is the only thing that turns that dict back into a typed
        ``ModelEventEnvelope`` the orchestrator can read. Without it the handler
        receives a dict and crashes on ``envelope.event_type``; with it the
        workflow advances past hop 1 by emitting the workspace-validate command.
        """
        # The orchestrator's allowed roots are contract-declared via ${env.…} and
        # it fails closed if empty, so the lane env var must resolve to a real root.
        monkeypatch.setenv("ONEX_CODING_AGENT_WORKSPACE_ROOT", str(tmp_path))

        entry = _orchestrator_entry()
        seen_envelopes: list[object] = []

        class _RecordingOrchestrator(HandlerCodingAgentOrchestrator):
            """Records the runtime-supplied envelope to prove re-hydration."""

            async def handle(
                self, envelope: ModelEventEnvelope[object]
            ) -> ModelHandlerOutput[None]:
                seen_envelopes.append(envelope)
                return await super().handle(envelope)

        # cast: the orchestrator's handle() returns ModelHandlerOutput, which the
        # dispatch callback normalizes — the same shape the runtime wires in prod.
        callback = _make_dispatch_callback(
            cast("ProtocolHandleable", _RecordingOrchestrator()), entry.event_model
        )
        payload_type_matcher = (
            _make_payload_type_matcher(entry.event_model)
            if entry.event_model is not None
            else None
        )

        engine = MessageDispatchEngine()
        # The engine derives ``message_type`` from the envelope ``event_type``
        # (the topic alias) and falls back to the payload class name when no
        # event_type is set, so the dispatcher must accept both the topic alias
        # and the model name — the same set the contract wiring registers.
        engine.register_dispatcher(
            dispatcher_id="coding-agent-orchestrator",
            dispatcher=callback,
            category=EnumMessageCategory.COMMAND,
            message_types={_INVOKE_TOPIC, "ModelCodingAgentInvokeCommand"},
            payload_type_matcher=payload_type_matcher,
        )
        engine.register_route(
            ModelDispatchRoute(
                route_id="route.coding-agent-orchestrator",
                topic_pattern="*.cmd.omnibase-infra.coding-agent-invoke.*",
                message_category=EnumMessageCategory.COMMAND,
                dispatcher_id="coding-agent-orchestrator",
            )
        )
        engine.freeze()

        command = _invoke_command(workspace_path=str(tmp_path))
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=command,
            correlation_id=command.correlation_id,
            event_type=_INVOKE_TOPIC,
        )

        result = await engine.dispatch(topic=_INVOKE_TOPIC, envelope=envelope)

        # The dispatch must succeed (no AttributeError on a dict 'envelope').
        assert result.status == EnumDispatchStatus.SUCCESS, result.error_message

        # The handler received a TYPED envelope, not the materialized dict — this
        # is exactly what the event_model re-hydration restores.
        assert len(seen_envelopes) == 1
        received = seen_envelopes[0]
        assert isinstance(received, ModelEventEnvelope), (
            "orchestrator handler must receive a typed ModelEventEnvelope; got "
            f"{type(received).__name__} (a 'dict' means event_model wiring is broken)"
        )
        # ...carrying the re-hydrated typed payload and the workflow event_type.
        assert isinstance(received.payload, ModelCodingAgentInvokeCommand)
        assert received.event_type == _INVOKE_TOPIC

        # Hop 1: the workflow advanced and emitted the workspace-validate command.
        emitted_types = [
            getattr(event, "event_type", None) for event in result.output_events
        ]
        assert any(
            isinstance(t, str) and t.endswith(_WORKSPACE_VALIDATE_SUFFIX)
            for t in emitted_types
        ), (
            "orchestrator must advance past hop 1 by emitting the workspace-validate "
            f"command; output event types were {emitted_types}"
        )
