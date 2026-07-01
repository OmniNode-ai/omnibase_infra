# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Real-dispatch multi-topic routing proof for the coding-agent workflow (OMN-13247).

The e2e proof surfaced a routing defect the handler-isolation + golden-chain
suites could not see: those suites assert the orchestrator's *handler output*
``event.event_type`` is correct, but they NEVER run the dispatch result applier
that actually resolves the Kafka publish topic. The runtime publishes every
command the orchestrator emits via ``DispatchResultApplier.apply``, which (before
this fix) resolved the publish topic from ``event.topic`` (absent on a
``ModelEventEnvelope``), then the class-name ``topic_router`` / ``output_topic_map``
(no match — the emitted output_event is always a ``ModelEventEnvelope``), then the
single ``output_topic`` fallback — the contract's terminal_event
(``coding-agent-completed.v1``). So the orchestrator's FIRST emit (the
workspace-validate command, ``event_type`` correctly set to the validate topic)
was published to the TERMINAL topic, and the validate -> invoke -> capture chain
never executed.

These tests drive the REAL dispatch surface AND the REAL applier for every hop,
asserting the topic the bus actually receives — not merely the handler-output
``event_type``. The CANONICAL routing field an ONEX multi-step orchestrator sets
is ``ModelEventEnvelope.event_type`` = its destination topic (identical to
``node_redeploy_orchestrator`` in omnimarket). The applier must honor that field
as the publish topic when it is a contract-declared (allowed) topic.

Only the actual CLI subprocess is mocked (via the invoke effect's injected
``run_subprocess`` seam); everything else flows through the real dispatch engine,
auto-wired callbacks, payload-type matchers, and the real DispatchResultApplier.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import yaml

from omnibase_core.enums import EnumMessageCategory
from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumDispatchStatus
from omnibase_infra.models.coding_agent import (
    EnumAgentSandbox,
    EnumAgentStatus,
    EnumCodingAgent,
    ModelCodingAgentInvokeCommand,
    ModelCodingAgentResult,
    ModelWorkspaceValidateCommand,
    ModelWorkspaceValidateResult,
)
from omnibase_infra.models.dispatch.model_dispatch_result import ModelDispatchResult
from omnibase_infra.nodes.node_coding_agent_invoke_effect.handlers.handler_coding_agent_invoke import (
    HandlerCodingAgentInvoke,
    ModelSubprocessOutcome,
)
from omnibase_infra.nodes.node_coding_agent_orchestrator.handlers.handler_coding_agent_orchestrator import (
    HandlerCodingAgentOrchestrator,
)
from omnibase_infra.nodes.node_coding_agent_workspace_compute.handlers.handler_workspace_validate import (
    HandlerWorkspaceValidate,
)
from omnibase_infra.protocols import ProtocolEventBusLike
from omnibase_infra.runtime.auto_wiring.discovery import _parse_handler_routing
from omnibase_infra.runtime.auto_wiring.handler_wiring import (
    ProtocolHandleable,
    _make_dispatch_callback,
    _make_payload_type_matcher,
    _select_dispatch_result_output_topic,
)
from omnibase_infra.runtime.message_dispatch_engine import MessageDispatchEngine
from omnibase_infra.runtime.service_dispatch_result_applier import (
    DispatchResultApplier,
)

pytestmark = pytest.mark.unit

_NODES_ROOT = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra" / "nodes"

_INVOKE_TOPIC = "onex.cmd.omnibase-infra.coding-agent-invoke.v1"
_VALIDATE_TOPIC = "onex.cmd.omnibase-infra.coding-agent-workspace-validate.v1"
_EFFECT_INVOKE_TOPIC = "onex.cmd.omnibase-infra.coding-agent-effect-invoke.v1"
_FSM_ADVANCE_TOPIC = "onex.evt.omnibase-infra.coding-agent-fsm-advance.v1"
_TERMINAL_TOPIC = "onex.evt.omnibase-infra.coding-agent-completed.v1"
_INVOKE_COMPLETED_TOPIC = "onex.evt.omnibase-infra.coding-agent-invoke-completed.v1"
_CRED_HOME = "/home/omniinfra"


def _contract(node_dir: str) -> dict[str, object]:
    return cast(
        "dict[str, object]",
        yaml.safe_load(
            (_NODES_ROOT / node_dir / "contract.yaml").read_text(encoding="utf-8")
        ),
    )


def _publish_topics(raw: dict[str, object]) -> list[str]:
    event_bus = cast("dict[str, object]", raw.get("event_bus") or {})
    return cast("list[str]", list(event_bus.get("publish_topics") or ()))


def _orchestrator_entry() -> object:
    raw = _contract("node_coding_agent_orchestrator")
    routing = _parse_handler_routing(cast("dict[str, object]", raw["handler_routing"]))
    assert len(routing.handlers) == 1
    return routing.handlers[0]


def _discovered_contract(node_dir: str) -> object:
    """Parse one node's contract via the production discovery parser."""
    from omnibase_infra.runtime.auto_wiring.discovery import (
        discover_contracts_from_paths,
    )

    contract_path = _NODES_ROOT / node_dir / "contract.yaml"
    manifest = discover_contracts_from_paths([contract_path])
    return next(c for c in manifest.contracts if c.contract_path == contract_path)


def _real_applier_for(node_dir: str, mock_bus: object) -> DispatchResultApplier:
    """Build a DispatchResultApplier exactly as ``_subscribe_contract_topics`` does.

    Mirrors production auto-wiring: ``output_topic`` is the contract's
    terminal_event (when publishable) else the first publish topic; the
    published-events map and the allowed-topic allowlist come straight from the
    contract. This is the surface the misroute lives in.
    """
    from omnibase_infra.runtime.event_bus_subcontract_wiring import (
        load_published_events_map,
    )

    discovered = _discovered_contract(node_dir)
    output_topic = _select_dispatch_result_output_topic(discovered)  # type: ignore[arg-type]
    assert output_topic is not None
    assert discovered.event_bus is not None  # type: ignore[attr-defined]
    return DispatchResultApplier(
        event_bus=cast("ProtocolEventBusLike", mock_bus),
        output_topic=output_topic,
        output_topic_map=load_published_events_map(discovered.contract_path),  # type: ignore[attr-defined]
        allowed_output_topics=discovered.event_bus.publish_topics,  # type: ignore[attr-defined]
    )


def _entry_event_model(node_dir: str) -> object:
    """Return the single handler entry's parsed ``event_model`` for ``node_dir``."""
    raw = _contract(node_dir)
    routing = _parse_handler_routing(cast("dict[str, object]", raw["handler_routing"]))
    assert len(routing.handlers) == 1
    return routing.handlers[0].event_model


def _make_engine(
    *,
    handler: object,
    dispatcher_id: str,
    category: EnumMessageCategory,
    message_types: set[str],
    route_id: str,
    topic_pattern: str,
    event_model: object,
    use_payload_matcher: bool = True,
) -> MessageDispatchEngine:
    # The auto-wired callback re-hydrates the materialized dict into a typed
    # ModelEventEnvelope via ``event_model`` (the dispatch engine ALWAYS hands
    # dispatchers a dict). A single-handler orchestrator that subscribes to
    # several topics matches by topic/event_type, so its payload-type matcher is
    # disabled here (a validated-event dict would not match the entrypoint model).
    callback = _make_dispatch_callback(cast("ProtocolHandleable", handler), event_model)
    payload_type_matcher = (
        _make_payload_type_matcher(event_model)
        if use_payload_matcher and event_model is not None
        else None
    )
    engine = MessageDispatchEngine()
    engine.register_dispatcher(
        dispatcher_id=dispatcher_id,
        dispatcher=callback,
        category=category,
        message_types=message_types,
        payload_type_matcher=payload_type_matcher,
    )
    from omnibase_core.models.dispatch.model_dispatch_route import ModelDispatchRoute

    engine.register_route(
        ModelDispatchRoute(
            route_id=route_id,
            topic_pattern=topic_pattern,
            message_category=category,
            dispatcher_id=dispatcher_id,
        )
    )
    engine.freeze()
    return engine


def _published_topics(mock_bus: AsyncMock) -> list[str]:
    return [c.kwargs["topic"] for c in mock_bus.publish_envelope.call_args_list]


class TestOrchestratorHop1RoutesToValidateNotTerminal:
    """Hop (a): the orchestrator's validate command must publish to the VALIDATE
    topic, NOT the terminal topic. This is the headline misroute."""

    @pytest.mark.asyncio
    async def test_invoke_command_publishes_validate_to_validate_topic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ONEX_CODING_AGENT_WORKSPACE_ROOT", str(tmp_path))

        # Sanity: the production applier fallback IS the terminal topic, so a
        # misroute lands on the terminal topic (proving the bug is real).
        assert (
            _select_dispatch_result_output_topic(
                _discovered_contract("node_coding_agent_orchestrator")  # type: ignore[arg-type]
            )
            == _TERMINAL_TOPIC
        )

        entry = _orchestrator_entry()
        engine = _make_engine(
            handler=HandlerCodingAgentOrchestrator(),
            dispatcher_id="coding-agent-orchestrator",
            category=EnumMessageCategory.COMMAND,
            message_types={_INVOKE_TOPIC, "ModelCodingAgentInvokeCommand"},
            route_id="route.coding-agent-orchestrator",
            topic_pattern="*.cmd.omnibase-infra.coding-agent-invoke.*",
            event_model=entry.event_model,  # type: ignore[attr-defined]
        )

        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        applier = _real_applier_for("node_coding_agent_orchestrator", mock_bus)

        command = ModelCodingAgentInvokeCommand(
            correlation_id=uuid4(),
            agent=EnumCodingAgent.CLAUDE,
            prompt="add a docstring to foo.py",
            workspace_path=str(tmp_path),
            sandbox=EnumAgentSandbox.READ_ONLY,
            timeout_ms=60000,
        )
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=command,
            correlation_id=command.correlation_id,
            event_type=_INVOKE_TOPIC,
        )

        result = await engine.dispatch(topic=_INVOKE_TOPIC, envelope=envelope)
        assert result.status == EnumDispatchStatus.SUCCESS, result.error_message

        await applier.apply(result)

        topics = _published_topics(mock_bus)
        assert topics == [_VALIDATE_TOPIC], (
            "the orchestrator's first emit (workspace-validate command) must be "
            f"published to the VALIDATE topic, not {topics}; a value of "
            f"[{_TERMINAL_TOPIC!r}] is the OMN-13247 misroute"
        )
        # The published payload must be the inner validate command, not a
        # double-nested ModelEventEnvelope.
        published_payload = mock_bus.publish_envelope.call_args.kwargs[
            "envelope"
        ].payload
        assert isinstance(published_payload, ModelWorkspaceValidateCommand)


class TestComputeConsumesValidateAndEmitsValidated:
    """Hop (b): COMPUTE consumes the validate command and produces a verdict that
    routes to the workspace-validated topic."""

    @pytest.mark.asyncio
    async def test_compute_verdict_publishes_to_validated_topic(
        self, tmp_path: Path
    ) -> None:
        validated_topic = _publish_topics(
            _contract("node_coding_agent_workspace_compute")
        )[0]
        assert validated_topic.endswith("coding-agent-workspace-validated.v1")

        engine = _make_engine(
            handler=HandlerWorkspaceValidate(),
            dispatcher_id="coding-agent-workspace-compute",
            category=EnumMessageCategory.COMMAND,
            message_types={_VALIDATE_TOPIC, "ModelWorkspaceValidateCommand"},
            route_id="route.coding-agent-workspace-compute",
            topic_pattern="*.cmd.omnibase-infra.coding-agent-workspace-validate.*",
            event_model=_entry_event_model("node_coding_agent_workspace_compute"),
        )
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        applier = _real_applier_for("node_coding_agent_workspace_compute", mock_bus)

        command = ModelWorkspaceValidateCommand(
            correlation_id=uuid4(),
            workspace_path=str(tmp_path),
            allowed_roots=(str(tmp_path),),
            sandbox=EnumAgentSandbox.READ_ONLY,
            prompt="add a docstring to foo.py",
        )
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=command,
            correlation_id=command.correlation_id,
            event_type=_VALIDATE_TOPIC,
        )
        result = await engine.dispatch(topic=_VALIDATE_TOPIC, envelope=envelope)
        assert result.status == EnumDispatchStatus.SUCCESS, result.error_message

        await applier.apply(result)
        topics = _published_topics(mock_bus)
        assert topics == [validated_topic], topics
        verdict = mock_bus.publish_envelope.call_args.kwargs["envelope"].payload
        assert isinstance(verdict, ModelWorkspaceValidateResult)
        assert verdict.valid is True


class TestOrchestratorHop3RoutesToEffectInvoke:
    """Hop (c): on workspace-validated=ok the orchestrator's invoke command must
    publish to the EFFECT invoke topic, not the terminal/validate topic."""

    @pytest.mark.asyncio
    async def test_validated_ok_publishes_invoke_to_effect_topic(
        self, tmp_path: Path
    ) -> None:
        entry = _orchestrator_entry()
        engine = _make_engine(
            handler=HandlerCodingAgentOrchestrator(),
            dispatcher_id="coding-agent-orchestrator",
            category=EnumMessageCategory.EVENT,
            message_types={
                "onex.evt.omnibase-infra.coding-agent-workspace-validated.v1",
            },
            route_id="route.coding-agent-orchestrator-validated",
            topic_pattern="*.evt.omnibase-infra.coding-agent-workspace-validated.*",
            event_model=entry.event_model,  # type: ignore[attr-defined]
            # The validated event payload is a {verdict, command} dict, not the
            # entrypoint ModelCodingAgentInvokeCommand; this single-handler
            # orchestrator matches by topic, so disable the payload matcher.
            use_payload_matcher=False,
        )
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        applier = _real_applier_for("node_coding_agent_orchestrator", mock_bus)

        corr = uuid4()
        command = ModelCodingAgentInvokeCommand(
            correlation_id=corr,
            agent=EnumCodingAgent.CLAUDE,
            prompt="add a docstring to foo.py",
            workspace_path=str(tmp_path),
            sandbox=EnumAgentSandbox.READ_ONLY,
            timeout_ms=60000,
        )
        verdict = ModelWorkspaceValidateResult(
            correlation_id=corr, valid=True, resolved_path=str(tmp_path)
        )
        validated_topic = "onex.evt.omnibase-infra.coding-agent-workspace-validated.v1"
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload={
                "verdict": verdict.model_dump(mode="json"),
                "command": command.model_dump(mode="json"),
            },
            correlation_id=corr,
            event_type=validated_topic,
        )
        result = await engine.dispatch(topic=validated_topic, envelope=envelope)
        assert result.status == EnumDispatchStatus.SUCCESS, result.error_message

        await applier.apply(result)
        topics = _published_topics(mock_bus)
        assert topics == [_EFFECT_INVOKE_TOPIC], (
            "validated-ok must emit the invoke command to the EFFECT invoke topic; "
            f"got {topics}"
        )
        published_payload = mock_bus.publish_envelope.call_args.kwargs[
            "envelope"
        ].payload
        assert isinstance(published_payload, ModelCodingAgentInvokeCommand)


class TestEffectReachedAndTerminalCarriesResult:
    """Hops (d)+(f): the EFFECT path is reached (subprocess mocked) and the
    terminal coding-agent-completed event carries a ModelCodingAgentResult.

    The effect's terminal_event (invoke-completed) is the applier fallback, so
    its result routes to that topic; the FSM-driven terminal completed event is
    asserted via the reducer + applier below.
    """

    @pytest.mark.asyncio
    async def test_effect_result_publishes_to_invoke_completed_with_result_payload(
        self, tmp_path: Path
    ) -> None:
        class _Spy:
            def __init__(self) -> None:
                self.calls = 0

            def __call__(self, _invocation: object) -> ModelSubprocessOutcome:
                self.calls += 1
                return ModelSubprocessOutcome(
                    returncode=0, stdout="done", stderr="", timed_out=False
                )

        spy = _Spy()
        handler = HandlerCodingAgentInvoke(
            run_subprocess=spy,
            probe_head_sha=lambda _cwd: "abc1234",
            capture_diff=lambda _cwd: (("foo.py",), "diff --git a/foo.py b/foo.py"),
            which=lambda _b: "/usr/bin/claude",
            agent_credential_home=_CRED_HOME,
        )
        engine = _make_engine(
            handler=handler,
            dispatcher_id="coding-agent-invoke-effect",
            category=EnumMessageCategory.COMMAND,
            message_types={_EFFECT_INVOKE_TOPIC, "ModelCodingAgentInvokeCommand"},
            route_id="route.coding-agent-invoke-effect",
            topic_pattern="*.cmd.omnibase-infra.coding-agent-effect-invoke.*",
            event_model=_entry_event_model("node_coding_agent_invoke_effect"),
        )
        mock_bus = AsyncMock(spec=ProtocolEventBusLike)
        applier = _real_applier_for("node_coding_agent_invoke_effect", mock_bus)

        command = ModelCodingAgentInvokeCommand(
            correlation_id=uuid4(),
            agent=EnumCodingAgent.CLAUDE,
            prompt="add a docstring to foo.py",
            workspace_path=str(tmp_path),
            sandbox=EnumAgentSandbox.READ_ONLY,
            timeout_ms=60000,
        )
        envelope: ModelEventEnvelope[object] = ModelEventEnvelope(
            payload=command,
            correlation_id=command.correlation_id,
            event_type=_EFFECT_INVOKE_TOPIC,
        )
        result = await engine.dispatch(topic=_EFFECT_INVOKE_TOPIC, envelope=envelope)
        assert result.status == EnumDispatchStatus.SUCCESS, result.error_message
        assert spy.calls == 1, "the EFFECT subprocess seam must be reached"

        await applier.apply(result)
        topics = _published_topics(mock_bus)
        assert topics == [_INVOKE_COMPLETED_TOPIC], topics
        published = mock_bus.publish_envelope.call_args.kwargs["envelope"].payload
        assert isinstance(published, ModelCodingAgentResult)
        assert published.status == EnumAgentStatus.COMPLETED
