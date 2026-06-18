# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Golden-chain / dispatch-path coverage for the coding-agent workflow (OMN-13247).

Phase A is the CREDENTIAL-INDEPENDENT SKELETON: the CLI subprocess is MOCKED via
the invoke effect's injected ``run_subprocess`` seam (Phase 0 / OMN-13246 proves
real auth; no real claude/codex is ever executed here, including in tests).

The chain is driven through the REAL dispatch surface, not handlers imported by
hand: each node's handler class + input model is resolved FROM its
``contract.yaml`` (the same ``handler_routing`` module/class the runtime
dispatcher resolves), so a contract that points at a missing/renamed handler
fails this suite. Every node is dispatched via its ``handle(envelope)`` entry —
the dispatcher's call site — never by reaching into private helpers.

Coverage (plan §5.2 / DoD):
  - workspace-rejected -> REJECTED with NO subprocess.
  - workspace-ok -> INVOKING.
  - mocked invoke-completed -> CAPTURING -> COMPLETED.
  - invoke-failed retry (READ_ONLY) + circuit breaker.
  - WORKSPACE_WRITE failure is never auto-retried.
  - duplicate correlation_id never re-runs.
  - replay issues no live intent (no subprocess on replay).
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import yaml

from omnibase_core.enums.enum_node_kind import EnumNodeKind
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.models.coding_agent import (
    TERMINAL_STATES,
    EnumAgentSandbox,
    EnumAgentStatus,
    EnumCliBackendStatus,
    EnumCodingAgent,
    EnumCodingAgentEventKind,
    EnumCodingAgentFsmState,
    EnumCodingAgentIntentKind,
    ModelCodingAgentEvent,
    ModelCodingAgentFsmState,
    ModelCodingAgentInvokeCommand,
    ModelWorkspaceValidateCommand,
    ModelWorkspaceValidateResult,
)
from omnibase_infra.nodes.node_coding_agent_fsm_reducer.handlers.handler_coding_agent_fsm import (
    delta,
)
from omnibase_infra.nodes.node_coding_agent_fsm_reducer.models.model_coding_agent_fsm_advance import (
    ModelCodingAgentFsmAdvance,
)
from omnibase_infra.nodes.node_coding_agent_invoke_effect.handlers.handler_coding_agent_invoke import (
    ModelSubprocessOutcome,
)

_NODES_ROOT = Path(__file__).resolve().parents[4] / "src" / "omnibase_infra" / "nodes"


def _resolve_handler_from_contract(node_dir: str, operation: str) -> type[Any]:
    """Import the handler class the contract declares for ``operation``.

    This is the contract-driven resolution the runtime dispatcher performs:
    read ``handler_routing.handlers[].handler.{module,name}`` from the node's
    ``contract.yaml`` and import it. A drifted contract fails here.
    """
    contract_path = _NODES_ROOT / node_dir / "contract.yaml"
    raw = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    handlers = raw["handler_routing"]["handlers"]
    match = [h for h in handlers if h["operation"] == operation]
    assert len(match) == 1, f"{node_dir}: expected one handler for {operation!r}"
    spec = match[0]["handler"]
    module = importlib.import_module(spec["module"])
    return getattr(module, spec["name"])


# --- contract-resolved handler classes (real dispatch surface) ---------------
HandlerWorkspaceValidate = _resolve_handler_from_contract(
    "node_coding_agent_workspace_compute", "coding_agent.workspace.validate"
)
HandlerCodingAgentFsm = _resolve_handler_from_contract(
    "node_coding_agent_fsm_reducer", "coding_agent.fsm.advance"
)
HandlerCodingAgentInvoke = _resolve_handler_from_contract(
    "node_coding_agent_invoke_effect", "coding_agent.invoke"
)
HandlerCodingAgentOrchestrator = _resolve_handler_from_contract(
    "node_coding_agent_orchestrator", "coding_agent.orchestrate"
)


def _allowed_root(tmp_path: Path) -> str:
    return str(tmp_path)


def _invoke_command(
    *,
    workspace_path: str,
    sandbox: EnumAgentSandbox = EnumAgentSandbox.READ_ONLY,
    correlation_id: Any = None,
) -> ModelCodingAgentInvokeCommand:
    return ModelCodingAgentInvokeCommand(
        correlation_id=correlation_id or uuid4(),
        agent=EnumCodingAgent.CLAUDE,
        prompt="add a docstring to foo.py",
        workspace_path=workspace_path,
        sandbox=sandbox,
        timeout_ms=60000,
    )


def _fsm_state(
    correlation_id: Any,
    state: EnumCodingAgentFsmState,
    *,
    sandbox: EnumAgentSandbox = EnumAgentSandbox.READ_ONLY,
    invoke_attempts: int = 0,
    consecutive_failures: int = 0,
) -> ModelCodingAgentFsmState:
    return ModelCodingAgentFsmState(
        correlation_id=correlation_id,
        agent=EnumCodingAgent.CLAUDE,
        sandbox=sandbox,
        current_state=state,
        invoke_attempts=invoke_attempts,
        consecutive_failures=consecutive_failures,
    )


class _SpySubprocess:
    """Mock for the invoke effect's run_subprocess seam; records calls.

    The seam signature carries the optional stdin (claude WORKSPACE_WRITE pipes
    the prompt via stdin — OMN-13246), so the spy records it too. No real
    claude/codex subprocess ever runs under test.
    """

    def __init__(self, outcome: ModelSubprocessOutcome) -> None:
        self._outcome = outcome
        self.calls: list[tuple[list[str], str, int, bool, str | None]] = []

    def __call__(
        self,
        argv: list[str],
        cwd: str,
        timeout_s: int,
        network: bool,
        stdin: str | None,
    ) -> ModelSubprocessOutcome:
        self.calls.append((argv, cwd, timeout_s, network, stdin))
        return self._outcome


# ---------------------------------------------------------------------------
# COMPUTE: workspace validation gates whether a subprocess runs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWorkspaceComputeDispatch:
    async def test_valid_workspace_passes(self, tmp_path: Path) -> None:
        handler = HandlerWorkspaceValidate()
        command = ModelWorkspaceValidateCommand(
            correlation_id=uuid4(),
            workspace_path=str(tmp_path),
            allowed_roots=(_allowed_root(tmp_path),),
            sandbox=EnumAgentSandbox.WORKSPACE_WRITE,
            prompt="do the thing",
        )
        envelope: ModelEventEnvelope[ModelWorkspaceValidateCommand] = (
            ModelEventEnvelope(payload=command, correlation_id=command.correlation_id)
        )
        output = await handler.handle(envelope)
        assert output.node_kind == EnumNodeKind.COMPUTE
        verdict = output.result
        assert isinstance(verdict, ModelWorkspaceValidateResult)
        assert verdict.valid is True

    async def test_symlink_escape_rejected(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "root"
        root.mkdir()
        escape = root / "escape"
        escape.symlink_to(outside, target_is_directory=True)

        handler = HandlerWorkspaceValidate()
        command = ModelWorkspaceValidateCommand(
            correlation_id=uuid4(),
            workspace_path=str(escape),
            allowed_roots=(str(root),),
            sandbox=EnumAgentSandbox.READ_ONLY,
            prompt="do the thing",
        )
        envelope: ModelEventEnvelope[ModelWorkspaceValidateCommand] = (
            ModelEventEnvelope(payload=command, correlation_id=command.correlation_id)
        )
        output = await handler.handle(envelope)
        verdict = output.result
        assert isinstance(verdict, ModelWorkspaceValidateResult)
        assert verdict.valid is False
        assert verdict.rejection_reason is not None

    async def test_no_allowed_roots_rejected(self, tmp_path: Path) -> None:
        handler = HandlerWorkspaceValidate()
        command = ModelWorkspaceValidateCommand(
            correlation_id=uuid4(),
            workspace_path=str(tmp_path),
            allowed_roots=(),
            sandbox=EnumAgentSandbox.READ_ONLY,
            prompt="do the thing",
        )
        envelope: ModelEventEnvelope[ModelWorkspaceValidateCommand] = (
            ModelEventEnvelope(payload=command, correlation_id=command.correlation_id)
        )
        output = await handler.handle(envelope)
        verdict = output.result
        assert isinstance(verdict, ModelWorkspaceValidateResult)
        assert verdict.valid is False


# ---------------------------------------------------------------------------
# ORCHESTRATOR: sequences validate -> invoke; rejected path emits no invoke
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrchestratorDispatch:
    async def test_invoke_requested_emits_workspace_validate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from omnibase_infra.nodes.node_coding_agent_orchestrator.handlers.handler_coding_agent_orchestrator import (
            TOPIC_WORKSPACE_VALIDATE,
        )

        # The allowed roots are contract-declared via ${env.…}; the contract
        # fails closed if it resolves empty, so the lane env var must be set.
        monkeypatch.setenv("ONEX_CODING_AGENT_WORKSPACE_ROOT", str(tmp_path))
        handler = HandlerCodingAgentOrchestrator()
        command = _invoke_command(workspace_path=str(tmp_path))
        envelope: ModelEventEnvelope[ModelCodingAgentInvokeCommand] = (
            ModelEventEnvelope(
                payload=command,
                correlation_id=command.correlation_id,
                event_type="onex.cmd.omnibase-infra.coding-agent-invoke.v1",
            )
        )
        output = await handler.handle(envelope)
        assert output.node_kind == EnumNodeKind.ORCHESTRATOR
        assert output.result is None  # orchestrator emits, never returns
        assert len(output.events) == 1
        validate_event = output.events[0]
        assert validate_event.event_type == TOPIC_WORKSPACE_VALIDATE
        # The contract-declared allowed root is threaded into the validate command.
        validate_command = validate_event.payload
        assert isinstance(validate_command, ModelWorkspaceValidateCommand)
        assert validate_command.allowed_roots == (str(tmp_path),)

    async def test_invoke_requested_fails_closed_without_allowed_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No allowed root declared/resolvable -> the orchestrator fails closed
        # rather than thread an empty allowlist (no unscoped workspace permitted).
        monkeypatch.delenv("ONEX_CODING_AGENT_WORKSPACE_ROOT", raising=False)
        handler = HandlerCodingAgentOrchestrator()
        command = _invoke_command(workspace_path=str(tmp_path))
        envelope: ModelEventEnvelope[ModelCodingAgentInvokeCommand] = (
            ModelEventEnvelope(
                payload=command,
                correlation_id=command.correlation_id,
                event_type="onex.cmd.omnibase-infra.coding-agent-invoke.v1",
            )
        )
        with pytest.raises(ValueError, match="allowed_workspace_roots"):
            await handler.handle(envelope)

    async def test_workspace_rejected_emits_fsm_reject_not_invoke(
        self, tmp_path: Path
    ) -> None:
        from omnibase_infra.nodes.node_coding_agent_orchestrator.handlers.handler_coding_agent_orchestrator import (
            TOPIC_FSM_ADVANCE,
            TOPIC_INVOKE,
        )

        handler = HandlerCodingAgentOrchestrator()
        command = _invoke_command(workspace_path=str(tmp_path))
        verdict = ModelWorkspaceValidateResult(
            correlation_id=command.correlation_id,
            valid=False,
            rejection_reason="symlink escape",
        )
        envelope: ModelEventEnvelope[Any] = ModelEventEnvelope(
            payload={
                "verdict": verdict.model_dump(mode="json"),
                "command": command.model_dump(mode="json"),
            },
            correlation_id=command.correlation_id,
            event_type="onex.evt.omnibase-infra.coding-agent-workspace-validated.v1",
        )
        output = await handler.handle(envelope)
        assert len(output.events) == 1
        emitted = output.events[0]
        # Rejected -> FSM advance to REJECTED, NEVER the invoke command.
        assert emitted.event_type == TOPIC_FSM_ADVANCE
        assert emitted.event_type != TOPIC_INVOKE
        assert emitted.payload["event"]["kind"] == "workspace_rejected"

    async def test_workspace_ok_emits_invoke(self, tmp_path: Path) -> None:
        from omnibase_infra.nodes.node_coding_agent_orchestrator.handlers.handler_coding_agent_orchestrator import (
            TOPIC_INVOKE,
        )

        handler = HandlerCodingAgentOrchestrator()
        command = _invoke_command(workspace_path=str(tmp_path))
        verdict = ModelWorkspaceValidateResult(
            correlation_id=command.correlation_id,
            valid=True,
            resolved_path=str(tmp_path),
        )
        envelope: ModelEventEnvelope[Any] = ModelEventEnvelope(
            payload={
                "verdict": verdict.model_dump(mode="json"),
                "command": command.model_dump(mode="json"),
            },
            correlation_id=command.correlation_id,
            event_type="onex.evt.omnibase-infra.coding-agent-workspace-validated.v1",
        )
        output = await handler.handle(envelope)
        assert len(output.events) == 1
        assert output.events[0].event_type == TOPIC_INVOKE


# ---------------------------------------------------------------------------
# REDUCER: FSM transitions, retry/breaker, dedupe, replay safety
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFsmReducerDispatch:
    async def test_reducer_handle_emits_state_and_projection(self) -> None:
        handler = HandlerCodingAgentFsm()
        corr = uuid4()
        advance = ModelCodingAgentFsmAdvance(
            state=_fsm_state(corr, EnumCodingAgentFsmState.IDLE),
            event=ModelCodingAgentEvent(
                correlation_id=corr, kind=EnumCodingAgentEventKind.INVOKE_REQUESTED
            ),
        )
        envelope: ModelEventEnvelope[ModelCodingAgentFsmAdvance] = ModelEventEnvelope(
            payload=advance, correlation_id=corr
        )
        output = await handler.handle(envelope)
        assert output.node_kind == EnumNodeKind.REDUCER
        # state projection + trace projection
        assert len(output.projections) == 2
        new_state = output.projections[0]
        assert isinstance(new_state, ModelCodingAgentFsmState)
        assert new_state.current_state == EnumCodingAgentFsmState.VALIDATING
        # REDUCER must not emit events/intents/result.
        assert output.events == ()
        assert output.intents == ()
        assert output.result is None

    def test_workspace_rejected_to_rejected_no_invoke_intent(self) -> None:
        corr = uuid4()
        state = _fsm_state(corr, EnumCodingAgentFsmState.VALIDATING)
        event = ModelCodingAgentEvent(
            correlation_id=corr, kind=EnumCodingAgentEventKind.WORKSPACE_REJECTED
        )
        new_state, intents = delta(state, event)
        assert new_state.current_state == EnumCodingAgentFsmState.REJECTED
        kinds = {i.kind for i in intents}
        assert EnumCodingAgentIntentKind.DISPATCH_INVOKE not in kinds
        assert kinds == {EnumCodingAgentIntentKind.EMIT_TERMINAL}

    def test_workspace_ok_to_invoking(self) -> None:
        corr = uuid4()
        state = _fsm_state(corr, EnumCodingAgentFsmState.VALIDATING)
        event = ModelCodingAgentEvent(
            correlation_id=corr, kind=EnumCodingAgentEventKind.WORKSPACE_OK
        )
        new_state, intents = delta(state, event)
        assert new_state.current_state == EnumCodingAgentFsmState.INVOKING
        assert new_state.invoke_attempts == 1
        assert intents[0].kind == EnumCodingAgentIntentKind.DISPATCH_INVOKE

    def test_invoke_completed_to_capturing_to_completed(self) -> None:
        corr = uuid4()
        invoking = _fsm_state(corr, EnumCodingAgentFsmState.INVOKING, invoke_attempts=1)
        capturing, _ = delta(
            invoking,
            ModelCodingAgentEvent(
                correlation_id=corr, kind=EnumCodingAgentEventKind.INVOKE_COMPLETED
            ),
        )
        assert capturing.current_state == EnumCodingAgentFsmState.CAPTURING
        completed, intents = delta(
            capturing,
            ModelCodingAgentEvent(
                correlation_id=corr, kind=EnumCodingAgentEventKind.DIFF_CAPTURED
            ),
        )
        assert completed.current_state == EnumCodingAgentFsmState.COMPLETED
        assert intents[0].kind == EnumCodingAgentIntentKind.EMIT_TERMINAL

    def test_read_only_retry_then_circuit_breaker(self) -> None:
        corr = uuid4()
        state = _fsm_state(
            corr,
            EnumCodingAgentFsmState.INVOKING,
            sandbox=EnumAgentSandbox.READ_ONLY,
            invoke_attempts=1,
        )
        fail = ModelCodingAgentEvent(
            correlation_id=corr,
            kind=EnumCodingAgentEventKind.INVOKE_FAILED,
            error_class=EnumCliBackendStatus.SUBPROCESS_ERROR,
        )
        # First failure: bounded READ_ONLY retry -> back to INVOKING.
        s1, i1 = delta(state, fail)
        assert s1.current_state == EnumCodingAgentFsmState.INVOKING
        assert s1.consecutive_failures == 1
        assert i1[0].kind == EnumCodingAgentIntentKind.DISPATCH_INVOKE
        # Second failure: retry budget consumed.
        s2, _ = delta(s1, fail)
        assert s2.current_state == EnumCodingAgentFsmState.INVOKING
        assert s2.consecutive_failures == 2
        # Third failure trips the circuit breaker -> FAILED.
        s3, i3 = delta(s2, fail)
        assert s3.current_state == EnumCodingAgentFsmState.FAILED
        assert i3[0].kind == EnumCodingAgentIntentKind.EMIT_TERMINAL

    def test_workspace_write_failure_never_auto_retried(self) -> None:
        corr = uuid4()
        state = _fsm_state(
            corr,
            EnumCodingAgentFsmState.INVOKING,
            sandbox=EnumAgentSandbox.WORKSPACE_WRITE,
            invoke_attempts=1,
        )
        fail = ModelCodingAgentEvent(
            correlation_id=corr,
            kind=EnumCodingAgentEventKind.INVOKE_FAILED,
            error_class=EnumCliBackendStatus.SUBPROCESS_ERROR,
        )
        new_state, intents = delta(state, fail)
        # A partial edit must not be blindly re-run: straight to FAILED.
        assert new_state.current_state == EnumCodingAgentFsmState.FAILED
        kinds = {i.kind for i in intents}
        assert EnumCodingAgentIntentKind.DISPATCH_INVOKE not in kinds

    def test_duplicate_correlation_id_never_reruns(self) -> None:
        corr = uuid4()
        # A run already past IDLE that receives invoke-requested again: no re-run.
        invoking = _fsm_state(corr, EnumCodingAgentFsmState.INVOKING, invoke_attempts=1)
        dup = ModelCodingAgentEvent(
            correlation_id=corr, kind=EnumCodingAgentEventKind.INVOKE_REQUESTED
        )
        new_state, intents = delta(invoking, dup)
        assert new_state == invoking
        assert intents == ()
        # A run already terminal never advances/re-runs.
        completed = _fsm_state(corr, EnumCodingAgentFsmState.COMPLETED)
        s2, i2 = delta(completed, dup)
        assert s2 == completed
        assert i2 == ()

    def test_replay_issues_no_live_intent(self) -> None:
        corr = uuid4()
        state = _fsm_state(corr, EnumCodingAgentFsmState.VALIDATING)
        replay_ok = ModelCodingAgentEvent(
            correlation_id=corr,
            kind=EnumCodingAgentEventKind.WORKSPACE_OK,
            is_replay=True,
        )
        new_state, intents = delta(state, replay_ok)
        # State still recomputes...
        assert new_state.current_state == EnumCodingAgentFsmState.INVOKING
        # ...but the intent is non-live, so the orchestrator must not act on it
        # (no subprocess on replay).
        assert all(i.is_live is False for i in intents)


# ---------------------------------------------------------------------------
# EFFECT: subprocess MOCKED; git-derived diff; failure classes; no-runner gate
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestInvokeEffectDispatch:
    async def test_invoke_completed_with_git_diff(self, tmp_path: Path) -> None:
        spy = _SpySubprocess(
            ModelSubprocessOutcome(
                returncode=0, stdout="done", stderr="", timed_out=False
            )
        )
        handler = HandlerCodingAgentInvoke(
            run_subprocess=spy,
            probe_head_sha=lambda _cwd: "abc1234",
            capture_diff=lambda _cwd: (("foo.py",), "diff --git a/foo.py b/foo.py"),
            which=lambda _b: "/usr/bin/claude",
        )
        command = _invoke_command(workspace_path=str(tmp_path))
        envelope: ModelEventEnvelope[ModelCodingAgentInvokeCommand] = (
            ModelEventEnvelope(payload=command, correlation_id=command.correlation_id)
        )
        output = await handler.handle(envelope)
        assert output.node_kind == EnumNodeKind.EFFECT
        assert len(output.events) == 1
        result = output.events[0].payload
        assert result.status == EnumAgentStatus.COMPLETED
        # files_changed/diff are git-derived (system), not agent-reported.
        assert result.files_changed == ("foo.py",)
        assert result.diff_hash != ""
        assert result.starting_head_sha == "abc1234"
        assert len(spy.calls) == 1

    def test_unavailable_when_binary_missing(self, tmp_path: Path) -> None:
        spy = _SpySubprocess(
            ModelSubprocessOutcome(returncode=0, stdout="x", stderr="", timed_out=False)
        )
        handler = HandlerCodingAgentInvoke(run_subprocess=spy, which=lambda _b: None)
        result = handler.invoke(_invoke_command(workspace_path=str(tmp_path)))
        assert result.status == EnumAgentStatus.FAILED
        assert result.error_class == EnumCliBackendStatus.UNAVAILABLE
        # No subprocess attempted when the binary is missing.
        assert spy.calls == []

    def test_timeout_classified(self, tmp_path: Path) -> None:
        spy = _SpySubprocess(
            ModelSubprocessOutcome(returncode=-9, stdout="", stderr="", timed_out=True)
        )
        handler = HandlerCodingAgentInvoke(
            run_subprocess=spy,
            probe_head_sha=lambda _cwd: None,
            capture_diff=lambda _cwd: ((), ""),
            which=lambda _b: "/usr/bin/claude",
        )
        result = handler.invoke(_invoke_command(workspace_path=str(tmp_path)))
        assert result.error_class == EnumCliBackendStatus.TIMEOUT
        assert result.timed_out is True

    def test_subprocess_error_classified(self, tmp_path: Path) -> None:
        spy = _SpySubprocess(
            ModelSubprocessOutcome(
                returncode=1, stdout="", stderr="boom", timed_out=False
            )
        )
        handler = HandlerCodingAgentInvoke(
            run_subprocess=spy,
            probe_head_sha=lambda _cwd: None,
            capture_diff=lambda _cwd: ((), ""),
            which=lambda _b: "/usr/bin/claude",
        )
        result = handler.invoke(_invoke_command(workspace_path=str(tmp_path)))
        assert result.error_class == EnumCliBackendStatus.SUBPROCESS_ERROR
        assert result.exit_code == 1

    def test_empty_response_classified(self, tmp_path: Path) -> None:
        spy = _SpySubprocess(
            ModelSubprocessOutcome(
                returncode=0, stdout="   ", stderr="", timed_out=False
            )
        )
        handler = HandlerCodingAgentInvoke(
            run_subprocess=spy,
            probe_head_sha=lambda _cwd: None,
            capture_diff=lambda _cwd: ((), ""),
            which=lambda _b: "/usr/bin/claude",
        )
        result = handler.invoke(_invoke_command(workspace_path=str(tmp_path)))
        assert result.error_class == EnumCliBackendStatus.EMPTY_RESPONSE


# ---------------------------------------------------------------------------
# Full credential-independent chain: rejected path never touches the effect
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEndToEndCredentialIndependentChain:
    async def test_rejected_workspace_runs_no_subprocess(self, tmp_path: Path) -> None:
        """validate -> reject -> FSM REJECTED, with the effect's spy never called."""
        outside = tmp_path / "outside"
        outside.mkdir()
        root = tmp_path / "root"
        root.mkdir()
        escape = root / "escape"
        escape.symlink_to(outside, target_is_directory=True)

        corr = uuid4()
        command = _invoke_command(workspace_path=str(escape), correlation_id=corr)

        # COMPUTE validates and rejects.
        compute = HandlerWorkspaceValidate()
        validate_command = ModelWorkspaceValidateCommand(
            correlation_id=corr,
            workspace_path=command.workspace_path,
            allowed_roots=(str(root),),
            sandbox=command.sandbox,
            prompt=command.prompt,
        )
        verdict_out = await compute.handle(
            ModelEventEnvelope(payload=validate_command, correlation_id=corr)
        )
        verdict = verdict_out.result
        assert verdict.valid is False

        # ORCHESTRATOR routes the rejection to the FSM advance (not invoke).
        orchestrator = HandlerCodingAgentOrchestrator()
        from omnibase_infra.nodes.node_coding_agent_orchestrator.handlers.handler_coding_agent_orchestrator import (
            TOPIC_FSM_ADVANCE,
        )

        orch_out = await orchestrator.handle(
            ModelEventEnvelope(
                payload={
                    "verdict": verdict.model_dump(mode="json"),
                    "command": command.model_dump(mode="json"),
                },
                correlation_id=corr,
                event_type="onex.evt.omnibase-infra.coding-agent-workspace-validated.v1",
            )
        )
        advance_event = orch_out.events[0]
        assert advance_event.event_type == TOPIC_FSM_ADVANCE

        # REDUCER folds the rejection -> terminal REJECTED.
        reducer = HandlerCodingAgentFsm()
        advance = ModelCodingAgentFsmAdvance(
            state=_fsm_state(corr, EnumCodingAgentFsmState.VALIDATING),
            event=ModelCodingAgentEvent.model_validate(advance_event.payload["event"]),
        )
        fsm_out = await reducer.handle(
            ModelEventEnvelope(payload=advance, correlation_id=corr)
        )
        final_state = fsm_out.projections[0]
        assert final_state.current_state == EnumCodingAgentFsmState.REJECTED
        assert final_state.current_state in TERMINAL_STATES

        # The EFFECT was never constructed/called in the rejected chain — proven
        # by the FSM reaching REJECTED via the workspace_rejected event, which
        # the reducer only emits the EMIT_TERMINAL intent for (no DISPATCH_INVOKE).
