# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Coding-agent invoke EFFECT handler (OMN-13247, plan §5.1 / §5.7).

The ONLY I/O node in the workflow. Ported from
``node_llm_inference_effect.handlers.handler_llm_cli_subprocess`` and upgraded
from read-only single-shot inference to a workspace-write agentic invocation with
git-derived diff capture:

  - ``shutil.which`` availability gate (UNAVAILABLE if the CLI is missing).
  - structured failure classes (UNAVAILABLE / TIMEOUT / SUBPROCESS_ERROR /
    EMPTY_RESPONSE), reused from the inference handler's vocabulary.
  - process-group spawn; a timeout kills the WHOLE group (no orphan subprocess).
  - ``files_changed`` + ``diff`` are git-derived, captured AFTER the subprocess
    exits — never trusted from agent stdout (plan §5.4).
  - ``starting_head_sha`` recorded before invocation.

The sandbox -> CLI permission-mode argv mapping is finalized from the Phase 0
(OMN-13246) auth proof (evidence: omni_home PR #182, merged) — see
``build_agent_invocation``. The subprocess + git I/O remain behind injected seams
so tests mock them and the live process-group runner is wired here; the remaining
DoD is live e2e proof on dev after Phase B ships the binaries + creds mount.
"""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from omnibase_core.models.dispatch.model_handler_output import ModelHandlerOutput
from omnibase_core.models.events.model_event_envelope import ModelEventEnvelope
from omnibase_infra.enums import EnumHandlerType, EnumHandlerTypeCategory
from omnibase_infra.models.coding_agent.enum_agent_sandbox import EnumAgentSandbox
from omnibase_infra.models.coding_agent.enum_agent_status import EnumAgentStatus
from omnibase_infra.models.coding_agent.enum_cli_backend_status import (
    EnumCliBackendStatus,
)
from omnibase_infra.models.coding_agent.enum_coding_agent import EnumCodingAgent
from omnibase_infra.models.coding_agent.model_agent_invocation import (
    ModelAgentInvocation,
)
from omnibase_infra.models.coding_agent.model_coding_agent_invoke_command import (
    ModelCodingAgentInvokeCommand,
)
from omnibase_infra.models.coding_agent.model_coding_agent_result import (
    ModelCodingAgentResult,
)
from omnibase_infra.models.coding_agent.model_subprocess_invocation import (
    ModelSubprocessInvocation,
)
from omnibase_infra.models.coding_agent.model_subprocess_outcome import (
    ModelSubprocessOutcome,
)
from omnibase_infra.nodes.node_coding_agent_invoke_effect.contract_descriptor import (
    contract_agent_credential_home,
)

logger = logging.getLogger(__name__)

HANDLER_ID = "coding-agent-invoke-effect"

# The contract that declares descriptor.agent_credential_home (the ${env.…}
# overlay value the spawned subprocess uses as HOME). The descriptor is the single
# source of truth; the handler never hardcodes a credential-home path.
_CONTRACT = Path(__file__).resolve().parent.parent / "contract.yaml"

# Dispatch payloads are coerced at runtime; the protocol entry is generic over the
# envelope payload type (ProtocolMessageHandler.handle(ModelEventEnvelope[T])).
T = TypeVar("T")

# The CLI binary name per agent. The binary NAME is stable; the agentic argv that
# maps OmniNode sandbox -> CLI permission mode is finalized below (Phase 0 proof).
_CLI_BINARY_BY_AGENT: dict[EnumCodingAgent, str] = {
    EnumCodingAgent.CLAUDE: "claude",
    EnumCodingAgent.CODEX: "codex",
}


# Seam types: the EFFECT's I/O is injected so tests mock it. The runner takes a
# single ModelSubprocessInvocation bundling argv, cwd, timeout, network posture,
# the optional piped stdin (claude --add-dir pipes the prompt via stdin —
# OMN-13246), and the explicit child ``env`` whose HOME is the contract-resolved
# credential home (OMN-13247 Phase B dev-verify fix) — trivial to mock.
SubprocessRunner = Callable[[ModelSubprocessInvocation], ModelSubprocessOutcome]
GitProbe = Callable[[str], str | None]
GitDiffCapture = Callable[[str], tuple[tuple[str, ...], str]]


def build_agent_invocation(
    command: ModelCodingAgentInvokeCommand,
) -> ModelAgentInvocation:
    """Build the coding-agent CLI argv + optional stdin for one invocation.

    Argv mapping is finalized from the Phase 0 (OMN-13246) auth proof — evidence
    omni_home PR #182 (merged). Per-CLI, per-sandbox semantics (verbatim from the
    pinned proof; do NOT re-derive):

    Codex (the workspace IS the agent's cwd via ``-C``):
      - READ_ONLY:       ``codex exec -s read-only --json <prompt>``
      - WORKSPACE_WRITE: ``codex exec -C <ws> -s danger-full-access --json <prompt>``

      NOTE (OMN-13246): ``-s workspace-write`` FAILS in-container (bwrap cannot
      create a user namespace), so ``danger-full-access`` is the proven write mode
      and the CONTAINER is the sandbox boundary — our COMPUTE workspace guards,
      ``network: false``, and process-group isolation provide the safety envelope.
      ``-s workspace-write`` can be revisited if Phase B grants the container a
      user namespace.

    Claude:
      - READ_ONLY:       ``claude -p --output-format json --permission-mode plan
        <prompt>`` (``plan`` is genuinely read-only on claude 2.1.181).
      - WORKSPACE_WRITE: ``claude -p --output-format json --permission-mode
        acceptEdits --add-dir <ws>`` with the prompt delivered via STDIN —
        ``--add-dir`` is greedy on positionals and would swallow a trailing prompt
        arg, so the prompt MUST go via stdin (OMN-13246).

    ``command.model`` maps to the agent-native model flag (``codex --model`` /
    ``claude --model``) ONLY when provided; never an env read, never a
    ``*_PROVIDER``/``*_MODEL`` env. Auth is ambient (the CLI reads
    ``~/.codex``/``~/.claude``); this builds no API-key/endpoint argv.
    """
    binary = _CLI_BINARY_BY_AGENT.get(command.agent)
    if binary is None:
        raise ValueError(f"no CLI binary mapping for agent {command.agent!r}")

    workspace_write = command.sandbox is EnumAgentSandbox.WORKSPACE_WRITE

    if command.agent is EnumCodingAgent.CODEX:
        argv: list[str] = [binary, "exec"]
        if workspace_write:
            argv += ["-C", command.workspace_path, "-s", "danger-full-access"]
        else:
            argv += ["-s", "read-only"]
        argv += ["--json"]
        if command.model is not None:
            argv += ["--model", command.model]
        argv += [command.prompt]
        return ModelAgentInvocation(argv=argv, stdin=None)

    # EnumCodingAgent.CLAUDE
    argv = [binary, "-p", "--output-format", "json", "--permission-mode"]
    if workspace_write:
        argv += ["acceptEdits", "--add-dir", command.workspace_path]
        if command.model is not None:
            argv += ["--model", command.model]
        # Prompt via stdin: --add-dir is greedy on positionals (OMN-13246).
        return ModelAgentInvocation(argv=argv, stdin=command.prompt)
    argv += ["plan"]
    if command.model is not None:
        argv += ["--model", command.model]
    argv += [command.prompt]
    return ModelAgentInvocation(argv=argv, stdin=None)


def _diff_hash(diff: str) -> str:
    if not diff:
        return ""
    return hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16]


class HandlerCodingAgentInvoke:
    """Run a coding-agent CLI subprocess in the workspace and capture the git diff.

    The subprocess + git I/O are injected via the ``run_subprocess`` /
    ``probe_head_sha`` / ``capture_diff`` seams. Availability is gated by the
    injected ``which`` callable (defaults to ``shutil.which``). No bus access:
    this is an effect handler, never a publisher.
    """

    @property
    def handler_type(self) -> EnumHandlerType:
        return EnumHandlerType.NODE_HANDLER

    @property
    def handler_category(self) -> EnumHandlerTypeCategory:
        return EnumHandlerTypeCategory.EFFECT

    def __init__(
        self,
        *,
        run_subprocess: SubprocessRunner | None = None,
        probe_head_sha: GitProbe | None = None,
        capture_diff: GitDiffCapture | None = None,
        which: Callable[[str], str | None] | None = None,
        agent_credential_home: str | None = None,
    ) -> None:
        # Live process-group runner + git probes are the defaults (the argv
        # mapping is finalized from Phase 0 / OMN-13246). Tests inject mocks for
        # all three seams so no real claude/codex subprocess runs under test.
        self._run_subprocess = (
            run_subprocess if run_subprocess is not None else _run_subprocess_pgroup
        )
        self._probe_head_sha = (
            probe_head_sha if probe_head_sha is not None else _git_head_sha
        )
        self._capture_diff = (
            capture_diff if capture_diff is not None else _git_capture_diff
        )
        self._which = which if which is not None else _default_which
        # The credential HOME for the spawned subprocess. Resolved from the
        # contract descriptor (${env.CODING_AGENT_CRED_HOME}) by default; an
        # explicit override lets tests pin it without touching the environment.
        # Resolution is deferred to invoke-time (fail-closed there) so constructing
        # the handler never requires CODING_AGENT_CRED_HOME to be set.
        self._agent_credential_home = agent_credential_home

    def invoke(self, command: ModelCodingAgentInvokeCommand) -> ModelCodingAgentResult:
        """Run one coding-agent invocation; return the system-derived result."""
        binary = _CLI_BINARY_BY_AGENT.get(command.agent)
        if binary is None:
            return self._failed(
                command,
                EnumCliBackendStatus.UNAVAILABLE,
                f"no CLI binary mapping for agent {command.agent!r}",
            )

        if self._which(binary) is None:
            return self._failed(
                command,
                EnumCliBackendStatus.UNAVAILABLE,
                f"{binary} not found on PATH",
            )

        # Resolve the credential HOME and build the explicit child env BEFORE
        # spawning. Fail-closed: an unset CODING_AGENT_CRED_HOME raises (surfaced
        # as UNAVAILABLE) rather than letting the child inherit the runtime
        # container's HOME (/root under the dev runtime — no creds there).
        try:
            child_env = self._build_subprocess_env(command.agent)
        except ValueError as exc:
            return self._failed(
                command,
                EnumCliBackendStatus.UNAVAILABLE,
                str(exc),
            )

        starting_head_sha = self._probe_head_sha(command.workspace_path)

        invocation = build_agent_invocation(command)
        timeout_s = max(1, command.timeout_ms // 1000)

        start = time.monotonic()
        # The runner spawns in its own process group; on timeout it kills the
        # whole group and returns timed_out=True (no orphaned subprocess). The
        # prompt is piped via stdin for claude write mode (--add-dir is greedy).
        # The explicit child env carries HOME=<contract cred home> so claude/codex
        # find their ambient OAuth creds even when the runtime runs as root.
        outcome = self._run_subprocess(
            ModelSubprocessInvocation(
                argv=invocation.argv,
                cwd=command.workspace_path,
                timeout_s=timeout_s,
                network=command.network,
                stdin=invocation.stdin,
                env=child_env,
            )
        )
        duration_ms = (time.monotonic() - start) * 1000

        if outcome.timed_out:
            return self._failed(
                command,
                EnumCliBackendStatus.TIMEOUT,
                f"{binary} exceeded {timeout_s}s deadline",
                exit_code=outcome.returncode,
                starting_head_sha=starting_head_sha,
                timed_out=True,
                duration_ms=duration_ms,
            )

        if outcome.returncode != 0:
            stderr_preview = (outcome.stderr or "(no stderr)")[:200]
            return self._failed(
                command,
                EnumCliBackendStatus.SUBPROCESS_ERROR,
                f"exit {outcome.returncode}: {stderr_preview}",
                exit_code=outcome.returncode,
                starting_head_sha=starting_head_sha,
                duration_ms=duration_ms,
            )

        content = (outcome.stdout or "").strip()
        if not content:
            return self._failed(
                command,
                EnumCliBackendStatus.EMPTY_RESPONSE,
                "stdout empty after successful exit",
                exit_code=outcome.returncode,
                starting_head_sha=starting_head_sha,
                duration_ms=duration_ms,
            )

        # System-derived evidence: git diff captured AFTER the subprocess exits.
        # Never parse agent stdout for file changes (plan §5.4).
        files_changed, diff = self._capture_diff(command.workspace_path)

        return ModelCodingAgentResult(
            correlation_id=command.correlation_id,
            status=EnumAgentStatus.COMPLETED,
            exit_code=outcome.returncode,
            files_changed=files_changed,
            diff=diff,
            diff_hash=_diff_hash(diff),
            starting_head_sha=starting_head_sha,
            error_class=EnumCliBackendStatus.SUCCESS,
            timed_out=False,
            duration_ms=duration_ms,
            output=content,
        )

    def _resolve_credential_home(self) -> str:
        """Resolve the credential HOME (override > contract descriptor).

        Fails closed via ``contract_agent_credential_home`` when the descriptor
        resolves empty (``CODING_AGENT_CRED_HOME`` unset), so the subprocess never
        silently inherits ``/root``.
        """
        if self._agent_credential_home is not None:
            return self._agent_credential_home
        return contract_agent_credential_home(_CONTRACT)

    def _build_subprocess_env(self, agent: EnumCodingAgent) -> dict[str, str]:
        """Build the explicit child env for the spawned claude/codex subprocess.

        Copies the parent ``os.environ`` and overrides ``HOME`` with the
        contract-resolved credential home so the CLI reads its ambient OAuth creds
        from ``<cred_home>/.claude`` / ``<cred_home>/.codex`` instead of the
        runtime container's ``HOME`` (``/root`` under the dev runtime, where no
        creds are bind-mounted). For codex, ``CODEX_HOME`` is set to
        ``<cred_home>/.codex`` so it does not depend on codex's own ``$HOME``
        derivation. No API-key / endpoint env is added — auth stays ambient.
        """
        cred_home = self._resolve_credential_home()
        env = dict(os.environ)
        env["HOME"] = cred_home
        if agent is EnumCodingAgent.CODEX:
            env["CODEX_HOME"] = str(Path(cred_home) / ".codex")
        return env

    def _failed(
        self,
        command: ModelCodingAgentInvokeCommand,
        error_class: EnumCliBackendStatus,
        message: str,
        *,
        exit_code: int | None = None,
        starting_head_sha: str | None = None,
        timed_out: bool = False,
        duration_ms: float = 0.0,
    ) -> ModelCodingAgentResult:
        logger.debug("coding-agent invoke failed [%s]: %s", error_class.value, message)
        return ModelCodingAgentResult(
            correlation_id=command.correlation_id,
            status=EnumAgentStatus.FAILED,
            exit_code=exit_code,
            starting_head_sha=starting_head_sha,
            error_class=error_class,
            timed_out=timed_out,
            duration_ms=duration_ms,
            output=message,
        )

    async def handle(self, envelope: ModelEventEnvelope[T]) -> ModelHandlerOutput[None]:
        """Invoke the coding agent and emit the result as an effect event.

        Effect handlers emit events (never intents/projections, never publish to
        the bus directly). The runtime publishes the returned event envelope to
        the effect's contract-declared publish topic.
        """
        command = _coerce_command(envelope.payload)
        result = self.invoke(command)
        result_event: ModelEventEnvelope[ModelCodingAgentResult] = ModelEventEnvelope(
            payload=result,
            correlation_id=command.correlation_id,
        )
        return ModelHandlerOutput.for_effect(
            input_envelope_id=envelope.envelope_id,
            correlation_id=(
                envelope.correlation_id or command.correlation_id or uuid4()
            ),
            handler_id=HANDLER_ID,
            events=(result_event,),
        )


def _default_which(binary: str) -> str | None:
    import shutil

    return shutil.which(binary)


def _run_subprocess_pgroup(
    invocation: ModelSubprocessInvocation,
) -> ModelSubprocessOutcome:
    """Run ``invocation`` in its OWN process group; timeout kills the group.

    ``start_new_session=True`` puts the child in a fresh session + process group,
    so a runaway agent that forks children is reaped as a unit: on timeout (or
    cancel) we ``os.killpg(SIGTERM)`` then escalate to ``SIGKILL`` — no orphaned
    subprocess survives (plan §5.5). The prompt is piped via stdin when provided
    (claude WORKSPACE_WRITE; ``--add-dir`` is greedy on positionals — OMN-13246).

    ``invocation.env`` is the EXPLICIT child environment (a copy of the parent env
    with ``HOME`` — and ``CODEX_HOME`` for codex — overridden to the
    contract-resolved credential home) so claude/codex find their ambient OAuth
    creds even when the runtime runs as root (OMN-13247 Phase B dev-verify fix).
    It is passed verbatim to ``Popen(env=...)``; the child never inherits the
    parent env implicitly.

    ``invocation.network`` is carried for parity with the seam contract; network
    isolation is enforced by the container/namespace boundary (Phase B), not by
    this in-process runner, so it does not branch on the flag here.
    """
    stdin = invocation.stdin
    process = subprocess.Popen(
        invocation.argv,
        cwd=invocation.cwd,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        env=dict(invocation.env),
    )
    try:
        stdout, stderr = process.communicate(input=stdin, timeout=invocation.timeout_s)
        return ModelSubprocessOutcome(
            returncode=process.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        stdout, stderr = process.communicate()
        return ModelSubprocessOutcome(
            returncode=process.returncode if process.returncode is not None else -1,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=True,
        )


def _kill_process_group(process: subprocess.Popen[str]) -> None:
    """SIGTERM then SIGKILL the child's whole process group; never raise."""
    try:
        group = os.getpgid(process.pid)
    except (ProcessLookupError, OSError):
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except (ProcessLookupError, OSError):
            return
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
    """Run a read-only git command in ``cwd`` and return the completed process."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_head_sha(cwd: str) -> str | None:
    """Return ``git rev-parse HEAD`` in ``cwd``; None when not a git repo."""
    completed = _git(["rev-parse", "HEAD"], cwd)
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def _git_capture_diff(cwd: str) -> tuple[tuple[str, ...], str]:
    """Capture system-derived ``files_changed`` + ``diff`` AFTER the agent exits.

    ``files_changed`` = ``git diff --name-only`` (tracked edits) UNION the
    untracked paths reported by ``git status --porcelain`` (new files the agent
    created). ``diff`` = ``git diff``. Never parses agent stdout (plan §5.4).
    """
    changed: list[str] = []
    name_only = _git(["diff", "--name-only"], cwd)
    if name_only.returncode == 0:
        changed.extend(line for line in name_only.stdout.splitlines() if line.strip())
    status = _git(["status", "--porcelain"], cwd)
    if status.returncode == 0:
        for line in status.stdout.splitlines():
            # Untracked entries are prefixed with "?? " in porcelain v1.
            if line.startswith("?? "):
                path = line[3:].strip()
                if path:
                    changed.append(path)
    diff = _git(["diff"], cwd)
    diff_text = diff.stdout if diff.returncode == 0 else ""
    # De-dupe while preserving first-seen order.
    seen: dict[str, None] = {}
    for path in changed:
        seen.setdefault(path, None)
    return tuple(seen), diff_text


def _coerce_command(payload: object) -> ModelCodingAgentInvokeCommand:
    if isinstance(payload, ModelCodingAgentInvokeCommand):
        return payload
    if isinstance(payload, Mapping):
        return ModelCodingAgentInvokeCommand.model_validate(dict(payload))
    if hasattr(payload, "model_dump"):
        return ModelCodingAgentInvokeCommand.model_validate(
            payload.model_dump(mode="json")
        )
    raise TypeError(
        "coding-agent invoke payload must be ModelCodingAgentInvokeCommand or a "
        f"mapping; got {type(payload).__name__}"
    )


__all__: list[str] = [
    "HANDLER_ID",
    "GitDiffCapture",
    "GitProbe",
    "HandlerCodingAgentInvoke",
    "ModelAgentInvocation",
    "ModelSubprocessInvocation",
    "ModelSubprocessOutcome",
    "SubprocessRunner",
    "build_agent_invocation",
]
