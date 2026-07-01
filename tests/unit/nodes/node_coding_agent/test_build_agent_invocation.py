# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Exact-argv coverage for ``build_agent_invocation`` (OMN-13247, plan §5.7).

These assertions pin the headless CLI argv for all four (agent, sandbox) pairs
exactly as proven in Phase 0 (OMN-13246; evidence omni_home PR #182, merged). If
the proven mapping ever drifts, this suite fails — the argv is load-bearing
(claude WORKSPACE_WRITE prompt-via-stdin, codex WORKSPACE_WRITE
danger-full-access). No real claude/codex subprocess is executed here.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from omnibase_infra.models.coding_agent import (
    EnumAgentSandbox,
    EnumCodingAgent,
    ModelCodingAgentInvokeCommand,
)
from omnibase_infra.nodes.node_coding_agent_invoke_effect.handlers.handler_coding_agent_invoke import (
    build_agent_invocation,
)

_WORKSPACE = "/work/repo"
_PROMPT = "add a docstring to foo.py"


def _command(
    *,
    agent: EnumCodingAgent,
    sandbox: EnumAgentSandbox,
    model: str | None = None,
) -> ModelCodingAgentInvokeCommand:
    return ModelCodingAgentInvokeCommand(
        correlation_id=uuid4(),
        agent=agent,
        prompt=_PROMPT,
        workspace_path=_WORKSPACE,
        sandbox=sandbox,
        model=model,
        timeout_ms=60000,
    )


@pytest.mark.unit
class TestBuildAgentInvocationExactArgv:
    def test_codex_read_only(self) -> None:
        invocation = build_agent_invocation(
            _command(agent=EnumCodingAgent.CODEX, sandbox=EnumAgentSandbox.READ_ONLY)
        )
        assert invocation.argv == [
            "codex",
            "exec",
            "-s",
            "read-only",
            "--json",
            _PROMPT,
        ]
        # Codex carries the prompt as the final positional, never via stdin.
        assert invocation.stdin is None

    def test_codex_workspace_write(self) -> None:
        invocation = build_agent_invocation(
            _command(
                agent=EnumCodingAgent.CODEX,
                sandbox=EnumAgentSandbox.WORKSPACE_WRITE,
            )
        )
        # -C <ws> + danger-full-access (the proven in-container write mode;
        # workspace-write fails because bwrap cannot create a user namespace).
        assert invocation.argv == [
            "codex",
            "exec",
            "-C",
            _WORKSPACE,
            "-s",
            "danger-full-access",
            "--json",
            _PROMPT,
        ]
        assert invocation.stdin is None

    def test_claude_read_only(self) -> None:
        invocation = build_agent_invocation(
            _command(agent=EnumCodingAgent.CLAUDE, sandbox=EnumAgentSandbox.READ_ONLY)
        )
        # plan == genuinely read-only on claude 2.1.181; prompt is positional.
        assert invocation.argv == [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--permission-mode",
            "plan",
            _PROMPT,
        ]
        assert invocation.stdin is None

    def test_claude_workspace_write_prompt_via_stdin(self) -> None:
        invocation = build_agent_invocation(
            _command(
                agent=EnumCodingAgent.CLAUDE,
                sandbox=EnumAgentSandbox.WORKSPACE_WRITE,
            )
        )
        # acceptEdits == write; --add-dir is greedy on positionals, so the prompt
        # MUST be delivered via stdin and NEVER appear in argv (OMN-13246).
        assert invocation.argv == [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--permission-mode",
            "acceptEdits",
            "--add-dir",
            _WORKSPACE,
        ]
        assert invocation.stdin == _PROMPT
        assert _PROMPT not in invocation.argv

    def test_model_maps_to_native_flag_only_when_provided(self) -> None:
        # model None -> no --model flag (agent uses its own default).
        without = build_agent_invocation(
            _command(agent=EnumCodingAgent.CODEX, sandbox=EnumAgentSandbox.READ_ONLY)
        )
        assert "--model" not in without.argv

        # codex --model <id> when provided.
        codex = build_agent_invocation(
            _command(
                agent=EnumCodingAgent.CODEX,
                sandbox=EnumAgentSandbox.READ_ONLY,
                model="gpt-5-codex",
            )
        )
        assert codex.argv[codex.argv.index("--model") + 1] == "gpt-5-codex"

        # claude --model <id> when provided (read-only: still positional prompt).
        claude_ro = build_agent_invocation(
            _command(
                agent=EnumCodingAgent.CLAUDE,
                sandbox=EnumAgentSandbox.READ_ONLY,
                model="claude-opus-4",
            )
        )
        assert claude_ro.argv[claude_ro.argv.index("--model") + 1] == "claude-opus-4"

        # claude write + model: prompt still via stdin, model still mapped.
        claude_write = build_agent_invocation(
            _command(
                agent=EnumCodingAgent.CLAUDE,
                sandbox=EnumAgentSandbox.WORKSPACE_WRITE,
                model="claude-opus-4",
            )
        )
        assert claude_write.argv[claude_write.argv.index("--model") + 1] == (
            "claude-opus-4"
        )
        assert claude_write.stdin == _PROMPT
        assert _PROMPT not in claude_write.argv
