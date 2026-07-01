# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""Typed-model invariants for the coding-agent workflow (OMN-13247, plan §5.4).

Asserts the strongly-typed policy: frozen, extra=forbid, UUID + enums, and the
system-derived vs agent-reported field separation on the result model.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from omnibase_infra.models.coding_agent import (
    EnumAgentSandbox,
    EnumAgentStatus,
    EnumCliBackendStatus,
    EnumCodingAgent,
    ModelCodingAgentInvokeCommand,
    ModelCodingAgentResult,
)


@pytest.mark.unit
class TestModelCodingAgentInvokeCommand:
    def test_frozen_and_extra_forbid(self) -> None:
        command = ModelCodingAgentInvokeCommand(
            correlation_id=uuid4(),
            agent=EnumCodingAgent.CODEX,
            prompt="x",
            workspace_path="/tmp/ws",  # noqa: S108 # local-path-ok: test fixture
            sandbox=EnumAgentSandbox.READ_ONLY,
            timeout_ms=1000,
        )
        # frozen
        with pytest.raises(ValidationError):
            command.prompt = "mutated"  # type: ignore[misc]
        # extra=forbid
        with pytest.raises(ValidationError):
            ModelCodingAgentInvokeCommand(
                correlation_id=uuid4(),
                agent=EnumCodingAgent.CLAUDE,
                prompt="x",
                workspace_path="/tmp/ws",  # noqa: S108 # local-path-ok: test fixture
                sandbox=EnumAgentSandbox.READ_ONLY,
                timeout_ms=1000,
                unexpected="boom",  # type: ignore[call-arg]
            )

    def test_defaults(self) -> None:
        command = ModelCodingAgentInvokeCommand(
            correlation_id=uuid4(),
            agent=EnumCodingAgent.CLAUDE,
            prompt="x",
            workspace_path="/tmp/ws",  # noqa: S108 # local-path-ok: test fixture
            sandbox=EnumAgentSandbox.WORKSPACE_WRITE,
            timeout_ms=1000,
        )
        assert command.allow_dirty_tree is False
        assert command.network is False
        assert command.model is None

    def test_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            ModelCodingAgentInvokeCommand(
                correlation_id=uuid4(),
                agent=EnumCodingAgent.CLAUDE,
                prompt="x",
                workspace_path="/tmp/ws",  # noqa: S108 # local-path-ok: test fixture
                sandbox=EnumAgentSandbox.READ_ONLY,
                timeout_ms=0,
            )


@pytest.mark.unit
class TestModelCodingAgentResult:
    def test_provenance_fields_present(self) -> None:
        result = ModelCodingAgentResult(
            correlation_id=uuid4(),
            status=EnumAgentStatus.COMPLETED,
            exit_code=0,
            files_changed=("a.py", "b.py"),
            diff="diff",
            diff_hash="hash",
            starting_head_sha="sha",
            error_class=EnumCliBackendStatus.SUCCESS,
            output="agent said it edited files",
            usage={"input_tokens": 10},
        )
        # system-derived (authoritative) and agent-reported (advisory) coexist.
        assert result.files_changed == ("a.py", "b.py")
        assert result.output == "agent said it edited files"
        assert result.usage == {"input_tokens": 10}

    def test_frozen_and_extra_forbid(self) -> None:
        result = ModelCodingAgentResult(
            correlation_id=uuid4(), status=EnumAgentStatus.REJECTED
        )
        with pytest.raises(ValidationError):
            result.status = EnumAgentStatus.COMPLETED  # type: ignore[misc]
        with pytest.raises(ValidationError):
            ModelCodingAgentResult(
                correlation_id=uuid4(),
                status=EnumAgentStatus.FAILED,
                extra_field="boom",  # type: ignore[call-arg]
            )
