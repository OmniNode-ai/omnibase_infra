# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""Trace context reaches a typed handler through the real CLI and runtime.

Use the in-memory bus and an echo handler in place of omnimarket. Contract
loading, request validation, dispatch, terminal handling and receipt writing
run normally; no model endpoint or deployed lane is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import pytest
from click.testing import CliRunner
from pydantic import Field

from omnibase_infra.cli import cli_delegate
from tests.fixtures.handler_correlated_noop import (
    HandlerCorrelatedNoop,
    ModelCorrelatedNoopRequest,
    ModelDelegateSkillFixtureTerminal,
)
from tests.helpers.cli_registry_stand_in import STAND_IN_INSTALL_TENANT
from tests.helpers.cli_registry_stand_in.node_delegate_stand_in.model_stand_in_delegate_request import (
    ModelStandInDelegateRequest,
)

pytestmark = pytest.mark.integration


class ModelContextEchoRequest(ModelStandInDelegateRequest):
    """Accept the caller metadata the CLI includes on its requests."""

    metadata: dict[str, str] = Field(default_factory=dict)
    session_id: str | None = None


class ModelTraceContextRequest(ModelContextEchoRequest):
    """A closed request contract that declares both trace context fields."""

    trace_id: UUID | None = None
    span_id: UUID | None = None


class HandlerTraceContextEcho:
    """Echo the typed request's trace context through the completed terminal."""

    def handle(
        self, request: ModelStandInDelegateRequest
    ) -> ModelDelegateSkillFixtureTerminal:
        context = request.model_dump(
            mode="json", include={"trace_id", "span_id", "tenant_id"}
        )
        terminal = HandlerCorrelatedNoop().handle(
            ModelCorrelatedNoopRequest(
                correlation_id=str(request.correlation_id),
                prompt=request.prompt,
                task_type=request.task_type,
                requested_timeout_seconds=request.requested_timeout_seconds,
            )
        )
        return terminal.model_copy(update={"response": json.dumps(context)})


@pytest.mark.parametrize("declares_trace", [True, False])
def test_trace_context_survives_dispatch_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declares_trace: bool
) -> None:
    request_model = (
        ModelTraceContextRequest if declares_trace else ModelContextEchoRequest
    )
    model_path = f"{request_model.__module__}.{request_model.__name__}"
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "name: trace_context_echo\n"
        "node_type: compute\n"
        "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
        f"input_model: {model_path}\n"
        "handler:\n"
        f"  module: {__name__}\n"
        "  class: HandlerTraceContextEcho\n"
        f"  input_model: {model_path}\n"
        "handler_routing:\n"
        f"  default_handler: {__name__}:HandlerTraceContextEcho\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OMNI_HOME", raising=False)
    monkeypatch.delenv("ONEX_CONTRACTS_DIR", raising=False)
    monkeypatch.delenv("KAFKA_BOOTSTRAP_SERVERS", raising=False)
    monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
    monkeypatch.setattr(cli_delegate, "_delegate_request_model", lambda: request_model)
    monkeypatch.setattr(cli_delegate, "_resolve_packaged_contract", lambda _: contract)
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.chdir(tmp_path)

    contexts = []
    for invocation in ("first", "second"):
        state_root = tmp_path / invocation
        result = CliRunner().invoke(
            cli_delegate.delegate_command,
            [
                "Summarize this trace context.",
                "--task-type",
                "summarization",
                "--bus",
                "inmemory",
                "--locus",
                "in-process",
                "--state-root",
                str(state_root),
                "--emit-socket",
                str(tmp_path / "no-daemon.sock"),
                "--json",
            ],
            catch_exceptions=False,
        )
        assert result.exit_code == 0, result.output
        (workflow_path,) = (state_root / "runs").glob("*/workflow_result.json")
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        received = json.loads(workflow["handler_result"]["response"])
        (payload_path,) = (state_root / "tmp").glob("delegate-input-*.json")
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        assert received == {
            key: payload[key]
            for key in ("trace_id", "span_id", "tenant_id")
            if key in payload
        }
        assert UUID(received["tenant_id"]) == UUID(STAND_IN_INSTALL_TENANT)
        if declares_trace:
            trace_id, span_id = UUID(received["trace_id"]), UUID(received["span_id"])
            assert trace_id.int != 0 and span_id.int != 0
            assert trace_id != span_id
        else:
            assert "trace_id" not in payload
            assert "span_id" not in payload
            assert set(received) == {"tenant_id"}

        receipt = json.loads(
            (workflow_path.parent / "receipt.json").read_text(encoding="utf-8")
        )
        assert receipt["status"] == "success"
        assert receipt["correlation_id"] == payload["correlation_id"]
        assert (workflow_path.parent / "result.txt").read_text(encoding="utf-8") == (
            workflow["handler_result"]["response"]
        )
        contexts.append(received)

    if declares_trace:
        assert contexts[0]["trace_id"] != contexts[1]["trace_id"]
        assert contexts[0]["span_id"] != contexts[1]["span_id"]
