# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT
"""``onex delegate`` stamps its request with a trace id and a span id (OMN-19437).

The delegation terminal carried no trace or span id (every ``delegation_events``
trace_id read empty), so a run could not be joined to the trace that issued it.
The CLI is the publisher, so it mints both ids and writes them beside the tenant
it already stamps; the handler copies them onto the terminal.

What these tests pin:

* the request the publish path writes carries a non-null trace id and span id,
  both UUIDs, and a UUID tenant;
* each invocation mints its own ids;
* a contract whose request model does not yet declare the fields (an omnimarket
  older than the release that declares them) gets a request without them, so
  the command still validates there instead of being refused as an extra key.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import _write_payload, run_delegate
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from tests.helpers.cli_registry_stand_in import STAND_IN_INSTALL_TENANT
from tests.helpers.cli_registry_stand_in.node_delegate_stand_in.model_stand_in_delegate_request import (
    ModelStandInDelegateRequest,
)

pytestmark = pytest.mark.unit


class _ModelRequestDeclaringTraceContext(ModelStandInDelegateRequest):
    """The delegate request model once its release declares the trace context."""

    trace_id: UUID | None = None
    span_id: UUID | None = None


def _run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, declares_trace: bool
) -> dict[str, object]:
    """Run the command with the receipt layer faked; return the request it wrote."""
    captured: dict[str, object] = {}

    def _fake_run_receipt_mode(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    if declares_trace:
        monkeypatch.setattr(
            cli_delegate,
            "_delegate_request_model",
            lambda: _ModelRequestDeclaringTraceContext,
        )
    monkeypatch.setattr(
        cli_delegate,
        "_resolve_packaged_contract",
        lambda _name: tmp_path / "contract.yaml",
    )
    monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_run_receipt_mode)
    run_delegate(
        prompt="document the router",
        task_type="document",
        max_tokens=None,
        bus="kafka",
        locus=EnumDelegateLocus.IN_PROCESS,
        kafka_bootstrap="localhost:19092",
        state_root=tmp_path / "state",
        timeout=60,
        verbose=False,
        emit_socket=tmp_path / "no-daemon.sock",
    )
    input_path = captured["input_path"]
    assert isinstance(input_path, Path)
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


class TestPayload:
    def test_the_ids_are_written_as_trace_id_and_span_id(self, tmp_path: Path) -> None:
        trace_id, span_id = uuid4(), uuid4()
        path = _write_payload(
            prompt="Reply with exactly: OK",
            task_type="summarization",
            source="claude-code",
            state_root=tmp_path,
            run_id=uuid4(),
            correlation_id=uuid4(),
            max_tokens=None,
            trace_id=trace_id,
            span_id=span_id,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["trace_id"] == str(trace_id)
        assert payload["span_id"] == str(span_id)

    def test_no_ids_leave_both_fields_out_entirely(self, tmp_path: Path) -> None:
        path = _write_payload(
            prompt="Reply with exactly: OK",
            task_type="summarization",
            source="claude-code",
            state_root=tmp_path,
            run_id=uuid4(),
            correlation_id=uuid4(),
            max_tokens=None,
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "trace_id" not in payload
        assert "span_id" not in payload


class TestPublishPath:
    def test_the_published_request_carries_trace_span_and_a_uuid_tenant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _run(tmp_path, monkeypatch, declares_trace=True)

        trace_id = UUID(str(payload["trace_id"]))
        span_id = UUID(str(payload["span_id"]))
        assert trace_id != span_id
        assert UUID(str(payload["tenant_id"])) == UUID(STAND_IN_INSTALL_TENANT)

    def test_the_request_still_validates_against_the_contract_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = _run(tmp_path, monkeypatch, declares_trace=True)
        _ModelRequestDeclaringTraceContext.model_validate(payload)

    def test_each_invocation_mints_its_own_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        first = _run(tmp_path / "first", monkeypatch, declares_trace=True)
        second = _run(tmp_path / "second", monkeypatch, declares_trace=True)
        assert first["trace_id"] != second["trace_id"]
        assert first["span_id"] != second["span_id"]

    def test_a_model_that_does_not_declare_the_fields_gets_a_request_without_them(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Control: an older consumer would refuse the key, so none is sent."""
        payload = _run(tmp_path, monkeypatch, declares_trace=False)
        assert "trace_id" not in payload
        assert "span_id" not in payload
        ModelStandInDelegateRequest.model_validate(payload)
