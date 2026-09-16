# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""OMN-17295: a dispatched run that never produced a terminal says so.

The unit tests pin the join's arms directly. This one takes the whole path —
the real :func:`run_receipt_mode` driving the real ``RuntimeLocal`` in client
mode — for the shape measured live on 2026-09-16 against the ``.201`` dev lane:
the run ends before any terminal exists, so the only thing on disk is
``result`` / ``run_id`` / ``handler_locus``, with no terminal envelope and no
handler result.

Two live runs had exactly this shape (correlations ``e5bc2901`` and
``ea902bf4``, launched concurrently). Their terminal consumer never got a
partition assignment — the broker refused connections for ~30 s — so the bus
aborted with ``Timeout starting consumer ... after 30s`` and the command was
never published. Both receipts reported::

    OMN-17295 correlation-join refusal: no receipt found for this run's
    correlation <cid> — the stored receipt names no correlation id at all.

which is false twice over: nothing was joined, and nothing was stored to name
a correlation. The reporting lane read that as a receipt-join defect under
concurrency. The receipt must instead report the run's own outcome.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from omnibase_infra.cli.receipt_mode import run_receipt_mode

pytestmark = pytest.mark.integration

# A contract the runtime cannot bring up: the handler module does not exist, so
# the run ends before it can publish a command or receive a terminal. Standing
# in for the live cause (an unreachable broker) with something deterministic —
# what is under test is what the RECEIPT says when no terminal exists, not why.
_UNRESOLVABLE_HANDLER_CONTRACT = (
    "---\n"
    "name: proof_never_boots\n"
    "node_type: compute\n"
    "terminal_event: onex.evt.proof.never-arrives.v1\n"
    "handler:\n"
    "  module: tests.fixtures.module_that_does_not_exist\n"
    "  class: HandlerDoesNotExist\n"
    "handler_routing:\n"
    "  default_handler: tests.fixtures.module_that_does_not_exist:HandlerDoesNotExist\n"
)


def test_dispatched_run_with_no_terminal_reports_its_own_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    contract_path = tmp_path / "contract.yaml"
    contract_path.write_text(_UNRESOLVABLE_HANDLER_CONTRACT, encoding="utf-8")
    input_path = tmp_path / "input.json"
    input_path.write_text(
        json.dumps({"name": "omn17295", "count": 1}), encoding="utf-8"
    )
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))

    correlation = uuid.uuid4()
    exit_code = run_receipt_mode(
        node_name="proof_never_boots",
        contract_path=contract_path,
        input_path=input_path,
        state_root=tmp_path / "state",
        backend_overrides={"event_bus": "inmemory"},
        timeout=10,
        verbose=False,
        emit_socket=tmp_path / "no-daemon.sock",
        expected_correlation_id=correlation,
        host_handlers=False,
    )

    assert exit_code != 0, "a run that produced no result must fail loud"
    payload: object = json.loads(capsys.readouterr().out.strip())
    assert isinstance(payload, dict)
    body = payload["result"]
    assert isinstance(body, dict)

    assert payload["correlation_id"] == str(correlation)
    assert body["workflow_result"] == "failed", (
        "the receipt must report the runtime's own outcome, not a join verdict"
    )
    assert body["terminal_payload"] is None
    assert body["handler_result"] is None

    error_text = str(body.get("error", ""))
    assert "correlation-join refusal" not in error_text, (
        "nothing was joined, so nothing can be refused"
    )
    assert "produced no receipt" in error_text
    assert str(correlation) in error_text
    # The capture log is inlined on a non-success receipt, so the real cause
    # travels with the verdict instead of being replaced by it.
    assert "ModuleNotFoundError" in str(body.get("capture_log", ""))
