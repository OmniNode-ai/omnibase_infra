# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""The real timed runtime injects the untimed in-memory bus (OMN-20381)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_infra.cli.receipt_mode import DelegatePhaseStopwatch, run_receipt_mode

pytestmark = pytest.mark.integration


def test_in_process_handler_receives_the_untimed_in_memory_bus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        "---\n"
        "name: correlated_noop\n"
        "node_type: compute\n"
        "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
        "handler:\n"
        "  module: tests.fixtures.handler_event_bus_probe\n"
        "  class: HandlerEventBusProbe\n"
        "  input_model: tests.fixtures.handler_event_bus_probe"
        ".ModelCorrelatedNoopRequest\n"
        "handler_routing:\n"
        "  default_handler: tests.fixtures.handler_event_bus_probe"
        ":HandlerEventBusProbe\n",
        encoding="utf-8",
    )
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"correlation_id": str(uuid4())}))
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
    stopwatch = DelegatePhaseStopwatch()
    state_root = tmp_path / "state"

    exit_code = run_receipt_mode(
        node_name="correlated_noop",
        contract_path=contract,
        input_path=payload,
        state_root=state_root,
        backend_overrides={},
        timeout=5,
        verbose=False,
        emit_socket=tmp_path / "no-daemon.sock",
        phase_stopwatch=stopwatch,
    )

    assert exit_code == 0
    result_files = list((state_root / "runs").glob("*/workflow_result.json"))
    assert len(result_files) == 1
    workflow_result = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert workflow_result["handler_result"]["response"] == "EventBusInmemory"

    durations = stopwatch.durations()
    assert durations.bus_connect_seconds is not None
    assert durations.reply_subscribe_seconds is not None
    assert durations.publish_seconds is not None
    assert durations.terminal_wait_seconds is not None
