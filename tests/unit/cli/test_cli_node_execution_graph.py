# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""One ``onex node`` call runs a whole bus-hop sweep to its terminal (OMN-21071).

The caller's command is ``onex node <orchestrator> --input <file> --timeout <n>``
on the in-memory bus. The orchestrator's handler calls each stage its
``execution_graph`` names as a command on that stage's own topic and awaits the
stage's correlated terminal; no test supplies a consumer of those commands.
So every stage answer here comes from a stage node the runtime hosted for the
orchestrator's run, resolved by name from the registry. With that hosting
removed no stage answers and every test below fails.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from click.testing import CliRunner, Result

from omnibase_core.runtime.runtime_local import RuntimeLocal
from omnibase_infra.cli import contract_registry
from omnibase_infra.cli.cli_node import run_node_by_name
from tests.helpers.execution_graph_stand_in import install_execution_graph_stand_ins
from tests.helpers.execution_graph_stand_in.model_stand_in_sweep import (
    ModelStandInStageResult,
)

pytestmark = pytest.mark.unit

SWEEP = "node_stand_in_sweep_orchestrator"
STALLED_SWEEP = "node_stand_in_stalled_sweep_orchestrator"
CALLER_TIMEOUT_S = 900
# Short enough that the dod-verify base control, where no stage is hosted and
# every wait runs to this deadline, ends inside the per-check budget.
STAGE_DEADLINE_S = 2


def _invoke(
    tmp_path: Path,
    orchestrator: str,
    *,
    stage_deadline_s: float,
    extra: tuple[str, ...] = (),
) -> tuple[Result, uuid.UUID, float]:
    """Run the caller's exact command shape through the CLI."""
    correlation_id = uuid.uuid4()
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "correlation_id": str(correlation_id),
                "operand": "fleet-7",
                "stage_deadline_s": stage_deadline_s,
            }
        )
    )
    started = time.monotonic()
    result = CliRunner().invoke(
        run_node_by_name,
        [
            orchestrator,
            "--input",
            str(request),
            "--timeout",
            str(CALLER_TIMEOUT_S),
            "--state-root",
            str(tmp_path / "state"),
            *extra,
        ],
    )
    return result, correlation_id, time.monotonic() - started


def _terminal(tmp_path: Path) -> dict[str, object]:
    state = json.loads((tmp_path / "state" / "workflow_result.json").read_text())
    payload = state.get("terminal_payload")
    assert isinstance(payload, dict), state
    return payload


def test_one_call_receives_both_stage_terminals_and_exits_with_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_execution_graph_stand_ins(monkeypatch)

    result, correlation_id, _ = _invoke(
        tmp_path, SWEEP, stage_deadline_s=STAGE_DEADLINE_S
    )

    assert result.exit_code == 0, result.output
    terminal = _terminal(tmp_path)
    assert terminal["correlation_id"] == str(correlation_id)
    assert terminal["status"] == "completed", terminal
    assert terminal["stage_values"] == {
        "alpha": "node_stand_in_stage_alpha:fleet-7",
        "beta": "node_stand_in_stage_beta:fleet-7",
    }


def test_an_uncorrelated_stage_terminal_is_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Another sweep's answer on the stage's terminal topic reaches the
    orchestrator first and does not stand in for this sweep's answer."""
    install_execution_graph_stand_ins(monkeypatch)
    alpha = contract_registry.node_contract("node_stand_in_stage_alpha")
    dispatch = alpha["runtime_dispatch"]
    assert isinstance(dispatch, dict)
    alpha_command = str(dispatch["command_topic"])
    alpha_terminal = str(alpha["terminal_event"])
    decoys: list[uuid.UUID] = []
    create_bus = RuntimeLocal._create_event_bus

    def _bus_with_another_sweep(runtime: RuntimeLocal) -> object:
        bus = create_bus(runtime)
        start = bus.start

        async def _start() -> None:
            await start()

            async def _answer_for_another_sweep(_message: object) -> None:
                decoy = ModelStandInStageResult(
                    correlation_id=uuid.uuid4(), stage_key="alpha", value="decoy"
                )
                decoys.append(decoy.correlation_id)
                await bus.publish(
                    alpha_terminal, None, decoy.model_dump_json().encode("utf-8")
                )

            subscribe: Callable[..., Awaitable[object]] = bus.subscribe
            await subscribe(
                alpha_command,
                on_message=_answer_for_another_sweep,
                group_id="another-sweep",
            )

        monkeypatch.setattr(bus, "start", _start)
        return bus

    monkeypatch.setattr(RuntimeLocal, "_create_event_bus", _bus_with_another_sweep)

    result, correlation_id, _ = _invoke(
        tmp_path, SWEEP, stage_deadline_s=STAGE_DEADLINE_S
    )

    assert result.exit_code == 0, result.output
    assert len(decoys) == 1
    terminal = _terminal(tmp_path)
    assert terminal["correlation_id"] == str(correlation_id)
    assert terminal["ignored_terminals"] == 1
    assert terminal["stage_values"] == {
        "alpha": "node_stand_in_stage_alpha:fleet-7",
        "beta": "node_stand_in_stage_beta:fleet-7",
    }


def test_a_stage_terminal_that_never_arrives_is_a_typed_failure_before_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_execution_graph_stand_ins(monkeypatch)

    result, correlation_id, elapsed = _invoke(
        tmp_path, STALLED_SWEEP, stage_deadline_s=2
    )

    assert result.exit_code == 1, result.output
    terminal = _terminal(tmp_path)
    assert terminal["correlation_id"] == str(correlation_id)
    assert terminal["status"] == "failed"
    # Only the stage that never answers is missing: the hosted alpha answered.
    assert terminal["why"] == "stage-terminal-missing:silent"
    assert elapsed < CALLER_TIMEOUT_S / 10


def test_a_stage_node_the_registry_cannot_resolve_fails_the_run_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_execution_graph_stand_ins(
        monkeypatch, withhold=("node_stand_in_stage_beta",)
    )

    result, _, elapsed = _invoke(tmp_path, SWEEP, stage_deadline_s=STAGE_DEADLINE_S)

    assert result.exit_code == 1, result.output
    envelope = json.loads(result.output[result.output.index("{") :])
    assert envelope["error_type"] == "SkillRoutingError"
    assert "execution_graph" in envelope["message"]
    assert "node_stand_in_stage_beta" in envelope["message"]
    assert elapsed < CALLER_TIMEOUT_S / 10


def test_receipt_mode_runs_the_same_sweep_to_its_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller's receipt form (``--output receipt``) hosts the same stages."""
    install_execution_graph_stand_ins(monkeypatch)
    monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))

    result, correlation_id, _ = _invoke(
        tmp_path,
        SWEEP,
        stage_deadline_s=STAGE_DEADLINE_S,
        extra=(
            "--output",
            "receipt",
            "--emit-socket",
            str(tmp_path / "no-daemon.sock"),
        ),
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    assert receipt["exit_code"] == 0
    assert receipt["result"]["correlation_id"] == str(correlation_id)
    assert receipt["result"]["stage_values"] == {
        "alpha": "node_stand_in_stage_alpha:fleet-7",
        "beta": "node_stand_in_stage_beta:fleet-7",
    }
