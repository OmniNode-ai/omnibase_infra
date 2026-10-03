# SPDX-FileCopyrightText: 2025 OmniNode.ai Inc.
# SPDX-License-Identifier: MIT

"""``onex delegate`` records the duration of each client phase (OMN-19452 AC2).

Two layers, because the fields come from two places:

* the receipt writer, driven through the real ``run_delegate`` with a fake
  transport at the receipt-mode seam (the pattern of the rung-pin module), whose
  fake runs the phases the stopwatch it was handed is supposed to observe;
* the real ``run_receipt_mode`` and ``RuntimeLocal`` over the in-memory bus, to
  prove the runtime built for a delegation really routes its bus through the
  timing wrapper -- a fake transport cannot prove that.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

import pytest

from omnibase_core.enums.enum_skill_result_status import EnumSkillResultStatus
from omnibase_core.models.dispatch.model_skill_result import ModelSkillResult
from omnibase_core.protocols.runtime.protocol_local_runtime_bus import (
    UnsubscribeCallback,
)
from omnibase_core.protocols.runtime.protocol_local_runtime_message import (
    ProtocolLocalRuntimeMessage,
)
from omnibase_infra.cli import cli_delegate
from omnibase_infra.cli.cli_delegate import run_delegate
from omnibase_infra.cli.receipt_mode import (
    DelegatePhaseStopwatch,
    DelegatePhaseTimedBus,
    run_receipt_mode,
)
from omnibase_infra.enums.enum_delegate_locus import EnumDelegateLocus
from omnibase_infra.runtime_identity import collect_runtime_identity

pytestmark = pytest.mark.integration

_RESULT_MODEL = (
    "omnimarket.models.delegation.wire."
    "model_delegate_skill_response.ModelDelegateSkillCompleted"
)
_FIELDS = (
    "startup_seconds",
    "locus_probe_seconds",
    "bus_connect_seconds",
    "reply_subscribe_seconds",
    "publish_seconds",
    "terminal_wait_seconds",
)
_STEP = 0.02

_CONTRACT = """\
name: node_delegate_skill_orchestrator
terminal_event: onex.evt.omnimarket.delegate-skill-completed.v1
event_bus:
  publish_topics:
    - onex.evt.omnimarket.delegate-skill-completed.v1
  subscribe_topics:
    - onex.cmd.omnimarket.delegate-skill.v1
"""


class _SlowFakeBus:
    """A fake transport: every call takes a measurable, fixed time."""

    async def start(self) -> None:
        await asyncio.sleep(_STEP)

    async def close(self) -> None:
        await asyncio.sleep(0)

    async def publish(self, topic: str, key: object, value: bytes) -> object:
        await asyncio.sleep(_STEP)
        return None

    async def subscribe(
        self,
        topic: str,
        *,
        on_message: Callable[[ProtocolLocalRuntimeMessage], Awaitable[None]],
        group_id: str,
    ) -> UnsubscribeCallback:
        await asyncio.sleep(_STEP)

        async def _unsubscribe() -> None:
            return None

        return _unsubscribe


async def _run_the_client_phases(stopwatch: DelegatePhaseStopwatch) -> None:
    async def _noop(_: object) -> None:
        return None

    bus = DelegatePhaseTimedBus(_SlowFakeBus(), stopwatch)
    await bus.start()
    await bus.subscribe("terminal", on_message=_noop, group_id="run")
    await bus.publish("command", None, b"{}")
    await asyncio.sleep(_STEP)
    await bus.close()


def _receipt() -> ModelSkillResult[dict[str, object]]:
    return ModelSkillResult(
        skill_name="node_delegate_skill_orchestrator",
        node_name="node_delegate_skill_orchestrator",
        status=EnumSkillResultStatus.SUCCESS,
        correlation_id=uuid4(),
        run_id=uuid4(),
        exit_code=0,
        duration_ms=1200,
        result={
            "status": "completed",
            "task_type": "research",
            "model_name": "synthetic-model",
            "provider": "cheap_cloud",
            "response": "OK",
            "attempts": [
                {
                    "tier": "cheap_cloud",
                    "backend_id": "synthetic-backend",
                    "model_id": "synthetic-model",
                    "quality_gate_passed": True,
                    "quality_score": 1.0,
                    "cost_usd": 0.0,
                    "failure_class": None,
                    "error_message": "",
                    "acceptance_decision": "accept",
                    "acceptance_reason": "quality_bar_met",
                    "substituted_from_backend_id": None,
                }
            ],
            "terminal_failure_cause": None,
        },
        result_model=_RESULT_MODEL,
        runtime_identity=collect_runtime_identity(config_source="test"),
    )


class TestTheReceiptRecordsEveryClientPhase:
    def test_all_six_duration_fields_are_in_receipt_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract = tmp_path / "contract.yaml"
        contract.write_text(_CONTRACT, encoding="utf-8")
        receipt = _receipt()

        def _fake_receipt_mode(**kwargs: object) -> int:
            stopwatch = kwargs["phase_stopwatch"]
            assert isinstance(stopwatch, DelegatePhaseStopwatch)
            asyncio.run(_run_the_client_phases(stopwatch))
            callback = kwargs["receipt_callback"]
            assert callable(callback)
            callback(receipt)
            return 0

        monkeypatch.delenv("OMNI_HOME", raising=False)
        monkeypatch.setattr(cli_delegate, "check_omnimarket_drift", lambda **_: None)
        monkeypatch.setattr(
            cli_delegate, "_resolve_packaged_contract", lambda _name: contract
        )
        monkeypatch.setattr(cli_delegate, "run_receipt_mode", _fake_receipt_mode)

        exit_code = run_delegate(
            prompt="probe",
            task_type="research",
            max_tokens=None,
            bus="inmemory",
            locus=EnumDelegateLocus.IN_PROCESS,
            state_root=tmp_path / "state",
            timeout=5,
            verbose=False,
            emit_socket=tmp_path / "emit.sock",
        )

        assert exit_code == 0
        written = json.loads(
            (
                tmp_path / "state" / "runs" / str(receipt.run_id) / "receipt.json"
            ).read_text(encoding="utf-8")
        )
        durations = written["phase_durations"]
        assert set(durations) == set(_FIELDS)
        for field in _FIELDS:
            assert isinstance(durations[field], float), f"{field} was not recorded"
            assert durations[field] >= 0.0
        # The fake transport spends _STEP in each bus call and in the wait, so a
        # field that merely defaulted to zero would not clear this bound.
        for field in (
            "bus_connect_seconds",
            "reply_subscribe_seconds",
            "publish_seconds",
            "terminal_wait_seconds",
        ):
            assert durations[field] >= _STEP * 0.9, field


class TestTheRealRuntimeRoutesItsBusThroughTheStopwatch:
    def test_the_in_memory_runtime_reports_its_bus_phases(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        contract = tmp_path / "contract.yaml"
        contract.write_text(
            "---\n"
            "name: correlated_noop\n"
            "node_type: compute\n"
            "terminal_event: onex.evt.proof.correlated-noop-completed.v1\n"
            "handler:\n"
            "  module: tests.fixtures.handler_correlated_noop\n"
            "  class: HandlerCorrelatedNoop\n"
            "  input_model: tests.fixtures.handler_correlated_noop"
            ".ModelCorrelatedNoopRequest\n"
            "handler_routing:\n"
            "  default_handler: tests.fixtures.handler_correlated_noop"
            ":HandlerCorrelatedNoop\n",
            encoding="utf-8",
        )
        payload = tmp_path / "payload.json"
        payload.write_text(json.dumps({"correlation_id": str(uuid4())}))
        monkeypatch.setenv("ONEX_ARTIFACT_STORE_ROOT", str(tmp_path / "artifacts"))
        stopwatch = DelegatePhaseStopwatch()

        run_receipt_mode(
            node_name="correlated_noop",
            contract_path=contract,
            input_path=payload,
            state_root=tmp_path / "state",
            backend_overrides={},
            timeout=5,
            verbose=False,
            emit_socket=tmp_path / "no-daemon.sock",
            phase_stopwatch=stopwatch,
        )

        durations = stopwatch.durations()
        assert durations.bus_connect_seconds is not None
        assert durations.reply_subscribe_seconds is not None
        assert durations.publish_seconds is not None
        assert durations.terminal_wait_seconds is not None
